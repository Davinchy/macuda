// Turning a cubin into the block that goes into device memory.
//
// A cubin is not a loadable thing. The kernel's code, each constant bank and any __device__ data are separate ELF
// sections, and the addresses the GPU is handed - the program address in the descriptor, each constant buffer's address -
// are offsets into one contiguous block that has to be built first. Sections that declare an address are placed at it;
// the rest are appended in the order they appear, aligned, and told where they landed.
//
// Then the relocations. A cubin's code needs none - every kernel this driver has looked at has an empty .rela.text - but
// its constant banks do: a kernel using a __device__ lookup table reads the table's address out of a constant bank, and
// that address is only known once the block has one. Every quantised llama.cpp kernel of the IQ family works this way,
// eleven pointers each into a 26 KB table. Skipping this stage does not fail to load; it launches a kernel that reads a
// table at address zero.
//
// The layout must match the python oracle's exactly, because the numbers in the kernel descriptor are offsets into it
// and the descriptor is compared against the oracle's byte for byte. test_image checks that against the oracle's own
// elf_loader on the same cubins.
#include "cubin.h"
#include "internal.h"
#include <stdlib.h>
#include <string.h>

#define SHT_PROGBITS 1
#define SHT_SYMTAB 2
#define SHT_RELA 4
#define SHT_REL 9
#define CUDA_SECTION_ALIGN 128 // what the oracle loads cubins with, and a section may ask for more

// the CUDA relocation types that appear in a cubin, and what each one writes
#define R_CUDA_64 2         // the whole address, eight bytes
#define R_CUDA_32_LO 0x38   // its low half, four bytes, four bytes into the record's target
#define R_CUDA_32_HI 0x39   // its high half, likewise

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }
static void wr32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void wr64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }

// The name of the symbol a relocation names, from the string table its symbol table points at.
static const char *strtab_name(const tinynv_cubin_t *c, const tinynv_section_t *symtab, const uint8_t *sym) {
  if (symtab->link >= (uint32_t)c->nsections) return NULL;
  const tinynv_section_t *str = &c->sections[symtab->link];
  uint32_t at = rd32(sym);
  if (at >= str->size) return NULL;
  const char *n = (const char *)c->img + str->off + at;
  return memchr(n, 0, (size_t)(str->size - at)) ? n : NULL;   // must be terminated inside the table
}

static const char *after_prefix(const char *s, const char *prefix) {
  size_t n = strlen(prefix);
  return strncmp(s, prefix, n) ? NULL : s + n;
}

// ".nv.constant<N>" or ".nv.constant<N>.<kernel>": the bank number, or -1 if this is not a constant bank.
static int constbank_of(const char *name, const char **rest) {
  const char *p = after_prefix(name, ".nv.constant");
  if (!p || *p < '0' || *p > '9') return -1;
  int n = 0;
  while (*p >= '0' && *p <= '9') n = n * 10 + (*p++ - '0');
  *rest = p;
  return n;
}

static int grow(uint8_t **buf, size_t *len, size_t *cap, size_t want) {
  if (want <= *cap) return 0;
  size_t cap2 = *cap ? *cap : 4096;
  while (cap2 < want) cap2 *= 2;
  uint8_t *p = realloc(*buf, cap2);
  if (!p) return tinynv_fail("out of memory for a %zu byte image", cap2);
  memset(p + *len, 0, cap2 - *len);
  *buf = p;
  *cap = cap2;
  return 0;
}

int tinynv_cubin_image(const tinynv_cubin_t *c, const char *kernel, int with_bytes, tinynv_image_t *out) {
  memset(out, 0, sizeof(*out));
  if (!c->sections) return tinynv_fail("cubin: no section table; parse it first");

  // A private copy of the addresses: laying out assigns one to every section that had none, and the cubin itself stays
  // as it was read so a second kernel from the same cubin lays out from the same starting point.
  uint64_t *addr = calloc((size_t)c->nsections, sizeof(uint64_t));
  if (!addr) return tinynv_fail("out of memory for %d section addresses", c->nsections);
  for (int i = 0; i < c->nsections; i++) addr[i] = c->sections[i].addr;

  // Sections that declare an address fix the block's size before anything is appended.
  size_t len = 0, cap = 0;
  uint8_t *bytes = NULL;
  for (int i = 0; i < c->nsections; i++)
    if (c->sections[i].type == SHT_PROGBITS && addr[i] && addr[i] + c->sections[i].size > len) len = addr[i] + c->sections[i].size;
  if (with_bytes && grow(&bytes, &len, &cap, len)) { free(addr); return -1; }

  for (int i = 0; i < c->nsections; i++) {
    const tinynv_section_t *s = &c->sections[i];
    if (s->type != SHT_PROGBITS) continue;
    if (addr[i]) {
      if (with_bytes) memcpy(bytes + addr[i], c->img + s->off, s->size);
      continue;
    }
    uint64_t align = s->align > CUDA_SECTION_ALIGN ? s->align : CUDA_SECTION_ALIGN;
    size_t pad = (size_t)((align - len % align) % align);
    if (with_bytes && grow(&bytes, &len, &cap, len + pad + s->size)) { free(addr); free(bytes); return -1; }
    len += pad;
    if (with_bytes) memcpy(bytes + len, c->img + s->off, s->size);
    addr[i] = len;      // and this is what the relocations and the descriptor will refer to
    len += s->size;
  }

  // Where this kernel's code and constant banks ended up. Bank 0 belongs to the kernel and is named after it; the higher
  // banks belong to the module. The oracle takes whichever ".nv.constant0.*" it meets last, which is right for a cubin
  // holding one kernel and wrong for ours, where a single ggml cubin holds thousands - so bank 0 is matched by name.
  int found = 0;
  for (int i = 0; i < c->nsections; i++) {
    const char *rest, *sname = c->sections[i].sname;
    if (c->sections[i].type != SHT_PROGBITS) continue;
    if ((rest = after_prefix(sname, ".text.")) && !strcmp(rest, kernel)) { out->text_off = addr[i]; found = 1; }
    int bank = constbank_of(sname, &rest);
    if (bank < 0 || bank >= TINYNV_MAX_CONSTBUFS) continue;
    if (*rest && (*rest != '.' || strcmp(rest + 1, kernel))) continue; // another kernel's bank
    if (bank == 0 && !*rest) continue;                                 // a bank 0 with no kernel name is not one
    out->constbuf[bank].off = addr[i];
    out->constbuf[bank].size = c->sections[i].size;
    out->constbuf[bank].used = 1;
  }
  if (!found) { free(addr); free(bytes); return tinynv_fail("cubin: no code for a kernel called %s", kernel); }

  // The relocations, resolved to image offsets on both ends: where to write, and what to write there. A layout-only
  // request stops here: they are a property of the image, applied once when it is uploaded, and a caller asking where a
  // second kernel's code sits has no use for another copy of them.
  int cap_rel = 0;
  if (!with_bytes) { free(addr); out->len = len; return 0; }
  for (int i = 0; i < c->nsections; i++) {
    const tinynv_section_t *s = &c->sections[i];
    if (s->type != SHT_RELA && s->type != SHT_REL) continue;
    uint64_t ent = s->type == SHT_RELA ? 24 : 16;
    const char *target = after_prefix(s->sname, s->type == SHT_RELA ? ".rela" : ".rel");
    if (!target || !strcmp(target, ".eh_frame")) continue; // the oracle skips these: they relocate debug data, not code
    int t = -1;
    for (int j = 0; j < c->nsections; j++) if (!strcmp(c->sections[j].sname, target)) { t = j; break; }
    if (t < 0) continue;

    // the symbol table this relocation section indexes into
    if (s->link >= (uint32_t)c->nsections || c->sections[s->link].type != SHT_SYMTAB) continue;
    const tinynv_section_t *symtab = &c->sections[s->link];

    for (uint64_t off = 0; off + ent <= s->size; off += ent) {
      const uint8_t *r = c->img + s->off + off;
      uint64_t r_offset = rd64(r), r_info = rd64(r + 8);
      int64_t addend = s->type == SHT_RELA ? (int64_t)rd64(r + 16) : 0;
      uint32_t sym = (uint32_t)(r_info >> 32), type = (uint32_t)r_info;
      if ((uint64_t)sym * 24 + 24 > symtab->size) return tinynv_fail("cubin: relocation names symbol %u, past the table", sym);
      const uint8_t *sy = c->img + symtab->off + (uint64_t)sym * 24;
      uint16_t shndx = (uint16_t)(sy[6] | (sy[7] << 8));
      uint64_t target = 0;
      if (shndx && shndx < (uint16_t)c->nsections) target = addr[shndx] + rd64(sy + 8);
      else {
        // An undefined symbol: something the cubin references and does not contain. In a cubin there is exactly one
        // kind, and it is worth naming rather than refusing.
        //
        // Every one of the 183 cubins in a full ggml build was swept, and the only undefined symbol any of them
        // relocates against is `vprintf` - 115 times. It is CUDA's device side printf, reached from assert paths, and
        // the real driver supplies it out of its device runtime. This one does not have a device runtime, so there is
        // nothing truthful to point it at: it resolves to zero, which turns a kernel that actually prints into an
        // immediate fault on a null call rather than into a jump somewhere arbitrary.
        //
        // Refusing instead - which is what this did until a real op suite hit it - blocks every quantised model,
        // because one assert path in one header costs the entire decode path. Anything else undefined is still refused,
        // and now by name, because the next one might matter.
        const char *name = strtab_name(c, symtab, sy);
        if (!name || strcmp(name, "vprintf"))
          return tinynv_fail("cubin: relocation against '%s', a symbol this driver cannot resolve",
                             name && *name ? name : "an unnamed undefined symbol");
        out->unresolved++;
      }

      if (out->nrelocs == cap_rel) {
        cap_rel = cap_rel ? cap_rel * 2 : 16;
        tinynv_reloc_t *p = realloc(out->relocs, (size_t)cap_rel * sizeof(*p));
        if (!p) { free(addr); free(bytes); tinynv_image_free(out); return tinynv_fail("out of memory for relocations"); }
        out->relocs = p;
      }
      out->relocs[out->nrelocs++] = (tinynv_reloc_t){.at = addr[t] + r_offset, .target = target,
                                                    .type = type, .addend = addend};
    }
  }

  free(addr);
  out->bytes = bytes;
  out->len = len;
  return 0;
}

int tinynv_image_relocate(tinynv_image_t *im, uint64_t image_va) {
  for (int i = 0; i < im->nrelocs; i++) {
    const tinynv_reloc_t *r = &im->relocs[i];
    uint64_t v = image_va + r->target + (uint64_t)r->addend;
    switch (r->type) {
      case R_CUDA_64:
        if (r->at + 8 > im->len) return tinynv_fail("relocation at %#llx is past the image", (unsigned long long)r->at);
        wr64(im->bytes + r->at, v);
        break;
      // these two write into the word after the one the record names, which is the instruction's immediate field
      case R_CUDA_32_LO:
      case R_CUDA_32_HI:
        if (r->at + 8 > im->len) return tinynv_fail("relocation at %#llx is past the image", (unsigned long long)r->at);
        wr32(im->bytes + r->at + 4, (uint32_t)(r->type == R_CUDA_32_HI ? v >> 32 : v));
        break;
      default: return tinynv_fail("relocation type %#x is one this driver does not know", r->type);
    }
  }
  return 0;
}

void tinynv_image_free(tinynv_image_t *im) {
  free(im->bytes);
  free(im->relocs);
  memset(im, 0, sizeof(*im));
}
