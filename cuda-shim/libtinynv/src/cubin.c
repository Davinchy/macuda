// Cubin reader. A cubin is an ELF64 for EM_CUDA whose interesting parts are its attribute records: where in constant bank 0
// the kernel's parameters live, how wide each one is, how many registers the kernel needs. All of that is per cubin and
// per architecture - the parameter base is 0x160 on sm_8x and 0x380 on sm_12x - so it is always read, never assumed.
#include "cubin.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// attribute records: u8 format, u8 attribute, u16 (a value, or the length of the payload that follows)
#define EIFMT_HVAL 0x03
#define EIFMT_SVAL 0x04
#define EIATTR_PARAM_CBANK 0x0a
#define EIATTR_CBANK_PARAM_SIZE 0x19
#define EIATTR_KPARAM_INFO 0x17
#define EIATTR_MAX_THREADS 0x05
#define EIATTR_REGCOUNT 0x2f
#define EIATTR_MIN_STACK_SIZE 0x12

typedef tinynv_section_t sec_t;

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

static const char *str_at(const uint8_t *img, size_t len, uint64_t tab_off, uint64_t tab_size, uint32_t idx) {
  if (tab_off + tab_size > len || idx >= tab_size) return NULL;
  const char *s = (const char *)img + tab_off + idx;
  return memchr(s, 0, tab_size - idx) ? s : NULL; // must be terminated inside the table
}

// one kernel's per-function records
#define SHT_PROGBITS 1

// a kernel's machine code: the right section type, and the name starts exactly there rather than merely containing it
// the tail of a section name after a prefix, or NULL if it does not start with it
static const char *after_prefix(const char *s, const char *prefix) {
  size_t n = strlen(prefix);
  return strncmp(s, prefix, n) ? NULL : s + n;
}

static int is_kernel_text(const sec_t *s) { return s->type == SHT_PROGBITS && !strncmp(s->sname, ".text.", 6); }

static int parse_kernel_info(tinynv_kernel_desc_t *k, const uint8_t *p, const uint8_t *end) {
  while (p + 4 <= end) {
    uint8_t fmt = p[0], attr = p[1];
    uint16_t val = rd16(p + 2);
    const uint8_t *payload = p + 4;
    size_t plen = fmt == EIFMT_SVAL ? val : 0;
    if (payload + plen > end) return tinynv_fail("cubin: attribute %#x runs past its section", attr);

    if (attr == EIATTR_PARAM_CBANK && plen >= 8) {
      k->param_base = rd16(payload + 4); // and rd16(payload+6) is the size, which CBANK_PARAM_SIZE repeats
      if (!k->param_size) k->param_size = rd16(payload + 6);
    } else if (attr == EIATTR_CBANK_PARAM_SIZE && fmt == EIFMT_HVAL) {
      k->param_size = val;
    } else if (attr == EIATTR_KPARAM_INFO && plen >= 12) {
      // u32 index, u16 ordinal, u16 offset, then a packed word: logAlignment:8, space:4, cbank:6, size:...
      uint16_t ordinal = rd16(payload + 4), offset = rd16(payload + 6);
      uint32_t bits = rd32(payload + 8);
      if (ordinal >= TINYNV_MAX_PARAMS) return tinynv_fail("cubin: parameter %u is beyond the %d we handle", ordinal, TINYNV_MAX_PARAMS);
      k->params[ordinal].offset = offset;
      k->params[ordinal].size = bits >> 18;
      if (ordinal + 1 > (uint32_t)k->nparams) k->nparams = ordinal + 1;
    } else if (attr == EIATTR_MAX_THREADS && plen >= 12) {
      k->max_threads = rd32(payload) * rd32(payload + 4) * rd32(payload + 8);
    }
    p = payload + plen;
  }
  return 0;
}

int tinynv_cubin_parse(const void *data, size_t len, tinynv_cubin_t *out) {
  const uint8_t *img = data;
  memset(out, 0, sizeof(*out));
  if (len < 0x40 || memcmp(img, "\177ELF", 4) != 0 || img[4] != 2 || img[5] != 1)
    return tinynv_fail("cubin: not a little endian 64 bit elf");
  if (rd16(img + 0x12) != 190) return tinynv_fail("cubin: e_machine %u is not EM_CUDA", rd16(img + 0x12));

  out->img = img;
  out->len = len;
  out->sm_arch = (rd32(img + 0x30) >> 8) & 0xff; // e_flags holds the sm number in its second byte: 0x...5604 -> 86, 0x...7802 -> 120

  uint64_t shoff = rd64(img + 0x28);
  uint16_t shent = rd16(img + 0x3a), shnum = rd16(img + 0x3c), shstrndx = rd16(img + 0x3e);
  if (!shnum || shstrndx >= shnum || shoff + (uint64_t)shent * shnum > len) return tinynv_fail("cubin: section headers are out of range");

  sec_t *sec = calloc(shnum, sizeof(sec_t));
  uint64_t shstr_off = rd64(img + shoff + (uint64_t)shstrndx * shent + 0x18), shstr_size = rd64(img + shoff + (uint64_t)shstrndx * shent + 0x20);
  for (int i = 0; i < shnum; i++) {
    const uint8_t *sh = img + shoff + (uint64_t)i * shent;
    sec[i] = (sec_t){.name = rd32(sh), .type = rd32(sh + 4), .flags = rd64(sh + 8), .addr = rd64(sh + 0x10),
                     .off = rd64(sh + 0x18), .size = rd64(sh + 0x20), .link = rd32(sh + 0x28), .info = rd32(sh + 0x2c),
                     .align = rd64(sh + 0x30), .entsize = rd64(sh + 0x38)};
    sec[i].sname = str_at(img, len, shstr_off, shstr_size, sec[i].name);
    if (!sec[i].sname) sec[i].sname = "";
    if (sec[i].type != 8 /* NOBITS */ && sec[i].off + sec[i].size > len) { free(sec); return tinynv_fail("cubin: section %d is out of range", i); }
  }

  // every .text.<name> holding machine code is a kernel. the type test comes first on purpose: a cubin also carries
  // vendor sections whose names contain "text" (Mercury intermediate code), and they are not kernels.
  for (int i = 0; i < shnum; i++) if (is_kernel_text(&sec[i])) out->nkernels++;
  if (!out->nkernels) { free(sec); return tinynv_fail("cubin: no kernels"); }
  out->kernels = calloc(out->nkernels, sizeof(tinynv_kernel_desc_t));

  int n = 0;
  for (int i = 0; i < shnum; i++) {
    if (!is_kernel_text(&sec[i])) continue;
    tinynv_kernel_desc_t *k = &out->kernels[n++];
    if (!(k->name = strdup(sec[i].sname + 6))) { free(sec); tinynv_cubin_free(out); return tinynv_fail("cubin: out of memory"); }
    k->text_off = sec[i].off;
    k->text_size = sec[i].size;

    // The sections describing this kernel are named by prefix and then by the kernel's name. Match them by walking past
    // the prefix and comparing the remainder, rather than by building the name to compare against: a template
    // instantiation's name has no length anyone should be predicting, and the buffer that used to do this truncated.
    for (int j = 0; j < shnum; j++) {
      const char *rest;
      if ((rest = after_prefix(sec[j].sname, ".nv.constant0.")) && !strcmp(rest, k->name)) {
        k->const0_off = sec[j].off;
        k->const0_size = sec[j].size;
      }
      if ((rest = after_prefix(sec[j].sname, ".nv.shared.")) && !strcmp(rest, k->name)) k->static_smem = (uint32_t)sec[j].size;
      if ((rest = after_prefix(sec[j].sname, ".nv.info.")) && !strcmp(rest, k->name) &&
          parse_kernel_info(k, img + sec[j].off, img + sec[j].off + sec[j].size)) {
        free(sec);
        tinynv_cubin_free(out);
        return -1;
      }
    }
  }

  // the module wide .nv.info carries register and stack counts, addressed by symbol index
  for (int i = 0; i < shnum; i++) {
    if (strcmp(sec[i].sname, ".nv.info")) continue;
    const uint8_t *p = img + sec[i].off, *end = p + sec[i].size;
    while (p + 4 <= end) {
      uint8_t fmt = p[0], attr = p[1];
      uint16_t val = rd16(p + 2);
      const uint8_t *payload = p + 4;
      size_t plen = fmt == EIFMT_SVAL ? val : 0;
      if (payload + plen > end) break;
      if ((attr == EIATTR_REGCOUNT || attr == EIATTR_MIN_STACK_SIZE) && plen >= 8) {
        // resolve the symbol index against .symtab so a multi kernel cubin attributes these correctly
        uint32_t sym = rd32(payload), count = rd32(payload + 4);
        for (int s = 0; s < shnum; s++) {
          if (sec[s].type != 2 /* SYMTAB */ || sec[s].link >= (uint32_t)shnum) continue;
          uint64_t entsz = 24, nsyms = sec[s].size / entsz;
          if (sym >= nsyms) continue;
          const char *nm = str_at(img, len, sec[sec[s].link].off, sec[sec[s].link].size, rd32(img + sec[s].off + sym * entsz));
          for (int kk = 0; nm && kk < out->nkernels; kk++)
            if (!strcmp(out->kernels[kk].name, nm)) {
              if (attr == EIATTR_REGCOUNT) out->kernels[kk].regs = count;
              else out->kernels[kk].min_stack = count;
            }
        }
      }
      p = payload + plen;
    }
  }

  out->nsections = shnum;
  out->sections = sec;
  return 0;
}

void tinynv_cubin_free(tinynv_cubin_t *c) {
  for (int i = 0; i < c->nkernels; i++) free(c->kernels[i].name);
  free(c->kernels);
  free(c->sections);
  c->kernels = NULL;
  c->sections = NULL;
  c->nkernels = c->nsections = 0;
}

const tinynv_kernel_desc_t *tinynv_cubin_kernel(const tinynv_cubin_t *c, const char *name) {
  for (int i = 0; i < c->nkernels; i++) if (!strcmp(c->kernels[i].name, name)) return &c->kernels[i];
  tinynv_fail("cubin: no kernel named %s", name);
  return NULL;
}
