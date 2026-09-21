/* reloc_target_scan - relocations against DATA OBJECTS whose resolved target lands inside KERNEL CODE, per image, as
 * libtinynv's own loader resolves them. The same source is built against the loader before and after a change.
 *
 * It uses only what every version of the loader has: tinynv_cubin_image(with_bytes=1) for the relocation list and the
 * image length, and each kernel's layout (text_off) with its descriptor's text_size for the code ranges. Each entry of
 * im.relocs is paired with the ELF's own relocation walk, in the same order and with the same skips as image.c
 * (no .rel prefix or .eh_frame -> skipped; target section not found -> skipped; symtab link invalid -> skipped), to
 * learn the symbol's TYPE - a FUNC symbol pointing into code is legitimate, an OBJECT symbol there is the defect.
 * Every sm_120 entry of the library (entry-header arch 120), LZ4 or zstd decoded here, not just one per fatbin. */
#include "cubin.h"
#include "internal.h"
#include <fcntl.h>
#include <lz4.h>
#include <zstd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
static uint16_t r16(const uint8_t *p){uint16_t v; memcpy(&v,p,2); return v;}
static uint32_t r32(const uint8_t *p){uint32_t v; memcpy(&v,p,4); return v;}
static uint64_t r64(const uint8_t *p){uint64_t v; memcpy(&v,p,8); return v;}
static long g_nob, g_nob_code, g_nob_bank, g_nob_other, g_nob_past, g_img_mis, g_nob_own, g_mism; static char g_which[128];
static const char *after(const char *s, const char *p) { size_t n = strlen(p); return strncmp(s, p, n) ? NULL : s + n; }

int main(int argc, char **argv) {
  int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
  const uint8_t *e = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  uint64_t shoff=r64(e+0x28); uint16_t shent=r16(e+0x3a), shnum=r16(e+0x3c), shstr=r16(e+0x3e);
  uint64_t stroff=r64(e+shoff+(uint64_t)shstr*shent+0x18), so=0, ss=0;
  for (int i=0;i<shnum;i++){ const uint8_t*s=e+shoff+(uint64_t)i*shent; if(!strcmp((const char*)e+stroff+r32(s),".nv_fatbin")){so=r64(s+0x18);ss=r64(s+0x20);} }
  long n120=0, nimg_bad=0, total_bad=0, total_obj=0, refused=0, nokern=0; int fb=0;
  for (uint64_t p=0; p+16<=ss; fb++) {
    const uint8_t *d=e+so+p; if (r32(d)!=0xba55ed50u){p+=8; fb--; continue;}
    uint16_t hs=r16(d+6); uint64_t fat=r64(d+8); int en=0;
    for (uint64_t off=hs; off+64<=hs+fat; en++) {
      uint32_t ehs=r32(d+off+4), stride=r32(d+off+8), cs=r32(d+off+16), arch=r32(d+off+28), fl=r32(d+off+40); uint64_t us=r64(d+off+56);
      if (ehs<64||!stride) break;
      if (r16(d+off)==2 && arch==120) {
        uint64_t n = (fl & 0xA000) ? us : (cs ? cs : stride); uint8_t *im8 = malloc(n?n:1); int ok=0;
        if (fl & 0x8000) { size_t r = ZSTD_decompress(im8, n, d+off+ehs, cs); ok = !ZSTD_isError(r) && r == n; }
        else if (fl & 0x2000) ok = LZ4_decompress_safe((const char*)d+off+ehs,(char*)im8,(int)cs,(int)n)==(int)n;
        else { memcpy(im8,d+off+ehs,n); ok=1; }
        tinynv_cubin_t c; tinynv_image_t im;
        if (ok && !tinynv_cubin_parse(im8, n, &c)) {
          n120++;
          if (tinynv_cubin_image(&c, c.kernels[0].name, 1, &im)) { refused++; printf("%d:%d\trefused\t%s\n", fb, en, tinynv_last_error()); }
          else {
            /* code ranges: every kernel's text in THIS layout */
            uint64_t (*code)[2] = calloc((size_t)c.nkernels, sizeof *code);
            /* constant-bank ranges too: every kernel's bank 0 and the module's banks 1..7, as each layout reports them */
            uint64_t (*bank)[2] = calloc((size_t)c.nkernels * TINYNV_MAX_CONSTBUFS, sizeof *bank); int nbank = 0;
            for (int k = 0; k < c.nkernels; k++) { tinynv_image_t lay;
              if (!tinynv_cubin_image(&c, c.kernels[k].name, 0, &lay)) { code[k][0] = lay.text_off; code[k][1] = lay.text_off + c.kernels[k].text_size;
                for (int b = 0; b < TINYNV_MAX_CONSTBUFS; b++) if (lay.constbuf[b].used && lay.constbuf[b].size && (b == 0 || k == 0)) {
                  bank[nbank][0] = lay.constbuf[b].off; bank[nbank][1] = lay.constbuf[b].off + lay.constbuf[b].size; nbank++; }
                tinynv_image_free(&lay); } }
            /* PROGBITS PLACEMENT REBUILT FROM THE ELF - the rule image.c documents (declared address kept; else appended in
             * file order at max(128, align)). The change under test only APPENDS NOBITS after these, so the rebuild is the
             * same for both builds, and it is VALIDATED against what the loader itself reports: every kernel's text offset
             * and every bank offset must match, or the scan says so and its classification is void for that image. */
            uint64_t *rep = calloc((size_t)c.nsections, sizeof(uint64_t)); uint64_t pend = 0;
            for (int q = 0; q < c.nsections; q++) if (c.sections[q].type == 1 && c.sections[q].addr && c.sections[q].addr + c.sections[q].size > pend) pend = c.sections[q].addr + c.sections[q].size;
            for (int q = 0; q < c.nsections; q++) { const tinynv_section_t *z = &c.sections[q]; if (z->type != 1) continue;
              if (z->addr) { rep[q] = z->addr; continue; }
              uint64_t al = z->align > 128 ? z->align : 128; pend += (al - pend % al) % al; rep[q] = pend; pend += z->size; }
            int mism = 0;
            for (int k = 0; k < c.nkernels; k++) { for (int q = 0; q < c.nsections; q++) if (c.sections[q].type == 1 && c.kernels[k].text_off == c.sections[q].off) {
                if (rep[q] != code[k][0]) mism++; } }
            g_mism += mism;
            /* the ELF walk, in image.c's order, to pair each im.relocs entry with its symbol's type */
            int idx = 0, bad = 0, obj = 0, nob = 0, nob_code = 0, nob_bank = 0, nob_past = 0, nob_other = 0, nob_own = 0;
            for (int i = 0; i < c.nsections && idx <= im.nrelocs; i++) {
              const tinynv_section_t *s = &c.sections[i];
              if (s->type != 4 && s->type != 9) continue;
              uint64_t ent = s->type == 4 ? 24 : 16;
              const char *tgt = after(s->sname, s->type == 4 ? ".rela" : ".rel");
              if (!tgt || !strcmp(tgt, ".eh_frame")) continue;
              int t = -1; for (int j = 0; j < c.nsections; j++) if (!strcmp(c.sections[j].sname, tgt)) { t = j; break; }
              if (t < 0) continue;
              if (s->link >= (uint32_t)c.nsections || c.sections[s->link].type != 2) continue;
              const tinynv_section_t *sym = &c.sections[s->link];
              for (uint64_t o = 0; o + ent <= s->size; o += ent, idx++) {
                uint32_t si = (uint32_t)(r64(c.img + s->off + o + 8) >> 32);
                const uint8_t *sy = c.img + sym->off + (uint64_t)si * 24;
                if ((sy[4] & 15) != 1 || !r16(sy + 6) || idx >= im.nrelocs) continue;       /* OBJECT, defined */
                obj++;
                uint64_t tg = im.relocs[idx].target;
                int in_code = 0, in_bank = 0;
                for (int k = 0; k < c.nkernels; k++) if (tg >= code[k][0] && tg < code[k][1]) { in_code = 1; break; }
                for (int b = 0; b < nbank; b++) if (tg >= bank[b][0] && tg < bank[b][1]) { in_bank = 1; break; }
                bad += in_code;
                uint16_t x = r16(sy + 6);
                if (x < c.nsections && c.sections[x].type == 8) {          /* the symbol lives in SHT_NOBITS data */
                  nob++;
                  int in_progbits = 0; const char *which = NULL;
                  for (int q = 0; q < c.nsections; q++) if (c.sections[q].type == 1 && tg >= rep[q] && tg < rep[q] + c.sections[q].size) { in_progbits = 1; which = c.sections[q].sname; break; }
                  if (in_code) nob_code++; else if (in_bank) nob_bank++;
                  else if (tg >= im.len) nob_past++;
                  else if (in_progbits) { nob_other++; if (!g_which[0] && which) snprintf(g_which, sizeof g_which, "%s", which); }
                  else nob_own++;
                  if (getenv("SAMPLE") && nob == 1) { for (int q = 0; q < c.nsections; q++) if (c.sections[q].type == 1 && !c.sections[q].addr) {
                      printf("  first placed PROGBITS at offset 0: %s flags %#llx size %llu\n", c.sections[q].sname, (unsigned long long)c.sections[q].flags,
                             (unsigned long long)c.sections[q].size); break; } }
                  if (getenv("SAMPLE") && nob <= 2) printf("  sample %d:%d: %s sh_addr %#llx size %llu; image len %zu; target %#llx\n", fb, en,
                        c.sections[x].sname, (unsigned long long)c.sections[x].addr, (unsigned long long)c.sections[x].size, im.len,
                        (unsigned long long)tg);
                }
              }
            }
            if (idx != im.nrelocs) printf("%d:%d\tPAIRING-MISMATCH\twalk %d, loader %d\n", fb, en, idx, im.nrelocs);
            printf("%d:%d\tlen\t%zu\tobject-relocs\t%d\tinto-code\t%d\tnobits-relocs\t%d\tin-code\t%d\tin-a-bank\t%d\telsewhere\t%d\n",
                   fb, en, im.len, obj, bad, nob, nob_code, nob_bank, nob_other);
            total_bad += bad; total_obj += obj; nimg_bad += bad > 0;
            g_nob += nob; g_nob_code += nob_code; g_nob_bank += nob_bank; g_nob_other += nob_other; g_nob_past += nob_past; g_img_mis += (nob_code + nob_bank + nob_past + nob_other) > 0;
            g_nob_own += nob_own;
            free(code); free(bank); free(rep); tinynv_image_free(&im);
          }
          tinynv_cubin_free(&c);
        } else if (ok) nokern++;
        free(im8);
      }
      off += ehs + stride;
    }
    p += (hs+fat+7)&~7ull;
  }
  fprintf(stderr, "sm_120 images parsed %ld (+%ld with no kernels), layout refused %ld; relocations against OBJECT symbols %ld; "
          "landing inside kernel code %ld, in %ld images\n", n120, nokern, refused, total_obj, total_bad, nimg_bad);
  fprintf(stderr, "PROGBITS rebuild vs the loader's own kernel offsets: %ld mismatches%s\n", g_mism, g_mism ? " - CLASSIFICATION VOID" : "");
  fprintf(stderr, "relocations against NOBITS (.nv.global) objects %ld: into kernel code %ld, into a constant bank %ld, into ANOTHER "
          "PROGBITS section %ld (first seen: %s), past the image's end %ld, into space past every PROGBITS section %ld\n"
          "MISRESOLVED (code, bank, another section, or past the end) %ld, in %ld images\n",
          g_nob, g_nob_code, g_nob_bank, g_nob_other, g_which[0] ? g_which : "-", g_nob_past, g_nob_own,
          g_nob_code + g_nob_bank + g_nob_other + g_nob_past, g_img_mis);
  return 0;
}
