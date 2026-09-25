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
#define EIATTR_CTA_PER_CLUSTER 0x3d
#define EIATTR_EXPLICIT_CLUSTER 0x3e
#define EIATTR_MAX_CLUSTER_RANK 0x3f
#define EIATTR_NUM_BARRIERS 0x4c   // read from the CUTLASS FP8 cubin's own .nv.info (fmt 2, value 8) = cuobjdump's EIATTR_NUM_BARRIERS 0x8

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
    } else if (attr == EIATTR_CTA_PER_CLUSTER && plen >= 12) {
      k->cluster_dim[0] = rd32(payload); k->cluster_dim[1] = rd32(payload + 4); k->cluster_dim[2] = rd32(payload + 8);
    } else if (attr == EIATTR_EXPLICIT_CLUSTER) {
      k->explicit_cluster = 1;
    } else if (attr == EIATTR_MAX_CLUSTER_RANK && plen >= 4) {
      k->max_cluster_rank = rd32(payload);
    } else if (attr == EIATTR_NUM_BARRIERS) {
      k->num_barriers = fmt == EIFMT_SVAL ? 0 : (fmt == 2 ? p[2] : val);   // a one-byte value (fmt 2) or a 16-bit one
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
  // TWO CUDA ELF ABIs, AND THEY KEEP THE ARCHITECTURE IN DIFFERENT BYTES OF e_flags. OSABI 0x41 / ABI version 8 has it
  // in the SECOND byte (0x06005004 -> 80, 0x06007802 -> 120); OSABI 0x33 / ABI version 7 - what ptxas 12.x emits for
  // sm_80 and sm_90, and so cu128 torch's sm_80/sm_90 images - has it in the LOW byte (0x00500550 -> 80), where the
  // second byte is 5 for every architecture. Measured on one PTX compiled by ptxas 12.8 and 13.0
  // (test/fixtures/abi-*, README-abi.md) and on every image of two torch fatbins (libtinycudart/fatbin.c, elf_sm).
  // This read the second byte unconditionally until 2026-09-21, so a v7 sm_80 image came out as "sm_5".
  uint32_t eflags = rd32(img + 0x30);
  out->sm_arch = img[7] == 0x33 ? (eflags & 0xff) : ((eflags >> 8) & 0xff);

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

  // MODULE-WIDE ATTRIBUTES, each measured against NVIDIA's own driver (580.126.18, cuFuncGetAttribute on the 3090) for
  // every kernel of the sm_80 images of two torch builds - 15,238 (cu128, ELF ABI 7) and 14,934 (cu130, ABI 8), all
  // matching (B's survey and gate, cuda-shim-b 68b6070 vm/rmtrace/kernel-attrs-20260921/):
  //   CONST_SIZE_BYTES = the size of .nv.constant3, the module's user __constant__ data, the same for every function.
  //   PTX_VERSION = the virtual architecture: in ABI 8 the u16 at offset 2 of the note descriptor of .note.nv.cuinfo
  //     (CUDA 13) or .note.nv.cuver (CUDA 12.8); in ABI 7 bits 23:16 of e_flags. A compute_75 -> sm_86 build reads 75
  //     there and the driver reports 75, while BINARY_VERSION reports 86. The .note.nv.cuver reading (the guest's
  //     sm_120 images: 120) has NOT been checked against a driver - no sm_120 image loads on the 3090 - and rests on
  //     the note having the same type and layout as .note.nv.cuinfo.
  out->ptx_version = -1;
  for (int i = 0; i < shnum; i++) {
    if (!strcmp(sec[i].sname, ".nv.constant3")) out->const3_size = sec[i].size;
    if ((!strcmp(sec[i].sname, ".note.nv.cuinfo") || !strcmp(sec[i].sname, ".note.nv.cuver")) && sec[i].size >= 16) {
      uint32_t namesz = rd32(img + sec[i].off);
      uint64_t at = 12 + (((uint64_t)namesz + 3) & ~3ull) + 2;
      if (at + 2 <= sec[i].size) out->ptx_version = rd16(img + sec[i].off + at);
    }
  }
  if (out->ptx_version < 0 && img[7] == 0x33) out->ptx_version = (int)((rd32(img + 0x30) >> 16) & 0xff);

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

/* THE MOST THREADS ONE BLOCK OF THIS KERNEL CAN HAVE, which is 1024 only when its registers allow it.
 *
 * A kernel with no __launch_bounds__ was treated as good for 1024 threads. Its registers say otherwise: the register
 * file is shared by every warp of a block, and a block whose warps need more than the file holds is not refused by
 * the hardware in any way this driver can see - the shared-memory case, measured on this card, arrives as a launch
 * that is accepted and never scheduled. Torch's guest library has 1387 kernels with no bound whose registers do not
 * allow 1024 threads, down to 256 (B's sweep, D's rerun on the guest file: cuda-shim-d cc2b1c3).
 *
 * NVIDIA'S OWN RULE, NOT A REMEMBERED ONE - cuda-13/include/cuda_occupancy.h, the CUDA 13.0 occupancy calculator:
 *   cudaOccRegAllocationGranularity  (:680)   256 registers, compute capability 3 through 12
 *   cudaOccRegAllocationMaxPerThread (:650)   256 per thread, compute capability 7 through 12
 *   cudaOccSubPartitionsPerMultiprocessor (:708)  4, compute capability 7 through 12
 *   the per-block check (:1523-1545): regsAllocatedPerWarp = roundUp(numRegs * warpSize, granularity); a block of W
 *   warps fits only if regsPerBlock >= regsAllocatedPerWarp * roundUp(W, subPartitions)  ("the hardware check")
 *                  and regsPerBlock >= regsAllocatedPerWarp * W                          ("the software check")
 * regsPerBlock is 65536: the shim's CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, which it tags ARCH (a programming-
 * model constant, not read from this card - shim.c:345). The same calculator also needs the SM to hold the block's
 * warps; with regsPerMultiprocessor equal to regsPerBlock that follows from the check above.
 *
 * Every constant above is the same for every architecture this driver loads (sm_80 to sm_120), so none is keyed on
 * sm_arch; a compute capability outside 7..12 would need the table re-read, not these numbers reused. */
#define TNV_REGS_PER_BLOCK      65536u
#define TNV_REG_GRANULARITY     256u
#define TNV_REG_MAX_PER_THREAD  256u
#define TNV_SUBPARTITIONS       4u
uint32_t tinynv_kernel_thread_limit(const tinynv_kernel_desc_t *d) {
  uint32_t bound = d->max_threads && d->max_threads < 1024 ? d->max_threads : 1024;
  if (d->regs > TNV_REG_MAX_PER_THREAD) return 0;
  uint32_t per_warp = (d->regs * 32u + TNV_REG_GRANULARITY - 1) / TNV_REG_GRANULARITY * TNV_REG_GRANULARITY;
  if (!per_warp) return bound;
  uint32_t fit = 0;
  for (uint32_t w = 1; w <= 32; w++) {
    uint32_t rounded = (w + TNV_SUBPARTITIONS - 1) / TNV_SUBPARTITIONS * TNV_SUBPARTITIONS;
    if (per_warp * rounded > TNV_REGS_PER_BLOCK || per_warp * w > TNV_REGS_PER_BLOCK) break;
    fit = w * 32;
  }
  return fit < bound ? fit : bound;
}
