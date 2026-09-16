// A cubin is an ELF for EM_CUDA: one .text.<kernel> per __global__ function, a constant bank holding the driver's own
// parameters followed by the kernel's, and .nv.info sections of attribute records describing both. Everything the launch
// path needs (where parameters go, how many registers, how much shared memory) comes from here, never from a constant.
#ifndef TINYNV_CUBIN_H
#define TINYNV_CUBIN_H
#include <stddef.h>
#include <stdint.h>

#define TINYNV_MAX_PARAMS 128

typedef struct {
  // Owned, and however long it needs to be. A fixed buffer here truncated CUB's longest template instantiations at 255
  // characters, and because the constant, shared and info section names are built from this one, a truncated name meant
  // the parameter base, shared size and register count were silently read from sections that were never found. Two of
  // the 8012 kernels in a llama.cpp build are over 256 characters, which is how session A found it.
  char *name;
  uint64_t text_off, text_size;    // the kernel's code within the image
  uint64_t const0_off, const0_size; // .nv.constant0.<name>: driver parameters, then the kernel's
  uint32_t param_base;             // where the kernel's parameters start in constant bank 0. 0x160 on sm_8x, 0x380 on sm_12x
  uint32_t param_size;             // how many bytes of them
  uint32_t regs, static_smem, min_stack, max_threads;
  int nparams;
  struct { uint32_t offset, size; } params[TINYNV_MAX_PARAMS]; // by ordinal, offset relative to param_base
} tinynv_kernel_desc_t;

// One section header, kept because the loadable image is built out of them: which sections hold bytes, where each one
// wants to sit, and what the relocations point at.
typedef struct {
  const char *sname;                     // into the cubin's own string table, so it lives as long as the cubin
  uint32_t name, type, link, info;
  uint64_t flags, addr, off, size, align, entsize;
} tinynv_section_t;

typedef struct {
  const uint8_t *img;
  size_t len;
  int nsections;
  tinynv_section_t *sections;            // owned
  uint32_t sm_arch; // 120 for sm_120, from the ELF flags
  int nkernels;
  tinynv_kernel_desc_t *kernels;
} tinynv_cubin_t;

// The loadable image: what actually goes into device memory, which is not the cubin. The code, the constant banks and
// any __device__ data are separate sections in the file and have to be laid out into one contiguous block, because the
// addresses the GPU is given are offsets into that block.
#define TINYNV_MAX_CONSTBUFS 8

typedef struct { uint64_t at; uint64_t target; uint32_t type; int64_t addend; } tinynv_reloc_t;

typedef struct {
  uint8_t *bytes;                        // owned
  size_t len;
  uint64_t text_off;                     // the kernel's code within the image, which is not its offset in the file
  struct { uint64_t off, size; int used; } constbuf[TINYNV_MAX_CONSTBUFS];
  int nrelocs;
  tinynv_reloc_t *relocs;                // owned
  // Relocations against symbols the cubin does not contain and this driver does not provide - the device runtime's
  // printf, in practice. They resolve to zero, so a kernel that reaches one faults rather than jumping somewhere
  // arbitrary. Counted so a caller can say so rather than discover it.
  int unresolved;
} tinynv_image_t;

// Lays out one kernel's loadable image. Constant bank 0 is the named kernel's own; the other banks are the module's.
//
// The layout is a property of the cubin and the bytes are the same whichever kernel is named, so a module uploads once
// and every kernel after the first asks for `with_bytes` 0: it wants where its code and its banks are, not another copy
// of a four megabyte image. With it clear nothing is allocated at all - not the image, not the relocations - which
// matters because a ggml cubin holds hundreds of kernels and each would otherwise carry a private copy of both.
int tinynv_cubin_image(const tinynv_cubin_t *c, const char *kernel, int with_bytes, tinynv_image_t *out);
// Writes the addresses into the image, once the block has a device address. Relocation types are CUDA's, not the host's.
int tinynv_image_relocate(tinynv_image_t *im, uint64_t image_va);
void tinynv_image_free(tinynv_image_t *im);

int tinynv_cubin_parse(const void *data, size_t len, tinynv_cubin_t *out);
void tinynv_cubin_free(tinynv_cubin_t *c);
const tinynv_kernel_desc_t *tinynv_cubin_kernel(const tinynv_cubin_t *c, const char *name);

#endif
