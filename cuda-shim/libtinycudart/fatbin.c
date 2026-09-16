// Extract the device cubin (ELF) from the fatbin wrapper that clang's host code passes to __cudaRegisterFatBinary.
#include <stdint.h>
#include <string.h>
#include <stddef.h>
// wrapper clang/nvcc emits: {int magic=0x466243b1; int version; const void* data; void* filename}
struct fat_wrapper { int magic; int version; const void* data; void* filename; };
// find the embedded cubin ELF inside the fatbin blob. Scaffold: locate the ELF magic and size it from the ELF header.
// TODO(real): parse the fatbinary entry table (kind, sm_arch) to pick the matching arch instead of the first ELF.
int tinycudart_extract_cubin(const void* fatCubin, const void** out, size_t* out_len) {
  const struct fat_wrapper* w = (const struct fat_wrapper*)fatCubin;
  const unsigned char* data = (const unsigned char*)(w && w->magic == 0x466243b1 ? w->data : fatCubin);
  // scan up to 8 MB for the ELF magic \x7fELF
  const unsigned char elf[4] = {0x7f,'E','L','F'};
  for (size_t i = 0; i < (8u<<20); i++) {
    if (memcmp(data+i, elf, 4) == 0 && data[i+4]==2 /*ELFCLASS64*/) {
      const unsigned char* e = data + i;
      uint64_t shoff  = *(const uint64_t*)(e + 0x28);
      uint16_t shent  = *(const uint16_t*)(e + 0x3a);
      uint16_t shnum  = *(const uint16_t*)(e + 0x3c);
      *out = e; *out_len = (size_t)(shoff + (uint64_t)shent*shnum);
      return 0;
    }
  }
  return -1;
}

// --- the names of the kernels a cubin carries: PROGBITS sections named .text.<kernel> (the same rule the driver's
// reader uses, so the two never disagree about what is a kernel). Needed because clang on macOS and nvcc on Linux
// mangle the same kernel differently, and the runtime has to find nvcc's name for clang's.
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
int tinycudart_cubin_kernels(const void *cubin, size_t len, char ***names_out) {
  const unsigned char *e = cubin;
  if (len < 64 || memcmp(e, "\x7f" "ELF", 4) || e[4] != 2) return -1;
  uint64_t shoff; uint16_t shentsize, shnum, shstrndx;
  memcpy(&shoff, e + 40, 8); memcpy(&shentsize, e + 58, 2); memcpy(&shnum, e + 60, 2); memcpy(&shstrndx, e + 62, 2);
  if (shoff + (uint64_t)shnum * shentsize > len) return -1;
  uint64_t stroff; memcpy(&stroff, e + shoff + (uint64_t)shstrndx * shentsize + 24, 8);
  char **names = calloc(shnum, sizeof(char *)); int n = 0;
  for (uint16_t i = 0; i < shnum; i++) {
    const unsigned char *sh = e + shoff + (uint64_t)i * shentsize;
    uint32_t name, type; memcpy(&name, sh, 4); memcpy(&type, sh + 4, 4);
    if (type != 1 /* SHT_PROGBITS */) continue;
    const char *s = (const char *)e + stroff + name;
    if (strncmp(s, ".text.", 6)) continue;
    names[n++] = strdup(s + 6);
  }
  *names_out = names; return n;
}
