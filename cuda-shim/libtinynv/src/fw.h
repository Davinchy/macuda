// Loading NVIDIA's signed firmware images, and picking the pieces out of them.
//
// Three images boot a Blackwell card: the FMC that the FSP authorises and runs, GSP-RM itself, and the RISC-V bootloader
// that starts GSP-RM. Two of the three are ELF containers holding raw sections; the third is NVIDIA's own container.
#ifndef TINYNV_FW_H
#define TINYNV_FW_H
#include <stddef.h>
#include <stdint.h>

// The firmware release this driver is pinned to. Deliberately skewed against the headers (570.86.16) because that is the
// pairing the recorded boot proves works; see tools/fetch_nv_headers.sh.
#define TINYNV_FW_VER "570.144"

typedef struct { uint8_t *data; size_t size; } tinynv_blob_t;

// Where firmware images live, in order: a root the program set explicitly, then $TINYNV_FW_DIR, then the directory the
// library was built against. Never the working directory: a relative default means the same binary finds the firmware
// from one directory and silently does not from another, and "not found" here degrades into a test that passes without
// testing anything. Session A hit exactly that and nearly reported a boot as verified from a run that never reached it.
const char *tinynv_fw_root(void);
void tinynv_fw_set_root(const char *dir);

// Load <root>/nvidia/<chip_dir>/gsp/<name> and refuse it unless it hashes to sha256_hex. A firmware image is code the
// GPU's secure boot will execute, so "the file was there" is not good enough: the version is pinned by content.
int tinynv_fw_load(const char *chip_dir, const char *name, const char *sha256_hex, tinynv_blob_t *out);
void tinynv_fw_free(tinynv_blob_t *b);

// One ELF section by its exact name, pointing into the blob. Handles both 32 and 64 bit containers: NVIDIA ships the FMC
// as ELF32 and GSP-RM as ELF64.
int tinynv_elf_section(const tinynv_blob_t *b, const char *name, const uint8_t **out, size_t *out_len);

#endif
