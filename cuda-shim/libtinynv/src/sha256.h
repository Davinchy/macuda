// SHA-256 over a buffer. Used to check firmware images before they are handed to the GPU's secure boot.
#ifndef TINYNV_SHA256_H
#define TINYNV_SHA256_H
#include <stddef.h>
#include <stdint.h>

typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t n; } tinynv_sha256_t;

void tinynv_sha256_init(tinynv_sha256_t *c);
void tinynv_sha256_update(tinynv_sha256_t *c, const void *data, size_t n);
void tinynv_sha256_final(tinynv_sha256_t *c, uint8_t out[32]);
void tinynv_sha256(const void *data, size_t n, uint8_t out[32]);
void tinynv_sha256_hex(const uint8_t digest[32], char out[65]);

#endif
