// SHA-256, so the driver can check what it is about to hand the GPU's secure boot.
//
// Written out rather than linked, because the two hosts this runs on offer different libraries (CommonCrypto on macOS,
// OpenSSL or nothing on Linux) and a 200 line function is cheaper than a build-time choice. It is checked against the
// published NIST vectors in test/test_fw.c, which is what makes it trustworthy rather than merely present.
#include "sha256.h"
#include <string.h>

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define S0(x) (ROR(x, 2) ^ ROR(x, 13) ^ ROR(x, 22))
#define S1(x) (ROR(x, 6) ^ ROR(x, 11) ^ ROR(x, 25))
#define s0(x) (ROR(x, 7) ^ ROR(x, 18) ^ ((x) >> 3))
#define s1(x) (ROR(x, 17) ^ ROR(x, 19) ^ ((x) >> 10))

static const uint32_t K[64] = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static void block(tinynv_sha256_t *c, const uint8_t *p) {
  uint32_t w[64], a, b, cc, d, e, f, g, h;
  for (int i = 0; i < 16; i++) w[i] = ((uint32_t)p[i * 4] << 24) | ((uint32_t)p[i * 4 + 1] << 16) | ((uint32_t)p[i * 4 + 2] << 8) | p[i * 4 + 3];
  for (int i = 16; i < 64; i++) w[i] = s1(w[i - 2]) + w[i - 7] + s0(w[i - 15]) + w[i - 16];
  a = c->h[0]; b = c->h[1]; cc = c->h[2]; d = c->h[3]; e = c->h[4]; f = c->h[5]; g = c->h[6]; h = c->h[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = h + S1(e) + ((e & f) ^ (~e & g)) + K[i] + w[i];
    uint32_t t2 = S0(a) + ((a & b) ^ (a & cc) ^ (b & cc));
    h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
  }
  c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d; c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += h;
}

void tinynv_sha256_init(tinynv_sha256_t *c) {
  static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  memcpy(c->h, iv, sizeof(iv));
  c->len = 0;
  c->n = 0;
}

void tinynv_sha256_update(tinynv_sha256_t *c, const void *data, size_t n) {
  const uint8_t *p = data;
  c->len += n;
  while (n) {
    size_t take = 64 - c->n < n ? 64 - c->n : n;
    memcpy(c->buf + c->n, p, take);
    c->n += take;
    p += take;
    n -= take;
    if (c->n == 64) { block(c, c->buf); c->n = 0; }
  }
}

void tinynv_sha256_final(tinynv_sha256_t *c, uint8_t out[32]) {
  uint64_t bits = c->len * 8;
  uint8_t pad = 0x80;
  tinynv_sha256_update(c, &pad, 1);
  static const uint8_t zero[64] = {0};
  while (c->n != 56) tinynv_sha256_update(c, zero, 1);
  uint8_t tail[8];
  for (int i = 0; i < 8; i++) tail[i] = (uint8_t)(bits >> (56 - i * 8));
  tinynv_sha256_update(c, tail, 8);
  for (int i = 0; i < 8; i++) {
    out[i * 4] = (uint8_t)(c->h[i] >> 24);
    out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
    out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
    out[i * 4 + 3] = (uint8_t)c->h[i];
  }
}

void tinynv_sha256(const void *data, size_t n, uint8_t out[32]) {
  tinynv_sha256_t c;
  tinynv_sha256_init(&c);
  tinynv_sha256_update(&c, data, n);
  tinynv_sha256_final(&c, out);
}

void tinynv_sha256_hex(const uint8_t digest[32], char out[65]) {
  static const char *d = "0123456789abcdef";
  for (int i = 0; i < 32; i++) { out[i * 2] = d[digest[i] >> 4]; out[i * 2 + 1] = d[digest[i] & 0xf]; }
  out[64] = 0;
}
