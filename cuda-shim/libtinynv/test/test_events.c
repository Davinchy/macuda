// Event query / destroy / elapsed and mem_info through the public API, on the null device.
//
// What this can prove without a card: the null-device answers are the documented ones, an unrecorded event is
// complete, elapsed is refused BY NAME (never a made-up number), mem_info is refused by name where there is no
// allocator. What it cannot: that a query on the card never says "complete" before the work retires - that needs a
// kernel still running, and is a card gate (the query is tinynv_exec_reached, the predicate tinynv_exec_wait exits on).
#include "tinynv.h"
#include <stdio.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

int main(void) {
  tinynv_device_t dev = NULL; tinynv_stream_t s = NULL; tinynv_event_t a = NULL, b = NULL;
  CHECK(tinynv_init() == TINYNV_OK, "init");
  CHECK(tinynv_device_get(&dev, 0) == TINYNV_OK, "device_get");
  CHECK(tinynv_stream_create(dev, &s) == TINYNV_OK, "stream_create");
  CHECK(tinynv_event_create(dev, &a) == TINYNV_OK && tinynv_event_create(dev, &b) == TINYNV_OK, "event_create");
  int c = -1;
  CHECK(tinynv_event_query(a, &c) == TINYNV_OK && c == 1, "an event never recorded is complete (got %d)", c);
  CHECK(tinynv_event_record(a, s) == TINYNV_OK && tinynv_event_record(b, s) == TINYNV_OK, "record");
  c = -1;
  CHECK(tinynv_event_query(b, &c) == TINYNV_OK && c == 1, "on the null device a recorded event is complete (got %d)", c);
  CHECK(tinynv_event_query(NULL, &c) != TINYNV_OK, "a NULL event is refused");
  CHECK(tinynv_event_query(a, NULL) != TINYNV_OK, "a NULL answer pointer is refused");
  float ms = -1;
  tinynv_status_t st = tinynv_event_elapsed(&ms, a, b);
  CHECK(st != TINYNV_OK && strstr(tinynv_last_error(), "not recorded"), "elapsed must be refused by name, got status %d: %s",
        (int)st, tinynv_last_error());
  CHECK(ms == -1, "a refused elapsed must not write a number (wrote %f)", ms);
  uint64_t fr = 7, tot = 7;
  st = tinynv_mem_info(dev, &fr, &tot);
  CHECK(st == TINYNV_ERR_NODEV && strstr(tinynv_last_error(), "null device"), "mem_info on the null device: status %d: %s",
        (int)st, tinynv_last_error());
  CHECK(fr == 7 && tot == 7, "a refused mem_info must not write (wrote %llu/%llu)", (unsigned long long)fr, (unsigned long long)tot);
  CHECK(tinynv_event_destroy(a) == TINYNV_OK && tinynv_event_destroy(b) == TINYNV_OK, "destroy");
  CHECK(tinynv_event_destroy(NULL) != TINYNV_OK, "destroying NULL is refused");
  printf(fails ? "%d of %d checks failed\n" : "events and mem_info on the null device: all %d checks passed\n",
         fails ? fails : checks, checks);
  return fails ? 1 : 0;
}
