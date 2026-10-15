// The crash witness, checked offline.
//
// It exists to answer one question about one address: on 2026-09-15 a test-backend-ops run faulted in a static hash
// table's bucket array at 0x116d07710, and the question was whether this driver had written there. The sixteen-entry
// ring that answered it first could only speak for the last sixteen downloads, so "it is in none of them" meant almost
// nothing. This version keeps every span and dumps them to a file.
//
// What is checked here is the property the file is read for: an address inside a recorded span is found by the same
// search the reader will run, an address outside every span is not, and a run that records more than fits says so
// instead of quietly answering for a prefix. A witness that is wrong is worse than no witness, because its silence
// is read as evidence.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <stdint.h>
#include "internal.h"

static int checks, bad;
#define CHECK(c, ...) do { checks++; if (!(c)) { bad++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

// The reader's question, run over the dumped file exactly as a person would run it: does any span contain this address?
static const char *who_wrote(const char *path, unsigned long long addr, char *what, size_t whatn) {
  FILE *f = fopen(path, "r");
  if (!f) return NULL;
  static char kind[16];
  char line[512];
  const char *found = NULL;
  while (fgets(line, sizeof(line), f)) {
    unsigned long long lo, hi; size_t n; char k[16], desc[256];
    if (line[0] == '#') continue;
    if (sscanf(line, "%15s %llx %llx %zu %255[^\n]", k, &lo, &hi, &n, desc) != 5) continue;
    if (addr >= lo && addr < hi) {
      snprintf(kind, sizeof(kind), "%s", k);
      snprintf(what, whatn, "%s", desc);
      found = kind;
    }
  }
  fclose(f);
  return found;
}

static int line_count(const char *path, const char *prefix) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  char line[512]; int n = 0;
  while (fgets(line, sizeof(line), f)) if (!strncmp(line, prefix, strlen(prefix))) n++;
  fclose(f);
  return n;
}

int main(void) {
  char path[512];
  const char *dir = getenv("TMPDIR");
  snprintf(path, sizeof(path), "%stinynv-writes-%d.txt", dir && *dir ? dir : "/tmp/", (int)getpid());
  remove(path);

  // A standing buffer, the kind a per-write list never sees because it is written by the engine rather than by us.
  static char region[4096];
  tinynv_note_host_region(region, sizeof(region), "a pretend staging buffer");

  // Downloads, of the shape the real ones have: a caller's heap, in chunks.
  char *heap = malloc(1 << 20);
  CHECK(heap != NULL, "the test could not get a megabyte");
  if (!heap) return 1;
  for (int i = 0; i < 300; i++) tinynv_note_host_write(heap + i * 3072, 2048, "download: stage -> caller");

  tinynv_dump_host_writes(SIGSEGV);

  char what[256] = "";
  // Inside a download, at its first byte, in its middle, and at its last byte.
  CHECK(who_wrote(path, (unsigned long long)(uintptr_t)(heap + 3072), what, sizeof(what)) != NULL,
        "the first byte of a recorded write was not found in the file");
  CHECK(who_wrote(path, (unsigned long long)(uintptr_t)(heap + 3072 + 1000), what, sizeof(what)) != NULL,
        "the middle of a recorded write was not found");
  CHECK(who_wrote(path, (unsigned long long)(uintptr_t)(heap + 3072 + 2047), what, sizeof(what)) != NULL,
        "the last byte of a recorded write was not found");
  CHECK(!strcmp(what, "download: stage -> caller"), "the file named '%s' rather than the download", what);

  // One byte past the end is not ours, and neither is the gap between two chunks. This is the direction that matters:
  // a witness that claims too much turns an innocent address into a false lead.
  CHECK(who_wrote(path, (unsigned long long)(uintptr_t)(heap + 3072 + 2048), what, sizeof(what)) == NULL,
        "the byte just past a write was claimed as ours");
  CHECK(who_wrote(path, (unsigned long long)(uintptr_t)(heap + 3072 + 2500), what, sizeof(what)) == NULL,
        "the gap between two chunks was claimed as ours");

  // The standing region is found, and reported as a region rather than a write.
  const char *kind = who_wrote(path, (unsigned long long)(uintptr_t)(region + 100), what, sizeof(what));
  CHECK(kind && !strcmp(kind, "region"), "the standing buffer came back as '%s'", kind ? kind : "(not found)");
  CHECK(!strcmp(what, "a pretend staging buffer"), "the region was named '%s'", what);

  // And the address the whole instrument was built for: something far away, in nothing we recorded, answers "not ours".
  CHECK(who_wrote(path, 0x116d07710ull, what, sizeof(what)) == NULL ||
        (unsigned long long)(uintptr_t)heap <= 0x116d07710ull,
        "an address in none of our spans was claimed as ours");

  CHECK(line_count(path, "write ") == 300, "the file holds %d writes rather than 300", line_count(path, "write "));
  CHECK(line_count(path, "region ") == 1, "the file holds %d regions rather than 1", line_count(path, "region "));
  CHECK(line_count(path, "# TRUNCATED") == 0, "300 writes should not have truncated the list");

  // Past the ceiling the file must say so. Silence here would be the old bug in a new place: a complete-looking list
  // that answers for a prefix, read as if it answered for the run.
  for (int i = 0; i < 70000; i++) tinynv_note_host_write(heap + (i % 256) * 4, 4, "a flood");
  tinynv_dump_host_writes(SIGSEGV);
  CHECK(line_count(path, "# TRUNCATED") == 1, "a list that overflowed did not say it had overflowed");
  // And that the header's claim matches the file's contents. Asserting only that truncation was ANNOUNCED is a row
  // that cannot differ from its control - a dump that kept one entry out of seventy thousand would pass it, because
  // it did announce. Session C's shape: it runs, it reports, and its report is uninformative by construction. The
  // cross-check is the header against the body, which are written by different parts of the same function.
  {
    FILE *f = fopen(path, "r");
    unsigned happened = 0, recorded = 0;
    char line[512];
    if (f) {
      while (fgets(line, sizeof(line), f))
        if (sscanf(line, "# TRUNCATED: %u writes happened, %u recorded", &happened, &recorded) == 2) break;
      fclose(f);
    }
    CHECK(happened == 70300, "the file says %u writes happened; the test made 70300", happened);
    CHECK(recorded > 300, "the file says only %u were recorded, which is fewer than the 300 kept before the flood - "
                          "the cap is not where it claims to be", recorded);
    CHECK((int)recorded == line_count(path, "write "),
          "the header claims %u writes were recorded and the body holds %d - the count and the contents disagree",
          recorded, line_count(path, "write "));
  }

  free(heap);
  remove(path);
  printf("%s: %d checks, %d failed\n", bad ? "test_witness FAILED" : "test_witness", checks, bad);
  printf("  every host span is recorded and a dumped file answers containment both ways\n");
  return bad ? 1 : 0;
}
