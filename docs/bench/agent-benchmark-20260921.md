# Coding-agent benchmark: 3060 vs metal

**Headline: ten turns of a coding agent's conversation, 95.8 s on the RTX 3060 against 201.2 s on the M3 Max through
Metal.** The card takes less than half the time, and the gap is all in the prompt side: at 32k tokens of context the
3060 answers the first token in 2.3 s where Metal takes 13.2 s, while generation runs at 48 against 15 tokens/s. The
cache was reused on every turn on both sides. In the depth sweep the 3060 processes a 2,048-token prompt 2.6x faster
than Metal at depth 0 and 4.0x faster at depth 65,536; generation is 4% slower than Metal at depth 0 and 38-43% faster
at every depth from 16k up. Both card runs were made on an x4 link at 8 GT/s (Gen3, what the Thunderbolt enclosure
holds), sampled from the host every five seconds and constant through both. The morning's card runs on a Gen1 link
(2.5 GT/s, `logs/agentbench-3060.json`, `logs/replay-3060-20260921-144424.json`) were 4-26% slower row by row and
112.6 s for the replay; every flag identical (`-ngl 99 -fa 1 -p 2048 -n 256 -d 0,16384,32768,65536 -pg 16384,512 -r 5`),
one llama.cpp build (`b906d25`) on both sides.

Method: `tools/agentbench.sh` (llama-bench, JSON out, environment beside it) and `tools/agent-replay.sh` (llama-server with
`cache_prompt`, ten turns: an 8,000-token opening, then 2,000 tokens of tool output and 400 tokens back per turn, the
whole transcript resent every turn, sizes hit by tokenizing through the server). The protocol's 20k/3k/500 sizes were
scaled to 8k/2k/400 because Qwen2.5-7B's trained context is 32k and the tenth turn had to fit in it.

## 3060

- model: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (qwen2 7B Q4_K - Medium, 4.36 GiB)
- llama.cpp build: `b906d25` (#2), backend CUDA, ngl 99, flash attention 1, 12 threads
- driver build id: `1621873`; PCIe link during the run: x4 at 8.0 GT/s (before the run: x4 at 2.5 GT/s); 27.0 arm64
- date: 2026-09-21T15:16:51-07:00

### Sweep

| test | depth | tok/s | ± | spread |
|---|---:|---:|---:|---:|
| pp2048 | 0 | 2165.3 | 3.0 | 0.1% |
| tg256 | 0 | 66.7 | 0.1 | 0.1% |
| pp16384+tg512 | 0 | 875.5 | 3.3 | 0.4% |
| pp2048 | 16384 | 1260.8 | 0.9 | 0.1% |
| tg256 | 16384 | 55.1 | 0.0 | 0.1% |
| pp16384+tg512 | 16384 | 650.1 | 0.5 | 0.1% |
| pp2048 | 32768 | 900.4 | 0.9 | 0.1% |
| tg256 | 32768 | 48.0 | 0.1 | 0.1% |
| pp16384+tg512 | 32768 | 519.3 | 0.2 | 0.0% |
| pp2048 | 65536 | 572.9 | 0.6 | 0.1% |
| tg256 | 65536 | 38.2 | 0.1 | 0.2% |
| pp16384+tg512 | 65536 | 369.3 | 0.2 | 0.1% |

### Replay, ten turns

- PCIe link during the replay: x4 at 8.0 GT/s; context 32768; server up in 9 s

| turn | prompt tokens | processed | cached | TTFT s | prompt tok/s | gen tok/s | turn s | cache |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9510 | 9510 | 0 | 5.10 | 1865.9 | 57.62 | 12.04 | first |
| 2 | 12057 | 2148 | 9909 | 1.43 | 1509.1 | 56.2 | 8.55 | hit |
| 3 | 14556 | 2100 | 12456 | 1.50 | 1401.2 | 54.44 | 8.85 | hit |
| 4 | 17144 | 2189 | 14955 | 1.69 | 1299.3 | 54.92 | 8.97 | hit |
| 5 | 19714 | 2171 | 17543 | 1.78 | 1223.3 | 53.7 | 9.23 | hit |
| 6 | 22152 | 2039 | 20113 | 1.75 | 1168.0 | 52.46 | 9.38 | hit |
| 7 | 24790 | 2239 | 22551 | 2.07 | 1085.6 | 51.31 | 9.87 | hit |
| 8 | 27292 | 2103 | 25189 | 2.05 | 1030.4 | 50.21 | 10.02 | hit |
| 9 | 29919 | 2228 | 27691 | 2.30 | 974.5 | 49.08 | 10.45 | hit |
| 10 | 32483 | 2165 | 30318 | 2.34 | 929.9 | 48.11 | 8.26 | hit |
| **total** | | | | | | | **95.8** | |

## metal

- model: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (qwen2 7B Q4_K - Medium, 4.36 GiB)
- llama.cpp build: `b906d25` (#2), backend MTL, ngl 99, flash attention 1, 12 threads
- driver build id: `none (not a driver build)`; PCIe link during the run: - at -; 27.0 arm64
- date: 2026-09-21T13:12:41-07:00

### Sweep

| test | depth | tok/s | ± | spread |
|---|---:|---:|---:|---:|
| pp2048 | 0 | 819.9 | 0.9 | 0.1% |
| tg256 | 0 | 69.1 | 1.6 | 2.4% |
| pp16384+tg512 | 0 | 356.1 | 39.1 | 11.0% **>5%** |
| pp2048 | 16384 | 374.0 | 3.8 | 1.0% |
| tg256 | 16384 | 34.1 | 0.3 | 0.9% |
| pp16384+tg512 | 16384 | 255.6 | 7.1 | 2.8% |
| pp2048 | 32768 | 238.0 | 2.1 | 0.9% |
| tg256 | 32768 | 30.4 | 0.4 | 1.2% |
| pp16384+tg512 | 32768 | 188.6 | 12.7 | 6.7% **>5%** |
| pp2048 | 65536 | 142.5 | 2.8 | 2.0% |
| tg256 | 65536 | 21.9 | 0.6 | 2.7% |
| pp16384+tg512 | 65536 | 125.1 | 4.5 | 3.6% |

### Replay, ten turns

| turn | prompt tokens | processed | cached | TTFT s | prompt tok/s | gen tok/s | turn s | cache |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9510 | 9510 | 0 | 13.52 | 703.5 | 57.02 | 20.54 | first |
| 2 | 12057 | 2148 | 9909 | 4.05 | 531.2 | 55.49 | 11.26 | hit |
| 3 | 14557 | 2101 | 12456 | 4.79 | 438.9 | 55.2 | 12.04 | hit |
| 4 | 17145 | 2189 | 14956 | 6.04 | 362.9 | 47.78 | 14.41 | hit |
| 5 | 19715 | 2171 | 17544 | 5.80 | 374.7 | 42.95 | 15.12 | hit |
| 6 | 22153 | 2039 | 20114 | 5.38 | 379.3 | 40.4 | 15.29 | hit |
| 7 | 24792 | 2240 | 22552 | 6.46 | 347.0 | 32.82 | 18.65 | hit |
| 8 | 27295 | 2104 | 25191 | 7.46 | 282.4 | 23.92 | 24.18 | hit |
| 9 | 29923 | 2229 | 27694 | 10.34 | 215.9 | 14.51 | 37.90 | hit |
| 10 | 32487 | 2165 | 30322 | 13.17 | 164.6 | 15.28 | 31.56 | hit |
| **total** | | | | | | | **201.2** | |

## metal relative to 3060

| test | depth | 3060 | metal | difference |
|---|---:|---:|---:|---:|
| pp2048 | 0 | 2165.3 | 819.9 | -62.1% |
| tg256 | 0 | 66.7 | 69.1 | +3.6% |
| pp16384+tg512 | 0 | 875.5 | 356.1 | -59.3% |
| pp2048 | 16384 | 1260.8 | 374.0 | -70.3% |
| tg256 | 16384 | 55.1 | 34.1 | -38.1% |
| pp16384+tg512 | 16384 | 650.1 | 255.6 | -60.7% |
| pp2048 | 32768 | 900.4 | 238.0 | -73.6% |
| tg256 | 32768 | 48.0 | 30.4 | -36.6% |
| pp16384+tg512 | 32768 | 519.3 | 188.6 | -63.7% |
| pp2048 | 65536 | 572.9 | 142.5 | -75.1% |
| tg256 | 65536 | 38.2 | 21.9 | -42.7% |
| pp16384+tg512 | 65536 | 369.3 | 125.1 | -66.1% |
| replay total | | 95.8 s | 201.2 s | +110.0% (time: negative is faster) |

## Anomalies

- 3060: three llama-server starts in the four minutes after the sixteen-minute sweep (15:33, 15:34, 15:36) aborted in
  the server's warm-up decode with an MMU fault the firmware reported (event 4101, empty payload; the compute channel and
  the SMs clean by the debugger's account), at Gen3 and at Gen1 alike. Not the build: the same binary came up at 15:41,
  and the enclosing tree's came up at 15:40. Not a lost configuration: no window loss, no retry. The boots after a few
  minutes of cool-down and the replay reported here (15:42) were clean; the card read 60 C at 170 W, its power cap, during
  the 15:40 run. Open in the handoff.
- metal: pp16384+tg512 at depth 0: repetitions spread 11.0% of the mean
- metal: pp16384+tg512 at depth 32768: repetitions spread 6.7% of the mean
