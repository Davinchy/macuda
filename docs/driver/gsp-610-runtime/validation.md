# Validation record

## Offline

The runtime bundle and adapter suite validate:

- all eight required assets and the deterministic manifest;
- BTL1 container geometry, signatures, and fuse-selection bounds;
- hash, size, alignment, truncation, wrong-chip, and missing-asset failures;
- 570 and 610 firmware path and layout selection;
- 610 boot arguments and carveout placement;
- MCTP/NVDM framing, continuation records, empty replies, and RPC size;
- RM allocation/control payloads, channels, VA spaces, RUSD, and GR geometry;
- load-exec host requests and aperture restoration on failure;
- a complete boot/object-tree lifecycle against a mock GSP-RM peer;
- retry, per-stage cleanup, profile retention, unload, and 570 rollback; and
- unchanged 570 header, firmware, and memory behavior.

## Hardware

Maintainer validation on an RTX 4090 exercised boot, resource creation,
known-answer BF16 computation, and cold unload under both profiles.

| Profile | Lifecycle cycles | Compute check | Required result |
|---|---:|---:|---|
| 570.144 | 5 open/compute/cold-unload | passed | stock tensor state, clean DART state |
| 610.43.03 | 5 open/compute/cold-unload | passed | stock tensor state, clean DART state |

The 610 path reported `GSP-RM is up`, created the resource-management objects,
ran the compute workload, verified the firmware carveout, and left the card
cold after each cycle. A separate default-570 process then opened the card
successfully.

The public branch includes a hardware-independent reproduction of the boot,
resource, retry, and unload path.

## Reproduce

```sh
sh tools/gsp610-runtime/run_prepare.sh

make -C cuda-shim/libtinynv \
  BUILD=build-gsp610 \
  BUNDLE610="$PWD/cuda-shim/build/gsp610-runtime/bundle" \
  test-runtime-profile

sh tools/gsp610-live.sh preflight
sh tools/gsp610-live.sh boot610
sh tools/gsp610-live.sh lifecycle610
sh tools/gsp610-live.sh lifecycle570
```

## Limits

The hardware result covers one RTX 4090 and the BF16 workload above. The
standalone lifecycle runner reproduces boot and unload only. The result does
not establish support for another device, independent live FP16/FP32
coverage, a performance improvement, or crash-safe retention.
