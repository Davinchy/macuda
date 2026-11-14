# RTX 4090 tensor unlock validation plan

## Implemented

- deterministic AD102 bundle preparation with immutable input and output pins;
- strict, default-off `TINYNV_TENSOR_UNLOCK` selection;
- standalone preflight and supervised unlock transaction;
- transaction integration before normal device initialization;
- verified runtime handoff and postboot selector checks;
- bounded socket operations and a recovery hold;
- BF16 correctness and multi-shape timing harnesses;
- mock transport tests for transaction, timeout, cleanup, and signal behavior;
- userspace-only operation with no TinyGPU application, server, or dext changes.

## Evidence completed

The first supervised live transaction reached `applied-and-verified`. The normal
570.144 runtime boot retained the override, the BF16 workload passed its existing
numerical tolerance, and seven open/workload/close cycles ended with clean cold
unloads. See [validation.md](validation.md) for the recorded matrix.

## Characterization still required

1. Power-cycle the enclosure and run a stock baseline on the full seven-shape
   suite.
2. Repeat matched stock/unlocked sessions across at least three independent power
   cycles.
3. Complete ten sequential lifecycle cycles.
4. Record an interleaved stock/unlocked performance sequence with matched clocks,
   temperature, power policy, build, and cubin.
5. Add a register-resident tensor-instruction probe.
6. Run end-to-end BF16 prefill with a model and prompt hash.
7. Confirm recovery behavior on a controlled failure when environmental conditions
   make that test safe.

The implementation remains experimental until these measurements are complete.

## Live protocol

Use [usage.md](usage.md). Every run starts with read-only preflight and the
exclusive GPU lock. Do not kill a process that has entered the recovery hold.

Stop on:

- an unexplained DART or MMU fault;
- lost PCI or BAR access;
- a failed reset containment gate;
- incorrect BF16 output;
- selector loss after reset or runtime boot; or
- any ambiguous firmware state.

## Acceptance

An open PR must prove:

- the default path is unchanged and inert;
- malformed or mismatched assets fail before device writes;
- the standalone transaction reaches every required hardware gate;
- runtime handoff preserves the override and produces correct BF16 output;
- cleanup is ordered and testable;
- the remaining performance and lifecycle limits are reported without claims
  beyond the evidence; and
- all relevant offline tests and shell checks pass.
