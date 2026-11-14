# TinyGPU transport constraints

The unlock uses the installed TinyGPU socket transport without changing the
application, server, or signed DriverKit extension. This document records the
constraints enforced by macuda userspace.

## Operations used

The transaction relies on:

- PCI configuration-space reads and writes;
- BAR0 MMIO reads and writes;
- BAR1 views for VRAM access;
- temporary shared-memory mappings with DART IOVAs;
- a PCI function-reset operation; and
- bus-master quiesce followed by configuration readback.

The server accepts one client at a time. macuda still uses the project GPU lock
because the dext enables bus mastering when the provider starts, before a client
opens the transport.

## Address spaces

| Address | Owner | Use |
|---|---|---|
| CPU pointer | Client | Writable shared-memory view |
| DART IOVA | Device | Address published to firmware DMA |
| GPU physical | GPU allocator | VRAM address used by Falcon fetch and page tables |

The client and device address spaces do not alias. The BAR mapping returned by
the server is a server virtual address, not a PCI bus address.

## Buffer requirements

Before publishing any host address, the transaction verifies:

- every IOVA is below `2^32`;
- pages are contiguous for the Falcon DMA length;
- required alignments are preserved;
- the segment table fits the transport's limits;
- the first page is parsed before it is overwritten; and
- a release fence orders CPU stores before a device doorbell or MMIO write.

Malformed, truncated, or non-contiguous mappings fail the transaction. VRAM
physical addresses follow a separate allocation and BAR1-window protocol.

## Reset semantics

The transport reset requests `kIOPCIDeviceResetTypeFunctionReset`. The platform
reset path restores PCI configuration, but the RPC does not expose enough status
for success to stand alone.

After reset, macuda verifies:

- PCI identity and BAR accessibility;
- chip identity and firmware boot state;
- SEC2 and GSP halt state;
- WPR2 down and no pending transaction;
- COMMAND and BAR restoration; and
- FEAT, effective readout, SS0, and SS1 retention.

The same selector checks run after every runtime-boot retry and after the normal
570.144 firmware boot.

## Waiting and errors

All device waits have bounded deadlines. The TinyGPU client sets a receive
timeout so a wedged server cannot block forever. A catchable signal that
interrupts a socket transfer retries only when the protocol guarantees no bytes
were consumed.

MMIO writes are followed by readback because an out-of-range or unmapped write can
otherwise be lost. Register reads that return all ones or documented hardware
error patterns invalidate the stage.

## Containment

DMA cannot be released per buffer. Buffers remain valid until the session ends.
Before ending a session after firmware launch, the transaction:

1. disables bus mastering and reads the command register back;
2. resets the device;
3. verifies that SEC2 and GSP stopped and WPR2 is down; and
4. restores PCI state before releasing the transport.

If the halt gates fail, the process enters a recovery hold rather than closing the
socket. This hold protects against catchable termination and protocol errors, but
the installed transport cannot distinguish a safe disconnect from process death.
A `SIGKILL`, client crash, server death, or host failure can still unmap live DMA.
The only available recovery is physical enclosure power-off and the feature's
documentation must not claim stronger crash safety.
