# RTX 4090 tensor unlock

Experimental userspace support for enabling the AD102 tensor speed override on a
desktop RTX 4090 reached through the existing TinyGPU transport.

Start with:

- [Specification](spec.md)
- [Design](design.md)
- [Usage and recovery](usage.md)
- [Validation record](validation.md)
- [Transport constraints](transport.md)
- [Asset provenance](assets.md)

The feature is opt-in with `TINYNV_TENSOR_UNLOCK=1`, off by default, and limited
to the tested RTX 4090. It changes no TinyGPU application, server, DriverKit
extension, CUDA API, or kernel arithmetic.
