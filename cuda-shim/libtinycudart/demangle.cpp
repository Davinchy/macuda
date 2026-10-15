// LLVM's Itanium demangler, for matching kernel names across the macOS-host / Linux-device seam. Apple's __cxa_demangle
// cannot parse the function-pointer template arguments (Xad...E) that ggml's dequantize kernels carry; LLVM's can, and it
// is the same demangler clang itself uses. Returns a malloc'd string (free with free()), or NULL.
#include <llvm/Demangle/Demangle.h>
#include <string_view>
extern "C" char *tinycudart_demangle(const char *mangled) { return llvm::itaniumDemangle(std::string_view(mangled)); }
