// AVX2 + FMA kernels; compiled with /arch:AVX2 or -mavx2 -mfma (see CMake).
#define GSIC_KERNEL_NAMESPACE avx2
#define GSIC_KERNEL_NAME "avx2"
#include "kernels.inl"
