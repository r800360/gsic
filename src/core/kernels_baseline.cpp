// Portable kernels: SSE2 on x86-64, NEON on ARM (via autovectorization).
#define GSIC_KERNEL_NAMESPACE baseline
#define GSIC_KERNEL_NAME "baseline"
#include "kernels.inl"
