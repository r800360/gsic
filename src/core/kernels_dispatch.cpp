// Picks the best kernel set for the CPU at runtime, so one binary runs
// everywhere: AVX2+FMA machines (2013+) get the wide path, everything else
// gets the portable baseline.
#include "kernels.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  #include <intrin.h>
#endif

namespace gsic {

namespace baseline { extern const Kernels k; }
#if defined(GSIC_HAVE_AVX2_TU)
namespace avx2 { extern const Kernels k; }
#endif

#if defined(GSIC_HAVE_AVX2_TU)
static bool cpu_has_avx2_fma() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuidex(r, 1, 0);
    const bool fma = (r[2] & (1 << 12)) != 0;
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    if (!fma || !osxsave) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;  // OS saves YMM state
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;                // AVX2
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
#endif

const Kernels& kernels() {
    // Held as a pointer rather than a reference. Both referents have static
    // storage duration, so a reference here is in fact safe, but GCC 13 cannot
    // see that through the lambda and reports -Wdangling-reference. A warning
    // that is always wrong is worse than no warning: it trains the reader to
    // skim past this file. The pointer says the same thing without the noise.
    static const Kernels* chosen = []() -> const Kernels* {
#if defined(GSIC_HAVE_AVX2_TU)
        if (cpu_has_avx2_fma()) return &avx2::k;
#endif
        return &baseline::k;
    }();
    return *chosen;
}

} // namespace gsic
