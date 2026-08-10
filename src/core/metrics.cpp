#include "metrics.h"

#include "thread_pool.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace gsic {

double psnr(const Image& a, const Image& b) {
    if (a.w != b.w || a.h != b.h || a.c != b.c || a.empty()) return 0.0;
    double se = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const double d = std::clamp(a.data[i], 0.f, 1.f) - std::clamp(b.data[i], 0.f, 1.f);
        se += d * d;
    }
    const double mse = se / double(a.data.size());
    if (mse <= 1e-12) return 99.0;
    return 10.0 * std::log10(1.0 / mse);
}

// Separable gaussian blur used by SSIM (sigma 1.5, radius 5).
static void blur(const float* src, float* dst, float* tmp, int w, int h) {
    static constexpr int R = 5;
    static const auto kernel = [] {
        std::array<float, 2 * R + 1> k{};
        float sum = 0.f;
        for (int i = -R; i <= R; ++i) sum += k[i + R] = std::exp(-float(i * i) / (2.f * 1.5f * 1.5f));
        for (auto& v : k) v /= sum;
        return k;
    }();
    auto& pool = ThreadPool::instance();
    pool.parallel_for(h, 64, [&](std::int64_t y0, std::int64_t y1) {
        for (int y = int(y0); y < int(y1); ++y) {
            const std::size_t row = std::size_t(y) * w;
            for (int x = 0; x < w; ++x) {
                float s = 0.f;
                for (int i = -R; i <= R; ++i)
                    s += kernel[i + R] * src[row + std::size_t(std::clamp(x + i, 0, w - 1))];
                tmp[row + x] = s;
            }
        }
    });
    pool.parallel_for(h, 64, [&](std::int64_t y0, std::int64_t y1) {
        for (int y = int(y0); y < int(y1); ++y) {
            const std::size_t row = std::size_t(y) * w;
            for (int x = 0; x < w; ++x) {
                float s = 0.f;
                for (int i = -R; i <= R; ++i)
                    s += kernel[i + R] *
                         tmp[std::size_t(std::clamp(y + i, 0, h - 1)) * w + x];
                dst[row + x] = s;
            }
        }
    });
}

double ssim(const Image& a, const Image& b) {
    if (a.w != b.w || a.h != b.h || a.c != b.c || a.empty()) return 0.0;
    const int w = a.w, h = a.h;
    // Image size is capped so this always fits (see gaussians.h); indexing in
    // size_t keeps 32-bit builds free of silent truncation.
    const std::size_t n = std::size_t(a.pixels());
    constexpr double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;

    std::vector<float> mu_a(n), mu_b(n), aa(n), bb(n), ab(n), t1(n), t2(n), t3(n), tmp(n);
    double total = 0.0;
    for (int ch = 0; ch < a.c; ++ch) {
        const float* pa = a.plane(ch);
        const float* pb = b.plane(ch);
        for (std::size_t i = 0; i < n; ++i) {
            const float xa = std::clamp(pa[i], 0.f, 1.f), xb = std::clamp(pb[i], 0.f, 1.f);
            aa[i] = xa * xa; bb[i] = xb * xb; ab[i] = xa * xb;
            t1[i] = xa; t2[i] = xb;
        }
        blur(t1.data(), mu_a.data(), tmp.data(), w, h);
        blur(t2.data(), mu_b.data(), tmp.data(), w, h);
        blur(aa.data(), t1.data(), tmp.data(), w, h);   // E[a^2]
        blur(bb.data(), t2.data(), tmp.data(), w, h);   // E[b^2]
        blur(ab.data(), t3.data(), tmp.data(), w, h);   // E[ab]
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double ma = mu_a[i], mb = mu_b[i];
            const double va = t1[i] - ma * ma, vb = t2[i] - mb * mb, cab = t3[i] - ma * mb;
            sum += ((2 * ma * mb + C1) * (2 * cab + C2)) /
                   ((ma * ma + mb * mb + C1) * (va + vb + C2));
        }
        total += sum / double(n);
    }
    return total / a.c;
}

} // namespace gsic
