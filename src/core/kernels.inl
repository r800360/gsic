// Hot-loop implementations, compiled once per instruction set. The including
// TU defines GSIC_KERNEL_NAMESPACE (e.g. baseline, avx2) and arch flags; the
// AVX2 TU picks up the intrinsic paths via __AVX2__.
//
// Rendering model (per pixel p, channel c):
//   out_c(p) = sum_i alpha_i(p) * color_ic,
//   alpha    = max(exp(-sigma) - kAlphaBias, 0),   sigma = 0.5 * d^T Conic d.
// The sum is unnormalized (GaussianImage-style), which makes every gaussian's
// gradient independent of the others: backward is a per-gaussian gather with
// no atomics. The bias keeps alpha continuous at the cutoff ellipse
// (d alpha / d sigma is still just -exp(-sigma)).
//
// Along one pixel row (fixed dy), sigma is a quadratic in x:
//   sigma(x) = qa * dx^2 + qb * dx + qc,   dx = x + 0.5 - center_x,
// so each row's live pixels form one interval, found by solving
// sigma(x) = cutoff. Rows outside the ellipse are skipped entirely and the
// interval trims the ~25% of bounding-box pixels that fall outside it.

#include "gaussians.h"
#include "kernels.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

namespace gsic {
namespace GSIC_KERNEL_NAMESPACE {

// ------------------------------------------------------------------ fast exp
// exp(-s) for s >= 0 via 2^t split into integer/fraction parts; the fraction
// uses a degree-4 polynomial (rel. error ~1e-4, far below what 8..16-bit
// color quantization can resolve). Returns 0 where s >= kSigmaCutoff.
inline float exp_neg_masked(float s) {
    if (s >= kSigmaCutoff) return 0.f;
    float t = -1.44269504f * s;
    int i = int(t);
    i -= (t < float(i));                       // floor for negative t
    float f = t - float(i);
    float p = 1.f + f * (0.69314718f + f * (0.24022651f + f * (0.05550411f + f * 0.00961813f)));
    return std::bit_cast<float>((i + 127) << 23) * p;
}

// The live x-interval [out_x0, out_x1) of one row, clipped to [x0, x1).
// Returns false when the row misses the ellipse.
inline bool row_interval(float qa, float qb, float qc, float cx, int x0, int x1,
                         int& out_x0, int& out_x1) {
    const float disc = qb * qb - 4.f * qa * (qc - kSigmaCutoff);
    if (disc <= 0.f) return false;
    const float sq = std::sqrt(disc);
    const float inv2a = 0.5f / qa;
    const float base = cx - 0.5f;
    out_x0 = std::max(x0, int(std::ceil(base + (-qb - sq) * inv2a)));
    out_x1 = std::min(x1, int(std::floor(base + (-qb + sq) * inv2a)) + 1);
    return out_x0 < out_x1;
}

#if defined(__AVX2__)
// 8-lane exp(-s), masked to zero where s >= cutoff.
inline __m256 exp_neg8(__m256 s) {
    const __m256 t = _mm256_mul_ps(s, _mm256_set1_ps(-1.44269504f));
    const __m256 fi = _mm256_floor_ps(t);
    const __m256 f = _mm256_sub_ps(t, fi);
    __m256 p = _mm256_fmadd_ps(f, _mm256_set1_ps(0.00961813f), _mm256_set1_ps(0.05550411f));
    p = _mm256_fmadd_ps(f, p, _mm256_set1_ps(0.24022651f));
    p = _mm256_fmadd_ps(f, p, _mm256_set1_ps(0.69314718f));
    p = _mm256_fmadd_ps(f, p, _mm256_set1_ps(1.f));
    const __m256i ei = _mm256_add_epi32(_mm256_cvtps_epi32(fi), _mm256_set1_epi32(127));
    const __m256 e = _mm256_mul_ps(_mm256_castsi256_ps(_mm256_slli_epi32(ei, 23)), p);
    return _mm256_and_ps(e, _mm256_cmp_ps(s, _mm256_set1_ps(kSigmaCutoff), _CMP_LT_OQ));
}

inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// Load mask for the last (n % 8) lanes of a row.
inline __m256i tail_mask(int live) {
    alignas(32) static const std::int32_t bits[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                                      0, 0, 0, 0, 0, 0, 0, 0};
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bits + 8 - live));
}
#endif

// ------------------------------------------------------------------- forward
// Each tile accumulates its binned gaussians into a local buffer (stays in
// L1). Tiles partition the image, so destination planes never need clearing.
template <int C>
static void accumulate_tile(const KernelCtx& k, int tile, float* acc, int x0, int y0, int th) {
    constexpr int T = kTileSize;
    {
        const std::uint32_t* items = k.tile_items + k.tile_offsets[tile];
        const int count = int(k.tile_offsets[tile + 1] - k.tile_offsets[tile]);
        for (int it = 0; it < count; ++it) {
            const int g = int(items[it]);
            const float cx = k.cx[g], cy = k.cy[g];
            const float a = k.conic_a[g], b = k.conic_b[g], c = k.conic_c[g];
            const float* col = k.color + size_t(g) * C;
            const int ry0 = std::max(y0, k.bbox[g * 4 + 1]);
            const int ry1 = std::min(y0 + th, k.bbox[g * 4 + 3]);
            const float dx0 = float(x0) + 0.5f - cx;
            const float qa = 0.5f * a;
#if defined(__AVX2__)
            const __m256 vqa = _mm256_set1_ps(qa);
            const __m256 vbias = _mm256_set1_ps(kAlphaBias);
            __m256 vcol[C];
            for (int ch = 0; ch < C; ++ch) vcol[ch] = _mm256_set1_ps(col[ch]);
            const __m256 vdx_lo =
                _mm256_add_ps(_mm256_set1_ps(dx0), _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7));
            const __m256 vdx_hi = _mm256_add_ps(vdx_lo, _mm256_set1_ps(8.f));
#endif
            for (int y = ry0; y < ry1; ++y) {
                const float dy = float(y) + 0.5f - cy;
                const float qb = b * dy, qc = 0.5f * c * dy * dy;
                int rx0, rx1;
                if (!row_interval(qa, qb, qc, cx, x0, x0 + T, rx0, rx1)) continue;
                const int ly = y - y0;
#if defined(__AVX2__)
                const __m256 vqb = _mm256_set1_ps(qb), vqc = _mm256_set1_ps(qc);
                // Two 8-lane blocks cover the tile row; skip blocks the
                // ellipse interval misses (masking handles partial lanes).
                if (rx0 < x0 + 8) {
                    __m256 s = _mm256_fmadd_ps(_mm256_fmadd_ps(vqa, vdx_lo, vqb), vdx_lo, vqc);
                    __m256 al = _mm256_max_ps(_mm256_sub_ps(exp_neg8(s), vbias),
                                              _mm256_setzero_ps());
                    for (int ch = 0; ch < C; ++ch) {
                        float* dst = acc + (ch * T + ly) * T;
                        _mm256_store_ps(dst, _mm256_fmadd_ps(al, vcol[ch], _mm256_load_ps(dst)));
                    }
                }
                if (rx1 > x0 + 8) {
                    __m256 s = _mm256_fmadd_ps(_mm256_fmadd_ps(vqa, vdx_hi, vqb), vdx_hi, vqc);
                    __m256 al = _mm256_max_ps(_mm256_sub_ps(exp_neg8(s), vbias),
                                              _mm256_setzero_ps());
                    for (int ch = 0; ch < C; ++ch) {
                        float* dst = acc + (ch * T + ly) * T + 8;
                        _mm256_store_ps(dst, _mm256_fmadd_ps(al, vcol[ch], _mm256_load_ps(dst)));
                    }
                }
#else
                for (int x = rx0; x < rx1; ++x) {
                    const float dx = dx0 + float(x - x0);
                    const float s = (qa * dx + qb) * dx + qc;
                    const float al = std::max(exp_neg_masked(s) - kAlphaBias, 0.f);
                    for (int ch = 0; ch < C; ++ch) acc[(ch * T + ly) * T + (x - x0)] += al * col[ch];
                }
#endif
            }
        }
    }
}

template <int C>
static void forward_tiles_impl(const KernelCtx& k, int begin, int end) {
    constexpr int T = kTileSize;
    alignas(32) float acc[C * T * T];
    for (int tile = begin; tile < end; ++tile) {
        const int tx = tile % k.tiles_x, ty = tile / k.tiles_x;
        const int x0 = tx * T, y0 = ty * T;
        const int tw = std::min(T, k.w - x0), th = std::min(T, k.h - y0);
        std::memset(acc, 0, sizeof(acc));
        accumulate_tile<C>(k, tile, acc, x0, y0, th);
        for (int ch = 0; ch < C; ++ch) {
            float* plane = k.render[ch];
            for (int y = 0; y < th; ++y)
                std::memcpy(plane + size_t(y0 + y) * k.w + x0, acc + (ch * T + y) * T,
                            size_t(tw) * sizeof(float));
        }
    }
}

// Training fast path: the freshly accumulated tile becomes a grad tile
// in-register; the full-image render is never materialized.
template <int C>
static void forward_loss_tiles_impl(const KernelCtx& k, int begin, int end, double* l1,
                                    double* l2) {
    constexpr int T = kTileSize;
    alignas(32) float acc[C * T * T];
    float sum1 = 0.f, sum2 = 0.f;
    for (int tile = begin; tile < end; ++tile) {
        const int tx = tile % k.tiles_x, ty = tile / k.tiles_x;
        const int x0 = tx * T, y0 = ty * T;
        const int tw = std::min(T, k.w - x0), th = std::min(T, k.h - y0);
        std::memset(acc, 0, sizeof(acc));
        accumulate_tile<C>(k, tile, acc, x0, y0, th);
        for (int ch = 0; ch < C; ++ch) {
            const float* tplane = k.target[ch];
            float* gplane = k.grad[ch];
            for (int y = 0; y < th; ++y) {
                const float* row = acc + (ch * T + y) * T;
                const float* t = tplane + size_t(y0 + y) * k.w + x0;
                float* g = gplane + size_t(y0 + y) * k.w + x0;
                int x = 0;
#if defined(__AVX2__)
                const __m256 vw1 = _mm256_set1_ps(k.w_l1);
                const __m256 v2w2 = _mm256_set1_ps(2.f * k.w_l2);
                const __m256 vinv = _mm256_set1_ps(k.inv_norm);
                const __m256 sign_bit = _mm256_set1_ps(-0.f);
                __m256 vs1 = _mm256_setzero_ps(), vs2 = _mm256_setzero_ps();
                for (; x + 8 <= tw; x += 8) {
                    const __m256 d = _mm256_sub_ps(_mm256_load_ps(row + x), _mm256_loadu_ps(t + x));
                    vs1 = _mm256_add_ps(vs1, _mm256_andnot_ps(sign_bit, d));
                    vs2 = _mm256_fmadd_ps(d, d, vs2);
                    const __m256 sw1 = _mm256_or_ps(vw1, _mm256_and_ps(sign_bit, d));
                    _mm256_storeu_ps(g + x, _mm256_mul_ps(vinv, _mm256_fmadd_ps(v2w2, d, sw1)));
                }
                sum1 += hsum8(vs1);
                sum2 += hsum8(vs2);
#endif
                for (; x < tw; ++x) {
                    const float d = row[x] - t[x];
                    sum1 += std::fabs(d);
                    sum2 += d * d;
                    g[x] = k.inv_norm * (std::copysign(k.w_l1, d) + 2.f * k.w_l2 * d);
                }
            }
        }
    }
    *l1 += sum1;
    *l2 += sum2;
}

static void forward_tiles(const KernelCtx& k, int begin, int end) {
    switch (k.channels) {
        case 1: forward_tiles_impl<1>(k, begin, end); break;
        case 2: forward_tiles_impl<2>(k, begin, end); break;
        case 3: forward_tiles_impl<3>(k, begin, end); break;
        default: forward_tiles_impl<4>(k, begin, end); break;
    }
}

static void forward_loss_tiles(const KernelCtx& k, int begin, int end, double* l1, double* l2) {
    switch (k.channels) {
        case 1: forward_loss_tiles_impl<1>(k, begin, end, l1, l2); break;
        case 2: forward_loss_tiles_impl<2>(k, begin, end, l1, l2); break;
        case 3: forward_loss_tiles_impl<3>(k, begin, end, l1, l2); break;
        default: forward_loss_tiles_impl<4>(k, begin, end, l1, l2); break;
    }
}

// ----------------------------------------------------------------- loss grad
// grad = inv_norm * (w_l1 * sign(d) + 2 * w_l2 * d), d = render - target.
static void loss_grad_rows(const KernelCtx& k, int y0, int y1, double* l1, double* l2) {
    const std::int64_t i0 = std::int64_t(y0) * k.w, i1 = std::int64_t(y1) * k.w;
    float sum1 = 0.f, sum2 = 0.f;  // per-call range is small enough for float
    for (int ch = 0; ch < k.channels; ++ch) {
        const float* r = k.render[ch];
        const float* t = k.target[ch];
        float* g = k.grad[ch];
        std::int64_t i = i0;
#if defined(__AVX2__)
        const __m256 vw1 = _mm256_set1_ps(k.w_l1);
        const __m256 v2w2 = _mm256_set1_ps(2.f * k.w_l2);
        const __m256 vinv = _mm256_set1_ps(k.inv_norm);
        const __m256 sign_bit = _mm256_set1_ps(-0.f);
        __m256 vs1 = _mm256_setzero_ps(), vs2 = _mm256_setzero_ps();
        for (; i + 8 <= i1; i += 8) {
            const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(r + i), _mm256_loadu_ps(t + i));
            const __m256 ad = _mm256_andnot_ps(sign_bit, d);
            vs1 = _mm256_add_ps(vs1, ad);
            vs2 = _mm256_fmadd_ps(d, d, vs2);
            // w_l1 with d's sign, plus the smooth L2 term.
            const __m256 sw1 = _mm256_or_ps(vw1, _mm256_and_ps(sign_bit, d));
            _mm256_storeu_ps(g + i, _mm256_mul_ps(vinv, _mm256_fmadd_ps(v2w2, d, sw1)));
        }
        sum1 += hsum8(vs1);
        sum2 += hsum8(vs2);
#endif
        for (; i < i1; ++i) {
            const float d = r[i] - t[i];
            sum1 += std::fabs(d);
            sum2 += d * d;
            g[i] = k.inv_norm * (std::copysign(k.w_l1, d) + 2.f * k.w_l2 * d);
        }
    }
    *l1 += sum1;
    *l2 += sum2;
}

// ------------------------------------------------------------------ backward
// Per-gaussian gather over its own bounding box. For one row (fixed dy), all
// five geometric gradients reduce to three moment sums over t = dLoss/dsigma:
//   S0 = sum t,  S1 = sum t*dx,  S2 = sum t*dx^2
//   dA += S2/2          dB += dy*S1         dC += dy^2*S0/2
//   dcx -= a*S1 + b*dy*S0                   dcy -= b*S1 + c*dy*S0
// which keeps the inner loop down to a handful of FMAs per pixel.
template <int C>
static void backward_impl(const KernelCtx& k, int begin, int end) {
    for (int g = begin; g < end; ++g) {
        const int bx0 = k.bbox[g * 4 + 0], by0 = k.bbox[g * 4 + 1];
        const int bx1 = k.bbox[g * 4 + 2], by1 = k.bbox[g * 4 + 3];
        const float cx = k.cx[g], cy = k.cy[g];
        const float a = k.conic_a[g], b = k.conic_b[g], c = k.conic_c[g];
        float col[C];
        for (int ch = 0; ch < C; ++ch) col[ch] = k.color[size_t(g) * C + ch];

        float dA = 0, dB = 0, dC = 0, dcx = 0, dcy = 0;
        float dcol[C] = {};
        const float qa = 0.5f * a;
#if defined(__AVX2__)
        const __m256 vqa = _mm256_set1_ps(qa);
        const __m256 vbias = _mm256_set1_ps(kAlphaBias);
        const __m256 vidx = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        __m256 vcol[C], vdcol[C];
        for (int ch = 0; ch < C; ++ch) {
            vcol[ch] = _mm256_set1_ps(col[ch]);
            vdcol[ch] = _mm256_setzero_ps();
        }
#endif
        for (int y = by0; y < by1; ++y) {
            const float dy = float(y) + 0.5f - cy;
            const float qb = b * dy, qc = 0.5f * c * dy * dy;
            int rx0, rx1;
            if (!row_interval(qa, qb, qc, cx, bx0, bx1, rx0, rx1)) continue;
            const float dxr = float(rx0) + 0.5f - cx;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f;
#if defined(__AVX2__)
            const __m256 vqb = _mm256_set1_ps(qb), vqc = _mm256_set1_ps(qc);
            __m256 vs0 = _mm256_setzero_ps(), vs1 = _mm256_setzero_ps(),
                   vs2 = _mm256_setzero_ps();
            const int n = rx1 - rx0;
            const float* grow[C];
            for (int ch = 0; ch < C; ++ch) grow[ch] = k.grad[ch] + std::int64_t(y) * k.w + rx0;
            for (int x = 0; x < n; x += 8) {
                const int live = std::min(8, n - x);
                const __m256 vdx = _mm256_add_ps(_mm256_set1_ps(dxr + float(x)), vidx);
                const __m256 s = _mm256_fmadd_ps(_mm256_fmadd_ps(vqa, vdx, vqb), vdx, vqc);
                const __m256 e = exp_neg8(s);
                const __m256 ap = _mm256_max_ps(_mm256_sub_ps(e, vbias), _mm256_setzero_ps());
                // Masked tail load keeps lanes past the row at zero, which
                // zeroes their contribution to every accumulator.
                __m256 w = _mm256_setzero_ps();
                for (int ch = 0; ch < C; ++ch) {
                    const __m256 gg = live == 8
                                          ? _mm256_loadu_ps(grow[ch] + x)
                                          : _mm256_maskload_ps(grow[ch] + x, tail_mask(live));
                    vdcol[ch] = _mm256_fmadd_ps(ap, gg, vdcol[ch]);
                    w = _mm256_fmadd_ps(vcol[ch], gg, w);
                }
                const __m256 t = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), e), w);
                const __m256 tdx = _mm256_mul_ps(t, vdx);
                vs0 = _mm256_add_ps(vs0, t);
                vs1 = _mm256_add_ps(vs1, tdx);
                vs2 = _mm256_fmadd_ps(tdx, vdx, vs2);
            }
            s0 = hsum8(vs0);
            s1 = hsum8(vs1);
            s2 = hsum8(vs2);
#else
            for (int x = rx0; x < rx1; ++x) {
                const float dx = dxr + float(x - rx0);
                const float s = (qa * dx + qb) * dx + qc;
                const float e = exp_neg_masked(s);
                const float ap = std::max(e - kAlphaBias, 0.f);
                float w = 0.f;
                for (int ch = 0; ch < C; ++ch) {
                    const float gg = k.grad[ch][std::int64_t(y) * k.w + x];
                    dcol[ch] += ap * gg;
                    w += col[ch] * gg;
                }
                const float t = -e * w;
                const float tdx = t * dx;
                s0 += t;
                s1 += tdx;
                s2 += tdx * dx;
            }
#endif
            dA += 0.5f * s2;
            dB += dy * s1;
            dC += 0.5f * dy * dy * s0;
            dcx -= a * s1 + b * dy * s0;
            dcy -= b * s1 + c * dy * s0;
        }
#if defined(__AVX2__)
        for (int ch = 0; ch < C; ++ch) dcol[ch] = hsum8(vdcol[ch]);
#endif

        // Chain rule from conic entries to the raw parameters.
        const float u = k.sinv_x[g], v = k.sinv_y[g];
        const float co = std::cos(k.rot[g]), si = std::sin(k.rot[g]);
        const float u2v2 = u * u - v * v;
        const float s2t = 2.f * si * co, c2t = co * co - si * si;  // sin/cos(2 theta)
        k.d_pos_x[g] = dcx * float(k.w);
        k.d_pos_y[g] = dcy * float(k.h);
        k.d_sinv_x[g] = 2.f * u * (dA * co * co + dB * si * co + dC * si * si);
        k.d_sinv_y[g] = 2.f * v * (dA * si * si - dB * si * co + dC * co * co);
        k.d_rot[g] = u2v2 * (-dA * s2t + dB * c2t + dC * s2t);
        for (int ch = 0; ch < C; ++ch) k.d_color[size_t(g) * C + ch] = dcol[ch];
    }
}

static void backward_gaussians(const KernelCtx& k, int begin, int end) {
    switch (k.channels) {
        case 1: backward_impl<1>(k, begin, end); break;
        case 2: backward_impl<2>(k, begin, end); break;
        case 3: backward_impl<3>(k, begin, end); break;
        default: backward_impl<4>(k, begin, end); break;
    }
}

// ---------------------------------------------------------------------- adam
static void adam_step(const AdamArgs& a) {
    const float ib1 = 1.f / a.bias1, ib2 = 1.f / a.bias2;
    std::int64_t i = 0;
#if defined(__AVX2__)
    const __m256 vb1 = _mm256_set1_ps(a.beta1), vb1c = _mm256_set1_ps(1.f - a.beta1);
    const __m256 vb2 = _mm256_set1_ps(a.beta2), vb2c = _mm256_set1_ps(1.f - a.beta2);
    const __m256 vlr = _mm256_set1_ps(a.lr), veps = _mm256_set1_ps(a.eps);
    const __m256 vib1 = _mm256_set1_ps(ib1), vib2 = _mm256_set1_ps(ib2);
    const __m256 vlo = _mm256_set1_ps(a.lo), vhi = _mm256_set1_ps(a.hi);
    for (; i + 8 <= a.n; i += 8) {
        const __m256 g = _mm256_loadu_ps(a.grad + i);
        const __m256 m = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(a.m + i), _mm256_mul_ps(vb1c, g));
        const __m256 v = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(a.v + i),
                                         _mm256_mul_ps(vb2c, _mm256_mul_ps(g, g)));
        _mm256_storeu_ps(a.m + i, m);
        _mm256_storeu_ps(a.v + i, v);
        const __m256 den =
            _mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(v, vib2)), veps);
        const __m256 stepv = _mm256_div_ps(_mm256_mul_ps(vlr, _mm256_mul_ps(m, vib1)), den);
        __m256 p = _mm256_sub_ps(_mm256_loadu_ps(a.param + i), stepv);
        p = _mm256_min_ps(_mm256_max_ps(p, vlo), vhi);
        _mm256_storeu_ps(a.param + i, p);
    }
#endif
    for (; i < a.n; ++i) {
        const float g = a.grad[i];
        a.m[i] = a.beta1 * a.m[i] + (1.f - a.beta1) * g;
        a.v[i] = a.beta2 * a.v[i] + (1.f - a.beta2) * g * g;
        const float p = a.param[i] - a.lr * (a.m[i] * ib1) / (std::sqrt(a.v[i] * ib2) + a.eps);
        a.param[i] = std::clamp(p, a.lo, a.hi);
    }
}

extern const Kernels k = {
    GSIC_KERNEL_NAME,
    &forward_tiles,
    &forward_loss_tiles,
    &loss_grad_rows,
    &backward_gaussians,
    &adam_step,
};

} // namespace GSIC_KERNEL_NAMESPACE
} // namespace gsic
