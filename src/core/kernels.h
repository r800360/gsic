#pragma once

#include <cstdint>

namespace gsic {

constexpr int kTileSize = 16;

// Everything the hot loops need, gathered in one struct so kernel entry
// points stay stable across CPU architectures. All pointers refer to planar
// float buffers owned by the renderer/trainer.
struct KernelCtx {
    int w = 0, h = 0, channels = 0;
    int tiles_x = 0, tiles_y = 0;

    // Image planes, each w*h floats.
    const float* target[4] = {};
    float*       render[4] = {};
    float*       grad[4]   = {};   // dLoss/dRender

    // Gaussian data.
    int n = 0;
    const float* color = nullptr;   // [n * channels]
    const float* sinv_x = nullptr;  // raw parameters, needed for the chain rule
    const float* sinv_y = nullptr;
    const float* rot = nullptr;

    // Projection results (computed per step by the renderer).
    const float* cx = nullptr;      // center, pixels
    const float* cy = nullptr;
    const float* conic_a = nullptr; // inverse covariance [[a, b], [b, c]]
    const float* conic_b = nullptr;
    const float* conic_c = nullptr;
    const std::int32_t* bbox = nullptr; // [n * 4] x0, y0, x1, y1 (clamped, half-open)

    // Tile binning: for tile t, gaussian ids are
    // tile_items[tile_offsets[t] .. tile_offsets[t + 1]).
    const std::uint32_t* tile_offsets = nullptr;
    const std::uint32_t* tile_items = nullptr;

    // Loss configuration. grad = inv_norm * (w_l1 * sign(d) + 2 * w_l2 * d).
    float w_l1 = 1.f, w_l2 = 0.f, inv_norm = 1.f;

    // Parameter gradients (outputs of backward), each [n] / [n * channels].
    float* d_pos_x = nullptr;
    float* d_pos_y = nullptr;
    float* d_sinv_x = nullptr;
    float* d_sinv_y = nullptr;
    float* d_rot = nullptr;
    float* d_color = nullptr;
};

// One Adam parameter-group update over params[0, n), with clamping.
struct AdamArgs {
    float* param; float* m; float* v;
    const float* grad;
    std::int64_t n;
    float lr, beta1, beta2, eps;
    float bias1, bias2;      // 1 - beta^t corrections
    float lo, hi;            // parameter clamp range
};

struct Kernels {
    const char* name;
    // Accumulate gaussians into render planes for tiles [begin, end).
    void (*forward_tiles)(const KernelCtx&, int begin, int end);
    // Training fast path: forward-render tiles [begin, end) and immediately
    // turn them into grad tiles against the target, accumulating L1/L2 sums.
    // Only grad planes are written; render planes are untouched (saves two
    // full-image passes of memory traffic per step).
    void (*forward_loss_tiles)(const KernelCtx&, int begin, int end, double* l1, double* l2);
    // Compute grad planes from render/target for rows [y0, y1);
    // adds this range's L1 and L2 sums to the accumulators.
    void (*loss_grad_rows)(const KernelCtx&, int y0, int y1, double* l1, double* l2);
    // Per-gaussian gradient gather for gaussians [begin, end).
    void (*backward_gaussians)(const KernelCtx&, int begin, int end);
    void (*adam_step)(const AdamArgs&);
};

// Best implementation for the CPU we are running on (AVX2+FMA when available,
// otherwise the portable baseline).
const Kernels& kernels();

} // namespace gsic
