#pragma once

#include "codec.h"
#include "gaussians.h"
#include "image.h"

#include <atomic>
#include <functional>
#include <memory>

namespace gsic {

enum class Backend { Auto, Cpu, Gpu };

// Defaults follow the Image-GS paper where applicable; steps/counts are the
// "balanced" preset.
struct EncodeSettings {
    // Quality / size trade-off.
    int num_gaussians = 0;          // 0 -> derived: pixels / pixels_per_gaussian
    int pixels_per_gaussian = 400;
    int max_steps = 4000;
    QuantSpec quant;

    // Loss (mean-normalized) and Adam learning rates.
    float w_l1 = 1.f, w_l2 = 0.f;
    float lr_pos = 5e-4f, lr_scale = 2e-3f, lr_rot = 2e-3f, lr_feat = 5e-3f;

    // Initialization.
    float init_scale_px = 5.f;
    float init_random_ratio = 0.3f;   // fraction placed uniformly at random

    // Largest allowed gaussian std-dev in pixels. Per-step cost is
    // proportional to total gaussian footprint, so this bounds the damage a
    // few background gaussians can do; it rarely limits quality at default
    // budgets.
    float max_scale_px = 32.f;

    // Note on coarse-to-fine: optimizing up a resolution pyramid is the
    // obvious way to cut the O(pixels) per-step cost, and it was measured
    // here at both 2K and 8K. It loses. Gradient-guided placement, color
    // initialization from the source, and error-guided progressive growth
    // already put gaussians in the right places with the right colors, so the
    // coarse levels have nothing left to teach and simply consume steps that
    // are worth more at full resolution. Spending the same time on fewer
    // full-resolution steps beat every pyramid schedule tried.

    // Error-guided progressive optimization: start with a fraction of the
    // gaussians and add the rest where reconstruction error is largest.
    bool progressive = true;
    float initial_ratio = 0.5f;
    int add_times = 4;

    // Plateau learning-rate decay + early stop.
    bool lr_decay = true;

    std::uint32_t seed = 123;
    Backend backend = Backend::Auto;
    int preview_interval_ms = 250;    // 0 disables preview snapshots
};

struct EncodeProgress {
    int step = 0, max_steps = 0;
    float loss = 0.f;
    int num_gaussians = 0;
    const Image* preview = nullptr;   // non-null when a snapshot was taken
};

struct EncodeStats {
    double psnr = 0, ssim = 0;
    double encode_seconds = 0;    // everything: init, optimize, quantize, measure
    double optimize_seconds = 0;  // the optimization loop alone
    int steps_run = 0;
    int num_gaussians = 0;
    const char* backend_used = "cpu";
    std::int64_t file_bytes = 0;
    std::int64_t source_bytes = 0;    // w * h * c at 8 bits per channel
};

struct EncodeResult {
    GaussianCloud cloud;
    std::vector<std::uint8_t> file;   // .gsi contents
    Image reconstruction;             // decoded from the quantized parameters
    EncodeStats stats;
    bool cancelled = false;
};

using ProgressFn = std::function<void(const EncodeProgress&)>;

// A compute backend runs the per-step math (forward, backward, Adam). The
// trainer owns the schedule, initialization and progressive additions.
class ITrainBackend {
public:
    virtual ~ITrainBackend() = default;
    virtual const char* name() const = 0;
    // (Re)binds target and parameters; resets Adam state. Called at start and
    // after gaussians are added.
    virtual bool prepare(const Image& target, GaussianCloud& cloud, const EncodeSettings& s) = 0;
    // One optimization step; t is the 1-based Adam step since prepare().
    // Returns the mean L1 loss of the step's render.
    virtual float step(int t, float lr_mult) = 0;
    // Copies the most recent render into `out` (allocated by callee).
    virtual void snapshot(Image& out) = 0;
    // Pulls current parameters back into the cloud (no-op on CPU).
    virtual void sync_cloud(GaussianCloud& cloud) = 0;
};

// Compress one image. Blocks; call from a worker thread for UI use.
// `cancel` may be flipped from another thread to stop early (partial result,
// result.cancelled = true).
EncodeResult encode_image(const Image& target, const EncodeSettings& settings,
                          const ProgressFn& on_progress = nullptr,
                          std::atomic<bool>* cancel = nullptr);

// Backend factories. The CPU factory always succeeds; the GPU factory
// returns null (with a reason) when unavailable. It is implemented in
// gpu_backend.cpp when built with GSIC_ENABLE_GPU.
std::unique_ptr<ITrainBackend> make_cpu_backend();
std::unique_ptr<ITrainBackend> make_gpu_backend(std::string* why_not);
// gpu_init / gpu_shutdown / gpu_render live in gpu.h.

} // namespace gsic
