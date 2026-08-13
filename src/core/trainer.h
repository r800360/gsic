#pragma once

#include "codec.h"
#include "gaussians.h"
#include "image.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>

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

    // Shortest gap between preview snapshots; 0 disables them entirely.
    // Treated as a lower bound rather than the actual rate, because a
    // snapshot renders the whole image and so costs more the larger the
    // image is. Holding the interval fixed while that cost grows spends an
    // unbounded fraction of the encode on previews: at 12 megapixels a
    // snapshot measures 132 ms, so a 250 ms interval was giving up around
    // 40% of the run to drawing pictures of it. See preview_time_fraction.
    int preview_interval_ms = 250;
    // Ceiling on the share of encode time spent producing previews. The
    // interval stretches automatically to respect it.
    double preview_time_fraction = 0.10;

    // Wall-clock target for the optimization loop, in seconds. Zero means no
    // limit, which is the right default for a library and a command line
    // tool: the step count is then fixed, so the encode is reproducible.
    //
    // The desktop application sets it, and accepts what that costs. Per-step
    // work scales with pixel count, so a step count chosen to look good on
    // one machine is a multi-minute wait on a slower one with a larger image,
    // and a person watching a progress bar reasonably concludes the feature
    // does not work. Quality against steps has sharp diminishing returns --
    // measured on a 12 MP image, the last half of a 3000 step run costs 143
    // seconds and buys 1.15 dB -- so trading the tail of that curve for a
    // predictable wait is a good exchange, and the alternative is not a
    // slower result but no result.
    double time_budget_seconds = 0.0;
    // Never cut below this many steps, whatever the budget says. Under about
    // 250 steps quality falls away quickly, and a finished-but-poor image is
    // not what anyone was asking for either.
    int min_budget_steps = 250;
};

struct EncodeProgress {
    int step = 0, max_steps = 0;
    float loss = 0.f;
    int num_gaussians = 0;
    const Image* preview = nullptr;   // non-null when a snapshot was taken
    double elapsed_seconds = 0.0;
    // Projected total for the optimization loop, from throughput so far.
    // Zero until enough steps have run to mean anything.
    double estimated_total_seconds = 0.0;
};

struct EncodeStats {
    double psnr = 0, ssim = 0;
    double encode_seconds = 0;    // everything: init, optimize, quantize, measure
    double optimize_seconds = 0;  // the optimization loop alone
    int steps_run = 0;
    // What the settings asked for, before any time budget cut it down. When
    // these differ, the encode traded quality for a bounded wait and the
    // interface should be able to say so.
    int steps_requested = 0;
    int num_gaussians = 0;
    const char* backend_used = "cpu";
    std::int64_t file_bytes = 0;
    std::int64_t source_bytes = 0;    // w * h * c at 8 bits per channel
    // Non-empty when the GPU was chosen, went wrong partway through, and the
    // encode was redone on the CPU. Worth surfacing: the result is correct,
    // but the machine has a GPU that cannot be trusted with this workload and
    // the user is paying for it in time on every image.
    std::string gpu_fallback_reason;
};

struct EncodeResult {
    GaussianCloud cloud;
    std::vector<std::uint8_t> file;   // .gsi contents
    Image reconstruction;             // decoded from the quantized parameters
    EncodeStats stats;
    bool cancelled = false;
    // Empty on success. Set when no usable file could be produced, which is
    // the only honest outcome for an encode that went wrong: writing bytes a
    // decoder rejects is worse than reporting the failure.
    std::string error;

    bool ok() const { return error.empty() && !cancelled && !file.empty(); }
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

    // False once the backend has detected that its results can no longer be
    // trusted. A backend that compiles and starts is not proof that it
    // computes correctly: drivers differ, and a compute dispatch can fail
    // partway through a run for reasons the caller cannot predict from the
    // hardware alone. Reporting it lets the trainer abandon this backend and
    // redo the encode on the CPU rather than write out whatever came back.
    // `why` receives a description the user can be shown.
    virtual bool healthy(std::string* why = nullptr) const {
        (void)why;
        return true;
    }
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

// Test seam. The mid-encode fallback can only be exercised by a backend that
// fails partway through, and a machine whose GPU works will never produce
// one. Installing a factory here lets the robustness suite inject that
// failure on any hardware, so the recovery path is covered everywhere rather
// than only on the machines where it is already broken.
using BackendFactory = std::function<std::unique_ptr<ITrainBackend>(std::string*)>;
void set_gpu_backend_factory_for_testing(BackendFactory factory);

} // namespace gsic
