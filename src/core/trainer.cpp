#include "trainer.h"

#include "gpu.h"
#include "kernels.h"
#include "metrics.h"
#include "profiling.h"
#include "renderer.h"
#include "thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <numbers>
#include <random>

namespace gsic {

namespace {

// Fraction of the run spent at full learning rate before the final decay
// begins, and the rate the decay ends at (as a fraction of the start).
constexpr float kLrDecayStart = 0.8f;
constexpr float kLrFloor = 0.1f;

// ------------------------------------------------------------- CPU backend
class CpuBackend final : public ITrainBackend {
public:
    const char* name() const override { return kernels().name; }

    bool prepare(const Image& target, GaussianCloud& cloud, const EncodeSettings& s) override {
        target_ = &target;
        cloud_ = &cloud;
        settings_ = s;
        renderer_.set_size(target.w, target.h, target.c);
        const size_t plane = size_t(target.w) * target.h;
        grad_.assign(plane * target.c, 0.f);
        const int n = cloud.count();
        const size_t np = size_t(n) * 5 + size_t(n) * target.c;  // 5 geometry groups + colors
        grads_.assign(np, 0.f);
        // Adam moments survive a pyramid level change: the gaussian count is
        // unchanged there, and resetting them would discard the optimizer's
        // learned per-parameter step sizes and cost hundreds of steps of
        // re-convergence. A progressive addition does change the count (and
        // reorders the cloud), so state starts fresh there.
        if (adam_m_.size() != np) {
            adam_m_.assign(np, 0.f);
            adam_v_.assign(np, 0.f);
        }
        // Scale bounds. The upper size cap matters for speed AND quality:
        // per-step cost is proportional to total gaussian footprint, and a few
        // image-sized background gaussians would dominate it while making the
        // sum ill-conditioned.
        sinv_lo_ = 1.f / std::min(s.max_scale_px, 0.5f * float(std::max(target.w, target.h)));
        sinv_hi_ = 1.f / kMinScalePx;
        return true;
    }

    float step(int t, float lr_mult) override {
        GSIC_ZONE("cpu_step");
        auto& pool = ThreadPool::instance();
        GaussianCloud& cloud = *cloud_;
        const Image& target = *target_;
        const int n = cloud.count();
        const int c = target.c;
        const size_t plane = size_t(target.w) * target.h;

        renderer_.project_and_bin(cloud);

        KernelCtx ctx;
        renderer_.fill_ctx(ctx, cloud);
        for (int ch = 0; ch < c; ++ch) {
            ctx.target[ch] = target.plane(ch);
            ctx.grad[ch] = grad_.data() + plane * ch;
        }
        ctx.w_l1 = settings_.w_l1;
        ctx.w_l2 = settings_.w_l2;
        ctx.inv_norm = 1.f / (float(plane) * float(c));
        float* gp = grads_.data();
        ctx.d_pos_x = gp; ctx.d_pos_y = gp + n;
        ctx.d_sinv_x = gp + 2 * size_t(n); ctx.d_sinv_y = gp + 3 * size_t(n);
        ctx.d_rot = gp + 4 * size_t(n);
        ctx.d_color = gp + 5 * size_t(n);

        const Kernels& kn = kernels();
        double l1 = 0.0, l2 = 0.0;
        {
            GSIC_ZONE("forward_loss");
            std::mutex m;
            pool.parallel_for(renderer_.tile_count(), 8, [&](std::int64_t b, std::int64_t e) {
                double p1 = 0.0, p2 = 0.0;
                kn.forward_loss_tiles(ctx, int(b), int(e), &p1, &p2);
                std::lock_guard lk(m);
                l1 += p1;
                l2 += p2;
            });
        }
        {
            GSIC_ZONE("backward");
            pool.parallel_for(n, 32, [&](std::int64_t b, std::int64_t e) {
                kn.backward_gaussians(ctx, int(b), int(e));
            });
        }
        {
            GSIC_ZONE("adam");
            const float b1 = 0.9f, b2 = 0.999f;
            const float bias1 = 1.f - std::pow(b1, float(t));
            const float bias2 = 1.f - std::pow(b2, float(t));
            const float inf = 1e30f;
            auto update = [&](float* p, int group, std::int64_t count, float lr, float lo, float hi) {
                AdamArgs a{p, adam_m_.data() + group_off_(group, n, c),
                           adam_v_.data() + group_off_(group, n, c),
                           gp + group_off_(group, n, c),
                           count, lr * lr_mult, b1, b2, 1e-8f, bias1, bias2, lo, hi};
                kn.adam_step(a);
            };
            update(cloud.pos_x.data(), 0, n, settings_.lr_pos, kPosLo, kPosHi);
            update(cloud.pos_y.data(), 1, n, settings_.lr_pos, kPosLo, kPosHi);
            update(cloud.sinv_x.data(), 2, n, settings_.lr_scale, sinv_lo_, sinv_hi_);
            update(cloud.sinv_y.data(), 3, n, settings_.lr_scale, sinv_lo_, sinv_hi_);
            update(cloud.rot.data(), 4, n, settings_.lr_rot, -inf, inf);
            update(cloud.color.data(), 5, std::int64_t(n) * c, settings_.lr_feat, kColorLo,
                   kColorHi);
        }
        return float(l1 / (double(plane) * c));
    }

    void snapshot(Image& out) override {
        // Training steps never materialize the render (fused forward+loss),
        // so previews render on demand from the latest projection.
        const Image& t = *target_;
        out = Image(t.w, t.h, t.c);
        float* planes[4] = {};
        for (int ch = 0; ch < t.c; ++ch) planes[ch] = out.plane(ch);
        renderer_.render(*cloud_, planes);
    }

    void sync_cloud(GaussianCloud&) override {}

private:
    // Gradient/moment layout: five geometry groups of n, then n*c colors.
    static size_t group_off_(int group, int n, int) {
        return size_t(std::min(group, 5)) * n;
    }

    const Image* target_ = nullptr;
    GaussianCloud* cloud_ = nullptr;
    EncodeSettings settings_;
    Renderer renderer_;
    std::vector<float> grad_, grads_, adam_m_, adam_v_;
    float sinv_lo_ = 0.f, sinv_hi_ = 1.f;
};

} // namespace

std::unique_ptr<ITrainBackend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

namespace {

// ------------------------------------------------------------ initialization

// Luma Sobel gradient magnitude squared; the paper samples initial gaussian
// centers from this distribution so detail gets more gaussians.
std::vector<float> gradient_weights(const Image& img) {
    const int w = img.w, h = img.h;
    std::vector<float> luma(size_t(w) * h, 0.f);
    const float inv_c = 1.f / img.c;
    for (int ch = 0; ch < img.c; ++ch) {
        const float* p = img.plane(ch);
        for (size_t i = 0; i < luma.size(); ++i) luma[i] += p[i] * inv_c;
    }
    std::vector<float> weight(size_t(w) * h, 0.f);
    ThreadPool::instance().parallel_for(h, 64, [&](std::int64_t y0, std::int64_t y1) {
        for (std::int64_t y = y0; y < y1; ++y) {
            const auto at = [&](std::int64_t yy, int xx) {
                return luma[size_t(std::clamp(yy, std::int64_t(0), std::int64_t(h - 1))) * w +
                            size_t(std::clamp(xx, 0, w - 1))];
            };
            for (int x = 0; x < w; ++x) {
                const float gx = at(y - 1, x + 1) + 2 * at(y, x + 1) + at(y + 1, x + 1) -
                                 at(y - 1, x - 1) - 2 * at(y, x - 1) - at(y + 1, x - 1);
                const float gy = at(y + 1, x - 1) + 2 * at(y + 1, x) + at(y + 1, x + 1) -
                                 at(y - 1, x - 1) - 2 * at(y - 1, x) - at(y - 1, x + 1);
                weight[size_t(y) * w + x] = gx * gx + gy * gy;  // |g|^2
            }
        }
    });
    return weight;
}

// Draws `count` distinct pixel indices with probability proportional to
// weight (weights may be zero; a uniform floor guarantees termination).
std::vector<std::int64_t> weighted_sample(const std::vector<float>& weight, int count,
                                          std::mt19937& rng, std::vector<std::uint8_t>& taken) {
    std::vector<double> cdf(weight.size());
    double sum = 0.0;
    for (size_t i = 0; i < weight.size(); ++i) cdf[i] = (sum += weight[i]);
    std::vector<std::int64_t> out;
    out.reserve(count);
    std::uniform_real_distribution<double> uni(0.0, sum);
    std::uniform_int_distribution<std::int64_t> uniform_px(0, std::int64_t(weight.size()) - 1);
    int attempts = 0;
    while (int(out.size()) < count) {
        std::int64_t idx;
        if (sum > 0.0 && attempts < count * 20) {
            const double r = uni(rng);
            idx = std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
            idx = std::min<std::int64_t>(idx, cdf.size() - 1);
        } else {
            idx = uniform_px(rng);  // fallback: uniform, always terminates
        }
        ++attempts;
        if (!taken[size_t(idx)]) {
            taken[size_t(idx)] = 1;
            out.push_back(idx);
        }
    }
    return out;
}

// Sorts gaussians along a Morton curve of their positions. Rendering is
// order-independent; the payoff is cache locality: neighboring gaussians end
// up in the same backward chunk, so their overlapping grad-plane reads hit
// cache instead of streaming from DRAM. Called whenever the optimizer state
// is rebuilt anyway (positions drift too slowly to need more).
void morton_sort(GaussianCloud& cloud) {
    const int n = cloud.count();
    const auto expand = [](std::uint32_t v) {
        std::uint64_t x = v & 0xFFFF;
        x = (x | (x << 8)) & 0x00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F;
        x = (x | (x << 2)) & 0x33333333;
        x = (x | (x << 1)) & 0x55555555;
        return x;
    };
    std::vector<std::pair<std::uint64_t, int>> order(n);
    for (int i = 0; i < n; ++i) {
        const auto q = [&](float p) {
            return std::uint32_t(std::clamp(p, 0.f, 1.f) * 65535.f);
        };
        order[i] = {(expand(q(cloud.pos_y[i])) << 1) | expand(q(cloud.pos_x[i])), i};
    }
    std::sort(order.begin(), order.end());
    GaussianCloud sorted;
    sorted.channels = cloud.channels;
    sorted.resize(n);
    for (int i = 0; i < n; ++i) {
        const int j = order[i].second;
        sorted.pos_x[i] = cloud.pos_x[j];
        sorted.pos_y[i] = cloud.pos_y[j];
        sorted.sinv_x[i] = cloud.sinv_x[j];
        sorted.sinv_y[i] = cloud.sinv_y[j];
        sorted.rot[i] = cloud.rot[j];
        for (int c = 0; c < cloud.channels; ++c)
            sorted.color[size_t(i) * cloud.channels + c] =
                cloud.color[size_t(j) * cloud.channels + c];
    }
    cloud = std::move(sorted);
}

// Appends gaussians at the given pixel indices. Colors come from
// `color_source`, a planar target-sized buffer (the image itself for the
// initial placement, the residual for progressive additions).
void append_gaussians(GaussianCloud& cloud, const Image& target,
                      const std::vector<std::int64_t>& pixels, float init_scale_px,
                      const float* color_source) {
    const int old_n = cloud.count();
    cloud.resize(old_n + int(pixels.size()));
    const float sinv = 1.f / init_scale_px;
    for (size_t i = 0; i < pixels.size(); ++i) {
        const int g = old_n + int(i);
        const std::int64_t px = pixels[i];
        cloud.pos_x[g] = (float(px % target.w) + 0.5f) / float(target.w);
        cloud.pos_y[g] = (float(px / target.w) + 0.5f) / float(target.h);
        cloud.sinv_x[g] = sinv;
        cloud.sinv_y[g] = sinv;
        cloud.rot[g] = 0.f;
        for (int c = 0; c < cloud.channels; ++c)
            cloud.color[size_t(g) * cloud.channels + c] =
                color_source[size_t(c) * target.pixels() + px];
    }
}

} // namespace

namespace {
// See set_gpu_backend_factory_for_testing in trainer.h. Read on the encode
// thread, written by tests before an encode starts.
BackendFactory g_gpu_factory_override;
}  // namespace

void set_gpu_backend_factory_for_testing(BackendFactory factory) {
    g_gpu_factory_override = std::move(factory);
}

#if !defined(GSIC_ENABLE_GPU)
std::unique_ptr<ITrainBackend> make_gpu_backend(std::string* why_not) {
    if (why_not) *why_not = "built without GPU support";
    return nullptr;
}
bool gpu_init(std::string* why_not) {
    if (why_not) *why_not = "built without GPU support";
    return false;
}
void gpu_shutdown() {}
std::optional<Image> gpu_render(const GaussianCloud&, int, int, std::string* error) {
    if (error) *error = "built without GPU support";
    return std::nullopt;
}
#endif

EncodeResult encode_image(const Image& target, const EncodeSettings& settings,
                          const ProgressFn& on_progress, std::atomic<bool>* cancel) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();
    EncodeResult result;
    if (target.empty() || target.c < 1 || target.c > kMaxChannels) return result;

    const std::int64_t pixels = target.pixels();
    int n_total = settings.num_gaussians > 0
                      ? settings.num_gaussians
                      : int(pixels / std::max(1, settings.pixels_per_gaussian));
    n_total = int(std::clamp<std::int64_t>(
        n_total, 16, std::min<std::int64_t>({pixels, 4'000'000, kMaxGaussians})));
    
    const int max_steps = std::max(64, settings.max_steps);
    const Image* train_target = &target;
    const EncodeSettings* stage = &settings;

    // One full optimization run on a given backend. Everything it touches is
    // derived from `settings.seed`, so a second call reproduces the first
    // exactly; that is what makes retrying on the CPU a correct recovery and
    // not a different encode. Returns false when the backend stopped being
    // trustworthy partway through, leaving `result` untouched for the retry.
    int step = 0;
    auto run_optimization = [&](ITrainBackend& backend, std::string* why) -> bool {
        // --- Initial placement: image-gradient guided + a uniform random share.
        std::mt19937 rng(settings.seed);
        GaussianCloud& cloud = result.cloud;
        cloud = GaussianCloud{};
        cloud.channels = target.c;
        const int n_initial =
            settings.progressive ? std::max(16, int(std::ceil(settings.initial_ratio * n_total)))
                                 : n_total;
        {
            const std::int64_t tpixels = train_target->pixels();
            std::vector<std::uint8_t> taken(size_t(tpixels), 0);
            auto weights = gradient_weights(*train_target);
            const int n_random = int(std::round(settings.init_random_ratio * n_initial));
            std::vector<float> uniform(size_t(tpixels), 1.f);
            auto random_px = weighted_sample(uniform, n_random, rng, taken);
            auto guided_px = weighted_sample(weights, n_initial - n_random, rng, taken);
            guided_px.insert(guided_px.end(), random_px.begin(), random_px.end());
            append_gaussians(cloud, *train_target, guided_px, stage->init_scale_px,
                             train_target->data.data());
        }

        morton_sort(cloud);
        if (!backend.prepare(*train_target, cloud, *stage)) {
            if (why) *why = "backend could not allocate its buffers";
            return false;
        }

        // --- Optimization schedule.
        //
        // `budget_steps` is what the run will actually do. It starts as what
        // was asked for and may be lowered once, early, by the wall-clock
        // budget; everything derived from it is recomputed at that point, so
        // the learning rate schedule and the progressive additions stay
        // consistent with the run that is really happening rather than the one
        // originally planned.
        int budget_steps = max_steps;
        const int adds_left_at_start = settings.progressive ? settings.add_times : 0;
        const auto interval_for = [&](int steps) {
            return adds_left_at_start > 0
                       ? std::max(32, steps * 2 / (5 * (adds_left_at_start + 1)))
                       : 0;
        };
        int add_interval = interval_for(budget_steps);
        int adds_left = adds_left_at_start;
        const int per_add =
            adds_left > 0 ? (n_total - cloud.count() + adds_left - 1) / adds_left : 0;

        // Enforcing the wall-clock budget.
        //
        // The obvious approach -- time a few early steps, divide, pick a step
        // count -- does not work here, and the way it fails is worth naming.
        // Per-step cost is not constant: the cloud doubles across the
        // progressive additions, and the gaussians spread out as they
        // optimize, so their total footprint (which is what the renderer
        // actually costs) grows through the run. Measured on a 12 MP image,
        // steps 8 to 24 run about five times faster than the run average, so
        // an early extrapolation promised 30 seconds and delivered 152. A
        // bound that is wrong by 5x is worse than no bound, because the
        // interface repeats it to the user.
        //
        // So the budget is enforced by the clock, not by arithmetic done once
        // at the start. `budget_steps` is a projection, re-estimated as the
        // run proceeds and only ever revised downward; it exists to keep the
        // learning-rate decay and the progressive additions pointed at a
        // realistic end. The guarantee comes from the hard stop below it.
        constexpr int kWarmupSteps = 8;
        constexpr int kReprojectEvery = 32;
        const bool budgeted = settings.time_budget_seconds > 0.0;
        const int floor_steps =
            std::min(max_steps, std::max(1, settings.min_budget_steps));
        const auto loop_start = clock::now();
        auto warmup_done = loop_start;

        float lr_mult = 1.f;
        int adam_t = 0;
        Image preview;
        auto last_preview = clock::now();
        double snapshot_seconds = 0.0;   // cost of the most recent snapshot

        for (step = 1; step <= budget_steps; ++step) {
            if (cancel && cancel->load(std::memory_order_relaxed)) {
                result.cancelled = true;
                break;
            }
            const float loss = backend.step(++adam_t, lr_mult);

            if (step == kWarmupSteps) warmup_done = clock::now();
            if (budgeted) {
                const double elapsed =
                    std::chrono::duration<double>(clock::now() - loop_start).count();

                // The guarantee, checked every step. Reading the clock costs
                // tens of nanoseconds against a step costing milliseconds, and
                // checking it only periodically would let the run overshoot by
                // however much a whole window of late steps costs -- which on
                // a large image is seconds, in the one situation where the
                // bound was the point.
                if (elapsed >= settings.time_budget_seconds && step >= floor_steps) {
                    budget_steps = step;
                    break;
                }

                // The projection. This does not enforce anything; it keeps the
                // learning-rate decay and the progressive additions aimed at
                // where the run will actually end, so the schedule is not
                // planned around a finish the run will never reach.
                if (step > kWarmupSteps && step % kReprojectEvery == 0) {
                    const double per_step = elapsed / double(step);
                    if (per_step > 0.0) {
                        const double remaining = settings.time_budget_seconds - elapsed;
                        const int projected =
                            std::clamp(step + int(remaining / per_step), floor_steps, max_steps);
                        // Downward only. Per-step cost rises as the cloud grows
                        // and the gaussians spread, so the average so far is
                        // always an optimistic estimate of what is left; a
                        // projection allowed to grow again would be chasing it.
                        if (projected < budget_steps) {
                            budget_steps = projected;
                            add_interval = interval_for(budget_steps);
                        }
                    }
                }
            }

            // Two ways a backend can stop being usable mid-run: it can say so
            // itself, or it can quietly start returning garbage. A loss that
            // is not a finite number is the earliest visible symptom of the
            // second, and continuing from it only produces a file whose own
            // decoder will reject it.
            if (!std::isfinite(loss)) {
                if (why) *why = "produced a non-finite loss";
                return false;
            }
            if (!backend.healthy(why)) return false;

            // Progressive addition: place new gaussians where error is largest.
            //
            // Scheduled by "have we reached the step this addition was due"
            // rather than by an exact multiple, because add_interval moves
            // when the budget is revised. An exact test would step straight
            // over an addition whose due step had just been recomputed, and
            // the run would quietly end with fewer gaussians than the settings
            // asked for -- visible to the user only as a file of the wrong
            // size.
            const int next_add_step = add_interval * (adds_left_at_start - adds_left + 1);
            if (adds_left > 0 && step >= next_add_step) {
                GSIC_ZONE("add_gaussians");
                const int add_n = std::min(per_add, n_total - cloud.count());
                if (add_n > 0) {
                    const Image& tt = *train_target;
                    const std::int64_t tpixels = tt.pixels();
                    backend.sync_cloud(cloud);
                    Image recon;
                    backend.snapshot(recon);
                    // Error map (mean |diff| over channels, squared) with a 3x3
                    // blur so isolated noisy pixels don't dominate the sampling.
                    std::vector<float> err(size_t(tpixels), 0.f);
                    std::vector<float> diff(size_t(tpixels) * tt.c);
                    const float inv_c = 1.f / tt.c;
                    for (int c = 0; c < tt.c; ++c) {
                        const float* r = recon.plane(c);
                        const float* t = tt.plane(c);
                        float* d = diff.data() + size_t(c) * tpixels;
                        for (size_t i = 0, np = size_t(tpixels); i < np; ++i) {
                            d[i] = t[i] - r[i];
                            err[i] += std::fabs(d[i]) * inv_c;
                        }
                    }
                    std::vector<float> err_blur(size_t(tpixels), 0.f);
                    for (int y = 0; y < tt.h; ++y)
                        for (int x = 0; x < tt.w; ++x) {
                            float s = 0.f;
                            for (int dy = -1; dy <= 1; ++dy)
                                for (int dx = -1; dx <= 1; ++dx)
                                    s += err[size_t(std::clamp(y + dy, 0, tt.h - 1)) * tt.w +
                                             size_t(std::clamp(x + dx, 0, tt.w - 1))];
                            err_blur[size_t(y) * tt.w + x] = s * s;  // squared like the paper
                        }
                    std::vector<std::uint8_t> taken(size_t(tpixels), 0);
                    auto px = weighted_sample(err_blur, add_n, rng, taken);
                    // New gaussians take the residual as color so they immediately
                    // reduce the error they were placed on.
                    append_gaussians(cloud, tt, px, stage->init_scale_px, diff.data());
                    morton_sort(cloud);
                    if (!backend.prepare(tt, cloud, *stage)) {
                        if (why) *why = "backend could not grow its buffers";
                        return false;
                    }
                    adam_t = 0;
                }
                --adds_left;
            }

            // Learning rate holds flat, then decays over the last stretch.
            // Decaying earlier measurably hurts: gaussians keep migrating across
            // the image for most of the run, and shrinking their steps too soon
            // freezes them before they reach the detail they should cover. The
            // final decay only settles what is already in place.
            if (settings.lr_decay) {
                const float progress = float(step) / float(budget_steps);
                if (progress > kLrDecayStart) {
                    const float t = (progress - kLrDecayStart) / (1.f - kLrDecayStart);
                    lr_mult = kLrFloor + (1.f - kLrFloor) * 0.5f *
                                             (1.f + std::cos(std::numbers::pi_v<float> * t));
                }
            }

            if (on_progress) {
                const auto now = clock::now();
                // How long to wait before the next snapshot. The configured
                // interval is the floor; above it, the gap stretches so that
                // producing previews stays within its share of the run. A
                // snapshot renders the entire image, so on a large one this is
                // the difference between a tenth of the encode and half of it.
                double min_gap = double(settings.preview_interval_ms) / 1000.0;
                const double fraction = std::clamp(settings.preview_time_fraction, 0.01, 1.0);
                if (fraction < 1.0)
                    min_gap = std::max(min_gap, snapshot_seconds * (1.0 - fraction) / fraction);

                const bool want_preview =
                    settings.preview_interval_ms > 0 &&
                    std::chrono::duration<double>(now - last_preview).count() >= min_gap;
                if (want_preview || step == budget_steps || step % 32 == 0) {
                    EncodeProgress p;
                    p.step = step;
                    p.max_steps = budget_steps;
                    p.loss = loss;
                    p.num_gaussians = cloud.count();
                    p.elapsed_seconds = std::chrono::duration<double>(now - t_start).count();
                    if (step > kWarmupSteps) {
                        const double since_warmup =
                            std::chrono::duration<double>(now - warmup_done).count();
                        const double per_step = since_warmup / double(step - kWarmupSteps);
                        p.estimated_total_seconds =
                            p.elapsed_seconds + per_step * double(budget_steps - step);
                    }
                    if (!want_preview) {
                        on_progress(p);
                    } else {
                        // Timed across the callback as well as the render. The
                        // render is only half of what a preview costs: the
                        // receiver still has to turn a planar float image into
                        // something displayable, which at 12 megapixels is
                        // another 48 MB of work. Budgeting for the render alone
                        // would leave that half unaccounted for.
                        const auto snap_start = clock::now();
                        backend.snapshot(preview);
                        p.preview = &preview;
                        on_progress(p);
                        snapshot_seconds =
                            std::chrono::duration<double>(clock::now() - snap_start).count();
                        last_preview = clock::now();
                    }
                }
            }
        }

        // The loop leaves `step` one past the last one it ran when it ends
        // normally, so the count reported to the caller is trimmed back to
        // what was actually done. It matters more than it used to: with a
        // budget, steps_run against steps_requested is how the interface knows
        // to tell the user the run was shortened.
        step = std::min(step, budget_steps);

        backend.sync_cloud(cloud);
        return backend.healthy(why);
    };

    // --- Backend selection, then the run, then the CPU retry if it went bad.
    //
    // Falling back before the first step (no GPU, shaders will not compile)
    // was already handled. What was not, and is the failure a user actually
    // sees, is a backend that starts fine and stops being correct partway
    // through: there is no error dialog for that, only a file that will not
    // open. The retry costs time on the machines that need it and nothing
    // anywhere else.
    const auto t_optimize = clock::now();
    std::unique_ptr<ITrainBackend> backend;
    if (settings.backend != Backend::Cpu) {
        std::string why;
        backend = g_gpu_factory_override ? g_gpu_factory_override(&why) : make_gpu_backend(&why);
    }
    bool ok = false;
    if (backend) {
        result.stats.backend_used = "gpu";
        std::string why = "stopped responding";
        ok = run_optimization(*backend, &why);
        if (!ok && !result.cancelled) {
            result.stats.gpu_fallback_reason = why;
            backend.reset();   // release the GPU before the CPU run starts
        }
    }
    if (!ok && !result.cancelled) {
        backend = std::make_unique<CpuBackend>();
        result.stats.backend_used = backend->name();
        std::string why;
        if (!run_optimization(*backend, &why)) {
            // The CPU path has no hardware to disagree with, so this means the
            // input itself drove the optimizer to a non-finite state. Report
            // it rather than writing a file nothing can open.
            result.error = why.empty() ? "optimization failed" : why;
            return result;
        }
    }
    GaussianCloud& cloud = result.cloud;

    result.stats.optimize_seconds =
        std::chrono::duration<double>(clock::now() - t_optimize).count();

    // Quantize exactly as the file stores it, then measure what a decoder
    // will actually see.
    if (!quantize_cloud(cloud, settings.quant)) {
        result.error = "the encoded parameters could not be stored in the file format";
        return result;
    }
    result.file = encode_gsi(cloud, target.w, target.h, settings.quant);
    // The encode is only finished when the bytes read back. Checking here
    // costs one decode of a file that is already in memory and is the
    // difference between "compression failed" and a .gsi on disk that nothing
    // will open -- including this application.
    if (result.file.empty() || !decode_gsi(result.file)) {
        result.file.clear();
        result.error = "the encoder produced a file that could not be read back";
        return result;
    }
    result.reconstruction = Renderer::render_image(cloud, target.w, target.h);

    result.stats.psnr = psnr(result.reconstruction, target);
    result.stats.ssim = ssim(result.reconstruction, target);
    result.stats.steps_run = std::min(step, max_steps);
    result.stats.steps_requested = max_steps;
    result.stats.num_gaussians = cloud.count();
    result.stats.file_bytes = std::int64_t(result.file.size());
    result.stats.source_bytes = pixels * target.c;
    result.stats.encode_seconds =
        std::chrono::duration<double>(clock::now() - t_start).count();
    return result;
}

} // namespace gsic
