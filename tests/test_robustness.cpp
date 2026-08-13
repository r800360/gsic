// Robustness tests: what happens when the machine is not this machine.
//
// The other three suites all answer "is the code right on the hardware I am
// sitting at". They passed on the developer's machine while the shipped
// application could not compress an image on a Surface Laptop 5, because
// nothing here was ever exercised: a compute backend that starts fine and
// then goes wrong, an optimizer that ends in a state the file format cannot
// represent, an output directory that cannot be written.
//
// None of those need the failing hardware to test. They need a way to inject
// the failure, which is what the seams used below are for. A recovery path
// that only runs on the machines where it is already broken is not a recovery
// path, it is a hope.

#include "core/codec.h"
#include "core/gpu.h"
#include "core/image.h"
#include "core/metrics.h"
#include "core/renderer.h"
#include "core/trainer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace gsic;

static int failures = 0;

static void fail(const std::string& what) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
}

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) fail(std::string(__FILE__ ":") + std::to_string(__LINE__) + ": " #cond); \
    } while (0)

namespace {

// --------------------------------------------------------------- fixtures
Image test_image(int w, int h) {
    Image img(w, h, 3);
    for (int c = 0; c < 3; ++c) {
        float* p = img.plane(c);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                p[size_t(y) * w + x] = 0.5f + 0.35f * std::sin(0.09f * float(x) + float(c)) *
                                                  std::cos(0.07f * float(y));
    }
    return img;
}

EncodeSettings quick_settings() {
    EncodeSettings s;
    s.num_gaussians = 400;
    s.max_steps = 240;
    s.preview_interval_ms = 0;
    s.seed = 123;
    return s;
}

// A stand-in for a compute backend that works and then stops working. It
// delegates every real computation to the CPU backend, so up to the injected
// failure it behaves exactly like a healthy one -- which is the point: the
// trainer must notice on the strength of what the backend reports, not
// because the numbers happen to look strange.
class FlakyBackend final : public ITrainBackend {
public:
    enum class Mode {
        UnhealthyAtStep,   // reports itself broken, the way a GL error does
        NanAtStep,         // silently returns garbage, the way bad hardware does
        PrepareFails,      // cannot allocate at all
    };

    FlakyBackend(Mode mode, int at_step) : mode_(mode), at_step_(at_step) {}

    const char* name() const override { return "flaky"; }

    bool prepare(const Image& target, GaussianCloud& cloud, const EncodeSettings& s) override {
        if (mode_ == Mode::PrepareFails) return false;
        return inner_->prepare(target, cloud, s);
    }

    float step(int t, float lr_mult) override {
        const float loss = inner_->step(t, lr_mult);
        if (++steps_ < at_step_) return loss;
        if (mode_ == Mode::NanAtStep) return std::numeric_limits<float>::quiet_NaN();
        healthy_ = false;
        return loss;
    }

    void snapshot(Image& out) override { inner_->snapshot(out); }
    void sync_cloud(GaussianCloud& cloud) override { inner_->sync_cloud(cloud); }

    bool healthy(std::string* why) const override {
        if (!healthy_ && why) *why = "injected failure";
        return healthy_;
    }

    static int made;

private:
    std::unique_ptr<ITrainBackend> inner_ = make_cpu_backend();
    Mode mode_;
    int at_step_ = 0;
    int steps_ = 0;
    bool healthy_ = true;
};

int FlakyBackend::made = 0;

struct FactoryGuard {
    explicit FactoryGuard(BackendFactory f) { set_gpu_backend_factory_for_testing(std::move(f)); }
    ~FactoryGuard() { set_gpu_backend_factory_for_testing(nullptr); }
};

// ------------------------------------------------- the accelerator fallback
//
// The bug this suite exists for. An accelerated backend that fails partway
// through used to run to completion regardless, and whatever its buffers
// happened to contain became the .gsi the user was handed.
void test_accelerator_failure_falls_back_to_cpu() {
    const Image img = test_image(128, 96);
    auto s = quick_settings();
    s.backend = Backend::Auto;

    // The reference: what the CPU alone produces for this input.
    auto cpu_only = quick_settings();
    cpu_only.backend = Backend::Cpu;
    const auto reference = encode_image(img, cpu_only);
    CHECK(reference.ok());

    struct Case { FlakyBackend::Mode mode; int at_step; const char* what; };
    const Case cases[] = {
        {FlakyBackend::Mode::UnhealthyAtStep, 1, "fails on the very first step"},
        {FlakyBackend::Mode::UnhealthyAtStep, 50, "fails midway through"},
        {FlakyBackend::Mode::UnhealthyAtStep, 239, "fails on the last step"},
        {FlakyBackend::Mode::NanAtStep, 30, "silently returns NaN"},
        {FlakyBackend::Mode::PrepareFails, 0, "cannot allocate its buffers"},
    };

    for (const Case& c : cases) {
        FactoryGuard guard([&](std::string*) -> std::unique_ptr<ITrainBackend> {
            return std::make_unique<FlakyBackend>(c.mode, c.at_step);
        });
        const auto result = encode_image(img, s);

        if (!result.ok()) {
            fail(std::string("accelerator that ") + c.what +
                 " produced no usable file (error: " + result.error + ")");
            continue;
        }
        if (!decode_gsi(result.file)) {
            fail(std::string("accelerator that ") + c.what + " produced an unreadable file");
            continue;
        }
        if (result.stats.gpu_fallback_reason.empty()) {
            fail(std::string("accelerator that ") + c.what + " was not reported as fallen back");
            continue;
        }
        // The retry restarts from the same seed, so recovery has to reproduce
        // the plain CPU encode exactly. Anything else means the failed run
        // left state behind and the user's file depends on when the hardware
        // gave up.
        if (result.file != reference.file) {
            fail(std::string("recovery after an accelerator that ") + c.what +
                 " did not reproduce the CPU result byte for byte");
            continue;
        }
        std::printf("  accelerator %-32s -> recovered on CPU, %.2f dB, %zu bytes\n", c.what,
                    result.stats.psnr, result.file.size());
    }
}

// The CPU is the last resort, so when it is the thing that fails there is
// nothing left to fall back to. The only correct outcome is a reported error
// and no file, rather than bytes nothing can open.
void test_cpu_failure_is_reported_not_written() {
    const Image img = test_image(96, 96);
    auto s = quick_settings();
    s.backend = Backend::Cpu;
    s.max_steps = 64;

    // Learning rates large enough to blow the optimizer up rather than
    // converge it: this is what "the input drove it non-finite" looks like.
    s.lr_pos = s.lr_scale = s.lr_rot = s.lr_feat = 1e30f;
    const auto result = encode_image(img, s);
    if (result.ok()) {
        // Diverging is not guaranteed -- the parameter clamps may hold it
        // together -- but if it did produce a file, that file must be real.
        CHECK(decode_gsi(result.file).has_value());
        std::puts("  divergent settings still produced a readable file (clamps held)");
        return;
    }
    CHECK(!result.error.empty());
    CHECK(result.file.empty());
    std::printf("  divergent settings reported \"%s\" and wrote nothing\n", result.error.c_str());
}

// --------------------------------------------------- the format guarantee
//
// encode_gsi promises that anything it returns, decode_gsi accepts. That
// promise is what stands between a hardware fault and a .gsi on the user's
// disk that this application itself refuses to open.
void test_encoder_never_emits_an_unreadable_file() {
    std::mt19937 rng(2027);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float poison[] = {nan, inf, -inf, 0.f, -0.f, 1e38f, -1e38f, 1e-38f};

    int checked = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int channels = 1 + trial % 4;
        const int n = 1 + int(uni(rng) * 200.f);
        GaussianCloud cloud;
        cloud.channels = channels;
        cloud.resize(n);
        for (int i = 0; i < n; ++i) {
            cloud.pos_x[i] = uni(rng);
            cloud.pos_y[i] = uni(rng);
            cloud.sinv_x[i] = 0.05f + uni(rng);
            cloud.sinv_y[i] = 0.05f + uni(rng);
            cloud.rot[i] = (uni(rng) - 0.5f) * 6.f;
            for (int c = 0; c < channels; ++c)
                cloud.color[size_t(i) * channels + c] = uni(rng) * 2.f - 0.5f;
        }
        // Poison a random subset, including the case where every value in a
        // field is poisoned and there is no finite value to build a range from.
        const int poisonings = trial % 7 == 0 ? n : int(uni(rng) * 5.f);
        for (int k = 0; k < poisonings; ++k) {
            const int i = trial % 7 == 0 ? k : int(uni(rng) * float(n)) % n;
            const float p = poison[size_t(uni(rng) * 8.f) % 8];
            switch ((trial + k) % 6) {
                case 0: cloud.pos_x[i] = p; break;
                case 1: cloud.pos_y[i] = p; break;
                case 2: cloud.sinv_x[i] = p; break;
                case 3: cloud.sinv_y[i] = p; break;
                case 4: cloud.rot[i] = p; break;
                default: cloud.color[size_t(i) * channels] = p; break;
            }
        }

        const QuantSpec q{16, 12, 10, 12};
        const auto bytes = encode_gsi(cloud, 320, 240, q);
        if (bytes.empty()) continue;   // refused up front, which is allowed
        ++checked;
        std::string err;
        if (!decode_gsi(bytes, &err)) {
            fail("encode_gsi produced a file decode_gsi rejects (\"" + err + "\") on trial " +
                 std::to_string(trial));
            return;
        }
    }
    std::printf("  %d encodes with poisoned parameters, every one decodable\n", checked);
}

// A cloud the format cannot hold must be refused, not written. These are the
// inputs that used to produce bytes with a non-finite range in the header,
// which decode_gsi correctly rejects -- an encoder disagreeing with its own
// decoder.
void test_encoder_refuses_what_it_cannot_represent() {
    GaussianCloud empty;
    empty.channels = 3;
    CHECK(encode_gsi(empty, 64, 64, QuantSpec{}).empty());

    GaussianCloud one;
    one.channels = 3;
    one.resize(4);
    for (int i = 0; i < 4; ++i) {
        one.sinv_x[i] = one.sinv_y[i] = 0.2f;
        one.pos_x[i] = one.pos_y[i] = 0.5f;
    }
    CHECK(encode_gsi(one, 0, 64, QuantSpec{}).empty());       // no width
    CHECK(encode_gsi(one, 64, -1, QuantSpec{}).empty());      // negative height
    CHECK(encode_gsi(one, 64, 64, QuantSpec{2, 2, 2, 2}).empty());   // bit depths out of range
    CHECK(!encode_gsi(one, 64, 64, QuantSpec{}).empty());     // and the valid case still works

    // quantize_cloud reports rather than silently leaving the cloud alone: a
    // caller that believes quantization happened will report a PSNR that the
    // file cannot deliver.
    GaussianCloud good = one;
    CHECK(quantize_cloud(good, QuantSpec{}));
    GaussianCloud bad = empty;
    CHECK(!quantize_cloud(bad, QuantSpec{}));
}

// ------------------------------------------------------- backend selection
//
// Asking for a backend the machine does not have must still compress the
// image. This is the path every user without a capable GPU takes.
void test_requesting_a_missing_gpu_still_encodes() {
    const Image img = test_image(96, 96);
    auto s = quick_settings();
    s.backend = Backend::Gpu;

    FactoryGuard guard([](std::string* why) -> std::unique_ptr<ITrainBackend> {
        if (why) *why = "no capable GPU on this machine";
        return nullptr;
    });
    const auto result = encode_image(img, s);
    CHECK(result.ok());
    CHECK(decode_gsi(result.file).has_value());
    CHECK(std::string(result.stats.backend_used) != "gpu");
    // Nothing failed midway, so there is nothing to report as a fallback.
    CHECK(result.stats.gpu_fallback_reason.empty());
    std::printf("  Backend::Gpu with no GPU -> %s, %.2f dB\n", result.stats.backend_used,
                result.stats.psnr);
}

// Whatever backend runs, the quality the user is shown has to be the quality
// the file actually delivers. A fallback that reported the failed run's
// numbers would be worse than no fallback.
void test_reported_quality_survives_a_fallback() {
    const Image img = test_image(128, 96);
    auto s = quick_settings();
    s.backend = Backend::Auto;

    FactoryGuard guard([](std::string*) -> std::unique_ptr<ITrainBackend> {
        return std::make_unique<FlakyBackend>(FlakyBackend::Mode::UnhealthyAtStep, 40);
    });
    const auto result = encode_image(img, s);
    if (!result.ok()) { fail("fallback produced no file"); return; }

    const auto file = decode_gsi(result.file);
    if (!file) { fail("fallback produced an unreadable file"); return; }
    const Image decoded = Renderer::render_image(file->cloud, file->width, file->height);
    const double actual = psnr(decoded, img);
    if (std::fabs(actual - result.stats.psnr) > 0.05) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "after a fallback the reported %.2f dB does not match the file's %.2f dB",
                      result.stats.psnr, actual);
        fail(buf);
        return;
    }
    std::printf("  after fallback: reported %.2f dB, file measures %.2f dB\n", result.stats.psnr,
                actual);
}

// ------------------------------------------------------ the real GPU, if any
//
// On a machine that has one, the accelerated path must reach the same file as
// the CPU. This is the check that would have caught a driver computing the
// wrong answer, and it is the same comparison gpu_init now makes at startup
// before letting the GPU touch a user's image.
void test_real_gpu_agrees_or_is_refused() {
    std::string why;
    if (!gpu_init(&why)) {
        std::printf("  (no usable GPU: %s -- which is itself the correct outcome)\n", why.c_str());
        return;
    }
    const Image img = test_image(128, 96);
    auto cpu_s = quick_settings();
    cpu_s.backend = Backend::Cpu;
    auto gpu_s = quick_settings();
    gpu_s.backend = Backend::Gpu;

    const auto cpu = encode_image(img, cpu_s);
    const auto gpu = encode_image(img, gpu_s);
    CHECK(cpu.ok());
    CHECK(gpu.ok());
    if (!cpu.ok() || !gpu.ok()) return;

    if (!gpu.stats.gpu_fallback_reason.empty()) {
        std::printf("  GPU withdrew mid-encode (%s) and the CPU finished the job\n",
                    gpu.stats.gpu_fallback_reason.c_str());
        return;
    }

    // What this can and cannot measure, because the number is noisier than it
    // looks. The two backends never agree exactly: the CPU approximates exp()
    // with a degree-4 polynomial while the shaders call the hardware's, and
    // gradient descent amplifies that difference chaotically over hundreds of
    // steps. The GPU is not even self-consistent -- tile lists are filled with
    // atomics, so the order gaussians accumulate in varies between runs of the
    // same binary on the same input.
    //
    // Measured across eight runs on one machine: the CPU is deterministic at
    // 37.67 dB while the GPU lands between 37.42 and 37.51, so a systematic
    // shortfall of about 0.2 dB with a further 0.1 dB of run-to-run spread. A
    // second implementation, Mesa's software rasteriser, sits at 0.79 dB. None
    // of that is a defect; it is the same arithmetic in a different order.
    //
    // So the threshold sits at roughly twice the widest implementation gap
    // observed, not just above it: 0.79 dB is a single sample from llvmpipe
    // and its run-to-run spread is unknown, and a limit that has to be raised
    // every time a new implementation appears is not a limit. The comparison
    // is also one-sided -- a GPU that beats the CPU is not a fault, and
    // testing the absolute difference would fail on good news.
    //
    // This stays a sharp test despite the width, because the failure worth
    // catching is not subtle. When the mid-encode fallback was deliberately
    // disabled and a backend allowed to go wrong partway through, the result
    // was 14.54 dB against the CPU's 37.67: a shortfall of 23 dB. Anything
    // that passes the eight-step startup check and then genuinely breaks
    // lands in that range, not in tenths of a decibel.
    constexpr double kMaxShortfall = 1.5;
    const double shortfall = cpu.stats.psnr - gpu.stats.psnr;
    if (shortfall > kMaxShortfall) {
        char buf[240];
        std::snprintf(buf, sizeof(buf),
                      "GPU passed its self-check but the finished image is %.2f dB worse than "
                      "the CPU's (%.2f dB against %.2f dB); limit is %.2f",
                      shortfall, gpu.stats.psnr, cpu.stats.psnr, kMaxShortfall);
        fail(buf);
        return;
    }
    std::printf("  real GPU: %.2f dB vs CPU %.2f dB (%+.2f dB, limit %.2f worse)\n",
                gpu.stats.psnr, cpu.stats.psnr, -shortfall, kMaxShortfall);
}

// ------------------------------------------------------------ time budgets
//
// Per-step cost is not constant -- the cloud doubles across the progressive
// additions and the gaussians spread as they optimize -- so a budget worked
// out once from a few early steps promises a bound it cannot keep. It did
// exactly that: on a 12 MP image an early extrapolation asked for 30 seconds
// and took 152. These check the property that matters, which is that the
// wall clock is respected whatever the cost curve does.
void test_a_time_budget_is_respected() {
    const Image img = test_image(384, 384);
    struct Case { double budget; float initial_ratio; const char* shape; };
    // The third case is the one that matters. Starting from 5% of the
    // gaussians and growing twentyfold makes late steps cost many times what
    // early ones do, which is the shape of the real 12 MP failure and the
    // shape that defeats any single up-front extrapolation. A flat cost curve
    // would let a projection-only implementation pass.
    const Case cases[] = {
        {1.0, 0.5f, "steady"},
        {2.5, 0.5f, "steady"},
        {1.5, 0.05f, "steeply rising"},
    };

    for (const Case& c : cases) {
        auto s = quick_settings();
        s.backend = Backend::Cpu;
        s.max_steps = 20000;          // far more than the budget can pay for
        s.min_budget_steps = 32;      // a floor low enough not to be the binding limit
        s.time_budget_seconds = c.budget;
        s.num_gaussians = 6000;
        s.initial_ratio = c.initial_ratio;

        const auto t0 = std::chrono::steady_clock::now();
        const auto result = encode_image(img, s);
        const double optimize = result.stats.optimize_seconds;
        (void)std::chrono::steady_clock::now();

        if (!result.ok()) { fail("a budgeted encode produced no file"); return; }
        CHECK(decode_gsi(result.file).has_value());
        // Generous slack: the check is against a bound being ignored, not
        // against scheduling noise on a loaded machine. It verifies that the
        // bound holds, not which of the two mechanisms held it -- the clock
        // check and the re-projection overlap by design, and the redundancy is
        // the point.
        if (optimize > c.budget * 2.0 + 1.0) {
            char buf[220];
            std::snprintf(buf, sizeof(buf),
                          "a %.1fs budget on a %s cost curve ran the optimizer for %.1fs "
                          "(%.1fx over)",
                          c.budget, c.shape, optimize, optimize / c.budget);
            fail(buf);
            return;
        }
        if (result.stats.steps_run >= s.max_steps) {
            fail("the budget did not shorten the run at all");
            return;
        }
        std::printf("  %.1fs budget, %-14s cost -> %.1fs, %d of %d steps, %.2f dB\n", c.budget,
                    c.shape, optimize, result.stats.steps_run, result.stats.steps_requested,
                    result.stats.psnr);
    }
}

// A budget must not be able to cut a run down to nothing. Below a few hundred
// steps quality falls away fast, and "finished instantly, looks wrong" is not
// a better outcome than "took a while".
void test_the_budget_floor_holds() {
    const Image img = test_image(256, 256);
    auto s = quick_settings();
    s.backend = Backend::Cpu;
    s.max_steps = 4000;
    s.min_budget_steps = 400;
    s.time_budget_seconds = 0.001;   // impossible on any hardware

    const auto result = encode_image(img, s);
    if (!result.ok()) { fail("an impossible budget produced no file"); return; }
    CHECK(result.stats.steps_run >= s.min_budget_steps);
    CHECK(result.stats.steps_run < s.max_steps);
    std::printf("  an impossible budget still ran %d steps (floor %d), %.2f dB\n",
                result.stats.steps_run, s.min_budget_steps, result.stats.psnr);
}

// No budget means no clock in the loop, so the same input gives the same file.
// The command line tool and every baseline in the regression suite depend on
// this; a budget that leaked into the default would make them all flaky.
void test_no_budget_means_reproducible() {
    const Image img = test_image(128, 128);
    auto s = quick_settings();
    s.backend = Backend::Cpu;
    CHECK(s.time_budget_seconds == 0.0);   // the default, and it must stay that way

    const auto a = encode_image(img, s);
    const auto b = encode_image(img, s);
    CHECK(a.ok() && b.ok());
    if (a.file != b.file) {
        fail("with no time budget two identical runs produced different files");
        return;
    }
    CHECK(a.stats.steps_run == a.stats.steps_requested);
    std::printf("  no budget: two runs byte-identical, all %d steps run\n", a.stats.steps_run);
}

// Previews render the whole image, so their cost grows with image size while
// the configured interval does not. Holding the interval fixed spends an
// unbounded share of the encode drawing progress pictures -- around 40% at 12
// megapixels. The interval must stretch to match what a preview actually
// costs, including the receiver's work, which is why the cost is timed across
// the callback and not just the render.
void test_preview_cost_stays_bounded() {
    // Long enough that the throttle has to hold across many previews. A run
    // short enough for one preview would pass without the throttle existing.
    const Image img = test_image(384, 384);
    auto s = quick_settings();
    s.backend = Backend::Cpu;
    s.max_steps = 4000;
    s.preview_interval_ms = 1;      // ask for previews as fast as possible
    s.preview_time_fraction = 0.10;

    // Stands in for an expensive receiver: at 12 MP the conversion to
    // displayable pixels is 48 MB of work on top of the render.
    constexpr auto kCallbackCost = std::chrono::milliseconds(25);
    int previews = 0;
    auto progress = [&](const EncodeProgress& p) {
        if (!p.preview) return;
        ++previews;
        std::this_thread::sleep_for(kCallbackCost);
    };

    const auto t0 = std::chrono::steady_clock::now();
    const auto result = encode_image(img, s, progress);
    const double total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!result.ok()) { fail("an encode with previews produced no file"); return; }

    const double spent_on_previews = double(previews) * 0.025;
    const double share = spent_on_previews / std::max(1e-9, total);
    std::printf("  %d previews, %.0f%% of a %.2fs encode spent on them\n", previews,
                100.0 * share, total);
    // The bound is the configured fraction with headroom: the throttle reacts
    // to the previous preview's cost, so the first one is always unbudgeted.
    if (share > 0.30) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "previews consumed %.0f%% of the encode against a %.0f%% target",
                      100.0 * share, 100.0 * s.preview_time_fraction);
        fail(buf);
    }
    if (previews == 0) fail("no previews were produced at all, so the throttle is not throttling");
}

} // namespace

int main() {
    std::puts("robustness suite");
    {   // See gpu.h: created here or not at all.
        std::string why;
        if (!gpu_init(&why)) std::printf("  (no GPU available: %s)\n", why.c_str());
    }
    test_accelerator_failure_falls_back_to_cpu();
    test_cpu_failure_is_reported_not_written();
    test_encoder_never_emits_an_unreadable_file();
    test_encoder_refuses_what_it_cannot_represent();
    test_requesting_a_missing_gpu_still_encodes();
    test_reported_quality_survives_a_fallback();
    test_a_time_budget_is_respected();
    test_the_budget_floor_holds();
    test_no_budget_means_reproducible();
    test_preview_cost_stays_bounded();
    test_real_gpu_agrees_or_is_refused();
    gpu_shutdown();

    if (failures == 0) {
        std::puts("robustness suite passed");
        return 0;
    }
    std::printf("%d robustness failure(s)\n", failures);
    return 1;
}
