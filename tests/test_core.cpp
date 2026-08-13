// Unit tests for the core library: kernel math (analytic vs numerical
// gradients), codec roundtrip + hostile-input rejection, and metrics sanity.
#include "core/codec.h"
#include "core/gpu.h"
#include "core/image.h"
#include "core/kernels.h"
#include "core/metrics.h"
#include "core/renderer.h"
#include "core/trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace gsic;

static int failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                              \
    do {                                                                   \
        const double va = (a), vb = (b);                                   \
        if (std::fabs(va - vb) > (tol)) {                                  \
            std::printf("FAIL %s:%d: %s=%g vs %s=%g\n", __FILE__, __LINE__, \
                        #a, va, #b, vb);                                   \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

namespace {

GaussianCloud random_cloud(int n, int channels, std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    GaussianCloud c;
    c.channels = channels;
    c.resize(n);
    for (int i = 0; i < n; ++i) {
        c.pos_x[i] = uni(rng);
        c.pos_y[i] = uni(rng);
        c.sinv_x[i] = 0.05f + uni(rng);
        c.sinv_y[i] = 0.05f + uni(rng);
        c.rot[i] = (uni(rng) - 0.5f) * 6.f;
        for (int ch = 0; ch < channels; ++ch)
            c.color[size_t(i) * channels + ch] = uni(rng) * 2.f - 0.5f;
    }
    return c;
}

// L2 loss of a rendered cloud against a target; used for finite differences.
double render_loss(const GaussianCloud& cloud, const Image& target) {
    Image out = Renderer::render_image(cloud, target.w, target.h);
    double sum = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        const double d = out.data[i] - target.data[i];
        sum += d * d;
    }
    return sum / double(out.data.size());
}

void test_gradients() {
    std::mt19937 rng(7);
    const int W = 48, H = 40, C = 3;
    Image target(W, H, C);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    for (auto& v : target.data) v = uni(rng);
    GaussianCloud cloud = random_cloud(6, C, rng);

    // Analytic gradients via the kernels (pure L2 so the loss is smooth).
    Renderer renderer;
    renderer.set_size(W, H, C);
    renderer.project_and_bin(cloud);
    std::vector<float> render(size_t(W) * H * C), grad(size_t(W) * H * C);
    const int n = cloud.count();
    std::vector<float> grads(size_t(n) * (5 + C));
    KernelCtx ctx;
    renderer.fill_ctx(ctx, cloud);
    for (int c = 0; c < C; ++c) {
        ctx.target[c] = target.plane(c);
        ctx.render[c] = render.data() + size_t(c) * W * H;
        ctx.grad[c] = grad.data() + size_t(c) * W * H;
    }
    ctx.w_l1 = 0.f;
    ctx.w_l2 = 1.f;
    ctx.inv_norm = 1.f / float(W * H * C);
    ctx.d_pos_x = grads.data();
    ctx.d_pos_y = grads.data() + n;
    ctx.d_sinv_x = grads.data() + 2 * n;
    ctx.d_sinv_y = grads.data() + 3 * n;
    ctx.d_rot = grads.data() + 4 * n;
    ctx.d_color = grads.data() + 5 * n;
    const Kernels& k = kernels();
    const int tiles = renderer.tile_count();
    k.forward_tiles(ctx, 0, tiles);
    double l1 = 0, l2 = 0;
    k.loss_grad_rows(ctx, 0, H, &l1, &l2);
    k.backward_gaussians(ctx, 0, n);

    // Finite differences on every parameter of gaussian 2.
    const int g = 2;
    auto fd = [&](float* param, float eps) {
        const float saved = *param;
        *param = saved + eps;
        const double up = render_loss(cloud, target);
        *param = saved - eps;
        const double dn = render_loss(cloud, target);
        *param = saved;
        return (up - dn) / (2.0 * eps);
    };
    struct Case { float* p; float analytic; float eps; const char* name; };
    const Case cases[] = {
        {&cloud.pos_x[g], grads[g], 1e-3f, "pos_x"},
        {&cloud.pos_y[g], grads[n + g], 1e-3f, "pos_y"},
        {&cloud.sinv_x[g], grads[2 * n + g], 5e-3f, "sinv_x"},
        {&cloud.sinv_y[g], grads[3 * n + g], 5e-3f, "sinv_y"},
        {&cloud.rot[g], grads[4 * n + g], 1e-2f, "rot"},
        {&cloud.color[size_t(g) * C + 1], grads[5 * n + size_t(g) * C + 1], 2e-2f, "color"},
    };
    for (const auto& c : cases) {
        const double numeric = fd(c.p, c.eps);
        const double denom = std::max({std::fabs(numeric), std::fabs(double(c.analytic)), 1e-7});
        const double rel = std::fabs(numeric - c.analytic) / denom;
        // Loose bounds: finite differences see float32 rendering noise plus
        // the (intentional) derivative discontinuity at the cutoff ellipse,
        // so this guards against sign/scale bugs rather than exact agreement.
        if (rel > 0.08 && std::fabs(numeric - c.analytic) > 5e-5) {
            std::printf("FAIL gradient %s: analytic %g vs numeric %g (rel %g)\n", c.name,
                        double(c.analytic), numeric, rel);
            ++failures;
        }
    }
}

void test_codec_roundtrip() {
    std::mt19937 rng(11);
    for (int channels : {1, 3, 4}) {
        GaussianCloud cloud = random_cloud(500, channels, rng);
        const QuantSpec q{16, 16, 16, 16};
        auto bytes = encode_gsi(cloud, 640, 480, q);
        auto decoded = decode_gsi(bytes);
        CHECK(decoded.has_value());
        if (!decoded) continue;
        CHECK(decoded->width == 640 && decoded->height == 480);
        CHECK(decoded->cloud.count() == cloud.count());
        CHECK(decoded->cloud.channels == channels);
        for (int i = 0; i < cloud.count(); ++i) {
            CHECK_NEAR(decoded->cloud.pos_x[i], cloud.pos_x[i], 1e-4);
            CHECK_NEAR(decoded->cloud.sinv_x[i], cloud.sinv_x[i], 1e-3);
            // Rotation may be shifted by pi (same gaussian); compare conics.
            const float d = std::fabs(std::remainder(decoded->cloud.rot[i] - cloud.rot[i],
                                                     3.14159265f));
            CHECK(d < 1e-3f);
            for (int c = 0; c < channels; ++c)
                CHECK_NEAR(decoded->cloud.color[size_t(i) * channels + c],
                           cloud.color[size_t(i) * channels + c], 1e-3);
        }
    }
}

void test_codec_rejects_garbage() {
    std::mt19937 rng(13);
    GaussianCloud cloud = random_cloud(64, 3, rng);
    auto bytes = encode_gsi(cloud, 100, 100, QuantSpec{});

    // Truncations at every prefix length must fail cleanly, never crash.
    for (size_t len = 0; len < bytes.size(); len += 7)
        CHECK(!decode_gsi({bytes.data(), len}).has_value());
    // Bit flips in the header.
    for (size_t pos = 0; pos < 24; ++pos) {
        auto copy = bytes;
        copy[pos] ^= 0xFF;
        (void)decode_gsi(copy);  // must not crash; may or may not decode
    }
    // Random buffers.
    std::uniform_int_distribution<int> byte(0, 255);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<std::uint8_t> junk(std::uniform_int_distribution<int>(0, 300)(rng));
        for (auto& b : junk) b = std::uint8_t(byte(rng));
        (void)decode_gsi(junk);
    }
}

// Decoding at a size the file was never stored at. This is the whole reason
// the format is worth having over a raster one, and it is now one
// implementation shared by the command line tool and the window rather than
// arithmetic copied into each. That matters most for the checking: a scale
// arriving from a text box is untrusted input in the same sense a file is, and
// 5000x on a 2K image asks for a hundred gigapixels.
void test_scaled_decode() {
    std::mt19937 rng(29);
    GaussianCloud cloud = random_cloud(200, 3, rng);
    auto bytes = encode_gsi(cloud, 320, 240, QuantSpec{});
    auto file = decode_gsi(bytes);
    CHECK(file.has_value());
    if (!file) return;

    // At the stored size nothing is touched.
    auto same = scale_gsi(*file, 1.f);
    CHECK(same.has_value());
    if (same) {
        CHECK(same->w == 320 && same->h == 240);
        for (int i = 0; i < same->cloud.count(); ++i)
            CHECK_NEAR(same->cloud.sinv_x[i], file->cloud.sinv_x[i], 1e-6);
    }

    // Larger and smaller: the output size follows, and gaussian sizes are
    // scaled with it. Leaving sinv alone would render the same picture with
    // half-size splats -- a 2x decode of dots on a grey field.
    for (float s : {0.5f, 2.f, 3.25f}) {
        std::string err;
        auto scaled = scale_gsi(*file, s, &err);
        CHECK(scaled.has_value());
        if (!scaled) {
            std::printf("  scale %.2f refused: %s\n", double(s), err.c_str());
            continue;
        }
        CHECK(scaled->w == std::max(1, int(320 * s + 0.5f)));
        CHECK(scaled->h == std::max(1, int(240 * s + 0.5f)));
        const float ratio = float(scaled->w) / 320.f;
        for (int i = 0; i < scaled->cloud.count(); ++i) {
            CHECK_NEAR(scaled->cloud.sinv_x[i], file->cloud.sinv_x[i] / ratio, 1e-5);
            CHECK_NEAR(scaled->cloud.sinv_y[i], file->cloud.sinv_y[i] / ratio, 1e-5);
        }
        // Positions are normalized, so they must not move.
        for (int i = 0; i < scaled->cloud.count(); ++i)
            CHECK_NEAR(scaled->cloud.pos_x[i], file->cloud.pos_x[i], 1e-6);
    }

    // Everything out of range comes back as a message, before any allocation.
    for (float bad : {0.f, -1.f, 1e9f, kMaxDecodeScale * 2.f, kMinDecodeScale * 0.5f}) {
        std::string err;
        CHECK(!scale_gsi(*file, bad, &err).has_value());
        CHECK(!err.empty());
    }
    {
        std::string err;
        CHECK(!scale_gsi(*file, std::numeric_limits<float>::quiet_NaN(), &err).has_value());
        CHECK(!err.empty());
    }
    // A scale inside the offered range that would still exceed the build's
    // pixel ceiling is refused for that reason instead of being attempted.
    {
        GsiFile huge = *file;
        huge.width = kMaxDimension;
        huge.height = kMaxDimension;
        std::string err;
        CHECK(!scale_gsi(huge, kMaxDecodeScale, &err).has_value());
        CHECK(!err.empty());
    }

    // And the convenience wrapper produces an image of exactly that size.
    Image rendered = render_gsi(*file, 2.f);
    CHECK(rendered.w == 640 && rendered.h == 480 && rendered.c == 3);
    std::string err;
    Image refused = render_gsi(*file, 0.f, &err);
    CHECK(refused.empty());
    CHECK(!err.empty());
}

// The size caps are the guard against size_t overflow on 32-bit builds and
// against an untrusted file driving a huge allocation. Both must hold.
void test_size_limits() {
    std::printf("  build is %d-bit: max %lld pixels, %d per side, %d gaussians\n",
                kIs32Bit ? 32 : 64, static_cast<long long>(kMaxPixels), kMaxDimension,
                kMaxGaussians);
    // Everything the caps allow must fit in size_t without wrapping.
    CHECK(kMaxPixels <= std::int64_t(SIZE_MAX / (kMaxChannels * sizeof(float))));
    CHECK(std::int64_t(kMaxGaussians) * (5 + kMaxChannels) <=
          std::int64_t(SIZE_MAX / sizeof(float)));
    CHECK(std::int64_t(kMaxDimension) * kMaxDimension >= kMaxPixels);

    // A header claiming more gaussians than the build allows must be refused
    // before anything is allocated.
    std::mt19937 rng(41);
    GaussianCloud cloud = random_cloud(32, 3, rng);
    auto bytes = encode_gsi(cloud, 64, 64, QuantSpec{});
    // Gaussian count sits at byte offset 20 (after magic, flags, bits, w, h).
    auto poke = [&](size_t off, std::uint32_t v) {
        for (int i = 0; i < 4; ++i) bytes[off + i] = std::uint8_t(v >> (8 * i));
    };
    poke(20, std::uint32_t(kMaxGaussians) + 1);
    std::string err;
    CHECK(!decode_gsi(bytes, &err).has_value());
    poke(12, std::uint32_t(kMaxDimension) + 1);   // width
    CHECK(!decode_gsi(bytes).has_value());
}

void test_metrics() {
    Image a(64, 64, 3);
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    for (auto& v : a.data) v = uni(rng);
    CHECK(psnr(a, a) > 90.0);
    CHECK_NEAR(ssim(a, a), 1.0, 1e-6);
    Image b = a;
    for (auto& v : b.data) v = std::min(1.f, v + 0.1f);
    CHECK(psnr(a, b) < 30.0);
    CHECK(ssim(a, b) < 1.0);
}

void test_renderer_basic() {
    // One isotropic gaussian in the middle: peak at center, symmetric.
    GaussianCloud cloud;
    cloud.channels = 1;
    cloud.resize(1);
    cloud.pos_x[0] = 0.5f;
    cloud.pos_y[0] = 0.5f;
    cloud.sinv_x[0] = cloud.sinv_y[0] = 1.f / 8.f;
    cloud.rot[0] = 0.f;
    cloud.color[0] = 1.f;
    Image img = Renderer::render_image(cloud, 64, 64);
    const float* p = img.plane(0);
    CHECK(p[32 * 64 + 32] > 0.9f);
    CHECK_NEAR(p[32 * 64 + 12], p[32 * 64 + 51], 2e-3);  // horizontal symmetry
    CHECK_NEAR(p[12 * 64 + 32], p[51 * 64 + 32], 2e-3);  // vertical symmetry
    CHECK(p[0] < 0.05f);
}

void test_end_to_end_tiny() {
    // A tiny smooth image should compress to high PSNR quickly.
    Image img(96, 80, 3);
    for (int c = 0; c < 3; ++c) {
        float* p = img.plane(c);
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x)
                p[size_t(y) * img.w + x] =
                    0.5f + 0.4f * std::sin(0.07f * x + c) * std::cos(0.05f * y);
    }
    EncodeSettings s;
    s.num_gaussians = 220;
    s.max_steps = 900;
    s.backend = Backend::Cpu;
    s.preview_interval_ms = 0;
    auto result = encode_image(img, s);
    std::printf("  tiny end-to-end: PSNR %.2f dB, SSIM %.4f\n", result.stats.psnr,
                result.stats.ssim);
    CHECK(result.stats.psnr > 30.0);
    CHECK(!result.file.empty());
    auto decoded = decode_gsi(result.file);
    CHECK(decoded.has_value());
}

// GPU backend must agree with the CPU backend step-for-step (small float
// noise allowed). Skipped when no capable GPU is present.
void test_gpu_matches_cpu() {
    std::string why;
    auto gpu = make_gpu_backend(&why);
    if (!gpu) {
        std::printf("  (GPU test skipped: %s)\n", why.c_str());
        return;
    }
    std::mt19937 rng(23);
    const int W = 160, H = 120, C = 3;
    Image target(W, H, C);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    for (auto& v : target.data) v = uni(rng);
    GaussianCloud a = random_cloud(300, C, rng);
    GaussianCloud b = a;

    EncodeSettings s;
    s.preview_interval_ms = 0;
    auto cpu = make_cpu_backend();
    CHECK(cpu->prepare(target, a, s));
    CHECK(gpu->prepare(target, b, s));
    for (int t = 1; t <= 20; ++t) {
        const float lc = cpu->step(t, 1.f);
        const float lg = gpu->step(t, 1.f);
        if (t <= 3 || t == 20) {
            const double rel = std::fabs(lc - lg) / std::max(1e-9f, lc);
            if (rel > 0.02) {
                std::printf("FAIL gpu loss diverged at step %d: cpu %.6f gpu %.6f\n", t,
                            double(lc), double(lg));
                ++failures;
            }
        }
    }
    gpu->sync_cloud(b);
    struct Group { const char* name; const std::vector<float>* pa; const std::vector<float>* pb; };
    const Group groups[] = {
        {"pos_x", &a.pos_x, &b.pos_x},   {"pos_y", &a.pos_y, &b.pos_y},
        {"sinv_x", &a.sinv_x, &b.sinv_x}, {"sinv_y", &a.sinv_y, &b.sinv_y},
        {"rot", &a.rot, &b.rot},         {"color", &a.color, &b.color},
    };
    for (const auto& g : groups) {
        double max_diff = 0;
        for (size_t i = 0; i < g.pa->size(); ++i)
            max_diff = std::max(max_diff, std::fabs(double((*g.pa)[i] - (*g.pb)[i])));
        std::printf("  gpu-vs-cpu after 20 steps: %-6s max param diff %.6f\n", g.name, max_diff);
        // Adam steps are bounded by lr; 20 steps of accumulated float noise
        // should stay well under 20 * lr.
        if (max_diff > 0.05) {
            std::printf("FAIL gpu param group %s diverged\n", g.name);
            ++failures;
        }
    }
}

// GPU decode must reproduce the CPU render. Skipped without a GPU.
void test_gpu_render_matches_cpu() {
    std::mt19937 rng(31);
    GaussianCloud cloud = random_cloud(400, 3, rng);
    auto gpu = gpu_render(cloud, 200, 150);
    if (!gpu) {
        std::puts("  (GPU render test skipped: no GPU)");
        return;
    }
    Image cpu = Renderer::render_image(cloud, 200, 150);
    CHECK(gpu->w == cpu.w && gpu->h == cpu.h && gpu->c == cpu.c);
    double max_diff = 0;
    for (size_t i = 0; i < cpu.data.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(double(cpu.data[i] - gpu->data[i])));
    std::printf("  gpu-vs-cpu render: max pixel diff %.5f\n", max_diff);
    // Only difference is the fast exp approximation on the CPU side.
    CHECK(max_diff < 0.01);
}

} // namespace

int main() {
    // From the main thread, once, before anything asks for a GPU backend:
    // the context may only be created by a thread that outlives every use of
    // it (see gpu.h). Tests that need it find it ready; if there is none, the
    // GPU-dependent cases below skip themselves.
    {
        std::string why;
        if (!gpu_init(&why)) std::printf("  (no GPU available: %s)\n", why.c_str());
    }
    test_gradients();
    test_codec_roundtrip();
    test_codec_rejects_garbage();
    test_scaled_decode();
    test_size_limits();
    test_metrics();
    test_renderer_basic();
    test_end_to_end_tiny();
    test_gpu_matches_cpu();
    test_gpu_render_matches_cpu();
    if (failures == 0) {
        std::puts("all tests passed");
        return 0;
    }
    std::printf("%d failure(s)\n", failures);
    return 1;
}
