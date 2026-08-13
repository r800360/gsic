// Regression tests: the checks that answer "did my change help, hurt, or do
// nothing", which unit tests do not.
//
// Everything here is reproducible from fixed constants. The corpus is
// generated procedurally rather than committed as image files, so the suite
// runs identically on any machine with nothing to download and no binaries in
// the repository. Each pattern targets a different part of the codec:
// smooth gradients, hard edges, fine texture, and flat regions.
//
// Baselines below were recorded on a known good build. If a change moves a
// number, that is the point: decide whether the move is an improvement before
// updating the constant, and say so in the commit.

#include "core/codec.h"
#include "core/gpu.h"
#include "core/image.h"
#include "core/metrics.h"
#include "core/renderer.h"
#include "core/trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

using namespace gsic;

static int failures = 0;

static void fail(const std::string& what) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
}

namespace {

// ------------------------------------------------------------------ corpus
// A deterministic value noise so "texture" is reproducible without an RNG
// whose implementation could vary between standard libraries.
float hash2(int x, int y) {
    std::uint32_t h = std::uint32_t(x) * 374761393u + std::uint32_t(y) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return float(h ^ (h >> 16)) / 4294967296.0f;
}

float smooth_noise(float x, float y) {
    const int xi = int(std::floor(x)), yi = int(std::floor(y));
    const float fx = x - xi, fy = y - yi;
    const float sx = fx * fx * (3.f - 2.f * fx), sy = fy * fy * (3.f - 2.f * fy);
    const float a = hash2(xi, yi), b = hash2(xi + 1, yi);
    const float c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
    return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
}

enum class Pattern { Gradient, Edges, Texture, Mixed };

Image make_image(Pattern p, int w, int h) {
    Image img(w, h, 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float u = float(x) / w, v = float(y) / h;
            float r = 0, g = 0, b = 0;
            switch (p) {
                case Pattern::Gradient:  // smooth, what gaussians handle best
                    r = u; g = v; b = 0.5f * (u + v);
                    break;
                case Pattern::Edges: {  // hard boundaries and flat plateaus
                    const bool cell = ((x / 32) + (y / 32)) % 2 == 0;
                    const bool disc = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f) < 0.08f;
                    r = disc ? 0.9f : (cell ? 0.15f : 0.75f);
                    g = disc ? 0.25f : (cell ? 0.6f : 0.2f);
                    b = disc ? 0.35f : (cell ? 0.8f : 0.5f);
                    break;
                }
                case Pattern::Texture:  // dense detail, the hardest case
                    r = smooth_noise(u * 24.f, v * 24.f);
                    g = smooth_noise(u * 24.f + 7.f, v * 24.f + 3.f);
                    b = smooth_noise(u * 24.f - 5.f, v * 24.f + 11.f);
                    break;
                case Pattern::Mixed:  // smooth background, sharp foreground, texture patch
                    r = 0.25f + 0.5f * u;
                    g = 0.35f + 0.4f * v;
                    b = 0.6f - 0.3f * u;
                    if (std::fabs(u - 0.35f) < 0.12f && std::fabs(v - 0.4f) < 0.2f) {
                        r = 0.95f; g = 0.15f; b = 0.2f;
                    }
                    if (u > 0.6f && v > 0.55f) {
                        const float n = smooth_noise(u * 40.f, v * 40.f);
                        r = g = b = n;
                    }
                    break;
            }
            img.plane(0)[size_t(y) * w + x] = r;
            img.plane(1)[size_t(y) * w + x] = g;
            img.plane(2)[size_t(y) * w + x] = b;
        }
    }
    return img;
}

struct Baseline {
    Pattern pattern;
    const char* name;
    double psnr;        // recorded on a known good build
    int max_bytes;      // file must not grow past this
};

// Recorded with the settings in encode_settings() below. Regenerate with
// `gsic_regression --record` after a deliberate quality change.
//
// The spread across patterns is itself informative: smooth gradients are what
// gaussians represent most efficiently, hard edges are the weakest case, and
// a change that moves one pattern without the others usually means the change
// helps one kind of content at another's expense.
constexpr Baseline kBaselines[] = {
    {Pattern::Gradient, "gradient", 53.80, 15894},
    {Pattern::Edges,    "edges",    27.86, 15961},
    {Pattern::Texture,  "texture",  43.42, 15969},
    {Pattern::Mixed,    "mixed",    37.20, 15818},
};

// Quality may not drop by more than this against the recorded baseline, and
// files may not grow by more than 5%. The tolerance absorbs the small
// differences between the SSE2 and AVX2 kernels and between backends.
constexpr double kPsnrTolerance = 0.35;

EncodeSettings encode_settings() {
    EncodeSettings s;
    s.num_gaussians = 1200;
    s.max_steps = 600;
    s.backend = Backend::Cpu;   // the path every machine has
    s.preview_interval_ms = 0;
    s.seed = 123;
    return s;
}

// -------------------------------------------------------------- the checks
void test_quality_baselines(bool record) {
    if (record) std::printf("\n// regenerated baselines:\n");
    for (const Baseline& b : kBaselines) {
        const Image img = make_image(b.pattern, 256, 256);
        const auto result = encode_image(img, encode_settings());
        const auto bytes = int(result.file.size());

        if (record) {
            std::printf("    {Pattern::%-9s \"%-9s %.2f, %d},\n", "?", b.name, result.stats.psnr,
                        int(bytes * 1.05));
            continue;
        }
        if (b.psnr <= 0.0) continue;  // not yet recorded

        if (result.stats.psnr < b.psnr - kPsnrTolerance) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "quality regression on '%s': %.2f dB, baseline %.2f dB (limit %.2f)",
                          b.name, result.stats.psnr, b.psnr, b.psnr - kPsnrTolerance);
            fail(buf);
        }
        if (bytes > b.max_bytes) {
            char buf[200];
            std::snprintf(buf, sizeof(buf), "size regression on '%s': %d bytes, limit %d", b.name,
                          bytes, b.max_bytes);
            fail(buf);
        }
        std::printf("  %-9s %6.2f dB (baseline %6.2f)  %6d bytes (limit %d)\n", b.name,
                    result.stats.psnr, b.psnr, bytes, b.max_bytes);
    }
}

// The same input must always give the same file. Without this, every other
// comparison in this suite is noise.
void test_determinism() {
    const Image img = make_image(Pattern::Mixed, 192, 192);
    const auto a = encode_image(img, encode_settings());
    const auto b = encode_image(img, encode_settings());
    if (a.file != b.file) {
        fail("encoding is not deterministic: two identical runs produced different files");
        return;
    }
    std::printf("  determinism: two runs produced byte-identical files (%zu bytes)\n",
                a.file.size());
}

// A different seed must actually change the result, otherwise the seed is
// silently ignored and the determinism check above proves nothing.
void test_seed_is_used() {
    const Image img = make_image(Pattern::Mixed, 192, 192);
    auto s = encode_settings();
    const auto a = encode_image(img, s);
    s.seed = 999;
    const auto b = encode_image(img, s);
    if (a.file == b.file) fail("changing the seed did not change the output");
}

// What a decoder sees must match what the encoder reported, or the quality
// numbers in the interface are fiction.
void test_reported_quality_matches_file() {
    const Image img = make_image(Pattern::Mixed, 256, 256);
    const auto result = encode_image(img, encode_settings());
    const auto file = decode_gsi(result.file);
    if (!file) {
        fail("could not decode the file the encoder just produced");
        return;
    }
    const Image decoded = Renderer::render_image(file->cloud, file->width, file->height);
    const double actual = psnr(decoded, img);
    if (std::fabs(actual - result.stats.psnr) > 0.05) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "reported quality %.2f dB does not match the decoded file's %.2f dB",
                      result.stats.psnr, actual);
        fail(buf);
    }
    std::printf("  reported %.2f dB, decoded file measures %.2f dB\n", result.stats.psnr, actual);
}

// Both backends must agree, so a GPU change cannot quietly alter output.
void test_backend_agreement() {
    std::string why;
    if (!gpu_init(&why)) {
        std::printf("  (backend agreement skipped: %s)\n", why.c_str());
        return;
    }
    const Image img = make_image(Pattern::Mixed, 256, 256);
    auto s = encode_settings();
    const auto cpu = encode_image(img, s);
    s.backend = Backend::Gpu;
    const auto gpu = encode_image(img, s);
    // One-sided, and the tolerance is measured rather than guessed; see the
    // long explanation on the equivalent check in the robustness suite. In
    // short: the backends differ by construction and the GPU is not even
    // deterministic between runs, so the only meaningful question is whether
    // it comes out materially worse.
    constexpr double kMaxShortfall = 1.5;
    const double shortfall = cpu.stats.psnr - gpu.stats.psnr;
    if (shortfall > kMaxShortfall) {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "GPU is %.2f dB worse than the CPU: %.2f dB against %.2f",
                      shortfall, gpu.stats.psnr, cpu.stats.psnr);
        fail(buf);
    }
    std::printf("  CPU %.2f dB, GPU %.2f dB (%+.2f dB)\n", cpu.stats.psnr, gpu.stats.psnr,
                -shortfall);
}

// Decoding at a larger size must stay faithful rather than degenerate, which
// is the property that makes the representation resolution independent.
void test_rescaled_decode() {
    const Image img = make_image(Pattern::Gradient, 192, 192);
    const auto result = encode_image(img, encode_settings());
    auto file = decode_gsi(result.file);
    if (!file) { fail("decode failed"); return; }
    GaussianCloud cloud = file->cloud;
    for (auto& v : cloud.sinv_x) v *= 0.5f;   // 2x output
    for (auto& v : cloud.sinv_y) v *= 0.5f;
    const Image big = Renderer::render_image(cloud, file->width * 2, file->height * 2);
    if (big.w != file->width * 2 || big.h != file->height * 2) { fail("wrong rescaled size"); return; }
    double mean = 0;
    for (float v : big.data) mean += v;
    mean /= double(big.data.size());
    if (!(mean > 0.05 && mean < 0.95)) {
        fail("rescaled decode produced a degenerate image");
        return;
    }
    std::printf("  rescaled decode: %dx%d, mean level %.3f\n", big.w, big.h, mean);
}

} // namespace

int main(int argc, char** argv) {
    const bool record = argc > 1 && std::string(argv[1]) == "--record";
    std::printf("regression suite (%s kernels)\n", kernels().name);
    {   // See gpu.h: created here on the main thread, or not at all.
        std::string why;
        if (!gpu_init(&why)) std::printf("  (no GPU available: %s)\n", why.c_str());
    }
    test_quality_baselines(record);
    if (record) return 0;
    test_determinism();
    test_seed_is_used();
    test_reported_quality_matches_file();
    test_backend_agreement();
    test_rescaled_decode();
    gpu_shutdown();

    if (failures == 0) {
        std::puts("regression suite passed");
        return 0;
    }
    std::printf("%d regression failure(s)\n", failures);
    return 1;
}
