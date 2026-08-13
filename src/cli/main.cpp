// gsic command line tool: compress, decompress, info, bench.
#include "core/codec.h"
#include "core/format.h"
#include "core/gpu.h"
#include "core/image.h"
#include "core/kernels.h"
#include "core/metrics.h"
#include "core/renderer.h"
#include "core/trainer.h"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifndef GSIC_VERSION
#define GSIC_VERSION "unknown"
#endif

namespace fs = std::filesystem;
using namespace gsic;

// On hybrid laptops, ask the driver for the discrete GPU.
#if defined(_WIN32)
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

namespace {

void print_usage() {
    std::puts(
        "gsic - gaussian splat image compressor\n"
        "\n"
        "usage:\n"
        "  gsic compress <images...> [options]     compress to .gsi\n"
        "  gsic decompress <files.gsi...> [options] decode to .png\n"
        "  gsic info <file.gsi>                    show file details\n"
        "  gsic bench [image] [options]            measure encode speed\n"
        "  gsic compare <a> <b>                    PSNR/SSIM between two images\n"
        "  gsic diagnose                           report what this machine will do\n"
        "\n"
        "compress options:\n"
        "  -o <dir>          output directory (default: next to input)\n"
        "  --preset <p>      fast | balanced | high (default: balanced)\n"
        "  -n <count>        gaussian count (overrides preset)\n"
        "  --steps <n>       optimization steps (overrides preset)\n"
        "  --bits <n>        set all four bit depths at once, 4-16\n"
        "  --bits-pos <n>    position bits (default: 16)\n"
        "  --bits-scale <n>  scale bits (default: 12)\n"
        "  --bits-rot <n>    rotation bits (default: 10)\n"
        "  --bits-color <n>  color bits (default: 12)\n"
        "  --cpu / --gpu     force backend (default: auto with CPU fallback)\n"
        "  --png             also write the decoded .png next to the .gsi\n"
        "  --seed <n>        random seed (default: 123)\n"
        "  --time-budget <s> stop early to finish within s seconds (default: no\n"
        "                    limit, which keeps the encode reproducible)\n"
        "\n"
        "decompress options:\n"
        "  -o <file|dir>     output path\n"
        "  --scale <s>       render at s times the stored resolution\n");
}

bool parse_int(const char* s, int& out) {
    return std::from_chars(s, s + std::strlen(s), out).ec == std::errc{};
}

bool parse_float(const char* s, float& out) {
    char* end = nullptr;
    out = std::strtof(s, &end);
    return end && *end == '\0';
}

std::vector<std::uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

bool write_file(const fs::path& p, const std::vector<std::uint8_t>& bytes) {
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    return bool(f);
}

void apply_preset(EncodeSettings& s, const std::string& preset) {
    if (preset == "fast") {
        s.pixels_per_gaussian = 450;
        s.max_steps = 1200;
    } else if (preset == "balanced") {
        s.pixels_per_gaussian = 300;
        s.max_steps = 3000;
    } else if (preset == "high") {
        s.pixels_per_gaussian = 190;
        s.max_steps = 8000;
    } else {
        std::fprintf(stderr, "unknown preset '%s'\n", preset.c_str());
        std::exit(2);
    }
}

int cmd_compress(const std::vector<std::string>& inputs, EncodeSettings settings,
                 const fs::path& out_dir, bool also_png) {
    int failures = 0;
    for (const auto& in : inputs) {
        const fs::path in_path(in);
        std::string err;
        auto img = Image::load(in_path, &err);
        if (!img) {
            std::fprintf(stderr, "%s: %s\n", in.c_str(), err.c_str());
            ++failures;
            continue;
        }
        std::printf("%s  %dx%d %dch\n", in_path.filename().string().c_str(), img->w, img->h,
                    img->c);

        int last_pct = -1;
        auto progress = [&](const EncodeProgress& p) {
            const int pct = p.step * 100 / p.max_steps;
            if (pct != last_pct) {
                std::printf("\r  optimizing %3d%%  (%d gaussians, loss %.5f)", pct,
                            p.num_gaussians, double(p.loss));
                std::fflush(stdout);
                last_pct = pct;
            }
        };
        EncodeSettings s = settings;
        s.preview_interval_ms = 0;
        auto result = encode_image(*img, s, progress);
        std::printf("\r%*s\r", 60, "");

        fs::path out = out_dir.empty() ? in_path : out_dir / in_path.filename();
        out.replace_extension(".gsi");
        std::error_code ec;
        if (!out_dir.empty()) fs::create_directories(out_dir, ec);
        if (!write_file(out, result.file)) {
            std::fprintf(stderr, "  cannot write %s\n", out.string().c_str());
            ++failures;
            continue;
        }
        if (also_png) {
            fs::path png = out;
            png.replace_extension(".decoded.png");
            result.reconstruction.save_png(png);
        }
        const auto& st = result.stats;
        std::printf(
            "  -> %s  %s (%.1f%% of 8-bit raw)  PSNR %.2f dB  SSIM %.4f\n"
            "     %d gaussians, %d steps, %.2fs [%s]\n",
            out.filename().string().c_str(), human_size(st.file_bytes).c_str(),
            100.0 * double(st.file_bytes) / double(st.source_bytes), st.psnr, st.ssim,
            st.num_gaussians, st.steps_run, st.encode_seconds, st.backend_used);
    }
    return failures == 0 ? 0 : 1;
}

// Decoding defaults to the CPU, and that is a measured choice rather than a
// missing feature. The GPU renders the image faster, but a decode has to end
// with the pixels in system memory, and reading a 2K RGB frame back across
// PCIe costs more than the render saves: 30 ms on the GPU against 20 ms on
// the CPU for the same 2048x2048 image. The GPU path stays available via
// --gpu because it wins when the result is wanted on the GPU anyway, and it
// pulls ahead on very large images where the render dominates the transfer.
Image decode_render(const GaussianCloud& cloud, int w, int h, Backend backend,
                    const char** used) {
    if (backend == Backend::Gpu) {
        if (auto img = gpu_render(cloud, w, h)) {
            *used = "gpu";
            return std::move(*img);
        }
    }
    *used = "cpu";
    return Renderer::render_image(cloud, w, h);
}

int cmd_decompress(const std::vector<std::string>& inputs, fs::path out_arg, float scale,
                   Backend backend) {
    const bool out_is_dir = !out_arg.empty() && (inputs.size() > 1 || fs::is_directory(out_arg));
    int failures = 0;
    for (const auto& in : inputs) {
        const fs::path in_path(in);
        auto bytes = read_file(in_path);
        std::string err;
        auto file = decode_gsi(bytes, &err);
        if (!file) {
            std::fprintf(stderr, "%s: %s\n", in.c_str(), err.c_str());
            ++failures;
            continue;
        }
        // The scaled-decode arithmetic and its range checking live in the
        // core, so the window and this tool cannot disagree about what
        // "--scale 2" means or about which scales are refused.
        auto scaled = scale_gsi(*file, scale, &err);
        if (!scaled) {
            std::fprintf(stderr, "%s: %s\n", in.c_str(), err.c_str());
            ++failures;
            continue;
        }
        const int w = scaled->w, h = scaled->h;
        GaussianCloud cloud = std::move(scaled->cloud);
        const char* used = "cpu";
        const auto t0 = std::chrono::steady_clock::now();
        Image img = decode_render(cloud, w, h, backend, &used);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();

        fs::path out;
        if (out_arg.empty()) {
            out = in_path;
            out.replace_extension(".png");
        } else if (out_is_dir) {
            out = out_arg / in_path.filename().replace_extension(".png");
        } else {
            out = out_arg;
        }
        std::error_code ec;
        if (out.has_parent_path()) fs::create_directories(out.parent_path(), ec);
        if (!img.save_png(out)) {
            std::fprintf(stderr, "  cannot write %s\n", out.string().c_str());
            ++failures;
            continue;
        }
        std::printf("%s -> %s  %dx%d, decoded in %.1f ms [%s]\n",
                    in_path.filename().string().c_str(), out.string().c_str(), w, h, ms, used);
    }
    return failures == 0 ? 0 : 1;
}

int cmd_info(const std::string& input) {
    auto bytes = read_file(input);
    std::string err;
    auto file = decode_gsi(bytes, &err);
    if (!file) {
        std::fprintf(stderr, "%s: %s\n", input.c_str(), err.c_str());
        return 1;
    }
    const auto& q = file->quant;
    const std::int64_t raw = std::int64_t(file->width) * file->height * file->cloud.channels;
    std::printf("%s\n", fs::path(input).filename().string().c_str());
    std::printf("  resolution   %d x %d, %d channel(s)\n", file->width, file->height,
                file->cloud.channels);
    std::printf("  gaussians    %d\n", file->cloud.count());
    std::printf("  bits         pos %d, scale %d, rot %d, color %d\n", q.pos, q.scale, q.rot,
                q.feat);
    std::printf("  file size    %s (%.3f bpp, %.1f%% of 8-bit raw)\n",
                human_size(std::int64_t(bytes.size())).c_str(),
                8.0 * double(bytes.size()) / (double(file->width) * file->height),
                100.0 * double(bytes.size()) / double(raw));
    return 0;
}

int cmd_compare(const std::string& a_path, const std::string& b_path) {
    std::string err;
    auto a = Image::load(a_path, &err);
    if (!a) {
        std::fprintf(stderr, "%s: %s\n", a_path.c_str(), err.c_str());
        return 1;
    }
    auto b = Image::load(b_path, &err);
    if (!b) {
        std::fprintf(stderr, "%s: %s\n", b_path.c_str(), err.c_str());
        return 1;
    }
    if (a->w != b->w || a->h != b->h || a->c != b->c) {
        std::fprintf(stderr, "images differ in size or channel count\n");
        return 1;
    }
    std::printf("PSNR %.2f dB  SSIM %.4f\n", psnr(*a, *b), ssim(*a, *b));
    return 0;
}

// Everything about this machine that decides how the application behaves on
// it, in one place.
//
// This exists because a bug report that says "compression does not work"
// leaves nowhere to start. Whether the GPU was used, whether it was rejected
// and for what, which CPU kernels were selected, how many threads there are,
// and what an encode of a realistic image actually costs are all facts that
// vary per machine, and none of them are visible from the outside. Asking
// someone to run one command and paste the output turns an unreproducible
// report into a reproducible one.
int cmd_diagnose() {
    std::printf("gsic %s diagnostics\n", GSIC_VERSION);
    std::printf("  build          %d-bit, %s\n", int(sizeof(void*) * 8),
                kIs32Bit ? "32-bit limits" : "64-bit limits");
    std::printf("  cpu kernels    %s\n", kernels().name);
    std::printf("  hardware threads %u\n", std::thread::hardware_concurrency());
    std::printf("  image limits   %lld pixels, %d per side, %d gaussians\n",
                static_cast<long long>(kMaxPixels), kMaxDimension, kMaxGaussians);

    std::string why;
    const bool gpu = gpu_init(&why);
    if (gpu)
        std::puts("  gpu            available, and it agreed with the CPU on the self-check");
    else
        std::printf("  gpu            NOT USED: %s\n", why.c_str());

    // A realistic size, not a toy one: the per-step cost scales with pixels,
    // so a 64x64 timing says nothing about what a photo will do.
    struct Case { int w, h; const char* label; };
    const Case cases[] = {{1024, 1024, "1 MP"}, {2048, 2048, "4 MP"}};
    for (const Case& c : cases) {
        Image img(c.w, c.h, 3);
        std::uint32_t state = 1;
        for (int ch = 0; ch < 3; ++ch) {
            float* p = img.plane(ch);
            for (int y = 0; y < img.h; ++y)
                for (int x = 0; x < img.w; ++x) {
                    state = state * 1664525u + 1013904223u;
                    p[size_t(y) * img.w + x] =
                        0.5f + 0.25f * std::sin(0.02f * (x + y * (ch + 1))) +
                        0.1f * (float(state >> 8) / float(1 << 24) - 0.5f);
                }
        }
        for (int pass = 0; pass < (gpu ? 2 : 1); ++pass) {
            EncodeSettings s;
            s.pixels_per_gaussian = 300;
            s.max_steps = 300;   // short: this measures throughput, not quality
            s.preview_interval_ms = 0;
            s.backend = pass == 0 ? Backend::Cpu : Backend::Gpu;
            const auto t0 = std::chrono::steady_clock::now();
            const auto r = encode_image(img, s);
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!r.ok()) {
                std::printf("  %-5s %-4s FAILED: %s\n", c.label, pass == 0 ? "cpu" : "gpu",
                            r.error.c_str());
                continue;
            }
            const double per_step = r.stats.optimize_seconds / std::max(1, r.stats.steps_run);
            std::printf("  %-5s %-4s %6.1f steps/s -> a default 3000-step encode would take"
                        " about %.0f s  (%.2f dB here)\n",
                        c.label, r.stats.backend_used, 1.0 / std::max(1e-9, per_step),
                        per_step * 3000.0, r.stats.psnr);
            if (!r.stats.gpu_fallback_reason.empty())
                std::printf("        the GPU %s partway through, so the CPU finished it\n",
                            r.stats.gpu_fallback_reason.c_str());
            (void)secs;
        }
    }
    gpu_shutdown();
    std::puts("\nIf compression seems not to work, paste all of the above into the report.");
    return 0;
}

int cmd_bench(const std::string& input, EncodeSettings settings) {
    Image img;
    if (!input.empty()) {
        auto loaded = Image::load(input);
        if (!loaded) {
            std::fprintf(stderr, "cannot load %s\n", input.c_str());
            return 1;
        }
        img = std::move(*loaded);
    } else {
        // Synthetic 1024x1024 RGB gradient+noise target.
        img = Image(1024, 1024, 3);
        std::uint32_t state = 1;
        for (int c = 0; c < 3; ++c) {
            float* p = img.plane(c);
            for (int y = 0; y < img.h; ++y)
                for (int x = 0; x < img.w; ++x) {
                    state = state * 1664525u + 1013904223u;
                    p[size_t(y) * img.w + x] =
                        0.5f + 0.25f * std::sin(0.02f * (x + y * (c + 1))) +
                        0.1f * (float(state >> 8) / float(1 << 24) - 0.5f);
                }
        }
    }
    std::printf("bench: %dx%d, %d channels\n", img.w, img.h, img.c);
    settings.preview_interval_ms = 0;

    const auto t0 = std::chrono::steady_clock::now();
    auto result = encode_image(img, settings);
    const double total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto t1 = std::chrono::steady_clock::now();
    Image decoded = Renderer::render_image(result.cloud, img.w, img.h);
    const double decode_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

    std::printf("  backend       %s\n", result.stats.backend_used);
    std::printf("  optimize      %d steps in %.2fs -> %.1f steps/s\n", result.stats.steps_run,
                result.stats.optimize_seconds,
                result.stats.steps_run / result.stats.optimize_seconds);
    std::printf("  end-to-end    %.2fs (load, optimize, quantize, measure, write)\n", total);
    std::printf("  gaussians     %d\n", result.stats.num_gaussians);
    std::printf("  decode        %.2f ms\n", decode_ms);
    std::printf("  PSNR          %.2f dB\n", result.stats.psnr);
    std::printf("  file          %s\n", human_size(result.stats.file_bytes).c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }
    const std::string cmd = argv[1];
    std::vector<std::string> inputs;
    EncodeSettings settings;
    fs::path out_dir;
    bool also_png = false;
    float scale = 1.f;
    std::string preset = "balanced";
    apply_preset(settings, preset);

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        int iv = 0;
        if (a == "-o") out_dir = next();
        else if (a == "--preset") apply_preset(settings, next());
        else if (a == "-n" && parse_int(next(), iv)) settings.num_gaussians = iv;
        else if (a == "--steps" && parse_int(next(), iv)) settings.max_steps = iv;
        else if (a == "--bits" && parse_int(next(), iv))
            settings.quant = QuantSpec{iv, iv, iv, iv};
        else if (a == "--bits-pos" && parse_int(next(), iv)) settings.quant.pos = iv;
        else if (a == "--bits-scale" && parse_int(next(), iv)) settings.quant.scale = iv;
        else if (a == "--bits-rot" && parse_int(next(), iv)) settings.quant.rot = iv;
        else if (a == "--bits-color" && parse_int(next(), iv)) settings.quant.feat = iv;
        else if (a == "--seed" && parse_int(next(), iv)) settings.seed = std::uint32_t(iv);
        else if (a == "--max-scale") { if (!parse_float(next(), settings.max_scale_px)) { std::fprintf(stderr, "bad --max-scale\n"); return 2; } }
        else if (a == "--time-budget") {
            float secs = 0.f;
            if (!parse_float(next(), secs) || secs < 0.f) {
                std::fprintf(stderr, "bad --time-budget\n");
                return 2;
            }
            settings.time_budget_seconds = double(secs);
        }
        else if (a == "--no-lr-decay") settings.lr_decay = false;
        else if (a == "--cpu") settings.backend = Backend::Cpu;
        else if (a == "--gpu") settings.backend = Backend::Gpu;
        else if (a == "--png") also_png = true;
        else if (a == "--scale") { if (!parse_float(next(), scale)) { std::fprintf(stderr, "bad --scale\n"); return 2; } }
        else if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        } else {
            inputs.push_back(a);
        }
    }
    if (!settings.quant.valid()) {
        std::fprintf(stderr, "bits must be between 4 and 16\n");
        return 2;
    }

    // The GPU context is created here, on the main thread, or not at all: see
    // gpu.h. Every command that might reach the GPU asks for it first rather
    // than letting a deeper call site create it wherever it happens to run.
    const auto want_gpu = [&](bool required) {
        if (settings.backend == Backend::Cpu) return;
        std::string why;
        if (!gpu_init(&why) && required)
            std::fprintf(stderr, "GPU unavailable: %s (falling back to CPU)\n", why.c_str());
    };

    if (cmd == "compress") {
        if (inputs.empty()) { print_usage(); return 2; }
        want_gpu(settings.backend == Backend::Gpu);
        return cmd_compress(inputs, settings, out_dir, also_png);
    }
    if (cmd == "decompress") {
        if (inputs.empty()) { print_usage(); return 2; }
        want_gpu(settings.backend == Backend::Gpu);
        return cmd_decompress(inputs, out_dir, scale, settings.backend);
    }
    if (cmd == "info") {
        if (inputs.size() != 1) { print_usage(); return 2; }
        return cmd_info(inputs[0]);
    }
    if (cmd == "compare") {
        if (inputs.size() != 2) { print_usage(); return 2; }
        return cmd_compare(inputs[0], inputs[1]);
    }
    if (cmd == "diagnose") return cmd_diagnose();
    if (cmd == "bench") {
        want_gpu(settings.backend == Backend::Gpu);
        return cmd_bench(inputs.empty() ? "" : inputs[0], settings);
    }
    print_usage();
    return 2;
}
