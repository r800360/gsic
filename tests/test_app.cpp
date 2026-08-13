// End-to-end tests for the desktop application's compress pipeline.
//
// This covers the exact sequence a person performs: add an image, wait, get a
// compressed file. It is the feature Microsoft reported as unusable, and
// until now it was the one thing no suite touched -- the library underneath
// was tested, the window on top was tested, and everything in between was
// only ever checked by opening the app and looking at it.
//
// Nothing here needs a window or a GL context. That is the point of splitting
// the pipeline out: what a user actually does is now something a machine can
// do too, on every platform the project builds on.

#include "pipeline.h"
#include "settings.h"

#include "core/codec.h"
#include "core/gpu.h"
#include "core/image.h"
#include "core/metrics.h"
#include "core/renderer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace gsic;
using namespace std::chrono_literals;

static int failures = 0;

static void fail(const std::string& what) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
}

#define CHECK(cond)                                                                          \
    do {                                                                                     \
        if (!(cond))                                                                         \
            fail(std::string(__FILE__ ":") + std::to_string(__LINE__) + ": " #cond);          \
    } while (0)

namespace {

fs::path g_dir;

// Collects everything the pipeline reports, so tests can assert that a
// rejection was actually communicated instead of just not happening.
class LogCapture {
public:
    Pipeline::LogFn sink() {
        return [this](const std::string& line) {
            std::lock_guard lk(m_);
            lines_.push_back(line);
        };
    }
    bool mentions(const std::string& needle) const {
        std::lock_guard lk(m_);
        for (const auto& l : lines_)
            if (l.find(needle) != std::string::npos) return true;
        return false;
    }
    size_t size() const {
        std::lock_guard lk(m_);
        return lines_.size();
    }
    void dump() const {
        std::lock_guard lk(m_);
        for (const auto& l : lines_) std::printf("      | %s\n", l.c_str());
    }

private:
    mutable std::mutex m_;
    std::vector<std::string> lines_;
};

// Small enough that a full encode is a fraction of a second, structured
// enough that the result is clearly an image and not noise.
fs::path write_test_png(const fs::path& p, int w = 96, int h = 72) {
    Image img(w, h, 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            img.plane(0)[size_t(y) * w + x] = float(x) / float(w);
            img.plane(1)[size_t(y) * w + x] = float(y) / float(h);
            img.plane(2)[size_t(y) * w + x] = ((x / 12 + y / 12) % 2) ? 0.85f : 0.15f;
        }
    fs::create_directories(p.parent_path());
    if (!img.save_png(p)) fail("could not write the fixture image " + p.string());
    return p;
}

EncodeOptions fast_options() {
    EncodeOptions o;
    o.preset = 3;             // custom, so the tests are not at the mercy of
    o.pixels_per_gaussian = 100;   // whatever the shipped presets cost
    o.steps = 200;            // enough gaussians and steps that the result is
    o.backend = 1;            // recognisably the picture, on images this small
    o.save_next_to_input = true;   // CPU: the path every machine has
    return o;
}

// Compresses one image through the real pipeline and returns the .gsi it
// wrote, or an empty path with the failure already reported. Several tests
// below need a genuine compressed file to open, and producing it the way the
// application does is the only way to be sure the file under test is the file
// users actually have.
fs::path compress_to_gsi(const fs::path& src) {
    LogCapture logs;
    Pipeline p;
    p.set_options(fast_options());
    p.start(logs.sink());
    if (p.add_files({src}) != 1) {
        fail("could not queue " + src.string());
        p.stop();
        return {};
    }
    if (!p.wait_until_idle(120s)) {
        fail("compressing " + src.filename().string() + " never finished");
        p.stop();
        return {};
    }
    p.stop();
    auto job = p.job(0);
    if (!job || job->status != JobStatus::Done) {
        if (job) {
            std::lock_guard lk(job->m);
            fail("could not produce a .gsi to open: " + job->error);
        }
        return {};
    }
    std::lock_guard lk(job->m);
    return job->output_path;
}

// Runs a queue to completion and hands back the job at `index`, or null with
// the failure reported. Every test below that drives one entry does the same
// four things, and doing them in one place keeps the tests about what they are
// actually asserting.
std::shared_ptr<Job> run_one(Pipeline& p, LogCapture& logs, const char* what,
                             int index = 0) {
    if (!p.wait_until_idle(120s)) {
        fail(std::string(what) + ": the queue never drained");
        logs.dump();
        p.stop();
        return nullptr;
    }
    p.stop();
    auto job = p.job(index);
    if (!job) fail(std::string(what) + ": the entry disappeared from the queue");
    return job;
}

// --------------------------------------------------------- the happy path
//
// Add an image, get a .gsi. The whole report from Microsoft is that this did
// not happen.
void test_adding_an_image_produces_a_readable_file() {
    const fs::path dir = g_dir / "happy";
    const fs::path src = write_test_png(dir / "photo.png");

    LogCapture logs;
    Pipeline p;
    p.set_options(fast_options());
    p.start(logs.sink());
    CHECK(p.add_files({src}) == 1);
    if (!p.wait_until_idle(120s)) {
        fail("the queue never drained: the image was accepted and then nothing happened");
        p.stop();
        return;
    }
    p.stop();

    auto job = p.job(0);
    if (!job) { fail("the job disappeared from the queue"); return; }
    if (job->status != JobStatus::Done) {
        std::lock_guard lk(job->m);
        fail("compressing a plain PNG failed: " + job->error);
        return;
    }

    // The bar is "the result is recognisably this picture", expressed against
    // what a flat grey image of the same size scores. A fixed dB threshold
    // would only be a statement about this fixture at these settings and
    // would need retuning every time either changed; this does not, and it
    // still fails loudly if the pipeline ever writes a valid file full of
    // nothing.
    const double grey_psnr = [&] {
        const Image source = *Image::load(src);
        Image grey(source.w, source.h, source.c);
        double mean = 0;
        for (float v : source.data) mean += v;
        mean /= double(source.data.size());
        for (float& v : grey.data) v = float(mean);
        return psnr(grey, source);
    }();

    fs::path out;
    {
        std::lock_guard lk(job->m);
        out = job->output_path;
        CHECK(job->has_stats);
        if (job->stats.psnr < grey_psnr + 3.0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "the compressed result is barely better than flat grey: %.2f dB "
                          "against grey's %.2f dB",
                          job->stats.psnr, grey_psnr);
            fail(buf);
        }
        std::printf("  quality: %.2f dB (flat grey scores %.2f dB on the same image)\n",
                    job->stats.psnr, grey_psnr);
        CHECK(job->stats.file_bytes > 0);
        CHECK(job->orig_w == 96 && job->orig_h == 72);
        CHECK(job->recon_w == 96 && job->recon_h == 72);
        // The preview buffers must match the sizes they are labelled with;
        // the texture upload trusts them.
        CHECK(job->orig_rgba.size() == size_t(job->orig_w) * size_t(job->orig_h) * 4);
        CHECK(job->recon_rgba.size() == size_t(job->recon_w) * size_t(job->recon_h) * 4);
    }
    CHECK(out == dir / "photo.gsi");
    CHECK(fs::exists(out));
    if (!fs::exists(out)) return;

    // Not "a file exists" but "a file this application can read back".
    std::vector<std::uint8_t> bytes;
    {
        std::ifstream f(out, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::string err;
    auto decoded = decode_gsi(bytes, &err);
    if (!decoded) { fail("the written .gsi does not decode: " + err); return; }
    CHECK(decoded->width == 96 && decoded->height == 72);
    // Nothing partial must be left behind.
    CHECK(!fs::exists(fs::path(out) += ".partial"));
    std::printf("  add -> compress -> %s (%d bytes, decodes to %dx%d)\n",
                out.filename().string().c_str(), int(bytes.size()), decoded->width,
                decoded->height);
}

// Several images in one go, which is what a drag-and-drop of a folder's worth
// of pictures looks like.
void test_a_batch_compresses_every_image() {
    const fs::path dir = g_dir / "batch";
    std::vector<fs::path> srcs;
    for (int i = 0; i < 4; ++i)
        srcs.push_back(write_test_png(dir / ("img" + std::to_string(i) + ".png"), 64, 64));

    LogCapture logs;
    Pipeline p;
    p.set_options(fast_options());
    p.start(logs.sink());
    CHECK(p.add_files(srcs) == 4);
    if (!p.wait_until_idle(180s)) { fail("batch never drained"); p.stop(); return; }
    p.stop();

    int done = 0;
    for (const auto& j : p.jobs())
        if (j->status == JobStatus::Done) ++done;
    if (done != 4) { fail("batch finished " + std::to_string(done) + " of 4 images"); return; }
    for (const auto& s : srcs) {
        fs::path out = s;
        out.replace_extension(".gsi");
        CHECK(fs::exists(out));
    }
    std::puts("  batch of 4 -> 4 .gsi files");
}

// ------------------------------------------------------ opening a .gsi
//
// The application used to write files it could not open. A compressed image
// existed only while the window that had just made it stayed open: close the
// application, or clear the queue, and there was no way back to the picture.
// The file association the package declares made it worse rather than better,
// because double-clicking a .gsi in Explorer started gsic and was told the
// user's own file "could not be read as an image".
//
// This is that whole story as a test: compress with one pipeline, throw it
// away, and open the result with a fresh one, exactly as a second launch of
// the application would.
void test_a_gsi_can_be_opened_again_later() {
    const fs::path dir = g_dir / "reopen";
    const fs::path src = write_test_png(dir / "keepsake.png", 96, 72);
    const fs::path gsi = compress_to_gsi(src);
    if (gsi.empty()) return;
    CHECK(gsi.extension() == ".gsi");

    // What the intake decides, before anything is queued.
    CHECK(has_gsi_extension(gsi));
    CHECK(looks_like_gsi(gsi));
    CHECK(!looks_like_gsi(src));
    CHECK(!has_gsi_extension(src));
    // Renamed away from .gsi, it is still recognised: the four magic bytes are
    // the authority, and the extension only decides what Explorer draws.
    const fs::path renamed = dir / "keepsake.backup";
    fs::copy_file(gsi, renamed, fs::copy_options::overwrite_existing);
    CHECK(looks_like_gsi(renamed));
    CHECK(!has_gsi_extension(renamed));

    LogCapture logs;
    Pipeline p;
    p.set_options(fast_options());
    p.start(logs.sink());
    CHECK(p.add_files({gsi}) == 1);
    auto job = run_one(p, logs, "reopening a .gsi");
    if (!job) return;

    if (job->kind != JobKind::Decode) {
        fail("a .gsi was queued as something to compress rather than something to look at");
        return;
    }
    if (job->status != JobStatus::Done) {
        std::lock_guard lk(job->m);
        fail("opening a .gsi this application wrote failed: " + job->error);
        return;
    }
    std::lock_guard lk(job->m);
    CHECK(job->has_info);
    CHECK(job->info.width == 96 && job->info.height == 72);
    CHECK(job->info.channels == 3);
    CHECK(job->info.gaussians > 0);
    CHECK(job->info.file_bytes == std::int64_t(fs::file_size(gsi)));
    CHECK(job->info.bpp > 0.0);
    CHECK(job->info.quant.valid());
    // And a picture to actually show, at the size it claims.
    CHECK(job->recon_w == 96 && job->recon_h == 72);
    CHECK(job->recon_rgba.size() == size_t(job->recon_w) * size_t(job->recon_h) * 4);
    // Opening one on its own leaves the other half of the divider empty;
    // there is no original to compare against yet.
    CHECK(job->orig_rgba.empty());
    // Viewing writes nothing.
    CHECK(job->output_path.empty());
    std::printf("  reopened %s: %dx%d, %d gaussians, %.3f bpp\n",
                gsi.filename().string().c_str(), job->info.width, job->info.height,
                job->info.gaussians, job->info.bpp);
}

// The other half of the promise the format makes: because the representation
// is continuous rather than a grid of pixels, it can be decoded at a size it
// was never stored at.
void test_a_gsi_decodes_at_another_size() {
    const fs::path dir = g_dir / "rescale";
    const fs::path src = write_test_png(dir / "small.png", 64, 48);
    const fs::path gsi = compress_to_gsi(src);
    if (gsi.empty()) return;

    LogCapture logs;
    Pipeline p;
    p.start(logs.sink());
    DecodeRequest twice;
    twice.input = gsi;
    twice.scale = 2.f;
    CHECK(p.add_decode(twice) != nullptr);
    auto job = run_one(p, logs, "decoding at 2x");
    if (!job) return;
    if (job->status != JobStatus::Done) {
        std::lock_guard lk(job->m);
        fail("a 2x decode failed: " + job->error);
        return;
    }
    {
        std::lock_guard lk(job->m);
        CHECK(job->recon_w == 128 && job->recon_h == 96);
        CHECK(job->recon_rgba.size() == size_t(128) * 96 * 4);
        // The file still describes itself at its stored size, whatever the
        // view is showing.
        CHECK(job->info.width == 64 && job->info.height == 48);
    }

    // A scale nobody meant to type must come back as a message, not as an
    // allocation the size of a hard disk.
    LogCapture logs2;
    Pipeline p2;
    p2.start(logs2.sink());
    DecodeRequest absurd;
    absurd.input = gsi;
    absurd.scale = 5000.f;
    CHECK(p2.add_decode(absurd) != nullptr);
    auto refused = run_one(p2, logs2, "decoding at an impossible scale");
    if (!refused) return;
    CHECK(refused->status == JobStatus::Failed);
    {
        std::lock_guard lk(refused->m);
        CHECK(!refused->error.empty());
        std::printf("  a 5000x decode -> refused, \"%s\"\n", refused->error.c_str());
    }
    std::puts("  a .gsi decodes at 2x, and an impossible scale is refused");
}

// Exporting is how a .gsi reaches every other program on the machine. Without
// it the format is a dead end from the window, whatever the command line tool
// can do.
void test_exporting_a_png_writes_a_readable_image() {
    const fs::path dir = g_dir / "export";
    const fs::path src = write_test_png(dir / "shot.png", 64, 64);
    const fs::path gsi = compress_to_gsi(src);
    if (gsi.empty()) return;

    struct Case { float scale; int w, h; const char* name; };
    const Case cases[] = {{1.f, 64, 64, "same-size.png"}, {2.f, 128, 128, "double.png"}};
    for (const Case& c : cases) {
        const fs::path out = dir / c.name;
        LogCapture logs;
        Pipeline p;
        p.start(logs.sink());
        DecodeRequest request;
        request.input = gsi;
        request.scale = c.scale;
        request.export_png = out;
        CHECK(p.add_decode(request) != nullptr);
        auto job = run_one(p, logs, "exporting a png");
        if (!job) return;
        if (job->status != JobStatus::Done) {
            std::lock_guard lk(job->m);
            fail(std::string("export at ") + std::to_string(c.scale) + "x failed: " + job->error);
            continue;
        }
        if (!fs::exists(out)) {
            fail(std::string("export at ") + std::to_string(c.scale) + "x wrote no file");
            continue;
        }
        // Nothing partial left where a finished file should be.
        CHECK(!fs::exists(fs::path(out) += ".partial"));
        auto decoded = Image::load(out);
        if (!decoded) {
            fail("the exported png cannot be read back");
            continue;
        }
        CHECK(decoded->w == c.w && decoded->h == c.h);
        {
            std::lock_guard lk(job->m);
            CHECK(job->output_path == out);
        }
        std::printf("  export at %.0fx -> %s (%dx%d)\n", double(c.scale), c.name, decoded->w,
                    decoded->h);
    }
}

// A .gsi is the one file this application invites people to keep, so it is
// also the one they will eventually hand it a damaged copy of. Every way that
// can go wrong has to end in a sentence, not a crash and not a blank window.
void test_a_damaged_gsi_fails_with_a_reason() {
    const fs::path dir = g_dir / "damaged";
    const fs::path src = write_test_png(dir / "ok.png", 64, 64);
    const fs::path gsi = compress_to_gsi(src);
    if (gsi.empty()) return;

    std::vector<char> good;
    {
        std::ifstream f(gsi, std::ios::binary);
        good.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    const auto write_bytes = [](const fs::path& p, const std::vector<char>& bytes) {
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        f.write(bytes.data(), std::streamsize(bytes.size()));
    };

    // Truncated halfway, every byte after the header flipped, empty, and an
    // ordinary PNG wearing the extension.
    const fs::path truncated = dir / "truncated.gsi";
    write_bytes(truncated, std::vector<char>(good.begin(), good.begin() + good.size() / 2));
    const fs::path corrupt = dir / "corrupt.gsi";
    {
        std::vector<char> bytes = good;
        for (size_t i = 40; i < bytes.size(); i += 13) bytes[i] = char(~bytes[i]);
        write_bytes(corrupt, bytes);
    }
    const fs::path empty = dir / "empty.gsi";
    write_bytes(empty, {});
    const fs::path impostor = dir / "impostor.gsi";
    fs::copy_file(src, impostor, fs::copy_options::overwrite_existing);

    for (const fs::path& bad : {truncated, corrupt, empty, impostor}) {
        LogCapture logs;
        Pipeline p;
        p.set_options(fast_options());
        p.start(logs.sink());
        CHECK(p.add_files({bad}) == 1);
        auto job = run_one(p, logs, bad.filename().string().c_str());
        if (!job) return;
        if (job->status != JobStatus::Failed) {
            fail(bad.filename().string() + " was not refused (status " +
                 std::to_string(int(job->status.load())) + ")");
            continue;
        }
        std::lock_guard lk(job->m);
        if (job->error.empty()) {
            fail(bad.filename().string() + " was refused without saying why");
            continue;
        }
        std::printf("  %-16s -> \"%s\"\n", bad.filename().string().c_str(), job->error.c_str());
    }
}

// Attaching the original to a decoded file: the measurement `gsic compare`
// prints, reached from the window, and what fills the other side of the A/B
// divider.
void test_a_decoded_file_can_be_measured_against_its_original() {
    const fs::path dir = g_dir / "measure";
    const fs::path src = write_test_png(dir / "source.png", 96, 72);
    const fs::path gsi = compress_to_gsi(src);
    if (gsi.empty()) return;

    LogCapture logs;
    Pipeline p;
    p.start(logs.sink());
    DecodeRequest request;
    request.input = gsi;
    request.reference = src;
    CHECK(p.add_decode(request) != nullptr);
    auto job = run_one(p, logs, "measuring a decode against its original");
    if (!job) return;
    if (job->status != JobStatus::Done) {
        std::lock_guard lk(job->m);
        fail("decoding with an original attached failed: " + job->error);
        return;
    }

    // Against flat grey, for the same reason the happy path uses it: a fixed
    // dB threshold would only be a statement about this fixture.
    const double grey_psnr = [&] {
        const Image source = *Image::load(src);
        Image grey(source.w, source.h, source.c);
        double mean = 0;
        for (float v : source.data) mean += v;
        mean /= double(source.data.size());
        for (float& v : grey.data) v = float(mean);
        return psnr(grey, source);
    }();

    std::lock_guard lk(job->m);
    CHECK(job->has_stats);
    CHECK(job->stats.psnr > grey_psnr + 3.0);
    CHECK(job->stats.ssim > 0.0);
    // Both sides are loaded and the same size, which is what the divider
    // requires; a mismatch here is a texture read past the end of a buffer.
    CHECK(job->orig_w == job->recon_w && job->orig_h == job->recon_h);
    CHECK(job->orig_rgba.size() == size_t(job->orig_w) * size_t(job->orig_h) * 4);
    CHECK(job->recon_rgba.size() == size_t(job->recon_w) * size_t(job->recon_h) * 4);
    std::printf("  decoded file measured at %.2f dB against its original (grey: %.2f dB)\n",
                job->stats.psnr, grey_psnr);
}

// Two images side by side with the numbers between them: everything `gsic
// compare` does, plus the part a terminal cannot do.
void test_comparing_two_images_reports_the_numbers() {
    const fs::path dir = g_dir / "compare";
    const fs::path a = write_test_png(dir / "a.png", 64, 64);
    const fs::path b = dir / "b.png";
    fs::copy_file(a, b, fs::copy_options::overwrite_existing);

    {
        LogCapture logs;
        Pipeline p;
        p.start(logs.sink());
        CHECK(p.add_compare(a, b) != nullptr);
        auto job = run_one(p, logs, "comparing two identical images");
        if (!job) return;
        if (job->status != JobStatus::Done) {
            std::lock_guard lk(job->m);
            fail("comparing two images failed: " + job->error);
            return;
        }
        std::lock_guard lk(job->m);
        CHECK(job->has_stats);
        // Identical inputs: the error is zero, so PSNR is capped rather than
        // infinite and SSIM is 1.
        CHECK(job->stats.psnr > 90.0);
        CHECK(job->stats.ssim > 0.999);
        CHECK(job->orig_rgba.size() == size_t(64) * 64 * 4);
        CHECK(job->recon_rgba.size() == size_t(64) * 64 * 4);
        std::printf("  identical images -> %.1f dB, SSIM %.4f\n", job->stats.psnr,
                    job->stats.ssim);
    }
    // Different sizes cannot be compared, and saying so beats a number that
    // means nothing.
    {
        const fs::path other = write_test_png(dir / "bigger.png", 96, 96);
        LogCapture logs;
        Pipeline p;
        p.start(logs.sink());
        CHECK(p.add_compare(a, other) != nullptr);
        auto job = run_one(p, logs, "comparing images of different sizes");
        if (!job) return;
        CHECK(job->status == JobStatus::Failed);
        std::lock_guard lk(job->m);
        CHECK(!job->error.empty());
        std::printf("  64x64 against 96x96 -> \"%s\"\n", job->error.c_str());
    }
    // A file that is not there at all is refused before anything is queued.
    {
        LogCapture logs;
        Pipeline p;
        p.start(logs.sink());
        CHECK(p.add_compare(a, dir / "missing.png") == nullptr);
        CHECK(logs.mentions("no such file"));
        CHECK(p.job_count() == 0);
        p.stop();
    }
}

// ------------------------------------------------------- the silent drops
//
// A file that is added and then vanishes without a word is, from where the
// user sits, identical to an application that does not work. Whatever the
// pipeline refuses, it has to say so.
void test_nothing_is_refused_silently() {
    const fs::path dir = g_dir / "refused";
    fs::create_directories(dir / "a-folder");
    // A text file with an image extension, and an image with no extension at
    // all: the first must fail with a reason, the second must still work,
    // because the reader identifies images by content.
    {
        std::ofstream f(dir / "not-an-image.png");
        f << "this is not a PNG";
    }
    write_test_png(dir / "extensionless_source.png", 48, 48);
    fs::rename(dir / "extensionless_source.png", dir / "no-extension");

    LogCapture logs;
    Pipeline p;
    p.set_options(fast_options());
    p.start(logs.sink());

    // A directory and a missing path are refused, but out loud.
    CHECK(p.add_files({dir / "a-folder"}) == 0);
    CHECK(logs.mentions("folder"));
    CHECK(p.add_files({dir / "definitely-missing.png"}) == 0);
    CHECK(logs.mentions("no such file"));

    // The two real files are both accepted for an attempt.
    CHECK(p.add_files({dir / "not-an-image.png", dir / "no-extension"}) == 2);
    if (!p.wait_until_idle(120s)) { fail("queue never drained"); p.stop(); return; }
    p.stop();

    auto jobs = p.jobs();
    if (jobs.size() != 2) { fail("expected two queued jobs"); return; }

    // The impostor fails with an explanation attached to a visible entry.
    CHECK(jobs[0]->status == JobStatus::Failed);
    {
        std::lock_guard lk(jobs[0]->m);
        CHECK(!jobs[0]->error.empty());
        std::printf("  a .png that is not a PNG -> failed, \"%s\"\n", jobs[0]->error.c_str());
    }
    // The extensionless image compresses, which the old suffix filter would
    // have made impossible without ever saying why.
    if (jobs[1]->status != JobStatus::Done) {
        std::lock_guard lk(jobs[1]->m);
        fail("an image without a file extension was not compressed: " + jobs[1]->error);
        return;
    }
    std::puts("  an image with no extension -> compressed anyway");
}

// ------------------------------------------------------- the queue itself
//
// add_files used to notify the worker's condition variable without holding
// the mutex that guards its predicate. A notify that lands between the
// worker's predicate check and the worker blocking is lost, and the job then
// sits in the queue forever while the window stays perfectly responsive -- an
// application that accepted your image and will not compress it.
//
// Be clear about what this test does and does not establish. The race window
// is a few instructions wide, so no amount of repetition reliably reproduces
// it; the fix is by construction, from the rule that the state a predicate
// reads must be published under the mutex the waiter holds. What the loop
// does catch, and did catch when checked against a deliberately broken
// version, is the whole family of gross regressions around it: a notify
// dropped, moved out of scope, or sent before the work is visible.
//
// It also covers the one timing that used to be genuinely unlucky rather than
// merely narrow -- a file arriving in the moment just after the worker
// starts -- which is what happens every time the app is launched by opening
// an image with it.
void test_queued_work_is_never_left_asleep() {
    const fs::path dir = g_dir / "race";
    const fs::path src = write_test_png(dir / "tiny.png", 32, 32);

    EncodeOptions o = fast_options();
    o.steps = 100;

    constexpr int kRounds = 60;
    for (int round = 0; round < kRounds; ++round) {
        LogCapture logs;
        Pipeline p;
        p.set_options(o);
        p.start(logs.sink());

        // Hand the file over at the moment the worker is settling down to
        // sleep, which is the interleaving that used to lose the wakeup.
        std::this_thread::sleep_for(std::chrono::microseconds(round * 17 % 400));
        CHECK(p.add_files({src}) == 1);

        if (!p.wait_until_idle(30s)) {
            fail("round " + std::to_string(round) +
                 ": the worker never woke up for a queued job");
            p.stop();
            return;
        }
        auto job = p.job(0);
        if (!job || job->status != JobStatus::Done) {
            fail("round " + std::to_string(round) + ": queued job did not complete");
            p.stop();
            return;
        }
        p.stop();
    }
    std::printf("  %d start-and-immediately-enqueue rounds, none stalled\n", kRounds);
}

// Adding while an encode is already running, repeatedly: the second job must
// start on its own once the first finishes.
void test_work_added_during_an_encode_still_runs() {
    const fs::path dir = g_dir / "concurrent";
    const fs::path a = write_test_png(dir / "a.png", 80, 80);
    const fs::path b = write_test_png(dir / "b.png", 80, 80);

    LogCapture logs;
    Pipeline p;
    p.set_options(fast_options());
    p.start(logs.sink());
    CHECK(p.add_files({a}) == 1);
    // Slip the second one in while the first is mid-flight.
    std::this_thread::sleep_for(15ms);
    CHECK(p.add_files({b}) == 1);
    if (!p.wait_until_idle(120s)) { fail("second job never ran"); p.stop(); return; }
    p.stop();

    for (const auto& j : p.jobs())
        if (j->status != JobStatus::Done) {
            std::lock_guard lk(j->m);
            fail("a job added during an encode did not finish: " + j->error);
            return;
        }
    std::puts("  a job added mid-encode still ran to completion");
}

// Cancelling must stop the work and leave nothing half-written.
void test_cancel_stops_cleanly() {
    const fs::path dir = g_dir / "cancel";
    const fs::path src = write_test_png(dir / "big.png", 192, 192);

    EncodeOptions o = fast_options();
    o.steps = 15000;   // long enough that cancelling lands mid-encode

    LogCapture logs;
    Pipeline p;
    p.set_options(o);
    p.start(logs.sink());
    p.add_files({src});
    std::this_thread::sleep_for(120ms);
    p.cancel_all();
    if (!p.wait_until_idle(60s)) { fail("cancel did not stop the encode"); p.stop(); return; }
    p.stop();

    auto job = p.job(0);
    if (!job) { fail("job vanished"); return; }
    CHECK(job->status == JobStatus::Cancelled);
    // A cancelled encode leaves no output at all, and above all no partial
    // file pretending to be one.
    CHECK(!fs::exists(dir / "big.gsi"));
    CHECK(!fs::exists(dir / "big.gsi.partial"));
    std::puts("  cancel -> stopped, no file and no leftover partial");
}

// ------------------------------------------------------- where files land
//
// An encode that succeeded and then could not be saved looks exactly like an
// encode that failed. plan_output is what stands between the two, and it is
// pure path logic, so it can be tested against a simulated file system that
// refuses the obvious destination.
void test_output_falls_back_when_the_first_choice_is_unwritable() {
    const fs::path input = g_dir / "readonly-source" / "picture.png";
    EncodeOptions o;
    o.save_next_to_input = true;

    // Everything writable: straight into the image's own folder.
    {
        const auto plan = plan_output(input, o, [](const fs::path&) { return true; });
        CHECK(plan.path == input.parent_path() / "picture.gsi");
        CHECK(plan.note.empty());
    }
    // The image's folder refuses writes, the way a read-only volume or a
    // folder a packaged application cannot reach would.
    {
        const auto plan = plan_output(input, o, [&](const fs::path& d) {
            return d != input.parent_path();
        });
        if (plan.path.empty()) { fail("gave up instead of finding somewhere writable"); return; }
        CHECK(plan.path.parent_path() != input.parent_path());
        CHECK(plan.path.filename() == "picture.gsi");
        CHECK(!plan.note.empty());          // and it says where it went
        CHECK(plan.fallback_from == input.parent_path());
        std::printf("  unwritable source folder -> %s\n", plan.path.string().c_str());
    }
    // A chosen output directory that does not exist yet is still the first
    // choice, and an empty one means "next to the input" rather than the
    // process's working directory, which under a packaged app is not a place
    // anything can be written.
    {
        EncodeOptions e = o;
        e.save_next_to_input = false;
        e.out_dir = "";
        const auto plan = plan_output(input, e, [](const fs::path&) { return true; });
        CHECK(plan.path == input.parent_path() / "picture.gsi");
    }
    {
        EncodeOptions e = o;
        e.save_next_to_input = false;
        e.out_dir = (g_dir / "chosen").string();
        const auto plan = plan_output(input, e, [](const fs::path&) { return true; });
        CHECK(plan.path == g_dir / "chosen" / "picture.gsi");
    }
    // Nowhere at all is writable: report it rather than inventing a path.
    {
        const auto plan = plan_output(input, o, [](const fs::path&) { return false; });
        CHECK(plan.path.empty());
    }
}

// The whole chain, through the real file system, with the destination
// genuinely not being the source folder.
void test_a_chosen_output_directory_is_used() {
    const fs::path dir = g_dir / "outdir";
    const fs::path src = write_test_png(dir / "in" / "shot.png", 64, 64);
    const fs::path dest = dir / "out";

    EncodeOptions o = fast_options();
    o.save_next_to_input = false;
    o.out_dir = dest.string();

    LogCapture logs;
    Pipeline p;
    p.set_options(o);
    p.start(logs.sink());
    p.add_files({src});
    if (!p.wait_until_idle(120s)) { fail("queue never drained"); p.stop(); return; }
    p.stop();

    auto job = p.job(0);
    if (!job || job->status != JobStatus::Done) {
        if (job) { std::lock_guard lk(job->m); fail("encode failed: " + job->error); }
        return;
    }
    CHECK(fs::exists(dest / "shot.gsi"));
    CHECK(!fs::exists(dir / "in" / "shot.gsi"));
    std::puts("  a chosen output folder is created and used");
}

// ----------------------------------------------------------- the settings
//
// Every preset the interface offers has to produce a usable encode. A preset
// that maps to zero gaussians or a run that never ends is a broken feature
// even though every other test passes.
void test_every_preset_and_precision_encodes() {
    const fs::path dir = g_dir / "presets";
    const fs::path src = write_test_png(dir / "p.png", 64, 64);

    for (int preset = 0; preset < 4; ++preset) {
        for (int precision = 0; precision < 3; ++precision) {
            EncodeOptions o;
            o.preset = preset;
            o.precision = precision;
            o.backend = 1;
            o.pixels_per_gaussian = 400;
            o.steps = 120;
            const EncodeSettings s = o.to_encode_settings();
            CHECK(s.max_steps > 0);
            CHECK(s.pixels_per_gaussian > 0);
            CHECK(s.quant.valid());
        }
    }
    // And one preset actually run end to end, to prove the mapping is not
    // just internally consistent.
    EncodeOptions o = fast_options();
    o.preset = 0;   // Fast
    LogCapture logs;
    Pipeline p;
    p.set_options(o);
    p.start(logs.sink());
    p.add_files({src});
    if (!p.wait_until_idle(180s)) { fail("Fast preset never finished"); p.stop(); return; }
    p.stop();
    auto job = p.job(0);
    if (!job || job->status != JobStatus::Done) {
        if (job) { std::lock_guard lk(job->m); fail("Fast preset failed: " + job->error); }
        return;
    }
    std::puts("  all 12 preset/precision combinations are valid; Fast runs end to end");
}

// The desktop app carries a time budget where the library and the command
// line tool do not, because someone watching a window is waiting and someone
// scripting a batch is not. Getting that default wrong in either direction is
// a real failure: too high and a large image looks like a hung application,
// absent and the app inherits the library's unbounded behaviour.
void test_the_app_defaults_to_a_bounded_wait() {
    const EncodeOptions defaults;
    CHECK(defaults.time_budget_seconds > 0.0);
    CHECK(defaults.time_budget_seconds == kDefaultTimeBudgetSeconds);
    CHECK(defaults.to_encode_settings().time_budget_seconds == defaults.time_budget_seconds);

    // Both bounds are policy, and both came down after measuring where the
    // quality curve flattens. Asserting them keeps a future edit to the slider
    // from quietly restoring a default wait nobody wants to sit through: half
    // a minute is already past the point where the remaining decibels arrive
    // slowly, and offering ten minutes was advertising a wait that should not
    // be encouraged.
    CHECK(kDefaultTimeBudgetSeconds <= 30.0);
    CHECK(kMaxTimeBudgetSeconds <= 240);
    CHECK(double(kMaxTimeBudgetSeconds) >= kDefaultTimeBudgetSeconds);
    // The slider's whole range has to be expressible, or the interface would
    // show a value the settings file then refuses to keep.
    AppSettings at_the_top;
    at_the_top.encode.time_budget_seconds = double(kMaxTimeBudgetSeconds);
    clamp_settings(at_the_top);
    CHECK(at_the_top.encode.time_budget_seconds == double(kMaxTimeBudgetSeconds));

    // And zero must pass straight through as "no limit" rather than being
    // clamped to something small.
    EncodeOptions unlimited = defaults;
    unlimited.time_budget_seconds = 0.0;
    CHECK(unlimited.to_encode_settings().time_budget_seconds == 0.0);

    // A negative value cannot come from the slider, but must not become a
    // budget so small that every encode stops at the floor.
    EncodeOptions negative = defaults;
    negative.time_budget_seconds = -5.0;
    CHECK(negative.to_encode_settings().time_budget_seconds == 0.0);
    std::printf("  app default time budget: %.0f s\n", defaults.time_budget_seconds);
}

// Values that could only arrive from outside the sliders must not produce a
// nonsensical encode.
void test_out_of_range_settings_are_clamped() {
    EncodeOptions o;
    o.preset = 3;
    o.pixels_per_gaussian = 0;
    o.steps = -5;
    EncodeSettings s = o.to_encode_settings();
    CHECK(s.pixels_per_gaussian >= 1);
    CHECK(s.max_steps >= 1);

    o.pixels_per_gaussian = 1 << 30;
    o.steps = 1 << 30;
    s = o.to_encode_settings();
    CHECK(s.pixels_per_gaussian <= 1200);
    CHECK(s.max_steps <= 20000);

    o.precision = 99;
    CHECK(o.to_encode_settings().quant.valid());
    o.precision = -3;
    CHECK(o.to_encode_settings().quant.valid());
}

// ------------------------------------------------- managing the queue
//
// Removing entries hands them back rather than dropping them, because the
// interface owns a GL texture on each one and has to delete it on the thread
// that holds the context. Before this the textures were never deleted at all:
// clearing a long batch left every one of them resident for the rest of the
// session.
void test_finished_entries_are_handed_back_when_removed() {
    const fs::path dir = g_dir / "removal";
    std::vector<fs::path> srcs;
    for (int i = 0; i < 3; ++i)
        srcs.push_back(write_test_png(dir / ("r" + std::to_string(i) + ".png"), 48, 48));

    LogCapture logs;
    Pipeline p;
    p.set_options(fast_options());
    p.start(logs.sink());
    CHECK(p.add_files(srcs) == 3);
    if (!p.wait_until_idle(120s)) { fail("queue never drained"); p.stop(); return; }

    // Standing in for the interface: mark each entry the way a texture upload
    // would, then check that removal is what returns it.
    for (const auto& j : p.jobs()) j->tex_recon = 1234;

    auto first = p.job(0);
    auto removed = p.remove_job(first);
    CHECK(removed == first);
    CHECK(removed && removed->tex_recon == 1234);
    CHECK(p.job_count() == 2);
    CHECK(p.index_of(first) == -1);

    const auto cleared = p.clear_finished();
    CHECK(cleared.size() == 2);
    CHECK(p.job_count() == 0);
    for (const auto& j : cleared) CHECK(j->tex_recon == 1234);
    p.stop();
    std::puts("  removed and cleared entries come back to the caller, textures and all");
}

// Removing something that has not started yet takes it out of the queue and
// stops it from ever running. Leaving it to run and be discarded afterwards is
// a strange thing to watch happen after pressing Remove.
void test_removing_a_queued_entry_stops_it_running() {
    const fs::path dir = g_dir / "remove-queued";
    const fs::path slow = write_test_png(dir / "slow.png", 160, 160);
    const fs::path waiting = write_test_png(dir / "waiting.png", 64, 64);

    EncodeOptions o = fast_options();
    o.steps = 6000;   // long enough that the second entry is still queued

    LogCapture logs;
    Pipeline p;
    p.set_options(o);
    p.start(logs.sink());
    CHECK(p.add_files({slow}) == 1);
    std::this_thread::sleep_for(80ms);
    CHECK(p.add_files({waiting}) == 1);

    auto second = p.job(1);
    if (!second || second->status != JobStatus::Queued) {
        // The first encode finished sooner than expected; nothing to assert.
        p.cancel_all();
        p.wait_until_idle(60s);
        p.stop();
        std::puts("  (skipped: the first encode finished before the second could be removed)");
        return;
    }
    CHECK(p.remove_job(second) == second);
    CHECK(p.job_count() == 1);
    p.cancel_all();
    if (!p.wait_until_idle(60s)) { fail("queue never drained"); p.stop(); return; }
    p.stop();
    CHECK(second->status == JobStatus::Queued || second->status == JobStatus::Cancelled);
    CHECK(!fs::exists(dir / "waiting.gsi"));
    std::puts("  an entry removed while queued never runs");
}

// Running something a second time, which is Retry for a failure and is how an
// original gets attached to an already-decoded file without leaving a
// duplicate entry behind.
void test_an_entry_can_be_run_again() {
    const fs::path dir = g_dir / "again";
    const fs::path src = write_test_png(dir / "twice.png", 64, 64);
    const fs::path gsi = compress_to_gsi(src);
    if (gsi.empty()) return;

    LogCapture logs;
    Pipeline p;
    p.start(logs.sink());
    DecodeRequest request;
    request.input = gsi;
    auto job = p.add_decode(request);
    CHECK(job != nullptr);
    if (!job) { p.stop(); return; }
    if (!p.wait_until_idle(60s)) { fail("first decode never finished"); p.stop(); return; }
    CHECK(job->status == JobStatus::Done);
    {
        std::lock_guard lk(job->m);
        CHECK(!job->has_stats);   // nothing to measure against yet
    }

    // Attach the original and run it again: same entry, now with numbers.
    job->reference = src;
    CHECK(p.requeue(job));
    if (!p.wait_until_idle(60s)) { fail("the repeat never finished"); p.stop(); return; }
    CHECK(p.job_count() == 1);   // and no second copy of it
    CHECK(job->status == JobStatus::Done);
    double measured = 0;
    {
        std::lock_guard lk(job->m);
        CHECK(job->has_stats);
        CHECK(job->stats.psnr > 0.0);
        CHECK(!job->orig_rgba.empty());
        measured = job->stats.psnr;
    }

    // Run it once more against an original that cannot be read. The previous
    // one has to go: keeping it would label one picture as another, and the
    // divider would be comparing the file against something it is not.
    job->reference = dir / "not-here.png";
    CHECK(p.requeue(job));
    if (!p.wait_until_idle(60s)) { fail("the third run never finished"); p.stop(); return; }
    p.stop();
    CHECK(job->status == JobStatus::Done);
    {
        std::lock_guard lk(job->m);
        CHECK(job->orig_rgba.empty());
        CHECK(job->orig_w == 0 && job->orig_h == 0);
        CHECK(!job->has_stats);   // and no stale measurement either
        CHECK(!job->recon_rgba.empty());
    }
    std::printf("  the same entry re-ran with an original attached: %.2f dB, and again "
                "without one\n", measured);
}

// The recovery a user needs most often, and the one that is easiest to leave
// out: the time limit was too small, or the preset was the wrong one, and they
// want to change it and try the same image again.
//
// Two things have to hold. The settings a re-run uses must be the ones showing
// now, not the ones the first run captured -- otherwise the button appears to
// do nothing. And a run cut short by the clock has to be distinguishable from
// one that simply finished, or the interface cannot know to offer the fix.
void test_changing_the_settings_and_running_again_uses_the_new_ones() {
    const fs::path dir = g_dir / "second-thoughts";
    const fs::path src = write_test_png(dir / "regret.png", 96, 96);

    // A limit no machine can meet, so the run stops at the floor rather than
    // at the end of its schedule. This is the situation exactly: a result the
    // user is unhappy with, produced by a setting rather than by the image.
    EncodeOptions rushed = fast_options();
    rushed.steps = 4000;
    rushed.time_budget_seconds = 0.001;

    LogCapture logs;
    Pipeline p;
    p.set_options(rushed);
    p.start(logs.sink());
    CHECK(p.add_files({src}) == 1);
    if (!p.wait_until_idle(120s)) { fail("the rushed encode never finished"); p.stop(); return; }
    auto job = p.job(0);
    if (!job || job->status != JobStatus::Done) {
        if (job) { std::lock_guard lk(job->m); fail("the rushed encode failed: " + job->error); }
        p.stop();
        return;
    }

    int rushed_steps = 0;
    double rushed_psnr = 0;
    {
        std::lock_guard lk(job->m);
        // The interface can tell this run was cut short by the clock, which is
        // what puts the offer of a longer limit on screen.
        if (!job->was_cut_short()) {
            fail("a run stopped by the time limit did not report itself as cut short");
            p.stop();
            return;
        }
        CHECK(job->stats.steps_run < job->stats.steps_requested);
        CHECK(job->used_time_budget == rushed.time_budget_seconds);
        rushed_steps = job->stats.steps_run;
        rushed_psnr = job->stats.psnr;
    }
    CHECK(logs.mentions("time limit"));

    // Now the fix: raise the limit, run the same entry again. Nothing is added
    // to the queue and nothing is re-picked from disk by the user.
    EncodeOptions patient = rushed;
    patient.time_budget_seconds = 0.0;   // no limit, the top of what the offer escalates to
    p.set_options(patient);
    CHECK(p.requeue(job));
    if (!p.wait_until_idle(180s)) { fail("the second attempt never finished"); p.stop(); return; }
    p.stop();

    CHECK(p.job_count() == 1);   // the same entry, not a second copy of it
    if (job->status != JobStatus::Done) {
        std::lock_guard lk(job->m);
        fail("the second attempt failed: " + job->error);
        return;
    }
    std::lock_guard lk(job->m);
    // The new settings reached the encoder: the run went the full distance
    // this time, and there is no longer anything to offer a fix for.
    CHECK(job->stats.steps_run == job->stats.steps_requested);
    CHECK(job->stats.steps_run > rushed_steps);
    CHECK(job->used_time_budget == 0.0);
    CHECK(!job->was_cut_short());
    // And more optimization actually bought something, which is the only
    // reason any of this is worth offering.
    CHECK(job->stats.psnr > rushed_psnr);
    // The file on disk is the new one.
    CHECK(fs::exists(dir / "regret.gsi"));
    std::printf("  cut short at %d steps (%.2f dB) -> run again unlimited: %d steps (%.2f dB)\n",
                rushed_steps, rushed_psnr, job->stats.steps_run, job->stats.psnr);
}

// Changing the preset and running everything again, which is what a folder
// compressed at the wrong setting needs. The same entries have to be reused;
// adding the files a second time would leave two of each in the queue.
void test_a_whole_batch_can_be_run_again_at_a_new_preset() {
    const fs::path dir = g_dir / "wrong-preset";
    std::vector<fs::path> srcs;
    for (int i = 0; i < 3; ++i)
        srcs.push_back(write_test_png(dir / ("b" + std::to_string(i) + ".png"), 64, 64));

    EncodeOptions coarse = fast_options();
    coarse.pixels_per_gaussian = 1200;   // very few gaussians: a poor result on purpose
    coarse.steps = 150;

    LogCapture logs;
    Pipeline p;
    p.set_options(coarse);
    p.start(logs.sink());
    CHECK(p.add_files(srcs) == 3);
    if (!p.wait_until_idle(180s)) { fail("the first pass never finished"); p.stop(); return; }

    std::vector<double> first;
    for (const auto& j : p.jobs()) {
        if (j->status != JobStatus::Done) { fail("the first pass did not finish an image"); }
        std::lock_guard lk(j->m);
        first.push_back(j->stats.psnr);
    }

    EncodeOptions fine = coarse;
    fine.pixels_per_gaussian = 100;   // many more gaussians
    fine.steps = 400;
    p.set_options(fine);

    // What the interface's "Run All Again" does.
    int requeued = 0;
    for (const auto& j : p.jobs())
        if (p.requeue(j)) ++requeued;
    CHECK(requeued == 3);
    if (!p.wait_until_idle(180s)) { fail("the second pass never finished"); p.stop(); return; }
    p.stop();

    CHECK(p.job_count() == 3);   // still three entries, not six
    auto jobs = p.jobs();
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (jobs[i]->status != JobStatus::Done) {
            std::lock_guard lk(jobs[i]->m);
            fail("an image failed on the second pass: " + jobs[i]->error);
            return;
        }
        std::lock_guard lk(jobs[i]->m);
        CHECK(jobs[i]->stats.steps_requested == 400);
        if (!(jobs[i]->stats.psnr > first[i])) {
            fail("running again at a finer preset did not improve the result");
            return;
        }
    }
    std::printf("  3 images re-run at a finer preset: %.2f -> %.2f dB, still 3 entries\n",
                first[0], [&] { std::lock_guard lk(jobs[0]->m); return jobs[0]->stats.psnr; }());
}

// What the interface offers when a limit turned out to be too small. Doubling
// keeps the next attempt proportionate to the last, and once that would leave
// the slider the only honest offer left is to let the encode finish.
void test_the_offered_next_time_limit_escalates_then_gives_up_limiting() {
    CHECK(next_time_budget_after(30.0) == 60.0);
    CHECK(next_time_budget_after(60.0) == 120.0);
    CHECK(next_time_budget_after(120.0) == 240.0);
    CHECK(next_time_budget_after(double(kMaxTimeBudgetSeconds)) == 0.0);
    // Anything it offers is either "no limit" or a value the slider can show,
    // so pressing the button never leaves the interface displaying a setting
    // it could not have produced.
    for (double used : {1.0, 5.0, 29.0, 30.0, 100.0, 239.0, 240.0, 1000.0}) {
        const double next = next_time_budget_after(used);
        CHECK(next == 0.0 || (next > used && next <= double(kMaxTimeBudgetSeconds)));
        AppSettings s;
        s.encode.time_budget_seconds = next;
        clamp_settings(s);
        CHECK(s.encode.time_budget_seconds == next);
    }
    // A run that already had no limit has nothing to escalate to.
    CHECK(next_time_budget_after(0.0) == 0.0);
    std::puts("  the offered next time limit doubles, then becomes no limit at all");
}

// A run that was never limited must not be reported as cut short, or the
// interface would send someone to raise a setting that was not the reason.
void test_an_unlimited_run_is_never_reported_as_cut_short() {
    const fs::path dir = g_dir / "unlimited";
    const fs::path src = write_test_png(dir / "full.png", 64, 64);

    EncodeOptions o = fast_options();
    o.time_budget_seconds = 0.0;

    LogCapture logs;
    Pipeline p;
    p.set_options(o);
    p.start(logs.sink());
    CHECK(p.add_files({src}) == 1);
    auto job = run_one(p, logs, "an unlimited encode");
    if (!job) return;
    CHECK(job->status == JobStatus::Done);
    std::lock_guard lk(job->m);
    CHECK(job->used_time_budget == 0.0);
    CHECK(!job->was_cut_short());
    CHECK(!logs.mentions("time limit"));
    std::puts("  an encode with no limit is not blamed on the clock");
}

// ---------------------------------------------------- remembered settings
//
// The application used to remember nothing, so every launch reset the preset,
// the backend, the time limit and above all the output folder, which is
// several clicks through a folder picker to set.
//
// The file is the one input a user is invited to edit by hand, so the parser
// is tested the way the .gsi parser is: with the things a person and a broken
// disk actually produce.
void test_settings_survive_a_round_trip() {
    AppSettings written;
    written.encode.preset = 2;
    written.encode.pixels_per_gaussian = 220;
    written.encode.steps = 5000;
    written.encode.num_gaussians = 12345;
    written.encode.precision = 3;
    written.encode.custom_quant = QuantSpec{15, 11, 9, 13};
    written.encode.seed = 777;
    written.encode.backend = 1;
    written.encode.save_next_to_input = false;
    written.encode.export_png = true;
    written.encode.time_budget_seconds = 45.0;
    written.encode.out_dir = "C:/pictures/out = final";   // an '=' in the path
    written.ui.view_mode = 3;
    written.ui.difference_gain = 12.f;
    written.ui.log_autoscroll = false;
    written.ui.window_w = 1500;
    written.ui.window_h = 900;

    AppSettings read;
    CHECK(parse_settings(serialize_settings(written), read));
    CHECK(read.encode.preset == 2);
    CHECK(read.encode.pixels_per_gaussian == 220);
    CHECK(read.encode.steps == 5000);
    CHECK(read.encode.num_gaussians == 12345);
    CHECK(read.encode.precision == 3);
    CHECK(read.encode.custom_quant.pos == 15 && read.encode.custom_quant.feat == 13);
    CHECK(read.encode.seed == 777);
    CHECK(read.encode.backend == 1);
    CHECK(read.encode.save_next_to_input == false);
    CHECK(read.encode.export_png == true);
    CHECK(read.encode.time_budget_seconds == 45.0);
    CHECK(read.encode.out_dir == "C:/pictures/out = final");
    CHECK(read.ui.view_mode == 3);
    CHECK(read.ui.difference_gain == 12.f);
    CHECK(read.ui.log_autoscroll == false);
    CHECK(read.ui.window_w == 1500 && read.ui.window_h == 900);

    // Through a real file, which is what the application does.
    const fs::path file = g_dir / "settings" / "settings.ini";
    CHECK(save_settings(written, file));
    AppSettings loaded;
    CHECK(load_settings(loaded, file));
    CHECK(loaded.encode.steps == 5000);
    CHECK(loaded.encode.out_dir == written.encode.out_dir);
    CHECK(!fs::exists(fs::path(file) += ".partial"));

    // A file that is not there leaves the defaults alone rather than
    // half-clearing them.
    AppSettings untouched;
    CHECK(!load_settings(untouched, g_dir / "settings" / "nope.ini"));
    CHECK(untouched.encode.time_budget_seconds == kDefaultTimeBudgetSeconds);

    // Hand-written, half-broken, and from some other program.
    AppSettings tolerant;
    const char* messy =
        "# a comment\n"
        "\n"
        "   preset=1   \n"
        "not a setting line\n"
        "steps = seventeen\n"          // unparseable: left alone
        "from_a_newer_version = 4\n"   // unknown: ignored
        "log_autoscroll = no\n"
        "time_budget_seconds = 12.5\r\n";   // a file edited on Windows
    CHECK(parse_settings(messy, tolerant));
    CHECK(tolerant.encode.preset == 1);
    CHECK(tolerant.encode.steps == EncodeOptions{}.steps);
    CHECK(tolerant.ui.log_autoscroll == false);
    CHECK(tolerant.encode.time_budget_seconds == 12.5);

    // Nothing recognisable at all is reported as such, so a foreign file is
    // told apart from an empty one.
    AppSettings nothing;
    CHECK(!parse_settings("", nothing));
    CHECK(!parse_settings("\xff\xfe\x00garbage", nothing));

    // Values no slider could have produced are brought back into range rather
    // than reaching the encoder.
    AppSettings hostile;
    CHECK(parse_settings("preset = 99\n"
                         "precision = -4\n"
                         "backend = 12\n"
                         "pixels_per_gaussian = 0\n"
                         "steps = 999999999\n"
                         "bits_pos = 40\n"
                         "bits_color = 0\n"
                         "time_budget_seconds = -60\n"
                         "difference_gain = 0\n"
                         "view_mode = 7\n"
                         "window_w = 3\n"
                         "window_h = 999999\n",
                         hostile));
    CHECK(hostile.encode.preset >= 0 && hostile.encode.preset <= 3);
    CHECK(hostile.encode.precision >= 0 && hostile.encode.precision <= 3);
    CHECK(hostile.encode.backend >= 0 && hostile.encode.backend <= 2);
    CHECK(hostile.encode.pixels_per_gaussian >= kMinPixelsPerGaussian);
    CHECK(hostile.encode.steps <= kMaxCustomSteps);
    CHECK(hostile.encode.custom_quant.valid());
    CHECK(hostile.encode.time_budget_seconds >= 0.0);
    CHECK(hostile.encode.time_budget_seconds <= double(kMaxTimeBudgetSeconds));
    CHECK(hostile.ui.difference_gain >= kMinDifferenceGain);
    CHECK(hostile.ui.view_mode >= 0 && hostile.ui.view_mode <= 3);
    CHECK(hostile.ui.window_w >= 640 && hostile.ui.window_h <= 16384);
    // And whatever survived that must still describe a legal encode.
    const EncodeSettings s = hostile.encode.to_encode_settings();
    CHECK(s.quant.valid());
    CHECK(s.max_steps > 0 && s.pixels_per_gaussian > 0);
    std::puts("  settings round-trip, and a hand-edited file cannot break the encoder");
}

// The interface offers per-attribute bit depths and an exact gaussian count so
// that everything the command line tool can ask for is reachable from the
// window. Both have to arrive at the encoder intact, and neither may arrive
// as something the codec would refuse.
void test_the_window_can_ask_for_everything_the_tool_can() {
    EncodeOptions o;
    o.preset = 3;
    o.pixels_per_gaussian = 250;
    o.steps = 1234;
    o.num_gaussians = 5000;
    o.precision = 3;
    o.custom_quant = QuantSpec{14, 10, 8, 11};
    o.seed = 4242;
    o.backend = 2;

    EncodeSettings s = o.to_encode_settings();
    CHECK(s.pixels_per_gaussian == 250);
    CHECK(s.max_steps == 1234);
    CHECK(s.num_gaussians == 5000);
    CHECK(s.quant.pos == 14 && s.quant.scale == 10 && s.quant.rot == 8 && s.quant.feat == 11);
    CHECK(s.seed == 4242u);
    CHECK(s.backend == Backend::Gpu);

    // Zero means "work it out from the image", which is what the density
    // slider means, and must not become a request for no gaussians at all.
    o.num_gaussians = 0;
    CHECK(o.to_encode_settings().num_gaussians == 0);

    // Bit depths outside the format's range are brought back in rather than
    // producing an encode that fails at the last step.
    o.custom_quant = QuantSpec{99, 0, -7, 1000};
    CHECK(o.to_encode_settings().quant.valid());

    // The three named precisions still map to valid, distinct allocations.
    std::vector<int> sizes;
    for (int p = 0; p < 3; ++p) {
        o.precision = p;
        const QuantSpec q = o.to_encode_settings().quant;
        CHECK(q.valid());
        sizes.push_back(q.pos + q.scale + q.rot + q.feat);
    }
    CHECK(sizes[0] > sizes[1] && sizes[1] > sizes[2]);
    std::puts("  bit depths, gaussian count, seed and backend all reach the encoder");
}

// Several pipelines in one process, one after another, each compressing an
// image with the backend left on automatic.
//
// This is a regression test for a stall, and the stall is worth describing
// because it is the shape of the original complaint. The GPU context is
// carried by a hidden window, and on Windows that window's message queue
// belongs to the thread that created it. The context used to be created
// lazily, by whichever thread first wanted a GPU backend -- an encoding
// worker. The first image compressed correctly. Then that pipeline shut down,
// its worker exited, and the window was left without a message queue; the
// next worker's attempt to make the context current blocked forever. One
// core busy, no error, no progress, an interface still responding to clicks.
//
// The fix is that the context is only ever created by a thread that asked for
// it at startup and outlives every encode. What this test enforces is the
// consequence: doing the same work twice in one process must behave like
// doing it once, twice.
// Must run before anything in this process calls gpu_init, which is why it is
// first in main. That ordering is the whole test: with the context created at
// startup by a long-lived thread the bug is unreachable, so reproducing it
// requires the situation where nobody did, and the correct behaviour there is
// to quietly use the CPU rather than to create a context on a worker.
void test_pipelines_stall_free_without_a_startup_gpu_init() {
    const fs::path dir = g_dir / "no-init";
    const fs::path src = write_test_png(dir / "cold.png", 96, 96);

    for (int round = 0; round < 2; ++round) {
        EncodeOptions o = fast_options();
        o.backend = 0;   // automatic, and nobody has initialized a GPU
        LogCapture logs;
        Pipeline p;
        p.set_options(o);
        p.start(logs.sink());
        CHECK(p.add_files({src}) == 1);
        if (!p.wait_until_idle(60s)) {
            // Reported and then abandoned without stopping the pipeline. A
            // worker blocked inside a graphics driver cannot be joined, so
            // calling stop() here would hang the suite instead of failing it;
            // leaving the process to be torn down keeps the message visible.
            // Under CTest this still ends the run, as a timeout rather than a
            // clean failure, which is the honest outcome for a deadlock.
            fail("pipeline " + std::to_string(round) +
                 " stalled with no startup gpu_init: a worker created the compute context "
                 "and the next one blocked on it");
            std::fflush(stdout);
            return;
        }
        auto job = p.job(0);
        if (!job || job->status != JobStatus::Done) {
            if (job) { std::lock_guard lk(job->m); fail("cold round failed: " + job->error); }
            p.stop();
            return;
        }
        p.stop();
    }
    std::puts("  without a startup gpu_init: 2 pipelines, no stall");
}

void test_repeated_pipelines_do_not_stall() {
    const fs::path dir = g_dir / "repeat";
    const fs::path src = write_test_png(dir / "again.png", 96, 96);

    for (int round = 0; round < 3; ++round) {
        EncodeOptions o = fast_options();
        o.backend = 0;   // automatic: use the GPU if this machine has one
        LogCapture logs;
        Pipeline p;
        p.set_options(o);
        p.start(logs.sink());
        CHECK(p.add_files({src}) == 1);
        if (!p.wait_until_idle(60s)) {
            fail("pipeline " + std::to_string(round) +
                 " stalled: the image was queued and never finished");
            logs.dump();
            p.stop();
            return;
        }
        auto job = p.job(0);
        if (!job || job->status != JobStatus::Done) {
            if (job) { std::lock_guard lk(job->m); fail("round failed: " + job->error); }
            p.stop();
            return;
        }
        if (round == 0) {
            std::lock_guard lk(job->m);
            std::printf("  repeated pipelines: backend %s\n", job->stats.backend_used);
        }
        p.stop();
    }
    std::puts("  3 pipelines in one process, none stalled");
}

// Starting and stopping without ever queueing anything must not hang or
// crash: it is what happens every time someone opens the app and closes it.
void test_start_and_stop_with_no_work() {
    for (int i = 0; i < 20; ++i) {
        LogCapture logs;
        Pipeline p;
        p.start(logs.sink());
        CHECK(p.wait_until_idle(5s));
        p.stop();
        p.stop();   // idempotent
    }
    std::puts("  20 start/stop cycles with an empty queue");
}

} // namespace

int main() {
    std::error_code ec;
    g_dir = fs::temp_directory_path(ec) / "gsic_app_test";
    fs::remove_all(g_dir, ec);
    fs::create_directories(g_dir, ec);
    std::puts("application pipeline suite");

    // Deliberately before gpu_init: see the comment on this test.
    test_pipelines_stall_free_without_a_startup_gpu_init();

    {   // See gpu.h: the context belongs to a thread that outlives every
        // encode, so it is created here and never by a pipeline worker.
        std::string why;
        if (!gpu_init(&why)) std::printf("  (no GPU available: %s)\n", why.c_str());
    }

    test_adding_an_image_produces_a_readable_file();
    test_a_batch_compresses_every_image();
    test_a_gsi_can_be_opened_again_later();
    test_a_gsi_decodes_at_another_size();
    test_exporting_a_png_writes_a_readable_image();
    test_a_damaged_gsi_fails_with_a_reason();
    test_a_decoded_file_can_be_measured_against_its_original();
    test_comparing_two_images_reports_the_numbers();
    test_nothing_is_refused_silently();
    test_queued_work_is_never_left_asleep();
    test_work_added_during_an_encode_still_runs();
    test_cancel_stops_cleanly();
    test_output_falls_back_when_the_first_choice_is_unwritable();
    test_a_chosen_output_directory_is_used();
    test_every_preset_and_precision_encodes();
    test_finished_entries_are_handed_back_when_removed();
    test_removing_a_queued_entry_stops_it_running();
    test_an_entry_can_be_run_again();
    test_changing_the_settings_and_running_again_uses_the_new_ones();
    test_a_whole_batch_can_be_run_again_at_a_new_preset();
    test_the_offered_next_time_limit_escalates_then_gives_up_limiting();
    test_an_unlimited_run_is_never_reported_as_cut_short();
    test_settings_survive_a_round_trip();
    test_the_window_can_ask_for_everything_the_tool_can();
    test_the_app_defaults_to_a_bounded_wait();
    test_out_of_range_settings_are_clamped();
    test_repeated_pipelines_do_not_stall();
    test_start_and_stop_with_no_work();

    fs::remove_all(g_dir, ec);
    if (failures == 0) {
        std::puts("application pipeline suite passed");
        return 0;
    }
    std::printf("%d application failure(s)\n", failures);
    return 1;
}
