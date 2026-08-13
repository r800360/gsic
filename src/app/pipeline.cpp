#include "pipeline.h"

#include "core/codec.h"
#include "core/format.h"
#include "core/metrics.h"
#include "core/renderer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <random>
#include <system_error>

namespace fs = std::filesystem;

namespace gsic {

namespace {

// Planar float image -> RGBA8 (gray and gray+alpha spread over RGB).
std::vector<std::uint8_t> to_rgba(const Image& img) {
    std::vector<std::uint8_t> out(size_t(img.w) * img.h * 4);
    const std::int64_t n = img.pixels();
    auto tob = [](float v) {
        // Non-finite values would convert to an unspecified byte. The encoder
        // will not ship a file built from them, but the preview is drawn
        // before anyone knows that, so it has to survive them too.
        if (!(v == v)) return std::uint8_t(0);
        return std::uint8_t(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
    };
    for (std::int64_t i = 0; i < n; ++i) {
        std::uint8_t r, g, b, a = 255;
        if (img.c >= 3) {
            r = tob(img.plane(0)[i]);
            g = tob(img.plane(1)[i]);
            b = tob(img.plane(2)[i]);
            if (img.c == 4) a = tob(img.plane(3)[i]);
        } else {
            r = g = b = tob(img.plane(0)[i]);
            if (img.c == 2) a = tob(img.plane(1)[i]);
        }
        out[i * 4 + 0] = r;
        out[i * 4 + 1] = g;
        out[i * 4 + 2] = b;
        out[i * 4 + 3] = a;
    }
    return out;
}

fs::path home_directory() {
#if defined(_WIN32)
    if (const char* p = std::getenv("USERPROFILE")) return fs::path(p);
#endif
    if (const char* p = std::getenv("HOME")) return fs::path(p);
    return {};
}

// Actually creates and removes a file. Permissions on Windows are not
// something you can read off a directory and believe: a folder can be listable
// and still refuse a write, and a packaged application sees a different view
// of the file system than the one it was developed in. The only reliable
// question is whether the write succeeds.
bool directory_accepts_a_file(const fs::path& dir) {
    std::error_code ec;
    if (dir.empty()) return false;
    fs::create_directories(dir, ec);
    if (!fs::is_directory(dir, ec)) return false;

    // A unique name, so two copies probing at once cannot delete each other's
    // probe and conclude the directory is unwritable.
    std::random_device rd;
    char name[64];
    std::snprintf(name, sizeof(name), ".gsic-write-probe-%08x%08x", rd(), rd());
    const fs::path probe = dir / name;
    {
        std::ofstream f(probe, std::ios::binary);
        if (!f) return false;
        f.put('\0');
        if (!f) return false;
    }
    fs::remove(probe, ec);
    return true;
}

std::vector<std::uint8_t> read_whole_file(const fs::path& p, std::string* error) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        if (error) *error = "cannot open file";
        return {};
    }
    std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(f),
                                    std::istreambuf_iterator<char>()};
    if (bytes.empty() && error) *error = "the file is empty";
    return bytes;
}

// Writes through a neighbouring temporary name and then moves it into place. A
// write that dies partway -- a full disk, a removed drive -- would otherwise
// leave a truncated file sitting where a valid one should be, and a truncated
// file is one the user has to discover is broken.
bool write_atomically(const fs::path& out, const std::function<bool(const fs::path&)>& write_to) {
    fs::path tmp = out;
    tmp += ".partial";
    std::error_code ec;
    if (out.has_parent_path()) fs::create_directories(out.parent_path(), ec);
    if (!write_to(tmp)) {
        fs::remove(tmp, ec);
        return false;
    }
    fs::remove(out, ec);
    fs::rename(tmp, out, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

bool write_bytes_atomically(const fs::path& out, const std::vector<std::uint8_t>& bytes) {
    return write_atomically(out, [&](const fs::path& tmp) {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        f.flush();
        return bool(f);
    });
}

GsiInfo describe(int width, int height, int channels, int gaussians, const QuantSpec& quant,
                 std::int64_t file_bytes) {
    GsiInfo info;
    info.width = width;
    info.height = height;
    info.channels = channels;
    info.gaussians = gaussians;
    info.quant = quant;
    info.file_bytes = file_bytes;
    const double pixels = double(width) * double(height);
    info.bpp = pixels > 0 ? 8.0 * double(file_bytes) / pixels : 0.0;
    const double raw = pixels * double(channels);
    info.percent_of_raw = raw > 0 ? 100.0 * double(file_bytes) / raw : 0.0;
    return info;
}

GsiInfo describe(const GsiFile& file, std::int64_t file_bytes) {
    return describe(file.width, file.height, file.cloud.channels, file.cloud.count(), file.quant,
                    file_bytes);
}

}  // namespace

// ------------------------------------------------------------------ options
EncodeSettings EncodeOptions::to_encode_settings() const {
    EncodeSettings s;
    switch (preset) {
        case 0: s.pixels_per_gaussian = 450; s.max_steps = 1200; break;
        case 1: s.pixels_per_gaussian = 300; s.max_steps = 3000; break;
        case 2: s.pixels_per_gaussian = 190; s.max_steps = 8000; break;
        default:
            // Custom. Clamp to the range the interface offers so a settings
            // value that arrived from anywhere else -- a saved file from an
            // older version, a hand-edited one -- cannot ask for zero
            // gaussians or a run that never ends.
            s.pixels_per_gaussian = std::clamp(pixels_per_gaussian, 100, 1200);
            s.max_steps = std::clamp(steps, 100, 20000);
            // An explicit count, the command line tool's -n. Zero keeps the
            // derived-from-density behaviour, which is what the slider means.
            s.num_gaussians = num_gaussians > 0 ? std::clamp(num_gaussians, 1, kMaxGaussians) : 0;
            break;
    }
    // Quality levels scale the default per-attribute allocation together;
    // position always keeps the most bits (see QuantSpec).
    static const QuantSpec kQuality[] = {
        {16, 12, 10, 12},   // best
        {14, 11, 9, 11},    // smaller
        {12, 10, 8, 10},    // smallest
    };
    if (precision == 3) {
        // Per-attribute bit depths, the command line tool's --bits-*. Clamped
        // to the format's range, then checked: anything the codec would refuse
        // falls back to the safe default rather than producing an encode that
        // fails at the last step.
        QuantSpec q = custom_quant;
        const auto fix = [](int b) { return std::clamp(b, 4, 16); };
        q = QuantSpec{fix(q.pos), fix(q.scale), fix(q.rot), fix(q.feat)};
        s.quant = q.valid() ? q : kQuality[0];
    } else {
        s.quant = kQuality[std::clamp(precision, 0, 2)];
    }
    s.seed = seed;
    s.backend = backend == 1 ? Backend::Cpu : (backend == 2 ? Backend::Gpu : Backend::Auto);
    s.time_budget_seconds = std::max(0.0, time_budget_seconds);
    return s;
}

// ------------------------------------------------------------------- intake
namespace {
std::string lowercase_extension(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    return ext;
}
}  // namespace

bool has_known_image_extension(const fs::path& p) {
    static const char* kExts[] = {".png",  ".jpg", ".jpeg", ".jpe", ".jfif",
                                  ".bmp",  ".tga", ".gif",  ".hdr", ".psd"};
    const std::string ext = lowercase_extension(p);
    return std::any_of(std::begin(kExts), std::end(kExts),
                       [&](const char* e) { return ext == e; });
}

bool has_gsi_extension(const fs::path& p) { return lowercase_extension(p) == ".gsi"; }

bool looks_like_gsi(const fs::path& p) {
    // The same four bytes the parser checks. Read here so a file dropped on
    // the window is routed to the decoder or the image reader on the strength
    // of what it contains, not what it is called: the extension decides how it
    // is displayed in Explorer and nothing more.
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    char magic[4] = {};
    f.read(magic, 4);
    return f.gcount() == 4 && magic[0] == 'G' && magic[1] == 'S' && magic[2] == 'I' &&
           magic[3] == '1';
}

// ------------------------------------------------------------------- output
OutputPlan plan_output(const fs::path& input, const EncodeOptions& options,
                       const std::function<bool(const fs::path&)>& is_writable) {
    const auto writable = is_writable ? is_writable : &directory_accepts_a_file;

    // Candidates in the order a user would expect, each one a place the file
    // could reasonably live. Saving beside the image is the default because it
    // is where people look; it is also the one most likely to be refused, on
    // read-only media, on a network share, or inside a folder a packaged
    // application is not permitted to write. An encode that finished should
    // not be thrown away over that, so the search continues and says where it
    // ended up.
    struct Candidate { fs::path dir; const char* note; };
    std::vector<Candidate> candidates;

    const fs::path configured = options.out_dir.empty() ? fs::path{} : fs::path(options.out_dir);
    if (options.save_next_to_input || configured.empty()) {
        candidates.push_back({input.parent_path(), nullptr});
        if (!configured.empty()) candidates.push_back({configured, "the chosen output folder"});
    } else {
        candidates.push_back({configured, nullptr});
        candidates.push_back({input.parent_path(), "the image's own folder"});
    }

    const fs::path home = home_directory();
    if (!home.empty()) {
        candidates.push_back({home / "Pictures" / "gsic", "Pictures\\gsic"});
        candidates.push_back({home / "gsic", "your home folder"});
    }
    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path(ec);
    if (!ec) candidates.push_back({tmp / "gsic", "a temporary folder"});

    OutputPlan plan;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const fs::path& dir = candidates[i].dir;
        if (dir.empty()) continue;
        if (!writable(dir)) continue;
        plan.path = dir / input.filename();
        plan.path.replace_extension(".gsi");
        if (i > 0) {
            plan.fallback_from = candidates[0].dir;
            plan.note = candidates[i].note ? candidates[i].note : plan.path.parent_path().string();
        }
        return plan;
    }
    return plan;   // nothing was writable; path stays empty
}

// ----------------------------------------------------------------- pipeline
Pipeline::Pipeline() = default;

Pipeline::~Pipeline() { stop(); }

void Pipeline::start(LogFn log) {
    log_ = std::move(log);
    shutdown_ = false;
    worker_ = std::thread([this] { worker_loop(); });
}

void Pipeline::stop() {
    if (!worker_.joinable()) return;
    cancel_all();
    {
        // The flag is part of the wait predicate, so it changes under the
        // mutex that guards it. Setting it outside would let a worker between
        // "predicate false" and "blocked" miss the notify and never join.
        std::lock_guard lk(work_m_);
        shutdown_ = true;
    }
    work_cv_.notify_all();
    worker_.join();
}

void Pipeline::log(const std::string& line) const {
    if (log_) log_(line);
}

void Pipeline::set_options(const EncodeOptions& options) {
    std::lock_guard lk(options_m_);
    options_ = options;
}

EncodeOptions Pipeline::options() const {
    std::lock_guard lk(options_m_);
    return options_;
}

void Pipeline::enqueue(std::vector<std::shared_ptr<Job>> accepted) {
    if (accepted.empty()) return;
    {
        std::lock_guard jl(jobs_m_);
        for (auto& j : accepted) jobs_.push_back(std::move(j));
    }
    {
        std::lock_guard lk(work_m_);   // see the comment on work_m_
    }
    work_cv_.notify_all();
}

int Pipeline::add_files(const std::vector<fs::path>& paths) {
    std::vector<std::shared_ptr<Job>> accepted;
    for (const auto& p : paths) {
        std::error_code ec;
        if (p.empty()) continue;
        if (fs::is_directory(p, ec)) {
            log(p.filename().string() + ": skipped, this is a folder");
            continue;
        }
        if (!fs::exists(p, ec)) {
            log(p.filename().string() + ": skipped, no such file");
            continue;
        }
        // Deliberately not gated on the extension. Filtering here is what made
        // an unrecognised file disappear with no message at all, and the
        // extension is the weaker authority anyway: the reader identifies
        // images by content, so it handles a .jfif or a misnamed file that a
        // list of suffixes would reject. Anything it genuinely cannot read
        // fails below with a reason attached to a visible queue entry.
        auto job = std::make_shared<Job>();
        job->input = p;
        // A compressed file dropped on the window is something to look at, not
        // something to compress again. Before this the file association the
        // package declares led straight to "could not be read as an image":
        // double-clicking a .gsi in Explorer started the application and then
        // told the user their own file was broken.
        if (has_gsi_extension(p) || looks_like_gsi(p)) job->kind = JobKind::Decode;
        accepted.push_back(std::move(job));
    }
    const int n = int(accepted.size());
    enqueue(std::move(accepted));
    return n;
}

std::shared_ptr<Job> Pipeline::add_decode(const DecodeRequest& request) {
    std::error_code ec;
    if (request.input.empty() || !fs::exists(request.input, ec)) {
        log(request.input.filename().string() + ": skipped, no such file");
        return nullptr;
    }
    auto job = std::make_shared<Job>();
    job->kind = JobKind::Decode;
    job->input = request.input;
    job->scale = request.scale;
    job->export_png = request.export_png;
    job->reference = request.reference;
    enqueue({job});
    return job;
}

std::shared_ptr<Job> Pipeline::add_compare(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    for (const auto& p : {a, b}) {
        if (p.empty() || !fs::exists(p, ec)) {
            log(p.filename().string() + ": skipped, no such file");
            return nullptr;
        }
    }
    auto job = std::make_shared<Job>();
    job->kind = JobKind::Compare;
    job->input = a;
    job->reference = b;
    enqueue({job});
    return job;
}

std::vector<std::shared_ptr<Job>> Pipeline::jobs() const {
    std::lock_guard lk(jobs_m_);
    return jobs_;
}

std::shared_ptr<Job> Pipeline::job(int index) const {
    std::lock_guard lk(jobs_m_);
    if (index < 0 || index >= int(jobs_.size())) return nullptr;
    return jobs_[size_t(index)];
}

int Pipeline::index_of(const std::shared_ptr<Job>& job) const {
    if (!job) return -1;
    std::lock_guard lk(jobs_m_);
    for (size_t i = 0; i < jobs_.size(); ++i)
        if (jobs_[i] == job) return int(i);
    return -1;
}

int Pipeline::job_count() const {
    std::lock_guard lk(jobs_m_);
    return int(jobs_.size());
}

void Pipeline::cancel_all() {
    std::lock_guard lk(jobs_m_);
    for (auto& j : jobs_) {
        const JobStatus st = j->status;
        if (st == JobStatus::Running || st == JobStatus::Queued) j->cancel = true;
    }
}

void Pipeline::cancel_job(const std::shared_ptr<Job>& job) {
    if (!job) return;
    const JobStatus st = job->status;
    if (st == JobStatus::Running || st == JobStatus::Queued) job->cancel = true;
}

std::vector<std::shared_ptr<Job>> Pipeline::clear_finished() {
    std::vector<std::shared_ptr<Job>> removed;
    std::lock_guard lk(jobs_m_);
    const auto it = std::stable_partition(jobs_.begin(), jobs_.end(), [](const auto& j) {
        const JobStatus st = j->status;
        return st == JobStatus::Queued || st == JobStatus::Running;
    });
    removed.assign(std::make_move_iterator(it), std::make_move_iterator(jobs_.end()));
    jobs_.erase(it, jobs_.end());
    return removed;
}

bool Pipeline::requeue(const std::shared_ptr<Job>& job) {
    if (!job) return false;
    {
        std::lock_guard lk(jobs_m_);
        if (std::find(jobs_.begin(), jobs_.end(), job) == jobs_.end()) return false;
        const JobStatus st = job->status;
        if (st == JobStatus::Queued) return true;   // already waiting; nothing to do
        if (st == JobStatus::Running) return false;
        {
            std::lock_guard jl(job->m);
            job->error.clear();
            job->has_stats = false;
            job->stats = EncodeStats{};
            job->used_time_budget = 0.0;
            job->output_path.clear();
        }
        job->cancel = false;
        job->progress = 0.f;
        job->elapsed_seconds.store(0.0, std::memory_order_relaxed);
        job->estimated_total_seconds.store(0.0, std::memory_order_relaxed);
        job->status = JobStatus::Queued;
    }
    {
        std::lock_guard lk(work_m_);   // see the comment on work_m_
    }
    work_cv_.notify_all();
    return true;
}

std::shared_ptr<Job> Pipeline::remove_job(const std::shared_ptr<Job>& job) {
    if (!job) return nullptr;
    std::lock_guard lk(jobs_m_);
    // The status is read under the same lock the worker claims jobs with, so
    // an entry here is either still queued -- and ours to take out -- or
    // already running, in which case it belongs to the worker and all we can
    // do is ask it to stop. Removing it under the worker's feet would leave
    // the interface unable to show that it is winding down.
    if (job->status == JobStatus::Running) {
        job->cancel = true;
        return nullptr;
    }
    const auto it = std::find(jobs_.begin(), jobs_.end(), job);
    if (it == jobs_.end()) return nullptr;
    job->cancel = true;
    auto out = *it;
    jobs_.erase(it);
    return out;
}

bool Pipeline::wait_until_idle(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::unique_lock lk(work_m_);
    return idle_cv_.wait_until(lk, deadline, [this] {
        if (worker_busy_.load(std::memory_order_relaxed)) return false;
        std::lock_guard jl(jobs_m_);
        return std::none_of(jobs_.begin(), jobs_.end(), [](const auto& j) {
            return j->status == JobStatus::Queued;
        });
    });
}

void Pipeline::worker_loop() {
    while (true) {
        std::shared_ptr<Job> next;
        {
            std::unique_lock lk(work_m_);
            work_cv_.wait(lk, [&] {
                if (shutdown_) return true;
                std::lock_guard jl(jobs_m_);
                for (auto& j : jobs_)
                    if (j->status == JobStatus::Queued) return true;
                return false;
            });
            if (shutdown_) return;
            std::lock_guard jl(jobs_m_);
            for (auto& j : jobs_)
                if (j->status == JobStatus::Queued) {
                    next = j;
                    // Claimed while both locks are held, so the same job can
                    // never be picked up twice.
                    j->status = JobStatus::Running;
                    break;
                }
            if (next) worker_busy_ = true;
        }
        if (!next) continue;
        run_job(next);
        {
            std::lock_guard lk(work_m_);
            worker_busy_ = false;
        }
        idle_cv_.notify_all();
    }
}

void Pipeline::run_job(const std::shared_ptr<Job>& job) {
    // A job cancelled while it sat in the queue must not start. Without this
    // the worker would pick it up, run it to completion, and only then notice
    // -- which is a strange thing to watch happen after pressing Cancel.
    if (job->cancel.load(std::memory_order_relaxed)) {
        job->status = JobStatus::Cancelled;
        return;
    }
    switch (job->kind) {
        case JobKind::Decode: decode_job(job); break;
        case JobKind::Compare: compare_job(job); break;
        case JobKind::Compress: encode_job(job); break;
    }
}

// ------------------------------------------------------------------- decode
//
// Opening a .gsi. Cheap next to an encode -- a 2K image renders in about 20 ms
// -- but it runs on the worker like everything else, because "cheap" is a
// statement about a typical file and this one arrives from the user's disk. A
// 4x decode of a large image is a real fraction of a second, and a fraction of
// a second on the drawing thread is a dropped frame the user can feel.
void Pipeline::decode_job(const std::shared_ptr<Job>& job) {
    job->status = JobStatus::Running;
    job->progress = 0.f;
    const std::string name = job->input.filename().string();

    const auto finish_failed = [&](const std::string& why) {
        std::lock_guard lk(job->m);
        job->error = why;
        job->status = JobStatus::Failed;
        log(name + ": " + why);
    };

    std::string err;
    const auto bytes = read_whole_file(job->input, &err);
    if (bytes.empty()) {
        finish_failed(err.empty() ? "the file could not be read" : err);
        return;
    }
    auto file = decode_gsi(bytes, &err);
    if (!file) {
        finish_failed(err.empty() ? "this is not a readable .gsi file" : err);
        return;
    }
    const GsiInfo info = describe(*file, std::int64_t(bytes.size()));
    {
        std::lock_guard lk(job->m);
        job->info = info;
        job->has_info = true;
        job->channels = info.channels;
    }
    job->progress = 0.25f;
    if (job->cancel) {
        job->status = JobStatus::Cancelled;
        log(name + ": cancelled");
        return;
    }

    auto scaled = scale_gsi(*file, job->scale, &err);
    if (!scaled) {
        finish_failed(err);
        return;
    }
    Image decoded = Renderer::render_image(scaled->cloud, scaled->w, scaled->h);
    job->progress = 0.7f;
    if (job->cancel) {
        job->status = JobStatus::Cancelled;
        log(name + ": cancelled");
        return;
    }
    {
        auto rgba = to_rgba(decoded);
        std::lock_guard lk(job->m);
        job->recon_w = decoded.w;
        job->recon_h = decoded.h;
        job->recon_rgba = std::move(rgba);
        job->recon_dirty = true;
    }

    // An original to hold it against, when one was supplied. This is the same
    // measurement `gsic compare` makes, reached from the window instead: a
    // person who kept the source can see what the compression actually cost.
    double psnr_db = 0, ssim_v = 0;
    bool measured = false;
    // Whatever was on the other side of the divider from a previous run of
    // this same entry goes now. Leaving it would label one original as another
    // the moment a second attempt failed to load its own.
    const auto forget_the_original = [&] {
        std::lock_guard lk(job->m);
        job->orig_rgba.clear();
        job->orig_w = job->orig_h = 0;
        job->orig_dirty = true;
    };
    forget_the_original();
    if (!job->reference.empty()) {
        auto ref = Image::load(job->reference, &err);
        if (!ref) {
            log(name + ": the original could not be read (" + err + "), showing the decoded "
                "image on its own");
        } else if (ref->w != decoded.w || ref->h != decoded.h || ref->c != decoded.c) {
            log(name + ": the original is " + std::to_string(ref->w) + "x" +
                std::to_string(ref->h) + " and this decodes to " + std::to_string(decoded.w) +
                "x" + std::to_string(decoded.h) + ", so they cannot be compared");
        } else {
            psnr_db = psnr(decoded, *ref);
            ssim_v = ssim(decoded, *ref);
            measured = true;
            auto rgba = to_rgba(*ref);
            std::lock_guard lk(job->m);
            job->orig_w = ref->w;
            job->orig_h = ref->h;
            job->orig_rgba = std::move(rgba);
            job->orig_dirty = true;
        }
    }

    fs::path written;
    if (!job->export_png.empty()) {
        if (!write_atomically(job->export_png,
                              [&](const fs::path& tmp) { return decoded.save_png(tmp); })) {
            finish_failed("decoded fine, but the image could not be written to " +
                          job->export_png.string());
            return;
        }
        written = job->export_png;
    }

    {
        std::lock_guard lk(job->m);
        if (measured) {
            job->stats.psnr = psnr_db;
            job->stats.ssim = ssim_v;
            job->has_stats = true;
        }
        job->stats.file_bytes = info.file_bytes;
        job->stats.source_bytes =
            std::int64_t(info.width) * info.height * std::max(1, info.channels);
        job->stats.num_gaussians = info.gaussians;
        job->output_path = written;
        job->status = JobStatus::Done;
    }
    job->progress = 1.f;

    char buf[512];
    if (!written.empty())
        std::snprintf(buf, sizeof(buf), "%s -> %s  (%dx%d at %.2fx)", name.c_str(),
                      written.filename().string().c_str(), decoded.w, decoded.h,
                      double(scaled->scale));
    else
        std::snprintf(buf, sizeof(buf), "%s  %dx%d, %d gaussians, %s (%.3f bpp)", name.c_str(),
                      info.width, info.height, info.gaussians,
                      human_size(info.file_bytes).c_str(), info.bpp);
    log(buf);
    if (measured) {
        std::snprintf(buf, sizeof(buf), "%s: PSNR %.2f dB, SSIM %.4f against %s", name.c_str(),
                      psnr_db, ssim_v, job->reference.filename().string().c_str());
        log(buf);
    }
}

// ------------------------------------------------------------------ compare
//
// Two images, the numbers between them, and both pictures loaded so the
// divider works. `gsic compare` prints the same two numbers; this adds the
// part a terminal cannot do, which is letting someone see where the difference
// actually is.
void Pipeline::compare_job(const std::shared_ptr<Job>& job) {
    job->status = JobStatus::Running;
    job->progress = 0.f;
    const std::string name = job->input.filename().string() + " vs " +
                             job->reference.filename().string();

    const auto finish_failed = [&](const std::string& why) {
        std::lock_guard lk(job->m);
        job->error = why;
        job->status = JobStatus::Failed;
        log(name + ": " + why);
    };

    std::string err;
    auto a = Image::load(job->input, &err);
    if (!a) {
        finish_failed(job->input.filename().string() + " could not be read: " + err);
        return;
    }
    job->progress = 0.33f;
    auto b = Image::load(job->reference, &err);
    if (!b) {
        finish_failed(job->reference.filename().string() + " could not be read: " + err);
        return;
    }
    job->progress = 0.66f;
    if (a->w != b->w || a->h != b->h || a->c != b->c) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "these cannot be compared: %dx%d with %d channel(s) against %dx%d with "
                      "%d channel(s)",
                      a->w, a->h, a->c, b->w, b->h, b->c);
        finish_failed(buf);
        return;
    }

    const double psnr_db = psnr(*a, *b);
    const double ssim_v = ssim(*a, *b);
    {
        auto rgba_a = to_rgba(*a);
        auto rgba_b = to_rgba(*b);
        std::lock_guard lk(job->m);
        job->orig_w = a->w;
        job->orig_h = a->h;
        job->orig_rgba = std::move(rgba_a);
        job->orig_dirty = true;
        job->recon_w = b->w;
        job->recon_h = b->h;
        job->recon_rgba = std::move(rgba_b);
        job->recon_dirty = true;
        job->channels = a->c;
        job->stats.psnr = psnr_db;
        job->stats.ssim = ssim_v;
        job->has_stats = true;
        job->status = JobStatus::Done;
    }
    job->progress = 1.f;

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s: PSNR %.2f dB, SSIM %.4f", name.c_str(), psnr_db, ssim_v);
    log(buf);
}

// ------------------------------------------------------------------- encode
void Pipeline::encode_job(const std::shared_ptr<Job>& job) {
    job->status = JobStatus::Running;
    job->progress = 0.f;
    const std::string name = job->input.filename().string();

    const auto finish_failed = [&](const std::string& why) {
        std::lock_guard lk(job->m);
        job->error = why;
        job->status = JobStatus::Failed;
        log(name + ": " + why);
    };

    std::string err;
    auto img = Image::load(job->input, &err);
    if (!img) {
        finish_failed(err.empty() ? "could not be read as an image" : err);
        return;
    }
    {
        auto rgba = to_rgba(*img);
        std::lock_guard lk(job->m);
        job->orig_w = img->w;
        job->orig_h = img->h;
        job->channels = img->c;
        job->orig_rgba = std::move(rgba);
        job->orig_dirty = true;
    }
    log(name + ": " + std::to_string(img->w) + "x" + std::to_string(img->h) + ", " +
        std::to_string(img->c) + " channel(s)");

    // Settings are snapshotted once, here, so a change made while this image
    // is encoding applies to the next one rather than to part of this one.
    const EncodeOptions options = this->options();
    EncodeSettings s = options.to_encode_settings();

    // Deciding where the output goes before spending the time on the encode.
    // Discovering the destination is unwritable after several minutes of work
    // is the difference between a warning and a wasted encode.
    const OutputPlan plan = plan_output(job->input, options);
    if (plan.path.empty()) {
        finish_failed("no writable folder was found to save the compressed file in");
        return;
    }
    if (!plan.note.empty())
        log(name + ": cannot write to " + plan.fallback_from.string() + ", saving to " +
            plan.note + " instead");

    auto progress = [&](const EncodeProgress& p) {
        job->progress = float(p.step) / float(std::max(1, p.max_steps));
        job->elapsed_seconds.store(p.elapsed_seconds, std::memory_order_relaxed);
        job->estimated_total_seconds.store(p.estimated_total_seconds,
                                           std::memory_order_relaxed);
        if (p.preview) {
            auto rgba = to_rgba(*p.preview);
            std::lock_guard lk(job->m);
            job->recon_w = p.preview->w;
            job->recon_h = p.preview->h;
            job->recon_rgba = std::move(rgba);
            job->recon_dirty = true;
        }
    };
    auto result = encode_image(*img, s, progress, &job->cancel);

    if (job->cancel || result.cancelled) {
        job->status = JobStatus::Cancelled;
        log(name + ": cancelled");
        return;
    }
    if (!result.ok()) {
        finish_failed(result.error.empty() ? "compression failed" : result.error);
        return;
    }
    if (!result.stats.gpu_fallback_reason.empty())
        log(name + ": the GPU " + result.stats.gpu_fallback_reason +
            ", so this image was compressed on the CPU");
    // Said out loud, because the alternative is a user comparing two machines
    // and finding quietly different results for the same image and settings.
    // Only claimed when there was a limit to stay inside: without one the
    // schedule decides when the run ends, and blaming the clock for that would
    // send someone to raise a setting that was never the reason.
    if (s.time_budget_seconds > 0.0 && result.stats.steps_requested > 0 &&
        result.stats.steps_run < result.stats.steps_requested)
        log(name + ": stopped at " + std::to_string(result.stats.steps_run) + " of " +
            std::to_string(result.stats.steps_requested) +
            " steps to stay inside the " + std::to_string(int(s.time_budget_seconds + 0.5)) +
            "s time limit; run it again with a longer one for more quality");

    const fs::path out = plan.path;
    if (!write_bytes_atomically(out, result.file)) {
        finish_failed("compressed fine, but the file could not be written to " + out.string());
        return;
    }
    if (options.export_png) {
        fs::path png = out;
        png.replace_extension(".decoded.png");
        if (!write_atomically(png, [&](const fs::path& tmp) {
                return result.reconstruction.save_png(tmp);
            }))
            log(name + ": the .gsi was saved but the decoded .png could not be written");
    }

    {
        auto rgba = to_rgba(result.reconstruction);
        const GsiInfo info =
            describe(result.reconstruction.w, result.reconstruction.h, result.cloud.channels,
                     result.cloud.count(), s.quant, result.stats.file_bytes);
        std::lock_guard lk(job->m);
        job->recon_w = result.reconstruction.w;
        job->recon_h = result.reconstruction.h;
        job->recon_rgba = std::move(rgba);
        job->recon_dirty = true;
        job->stats = result.stats;
        job->has_stats = true;
        job->used_time_budget = s.time_budget_seconds;
        job->output_path = out;
        job->info = info;
        job->has_info = true;
        job->status = JobStatus::Done;
    }
    job->progress = 1.f;

    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s -> %s  (%s, %.1f%% of raw, PSNR %.2f dB, %.1fs, %s)",
                  name.c_str(), out.filename().string().c_str(),
                  human_size(result.stats.file_bytes).c_str(),
                  100.0 * double(result.stats.file_bytes) /
                      double(std::max<std::int64_t>(1, result.stats.source_bytes)),
                  result.stats.psnr, result.stats.encode_seconds, result.stats.backend_used);
    log(buf);
}

} // namespace gsic
