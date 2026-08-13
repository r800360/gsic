#pragma once

// The desktop application's queue and the work it runs, with no window, no GL
// context and no ImGui.
//
// It lives apart from app.cpp for one reason: this is the code path a user
// exercises when they add a file and expect something to happen, and while it
// was tangled up with the interface there was no way to test it. The suites
// covered the library below it and the window on top of it, and the thing in
// between -- pick a file, decide where the output goes, write it, report what
// happened -- was only ever checked by hand.
//
// A queue entry is one of three things (see JobKind). They share one worker,
// one status model and one preview representation, so the interface draws them
// with the same code and the tests drive them the same way.

#include "core/codec.h"
#include "core/image.h"
#include "core/trainer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gsic {

enum class JobStatus { Queued, Running, Done, Failed, Cancelled };

// What a queue entry does.
//
// Decode is the one that was missing, and its absence was the application's
// worst feature: it wrote .gsi files it could not open. A compressed image
// existed only for as long as the window that had just made it stayed open,
// which makes the format -- and therefore the whole application -- useless for
// keeping anything. Opening one is now the same kind of operation as
// compressing one, down to the queue entry it produces.
enum class JobKind {
    Compress,   // an image in, a .gsi out
    Decode,     // a .gsi in, a picture to look at (and optionally a .png out)
    Compare,    // two images in, PSNR/SSIM and a side-by-side out
};

// What a .gsi says about itself. Everything `gsic info` prints, so the window
// can show the same facts.
struct GsiInfo {
    int width = 0, height = 0, channels = 0;
    int gaussians = 0;
    QuantSpec quant;
    std::int64_t file_bytes = 0;
    double bpp = 0;             // bits per pixel of the stored file
    double percent_of_raw = 0;  // against 8-bit-per-channel uncompressed
};

struct Job {
    JobKind kind = JobKind::Compress;
    std::filesystem::path input;
    // Decode: render at this multiple of the stored resolution. Compare: unused.
    float scale = 1.f;
    // Decode: also write the rendered image here, as a .png. Empty means the
    // decode exists only to be looked at.
    std::filesystem::path export_png;
    // Decode: the original image to measure this file against, which turns a
    // decode into a comparison and enables the A/B split. Compare: the second
    // image. Empty otherwise.
    std::filesystem::path reference;

    std::atomic<JobStatus> status{JobStatus::Queued};
    std::atomic<float> progress{0.f};
    std::atomic<bool> cancel{false};

    // Everything below is guarded by `m`.
    std::mutex m;
    std::string error;
    int channels = 0;
    // Each image carries its own dimensions. They used to share one pair,
    // which is only correct while the preview happens to be the same size as
    // the source; a preview at any other size would have had the texture
    // upload read past the end of the smaller buffer.
    int orig_w = 0, orig_h = 0;
    int recon_w = 0, recon_h = 0;
    std::vector<std::uint8_t> orig_rgba, recon_rgba;   // interleaved, w*h*4
    bool orig_dirty = false, recon_dirty = false;
    EncodeStats stats{};
    bool has_stats = false;
    // The time limit this run was actually given, which is not necessarily the
    // one in the Settings panel now. Kept so the interface can say what cut a
    // run short and offer a longer limit, rather than leaving the user to work
    // out from a quality number that the answer is "it did not get to finish".
    // Zero means the run had no limit, so nothing about it was cut short.
    double used_time_budget = 0.0;
    GsiInfo info{};
    bool has_info = false;
    std::filesystem::path output_path;   // the .gsi, or the exported .png

    // True when the time limit, not the schedule, decided when this run
    // stopped -- so running it again with a longer one would genuinely give a
    // better result.
    bool was_cut_short() const {
        return has_stats && used_time_budget > 0.0 && stats.steps_requested > 0 &&
               stats.steps_run < stats.steps_requested;
    }

    // Live timing, so the interface can say how long this will take rather
    // than only how far along it is. A bare progress bar on a multi-minute
    // encode does not tell someone whether to wait or give up.
    std::atomic<double> elapsed_seconds{0.0};
    std::atomic<double> estimated_total_seconds{0.0};

    // UI-thread only. The interface owns these and must delete them before it
    // drops its last reference to the job; see Pipeline::clear_finished.
    unsigned tex_orig = 0, tex_recon = 0;
    unsigned tex_diff = 0;
    bool diff_dirty = true;
    // Measured while the difference image is built, and kept per entry so
    // switching between two of them does not show one's numbers over the
    // other's picture.
    float diff_gain_built = 0.f;
    double diff_max = 0, diff_mean = 0;
};

// A .gsi to open, in the form the interface asks for it.
struct DecodeRequest {
    std::filesystem::path input;
    float scale = 1.f;
    std::filesystem::path export_png;   // empty: view only
    std::filesystem::path reference;    // empty: no quality comparison
};

// The range the Settings slider offers, named so the tests can assert on the
// policy rather than on a literal buried in a draw call.
//
// Both numbers came down after measuring where the quality curve flattens:
// past roughly half a minute the remaining decibels arrive slowly enough that
// a person watching the window is paying more for them than they are worth,
// and an upper bound of ten minutes was advertising a wait nobody should be
// encouraged to choose. Zero still means no limit, for the rare image where
// someone genuinely wants the tail.
inline constexpr double kDefaultTimeBudgetSeconds = 30.0;
inline constexpr int kMaxTimeBudgetSeconds = 240;

// The limit worth offering after one that turned out to be too short: twice
// what the run had, and once that passes the top of the slider, no limit at
// all. Past that point the honest offer is "let it finish" rather than another
// number that might also be too small.
//
// Returns 0 ("no limit") for a run that already had none, which is the only
// answer that means anything there.
inline double next_time_budget_after(double used) {
    if (!(used > 0.0)) return 0.0;
    const double doubled = used * 2.0;
    return doubled > double(kMaxTimeBudgetSeconds) ? 0.0 : doubled;
}

// The encode-affecting choices, in the form the interface holds them. Copied
// under a lock when a job starts, so the worker never reads a value the user
// is in the middle of changing.
struct EncodeOptions {
    int preset = 1;               // 0 fast, 1 balanced, 2 high, 3 custom
    int pixels_per_gaussian = 300;
    int steps = 3000;
    int num_gaussians = 0;        // custom preset: 0 derives from the above
    int precision = 0;            // 0 best, 1 smaller, 2 smallest, 3 custom
    QuantSpec custom_quant{};     // used when precision == 3
    unsigned seed = 123;
    int backend = 0;              // 0 auto, 1 cpu, 2 gpu
    bool save_next_to_input = true;
    std::string out_dir;
    bool export_png = false;
    // Wall-clock target per image, in seconds; 0 removes the limit. The
    // desktop app sets one by default and the command line tool does not,
    // because the two are used differently: someone watching a window wants
    // the result soon, and someone scripting a batch wants every last decibel
    // and is not sitting there.
    double time_budget_seconds = kDefaultTimeBudgetSeconds;

    EncodeSettings to_encode_settings() const;
};

// Where a finished .gsi should go, and where it ended up if the first choice
// was not writable. Separated out because it is pure logic over paths and is
// worth testing directly: an encode that succeeded and then could not be
// saved looks exactly like an encode that failed.
struct OutputPlan {
    std::filesystem::path path;      // empty if nowhere was writable
    std::filesystem::path fallback_from;   // set when the first choice failed
    std::string note;                // human-readable, empty when unremarkable
};

// Picks the output path and proves it is writable before the caller commits
// to it. `probe` exists so tests can simulate an unwritable directory without
// needing one; production passes nothing and a real file is attempted.
OutputPlan plan_output(const std::filesystem::path& input, const EncodeOptions& options,
                       const std::function<bool(const std::filesystem::path&)>& is_writable = {});

// True when the extension is one this build's image reader is expected to
// handle. Advisory only: a file that fails this is still queued, because the
// reader sniffs content and the only authority on whether an image loads is
// the reader itself.
bool has_known_image_extension(const std::filesystem::path& p);

// True when the path names a compressed file this application wrote.
//
// The extension is checked first because it is what a user sees, but the
// answer comes from the first four bytes. A .gsi that was renamed still opens,
// and a .png that was renamed to .gsi is still refused by the image reader
// with a reason rather than being fed to the parser. Neither costs more than
// one short read.
bool has_gsi_extension(const std::filesystem::path& p);
bool looks_like_gsi(const std::filesystem::path& p);

class Pipeline {
public:
    using LogFn = std::function<void(const std::string&)>;

    Pipeline();
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Starts the single worker. `log` is called from the worker thread.
    void start(LogFn log);
    // Cancels everything outstanding and joins the worker. Idempotent.
    void stop();

    // Queues paths, deciding for each one whether it is an image to compress
    // or a .gsi to open. Returns how many were accepted. Anything refused is
    // reported through the log with a reason -- a file that vanishes without
    // a word is indistinguishable, from the user's side, from an application
    // that does not work.
    int add_files(const std::vector<std::filesystem::path>& paths);

    // Explicit forms, for the interface's Open, Export and Compare commands.
    // Both return the queued job, or null when the request was refused (with
    // the reason logged).
    std::shared_ptr<Job> add_decode(const DecodeRequest& request);
    std::shared_ptr<Job> add_compare(const std::filesystem::path& a,
                                     const std::filesystem::path& b);

    // Puts a finished entry back in the queue, so it runs again with whatever
    // it now asks for. This is Retry for something that failed, and it is how
    // an original is attached to an already-decoded file without leaving a
    // second copy of the same entry in the list. Returns false when the job is
    // not in the queue, or is still running.
    bool requeue(const std::shared_ptr<Job>& job);

    void set_options(const EncodeOptions& options);
    EncodeOptions options() const;

    // Snapshot of the queue. The shared_ptrs keep jobs alive while the caller
    // reads them.
    std::vector<std::shared_ptr<Job>> jobs() const;
    std::shared_ptr<Job> job(int index) const;
    int index_of(const std::shared_ptr<Job>& job) const;
    int job_count() const;
    void cancel_all();
    void cancel_job(const std::shared_ptr<Job>& job);

    // Both hand the removed entries back rather than dropping them, because
    // the interface owns GL textures hanging off each one and has to delete
    // them on the thread that holds the context. Returning them also keeps the
    // job alive until the caller is finished with it.
    std::vector<std::shared_ptr<Job>> clear_finished();
    std::shared_ptr<Job> remove_job(const std::shared_ptr<Job>& job);

    bool busy() const { return worker_busy_.load(std::memory_order_relaxed); }

    // Blocks until nothing is queued or running, or the timeout expires.
    // Returns true if the queue drained. For tests and for shutdown.
    bool wait_until_idle(std::chrono::milliseconds timeout);

private:
    void worker_loop();
    void run_job(const std::shared_ptr<Job>& job);
    void encode_job(const std::shared_ptr<Job>& job);
    void decode_job(const std::shared_ptr<Job>& job);
    void compare_job(const std::shared_ptr<Job>& job);
    void enqueue(std::vector<std::shared_ptr<Job>> accepted);
    void log(const std::string& line) const;

    mutable std::mutex jobs_m_;
    std::vector<std::shared_ptr<Job>> jobs_;

    mutable std::mutex options_m_;
    EncodeOptions options_;

    // work_m_ guards the queue-has-work condition. Every change that can make
    // the predicate true is published under it, including the push in
    // add_files: notifying without it can arrive between a waiter's predicate
    // check and its wait, which loses the wakeup and leaves the job sitting in
    // the queue forever while the interface stays perfectly responsive.
    std::mutex work_m_;
    std::condition_variable work_cv_;
    std::condition_variable idle_cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> worker_busy_{false};
    std::thread worker_;
    LogFn log_;
};

} // namespace gsic
