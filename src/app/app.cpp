// gsic GUI: a queue on the left, the picture in the middle, the log at the
// bottom. One background worker runs the queue in order (each encode already
// uses every core or the GPU, so running two at once helps nobody).
//
// A queue entry is an image being compressed, a .gsi being looked at, or two
// images being compared -- see pipeline.h. The window draws all three the same
// way, because from where a user sits they are all "a thing I opened".
#include "app.h"

#include "pipeline.h"
#include "settings.h"

#include "core/codec.h"
#include "core/format.h"
#include "core/gpu.h"
#include "core/kernels.h"
#include "core/renderer.h"

#include <glad/glad.h>
// glad before GLFW.
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <nfd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace gsic {
namespace {

#ifndef GSIC_VERSION
#define GSIC_VERSION "1.0.1"
#endif

const nfdu8filteritem_t kImageFilter[] = {{"Images", "png,jpg,jpeg,jpe,jfif,bmp,tga,gif,hdr"}};
const nfdu8filteritem_t kGsiFilter[] = {{"Compressed images", "gsi"}};
const nfdu8filteritem_t kPngFilter[] = {{"PNG image", "png"}};

// Shows the file in whatever the desktop uses to browse files, with it
// selected where that is possible.
//
// Writing a file and then not being able to find it is one of the more
// annoying ways for an application to waste someone's time, and "next to the
// input" is only obvious when the user remembers where the input was. The log
// says the path; this saves them copying it out.
void reveal_in_file_manager(const fs::path& target) {
    if (target.empty()) return;
#if defined(_WIN32)
    // Quoted because a path may contain spaces, and passed as an argument
    // rather than through a shell so nothing in the name is interpreted.
    std::wstring args = L"/select,\"" + target.wstring() + L"\"";
    ShellExecuteW(nullptr, L"open", L"explorer.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
#else
    // fork/exec rather than system(): a file name may contain any character a
    // shell would treat as syntax, and there is no quoting scheme that is
    // correct for every shell. The intermediate child is reaped immediately
    // and the grandchild is inherited by init, so nothing here leaves zombies
    // behind or blocks the interface waiting for a file browser to close.
    const std::string path = target.string();
    const std::string dir = target.has_parent_path() ? target.parent_path().string() : ".";
    const pid_t first = fork();
    if (first == 0) {
        if (fork() == 0) {
#if defined(__APPLE__)
            execlp("open", "open", "-R", path.c_str(), static_cast<char*>(nullptr));
#else
            (void)path;
            execlp("xdg-open", "xdg-open", dir.c_str(), static_cast<char*>(nullptr));
#endif
            _exit(127);
        }
        _exit(0);
    }
    if (first > 0) {
        int status = 0;
        waitpid(first, &status, 0);
    }
#endif
}

// One place that asks for a file, so every dialog in the application behaves
// the same and none of them leak the string the library hands back.
std::vector<fs::path> ask_for_files(const nfdu8filteritem_t* filters, nfdfiltersize_t count,
                                    bool multiple) {   // NOLINT: mirrors the library's types
    std::vector<fs::path> files;
    if (multiple) {
        const nfdpathset_t* paths = nullptr;
        nfdopendialogu8args_t args{};
        args.filterList = filters;
        args.filterCount = count;
        if (NFD_OpenDialogMultipleU8_With(&paths, &args) != NFD_OKAY) return files;
        nfdpathsetsize_t n = 0;
        NFD_PathSet_GetCount(paths, &n);
        for (nfdpathsetsize_t i = 0; i < n; ++i) {
            nfdu8char_t* p = nullptr;
            if (NFD_PathSet_GetPathU8(paths, i, &p) == NFD_OKAY && p) {
                files.emplace_back(reinterpret_cast<char*>(p));
                NFD_PathSet_FreePathU8(p);
            }
        }
        NFD_PathSet_Free(paths);
        return files;
    }
    nfdu8char_t* p = nullptr;
    nfdopendialogu8args_t args{};
    args.filterList = filters;
    args.filterCount = count;
    if (NFD_OpenDialogU8_With(&p, &args) == NFD_OKAY && p) {
        files.emplace_back(reinterpret_cast<char*>(p));
        NFD_FreePathU8(p);
    }
    return files;
}

fs::path ask_where_to_save(const nfdu8filteritem_t* filters, nfdfiltersize_t count,
                           const std::string& suggested_name, const fs::path& suggested_dir) {
    nfdu8char_t* p = nullptr;
    nfdsavedialogu8args_t args{};
    args.filterList = filters;
    args.filterCount = count;
    const std::string dir = suggested_dir.string();
    if (!suggested_name.empty()) args.defaultName = suggested_name.c_str();
    if (!dir.empty()) args.defaultPath = dir.c_str();
    fs::path out;
    if (NFD_SaveDialogU8_With(&p, &args) == NFD_OKAY && p) {
        out = reinterpret_cast<char*>(p);
        NFD_FreePathU8(p);
    }
    return out;
}

const char* kind_label(JobKind kind) {
    switch (kind) {
        case JobKind::Decode: return "view";
        case JobKind::Compare: return "compare";
        case JobKind::Compress: break;
    }
    return "compress";
}

// ------------------------------------------------------------------- app
class App {
public:
    int run(const std::vector<fs::path>& initial_files, const AppOptions& options);
    // Both return an empty string on success, or a description of the first
    // thing found wrong: check_layout for what the window shows, and
    // check_compression for whether the application does its job.
    std::string check_layout() const;
    std::string check_compression();

private:
    // Intake and commands.
    void add_files(const std::vector<fs::path>& paths);
    void open_images_dialog();
    void open_gsi_dialog();
    void compare_images_dialog();
    void export_selected_as_png(float scale);
    void attach_original_to_selected();
    void retry_selected();
    void run_everything_again();
    void run_selected_again_with(double time_budget_seconds);
    void remove_selected();
    void reveal_selected();
    void run_speed_test();

    // Drawing.
    void draw_ui(bool& want_close);
    void draw_menu(bool& want_close);
    void draw_queue();
    void draw_settings();
    void draw_details();
    void draw_preview();
    void draw_log();
    void draw_about();
    void draw_diagnostics();
    void draw_export_popup();
    void draw_quit_popup(bool& want_close);
    void handle_shortcuts(bool& want_close);
    void setup_dock(ImGuiID dockspace);

    // Job and texture bookkeeping.
    void select(const std::shared_ptr<Job>& job);
    void upload_textures(Job& job);
    void rebuild_difference(Job& job);
    void release_textures(Job& job);
    void release_textures(const std::vector<std::shared_ptr<Job>>& jobs);
    void flush_released_textures();
    void reset_view();

    void push_options();
    void persist_settings();
    std::string diagnostics_report() const;
    void log(const std::string& line);
    static void drop_callback(GLFWwindow* win, int count, const char** paths);

    GLFWwindow* window_ = nullptr;

    // The queue and everything that happens to a file lives here; this class
    // is only the window around it.
    Pipeline pipeline_;
    // Held by pointer rather than by index: entries are removed from the
    // middle of the list, and an index quietly starts pointing at a different
    // file when that happens.
    std::shared_ptr<Job> selected_;

    std::mutex log_m_;
    std::deque<std::string> log_lines_;

    // The interface's copy of the settings, mirrored to disk.
    AppSettings settings_;
    fs::path settings_file_;
    bool settings_dirty_ = false;
    double settings_dirty_at_ = 0.0;
    char out_dir_[512] = "";
    std::string gpu_status_;
    bool gpu_ok_ = false;
    std::string gpu_reason_;

    // Preview state, reset when the selection changes.
    float split_ = 0.5f;
    float zoom_ = 0.f;            // 0 = fit
    ImVec2 pan_{0, 0};
    bool dragging_split_ = false;
    std::vector<unsigned> released_textures_;

    bool show_about_ = false;
    bool show_diagnostics_ = false;
    bool open_export_popup_ = false;
    bool open_quit_popup_ = false;
    float export_scale_ = 1.f;
    bool confirmed_quit_ = false;
    int idle_frames_ = 0;
};

App* g_app = nullptr;

void App::log(const std::string& line) {
    std::lock_guard lk(log_m_);
    log_lines_.push_back(line);
    if (log_lines_.size() > 2000) log_lines_.pop_front();
}

void App::push_options() {
    settings_.encode.out_dir = out_dir_;
    clamp_settings(settings_);
    pipeline_.set_options(settings_.encode);
    // Written out after a pause rather than on every keystroke: the output
    // folder is a text field, and saving on each character would mean a disk
    // write per letter typed.
    settings_dirty_ = true;
    settings_dirty_at_ = ImGui::GetTime();
}

void App::persist_settings() {
    if (!settings_dirty_) return;
    settings_dirty_ = false;
    if (settings_file_.empty()) return;
    save_settings(settings_, settings_file_);
}

// ------------------------------------------------------------------ intake
void App::drop_callback(GLFWwindow*, int count, const char** paths) {
    std::vector<fs::path> files;
    for (int i = 0; i < count; ++i) files.emplace_back(paths[i]);
    g_app->add_files(files);
}

void App::add_files(const std::vector<fs::path>& paths) {
    const int before = pipeline_.job_count();
    if (pipeline_.add_files(paths) <= 0) return;
    // Select the first thing that just arrived, so something is on screen
    // immediately instead of the empty state the user was already looking at.
    if (auto job = pipeline_.job(before)) select(job);
}

void App::open_images_dialog() {
    add_files(ask_for_files(kImageFilter, 1, true));
}

void App::open_gsi_dialog() {
    add_files(ask_for_files(kGsiFilter, 1, true));
}

void App::compare_images_dialog() {
    log("compare: choose the first image");
    auto a = ask_for_files(kImageFilter, 1, false);
    if (a.empty()) return;
    log("compare: choose the second image");
    auto b = ask_for_files(kImageFilter, 1, false);
    if (b.empty()) return;
    if (auto job = pipeline_.add_compare(a[0], b[0])) select(job);
}

void App::export_selected_as_png(float scale) {
    auto job = selected_;
    if (!job) return;
    // Exporting means decoding the compressed file, so it needs one to point
    // at: either the .gsi being viewed, or the one a compress just wrote.
    fs::path source;
    {
        std::lock_guard lk(job->m);
        source = job->kind == JobKind::Decode ? job->input : job->output_path;
    }
    if (source.empty() || !fs::exists(source)) {
        log("there is no compressed file to export from yet");
        return;
    }

    std::string name = source.filename().replace_extension("").string();
    if (scale != 1.f) {
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "_%gx", double(scale));
        name += suffix;
    }
    name += ".png";
    const fs::path out = ask_where_to_save(kPngFilter, 1, name, source.parent_path());
    if (out.empty()) return;

    DecodeRequest request;
    request.input = source;
    request.scale = scale;
    request.export_png = out;
    if (out.extension().empty()) request.export_png.replace_extension(".png");
    if (auto queued = pipeline_.add_decode(request)) select(queued);
}

void App::attach_original_to_selected() {
    auto job = selected_;
    if (!job || job->kind != JobKind::Decode) return;
    auto picked = ask_for_files(kImageFilter, 1, false);
    if (picked.empty()) return;
    job->reference = picked[0];
    // The same entry runs again with the original attached, rather than a
    // second copy of the same file appearing in the list.
    if (!pipeline_.requeue(job)) log("that file is busy; try again when it has finished");
}

void App::retry_selected() {
    auto job = selected_;
    if (!job) return;
    const JobStatus st = job->status;
    if (st != JobStatus::Failed && st != JobStatus::Cancelled && st != JobStatus::Done) return;
    if (pipeline_.requeue(job)) log(job->input.filename().string() + ": trying again");
}

// Everything finished, again, with whatever the Settings panel says now. This
// is the answer to picking the wrong preset for a folder of images: the
// alternative is adding them all a second time and then having two copies of
// each in the queue.
void App::run_everything_again() {
    int count = 0;
    for (const auto& j : pipeline_.jobs())
        if (pipeline_.requeue(j)) ++count;
    if (count > 0)
        log("running " + std::to_string(count) + " item" + (count == 1 ? "" : "s") +
            " again with the current settings");
}

// Raise the limit, then run this one again. The new value goes into the
// Settings panel rather than being applied invisibly to a single job: the user
// asked for more time because the last result was not good enough, and the
// next image they add almost certainly wants the same answer.
void App::run_selected_again_with(double time_budget_seconds) {
    auto job = selected_;
    if (!job) return;
    settings_.encode.time_budget_seconds = time_budget_seconds;
    push_options();
    if (!pipeline_.requeue(job)) return;
    char buf[160];
    if (time_budget_seconds > 0.0)
        std::snprintf(buf, sizeof(buf), "%s: running again with a %ds time limit",
                      job->input.filename().string().c_str(), int(time_budget_seconds + 0.5));
    else
        std::snprintf(buf, sizeof(buf), "%s: running again with no time limit",
                      job->input.filename().string().c_str());
    log(buf);
}

void App::remove_selected() {
    auto job = selected_;
    if (!job) return;
    const int index = pipeline_.index_of(job);
    if (auto removed = pipeline_.remove_job(job)) {
        release_textures(*removed);
        selected_.reset();
        auto jobs = pipeline_.jobs();
        if (!jobs.empty())
            select(jobs[size_t(std::clamp(index, 0, int(jobs.size()) - 1))]);
    }
}

void App::reveal_selected() {
    auto job = selected_;
    if (!job) return;
    fs::path target;
    {
        std::lock_guard lk(job->m);
        target = job->output_path.empty() ? job->input : job->output_path;
    }
    if (target.empty()) return;
    reveal_in_file_manager(target);
}

// The command line tool's `bench`, reached from the window: compress a fixed
// synthetic image and let the queue report what it cost. Someone deciding
// whether this machine is fast enough for their photographs should not have to
// find a terminal.
void App::run_speed_test() {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "gsic-speed-test";
    fs::create_directories(dir, ec);
    const fs::path src = dir / "speed-test.png";
    Image img(1024, 1024, 3);
    std::uint32_t state = 1;
    for (int c = 0; c < 3; ++c) {
        float* p = img.plane(c);
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x) {
                state = state * 1664525u + 1013904223u;
                p[size_t(y) * img.w + x] = 0.5f + 0.25f * std::sin(0.02f * (x + y * (c + 1))) +
                                           0.1f * (float(state >> 8) / float(1 << 24) - 0.5f);
            }
    }
    if (!img.save_png(src)) {
        log("the speed test could not write its scratch image");
        return;
    }
    log("speed test: compressing a 1024x1024 test image with the current settings");
    add_files({src});
}

// ------------------------------------------------------------- selection
void App::select(const std::shared_ptr<Job>& job) {
    if (selected_ == job) return;
    selected_ = job;
    reset_view();
}

void App::reset_view() {
    zoom_ = 0.f;
    pan_ = ImVec2(0, 0);
    split_ = 0.5f;
    dragging_split_ = false;
}

// --------------------------------------------------------------- textures
void App::upload_textures(Job& job) {
    std::lock_guard lk(job.m);
    auto upload = [&](unsigned& tex, std::vector<std::uint8_t>& rgba, bool& dirty, int w, int h) {
        // Each image is uploaded at its own size. Sharing one pair of
        // dimensions between the source and the reconstruction only holds
        // while they happen to match, and glTexImage2D reads w*h*4 bytes on
        // trust: the wrong pair here is an out-of-bounds read, not a stretched
        // picture.
        if (!dirty) return;
        // An entry that ran again and this time has nothing for this side --
        // a decode whose original could not be read -- must lose the picture
        // it had, not keep showing the previous one.
        if (rgba.empty() || w <= 0 || h <= 0) {
            if (tex) {
                released_textures_.push_back(tex);
                tex = 0;
                job.diff_dirty = true;
            }
            dirty = false;
            return;
        }
        if (rgba.size() < size_t(w) * size_t(h) * 4) return;
        if (!tex) glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        dirty = false;
        job.diff_dirty = true;
    };
    upload(job.tex_orig, job.orig_rgba, job.orig_dirty, job.orig_w, job.orig_h);
    upload(job.tex_recon, job.recon_rgba, job.recon_dirty, job.recon_w, job.recon_h);
}

// The difference view, built on the drawing thread from the two buffers the
// worker left behind.
//
// Compression error is a handful of levels per channel, which at 1x is a black
// rectangle. Amplifying it is the entire point of the view, so the gain is a
// control rather than a constant, and the numbers it measures on the way
// through are worth more than the picture to anyone tuning settings.
void App::rebuild_difference(Job& job) {
    const float gain = settings_.ui.difference_gain;
    if (!job.diff_dirty && job.tex_diff && job.diff_gain_built == gain) return;
    std::lock_guard lk(job.m);
    if (job.orig_w != job.recon_w || job.orig_h != job.recon_h || job.orig_w <= 0) return;
    const size_t n = size_t(job.orig_w) * size_t(job.orig_h) * 4;
    if (job.orig_rgba.size() < n || job.recon_rgba.size() < n) return;

    std::vector<std::uint8_t> diff(n);
    std::int64_t total = 0;
    int worst = 0;
    for (size_t i = 0; i < n; i += 4) {
        for (int c = 0; c < 3; ++c) {
            const int d = std::abs(int(job.orig_rgba[i + c]) - int(job.recon_rgba[i + c]));
            total += d;
            worst = std::max(worst, d);
            diff[i + c] = std::uint8_t(std::min(255.f, float(d) * gain));
        }
        diff[i + 3] = 255;
    }
    job.diff_max = double(worst);
    job.diff_mean = double(total) / double(std::max<size_t>(1, n / 4 * 3));

    if (!job.tex_diff) glGenTextures(1, &job.tex_diff);
    glBindTexture(GL_TEXTURE_2D, job.tex_diff);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, job.orig_w, job.orig_h, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, diff.data());
    job.diff_dirty = false;
    job.diff_gain_built = gain;
}

// Textures belong to the GL context, so they are deleted here, on the thread
// that holds it. Before this they were never deleted at all: clearing a long
// batch dropped every job and left its textures resident for the rest of the
// session, which on a run of large images is hundreds of megabytes of video
// memory the application had already finished with.
//
// The delete itself waits for the next frame. Removing an entry can happen
// after the panel that draws it has already emitted its commands -- a keyboard
// shortcut is handled at the end of the frame -- and those commands still name
// the texture. Deleting it before the frame is submitted would draw from an
// identifier that no longer exists.
void App::release_textures(Job& job) {
    for (unsigned* t : {&job.tex_orig, &job.tex_recon, &job.tex_diff}) {
        if (*t) released_textures_.push_back(*t);
        *t = 0;
    }
    job.diff_dirty = true;
}

void App::release_textures(const std::vector<std::shared_ptr<Job>>& jobs) {
    for (const auto& j : jobs)
        if (j) release_textures(*j);
}

void App::flush_released_textures() {
    if (released_textures_.empty()) return;
    glDeleteTextures(GLsizei(released_textures_.size()), released_textures_.data());
    released_textures_.clear();
}

// ------------------------------------------------------------------- dock
void App::setup_dock(ImGuiID dockspace) {
    ImGui::DockBuilderRemoveNode(dockspace);
    ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace, ImGui::GetMainViewport()->WorkSize);
    ImGuiID left, center, left_bottom, bottom;
    ImGui::DockBuilderSplitNode(dockspace, ImGuiDir_Left, 0.27f, &left, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.20f, &bottom, &center);
    ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, 0.50f, &left_bottom, &left);
    ImGui::DockBuilderDockWindow("Queue", left);
    // Tabbed together: the details of the selected file are read
    // occasionally, and giving them their own permanent strip would take room
    // from the picture, which is what the window is for.
    ImGui::DockBuilderDockWindow("Details", left_bottom);
    ImGui::DockBuilderDockWindow("Settings", left_bottom);
    ImGui::DockBuilderDockWindow("Preview", center);
    ImGui::DockBuilderDockWindow("Log", bottom);
    ImGui::DockBuilderFinish(dockspace);
}

// ------------------------------------------------------------------ queue
void App::draw_queue() {
    ImGui::Begin("Queue");

    if (ImGui::Button("Add Images...")) open_images_dialog();
    ImGui::SetItemTooltip("Pick images to compress (Ctrl+O)");
    ImGui::SameLine();
    if (ImGui::Button("Open .gsi...")) open_gsi_dialog();
    ImGui::SetItemTooltip("Open a compressed file to look at it (Ctrl+Shift+O)");

    auto jobs = pipeline_.jobs();
    const int active = int(std::count_if(jobs.begin(), jobs.end(), [](auto& j) {
        const JobStatus st = j->status;
        return st == JobStatus::Running || st == JobStatus::Queued;
    }));
    const int finished = int(jobs.size()) - active;

    ImGui::BeginDisabled(active == 0);
    if (ImGui::Button("Cancel All")) pipeline_.cancel_all();
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(finished == 0);
    if (ImGui::Button("Clear Finished")) {
        const auto removed = pipeline_.clear_finished();
        release_textures(removed);
        // The selection follows the list rather than an index into it: if what
        // was selected has just been cleared, fall back to whatever is left.
        const bool kept = std::any_of(removed.begin(), removed.end(),
                                      [&](const auto& j) { return j == selected_; });
        if (kept) {
            selected_.reset();
            auto rest = pipeline_.jobs();
            if (!rest.empty()) select(rest.front());
        }
        jobs = pipeline_.jobs();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(finished == 0);
    if (ImGui::Button("Run All Again")) run_everything_again();
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Do every finished item again with the settings as they are now.\n"
                          "Use this after changing the preset or the time limit.");

    if (jobs.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("Nothing queued.");
        ImGui::TextWrapped("Drag images here to compress them, or a .gsi to open it.");
        ImGui::End();
        return;
    }

    ImGui::Separator();
    if (active > 0)
        ImGui::TextDisabled("%d of %d finished, %d to go", finished, int(jobs.size()), active);
    else
        ImGui::TextDisabled("%d item%s", int(jobs.size()), jobs.size() == 1 ? "" : "s");
    ImGui::Separator();

    ImGui::BeginChild("queue-list", ImVec2(0, 0), ImGuiChildFlags_None);
    for (int i = 0; i < int(jobs.size()); ++i) {
        const std::shared_ptr<Job>& job_ptr = jobs[size_t(i)];
        Job& job = *job_ptr;
        ImGui::PushID(i);
        const JobStatus st = job.status;
        const ImVec4 color = st == JobStatus::Done       ? ImVec4(0.35f, 0.80f, 0.42f, 1)
                             : st == JobStatus::Running  ? ImVec4(0.36f, 0.62f, 1.00f, 1)
                             : st == JobStatus::Failed   ? ImVec4(0.95f, 0.42f, 0.38f, 1)
                             : st == JobStatus::Cancelled? ImVec4(0.65f, 0.55f, 0.35f, 1)
                                                         : ImVec4(0.55f, 0.55f, 0.58f, 1);
        const char* badge = st == JobStatus::Done       ? "done"
                            : st == JobStatus::Running  ? "busy"
                            : st == JobStatus::Failed   ? "fail"
                            : st == JobStatus::Cancelled? "stop"
                                                        : "wait";
        ImGui::TextColored(color, "[%s]", badge);
        ImGui::SameLine();

        std::string label = job.input.filename().string();
        if (job.kind == JobKind::Compare) label += "  vs  " + job.reference.filename().string();
        if (ImGui::Selectable(label.c_str(), selected_ == job_ptr,
                              ImGuiSelectableFlags_AllowOverlap))
            select(job_ptr);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s\n%s", job.input.string().c_str(), kind_label(job.kind));
        }

        // Everything a user might want to do with one entry, where they will
        // look for it.
        if (ImGui::BeginPopupContextItem("job-menu")) {
            select(job_ptr);
            const bool done = st == JobStatus::Done;
            if (ImGui::MenuItem("Export as PNG...", nullptr, false,
                                done && job.kind != JobKind::Compare))
                open_export_popup_ = true;
            if (ImGui::MenuItem("Compare with original...", nullptr, false,
                                done && job.kind == JobKind::Decode))
                attach_original_to_selected();
            if (ImGui::MenuItem("Show in folder", nullptr, false, true)) reveal_selected();
            ImGui::Separator();
            if (ImGui::MenuItem("Run again", nullptr, false,
                                st != JobStatus::Running && st != JobStatus::Queued))
                retry_selected();
            if (ImGui::MenuItem("Cancel", nullptr, false,
                                st == JobStatus::Running || st == JobStatus::Queued))
                pipeline_.cancel_job(job_ptr);
            if (ImGui::MenuItem("Remove", "Del", false, st != JobStatus::Running))
                remove_selected();
            ImGui::EndPopup();
        }

        if (st == JobStatus::Running) {
            ImGui::SameLine();
            const double total = job.estimated_total_seconds.load(std::memory_order_relaxed);
            const double done = job.elapsed_seconds.load(std::memory_order_relaxed);
            if (total > done + 1.0) {
                const std::string left = human_duration(total - done);
                ImGui::ProgressBar(job.progress, ImVec2(120, 0), left.c_str());
            } else {
                ImGui::ProgressBar(job.progress, ImVec2(120, 0));
            }
        } else if (st == JobStatus::Done) {
            std::lock_guard jl(job.m);
            ImGui::SameLine();
            if (job.kind == JobKind::Compress && job.has_stats)
                ImGui::TextDisabled("%s | %.1f dB", human_size(job.stats.file_bytes).c_str(),
                                    job.stats.psnr);
            else if (job.kind == JobKind::Decode && job.has_info)
                ImGui::TextDisabled("%dx%d | %s", job.info.width, job.info.height,
                                    human_size(job.info.file_bytes).c_str());
            else if (job.has_stats)
                ImGui::TextDisabled("%.1f dB", job.stats.psnr);
            else
                ImGui::TextDisabled(" ");
        } else if (st == JobStatus::Failed) {
            std::lock_guard jl(job.m);
            ImGui::SameLine();
            ImGui::TextColored(color, "%s", job.error.c_str());
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::End();
}

// --------------------------------------------------------------- settings
void App::draw_settings() {
    ImGui::Begin("Settings");
    EncodeOptions& o = settings_.encode;
    bool changed = false;

    const char* presets[] = {"Fast", "Balanced", "High quality", "Custom"};
    const int previous_preset = o.preset;
    if (ImGui::Combo("Preset", &o.preset, presets, 4)) {
        changed = true;
        // Switching to Custom starts from whatever the last preset actually
        // was, so the sliders open where the user left off rather than at a
        // pair of numbers they never chose.
        if (o.preset == 3 && previous_preset != 3) {
            const EncodeOptions from_preset = [&] {
                EncodeOptions e = o;
                e.preset = previous_preset;
                return e;
            }();
            const EncodeSettings s = from_preset.to_encode_settings();
            o.pixels_per_gaussian = s.pixels_per_gaussian;
            o.steps = s.max_steps;
        }
    }
    ImGui::SetItemTooltip(
        "Fast is quickest, High quality spends longer for a bit more detail.\n"
        "Whatever you pick, the time limit below still applies.");
    if (o.preset == 3) {
        changed |= ImGui::SliderInt("Pixels / gaussian", &o.pixels_per_gaussian,
                                    kMinPixelsPerGaussian, kMaxPixelsPerGaussian, "%d",
                                    ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Fewer pixels per gaussian means more gaussians: "
                              "more detail and a larger file.");
        changed |= ImGui::SliderInt("Steps", &o.steps, kMinCustomSteps, kMaxCustomSteps, "%d",
                                    ImGuiSliderFlags_Logarithmic);
        int count = o.num_gaussians;
        // The upper bound is the build's own ceiling, so a 32-bit build does
        // not offer a number it would then quietly clamp.
        if (ImGui::DragInt("Gaussian count", &count, 50.f, 0, kMaxGaussians,
                           count == 0 ? "from the density above" : "%d")) {
            o.num_gaussians = std::max(0, count);
            changed = true;
        }
        ImGui::SetItemTooltip("Set this to use an exact number instead of deriving it from\n"
                              "the image size. Zero derives it.");
        int seed = int(o.seed);
        if (ImGui::DragInt("Seed", &seed, 1.f, 0, 1'000'000)) {
            o.seed = unsigned(std::max(0, seed));
            changed = true;
        }
        ImGui::SetItemTooltip("Where the first gaussians are placed. The same seed and the\n"
                              "same settings give the same file every time.");
    }

    const char* bit_names[] = {"Best quality", "Smaller file", "Smallest file", "Custom"};
    changed |= ImGui::Combo("Precision", &o.precision, bit_names, 4);
    ImGui::SetItemTooltip("How finely each gaussian is stored. Lower precision is a\n"
                          "smaller file at some cost in quality.");
    if (o.precision == 3) {
        changed |= ImGui::SliderInt("Position bits", &o.custom_quant.pos, 4, 16);
        changed |= ImGui::SliderInt("Scale bits", &o.custom_quant.scale, 4, 16);
        changed |= ImGui::SliderInt("Rotation bits", &o.custom_quant.rot, 4, 16);
        changed |= ImGui::SliderInt("Colour bits", &o.custom_quant.feat, 4, 16);
    }

    const char* backends[] = {"Auto (GPU if available)", "CPU", "GPU"};
    changed |= ImGui::Combo("Backend", &o.backend, backends, 3);
    if (!gpu_ok_ && o.backend == 2)
        ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.35f, 1),
                           "No usable GPU here, so the CPU will do the work.");

    ImGui::Separator();

    // A time limit rather than a step count, because the step count that is
    // pleasant on one machine is a several-minute wait on another with a
    // bigger image. Quality against steps flattens out hard, so the tail this
    // gives up is small and the wait it removes is not.
    int budget = int(o.time_budget_seconds + 0.5);
    if (ImGui::SliderInt("Time limit", &budget, 0, kMaxTimeBudgetSeconds,
                         budget == 0 ? "no limit" : "%d s")) {
        o.time_budget_seconds = double(budget);
        changed = true;
    }
    ImGui::SetItemTooltip(
        "Large images stop early to finish within this. Most pictures reach\n"
        "their useful quality well before the limit. Set it to 0 to let every\n"
        "encode run to the end, however long that takes.");

    ImGui::Separator();
    changed |= ImGui::Checkbox("Save next to the original", &o.save_next_to_input);
    if (!o.save_next_to_input) {
        changed |= ImGui::InputTextWithHint("##outdir", "output folder", out_dir_,
                                            sizeof(out_dir_));
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            nfdu8char_t* p = nullptr;
            nfdpickfolderu8args_t args{};
            if (NFD_PickFolderU8_With(&p, &args) == NFD_OKAY && p) {
                std::snprintf(out_dir_, sizeof(out_dir_), "%s", reinterpret_cast<char*>(p));
                NFD_FreePathU8(p);
                changed = true;
            }
        }
    }
    changed |= ImGui::Checkbox("Also write a decoded .png", &o.export_png);
    ImGui::SetItemTooltip("Writes an ordinary .png beside each .gsi, for programs\n"
                          "that cannot open a .gsi.");

    if (changed) push_options();

    ImGui::Separator();
    ImGui::TextDisabled("Compute: %s", gpu_status_.c_str());
    if (!gpu_ok_ && !gpu_reason_.empty() && ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", gpu_reason_.c_str());
    ImGui::TextDisabled("Settings are remembered between launches.");
    ImGui::End();
}

// ---------------------------------------------------------------- details
//
// Everything `gsic info` prints, and everything a compress produced, for
// whatever is selected. The window used to show a single dense line above the
// picture and nothing else, which meant the file's own properties -- how many
// gaussians, at what precision, how many bits per pixel -- were only reachable
// from a terminal.
void App::draw_details() {
    if (!ImGui::Begin("Details")) {
        ImGui::End();
        return;
    }
    auto job = selected_;
    if (!job) {
        ImGui::TextDisabled("Nothing selected.");
        ImGui::End();
        return;
    }

    const auto row = [](const char* label, const std::string& value) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", label);
        ImGui::TableSetColumnIndex(1);
        // Wrapped rather than clipped: the one value here that does not fit is
        // the output path, which is exactly the value someone opens this panel
        // to read.
        ImGui::TextWrapped("%s", value.c_str());
    };
    char buf[256];
    std::lock_guard lk(job->m);

    if (ImGui::BeginTable("details", 2, ImGuiTableFlags_SizingStretchProp)) {
        // The label column takes what it needs and no more; letting it stretch
        // proportionally truncated "Gaussians" to "Gauss".
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
        row("File", job->input.filename().string());
        row("Kind", kind_label(job->kind));
        if (job->kind == JobKind::Compare) row("Against", job->reference.filename().string());

        if (job->has_info) {
            std::snprintf(buf, sizeof(buf), "%d x %d, %d channel(s)", job->info.width,
                          job->info.height, job->info.channels);
            row("Image", buf);
            std::snprintf(buf, sizeof(buf), "%d", job->info.gaussians);
            row("Gaussians", buf);
            std::snprintf(buf, sizeof(buf), "pos %d, scale %d, rot %d, colour %d",
                          job->info.quant.pos, job->info.quant.scale, job->info.quant.rot,
                          job->info.quant.feat);
            row("Bits", buf);
            std::snprintf(buf, sizeof(buf), "%s  (%.3f bpp, %.1f%% of raw)",
                          human_size(job->info.file_bytes).c_str(), job->info.bpp,
                          job->info.percent_of_raw);
            row("Compressed", buf);
        } else if (job->orig_w > 0) {
            std::snprintf(buf, sizeof(buf), "%d x %d, %d channel(s)", job->orig_w, job->orig_h,
                          job->channels);
            row("Image", buf);
        }

        if (job->has_stats) {
            std::snprintf(buf, sizeof(buf), "%.2f dB", job->stats.psnr);
            row("PSNR", buf);
            std::snprintf(buf, sizeof(buf), "%.4f", job->stats.ssim);
            row("SSIM", buf);
        }
        if (job->kind == JobKind::Compress && job->has_stats) {
            std::snprintf(buf, sizeof(buf), "%d of %d", job->stats.steps_run,
                          job->stats.steps_requested);
            row("Steps", buf);
            std::snprintf(buf, sizeof(buf), "%.1f s on the %s", job->stats.encode_seconds,
                          job->stats.backend_used);
            row("Encoded in", buf);
            if (!job->stats.gpu_fallback_reason.empty())
                row("Note", "the GPU " + job->stats.gpu_fallback_reason +
                                ", so the CPU did the work");
        }
        if (!job->output_path.empty()) row("Written to", job->output_path.string());
        if (!job->error.empty()) row("Problem", job->error);
        ImGui::EndTable();
    }

    if (job->kind == JobKind::Compress && job->has_stats) {
        ImGui::Separator();
        ImGui::TextWrapped("PSNR above about 35 dB is usually indistinguishable from the "
                           "original at normal viewing size.");
    }
    ImGui::End();
}

// ---------------------------------------------------------------- preview
void App::draw_preview() {
    ImGui::Begin("Preview");
    auto job_ptr = selected_;
    if (!job_ptr) {
        // The empty state has to say what the application is for and what to
        // do next. "Add images to the queue to get started" said neither that
        // compressed files could be opened nor that anything could be dropped.
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        ImGui::Dummy(ImVec2(0, avail.y * 0.3f));
        const char* lines[] = {
            "Drop an image here to compress it.",
            "Drop a .gsi here to look at one you compressed earlier.",
        };
        for (const char* line : lines) {
            const float w = ImGui::CalcTextSize(line).x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.f, (avail.x - w) * 0.5f));
            ImGui::TextDisabled("%s", line);
        }
        ImGui::Dummy(ImVec2(0, 12));
        const float button_w = 260.f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.f, (avail.x - button_w) * 0.5f));
        if (ImGui::Button("Add Images...", ImVec2(125, 0))) open_images_dialog();
        ImGui::SameLine();
        if (ImGui::Button("Open .gsi...", ImVec2(125, 0))) open_gsi_dialog();
        ImGui::End();
        return;
    }
    Job& job = *job_ptr;
    upload_textures(job);

    // What is on each side of the divider. A compress has the original on the
    // left and its result on the right; a decoded .gsi has only a result until
    // an original is attached; a comparison has one image on each side.
    unsigned tex_left = 0, tex_right = 0;
    int view_w = 0, view_h = 0;
    bool both = false;
    const char* left_label = "original";
    const char* right_label = "compressed";
    {
        std::lock_guard lk(job.m);
        tex_left = job.tex_orig;
        tex_right = job.tex_recon;
        both = tex_left != 0 && tex_right != 0 && job.orig_w == job.recon_w &&
               job.orig_h == job.recon_h;
        view_w = tex_right ? job.recon_w : job.orig_w;
        view_h = tex_right ? job.recon_h : job.orig_h;
        if (job.kind == JobKind::Compare) {
            left_label = "A";
            right_label = "B";
        } else if (job.kind == JobKind::Decode) {
            right_label = "decoded";
        }
    }

    // --- header
    {
        std::lock_guard lk(job.m);
        ImGui::TextUnformatted(job.input.filename().string().c_str());
        if (!job.error.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.42f, 0.38f, 1), "- %s", job.error.c_str());
        } else if (job.has_stats) {
            ImGui::SameLine();
            if (job.kind == JobKind::Compress)
                ImGui::TextDisabled("%s -> %s (%.1f%%)   PSNR %.2f dB   SSIM %.4f   %.1fs [%s]",
                                    human_size(job.stats.source_bytes).c_str(),
                                    human_size(job.stats.file_bytes).c_str(),
                                    100.0 * double(job.stats.file_bytes) /
                                        double(std::max<std::int64_t>(1, job.stats.source_bytes)),
                                    job.stats.psnr, job.stats.ssim, job.stats.encode_seconds,
                                    job.stats.backend_used);
            else
                ImGui::TextDisabled("PSNR %.2f dB   SSIM %.4f", job.stats.psnr, job.stats.ssim);
        } else if (job.has_info) {
            ImGui::SameLine();
            ImGui::TextDisabled("%d x %d   %d gaussians   %s   %.3f bpp", job.info.width,
                                job.info.height, job.info.gaussians,
                                human_size(job.info.file_bytes).c_str(), job.info.bpp);
        } else if (view_w > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("%d x %d", view_w, view_h);
        }
    }
    if (job.status == JobStatus::Running) {
        ImGui::ProgressBar(job.progress, ImVec2(200, 0));
        const double total = job.estimated_total_seconds.load(std::memory_order_relaxed);
        const double done = job.elapsed_seconds.load(std::memory_order_relaxed);
        if (total > done + 1.0) {
            ImGui::SameLine();
            ImGui::TextDisabled("about %s left", human_duration(total - done).c_str());
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Cancel")) pipeline_.cancel_job(job_ptr);
    }

    // --- the run was cut short
    //
    // A time limit that turned out to be too small produces a result that
    // looks like the compressor is simply not very good. The log said so, but
    // the log is at the bottom of the window and scrolls, and "raise it in
    // Settings and add the image again" is four steps away from the place the
    // disappointing picture is. So the picture says it, and offers the fix.
    {
        bool cut_short = false;
        int ran = 0, asked = 0;
        double used = 0;
        {
            std::lock_guard lk(job.m);
            cut_short = job.was_cut_short();
            ran = job.stats.steps_run;
            asked = job.stats.steps_requested;
            used = job.used_time_budget;
        }
        if (cut_short) {
            ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.35f, 1),
                               "Stopped at %d of %d steps to stay inside the %d s limit.", ran,
                               asked, int(used + 0.5));
            ImGui::SameLine();
            const double more = next_time_budget_after(used);
            char label[64];
            if (more > 0.0)
                std::snprintf(label, sizeof(label), "Give it %d s and run again",
                              int(more + 0.5));
            else
                std::snprintf(label, sizeof(label), "Run again with no limit");
            if (ImGui::SmallButton(label)) run_selected_again_with(more);
            ImGui::SetItemTooltip("Raises the time limit in Settings and compresses this "
                                  "image again.");
        }
    }

    // --- toolbar
    const bool can_export =
        job.status == JobStatus::Done && job.kind != JobKind::Compare;
    ImGui::BeginDisabled(!can_export);
    if (ImGui::SmallButton("Export PNG...")) open_export_popup_ = true;
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Write an ordinary .png, optionally at a different size (Ctrl+E)");
    ImGui::SameLine();
    if (ImGui::SmallButton("Show in folder")) reveal_selected();
    ImGui::SameLine();
    // Change a setting, press this. Without it, trying a different preset on
    // an image already in the queue meant adding the same file a second time
    // and then living with two entries for it.
    ImGui::BeginDisabled(job.status == JobStatus::Running || job.status == JobStatus::Queued);
    if (ImGui::SmallButton("Run again")) retry_selected();
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Do this one again with the settings as they are now (Ctrl+R)");
    if (job.kind == JobKind::Decode && job.status == JobStatus::Done) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Compare with original...")) attach_original_to_selected();
        ImGui::SetItemTooltip("Pick the picture this was made from, to see what it cost");
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140);
    const char* modes[] = {"Split", left_label, right_label, "Difference"};
    // With only one picture there is nothing to compare, so the control shows
    // what is actually on screen and is not offered. Writing that back into
    // the saved preference would silently discard the user's real choice the
    // next time they open something with two sides.
    int shown_mode = both ? settings_.ui.view_mode : (tex_right ? 2 : 1);
    ImGui::BeginDisabled(!both);
    if (ImGui::Combo("##view", &shown_mode, modes, 4)) {
        settings_.ui.view_mode = shown_mode;
        push_options();
    }
    ImGui::EndDisabled();
    if (both && settings_.ui.view_mode == 3) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        if (ImGui::SliderFloat("gain", &settings_.ui.difference_gain, kMinDifferenceGain,
                               kMaxDifferenceGain, "%.0fx", ImGuiSliderFlags_Logarithmic))
            push_options();
        ImGui::SameLine();
        ImGui::TextDisabled("max %.0f, mean %.2f of 255", job.diff_max, job.diff_mean);
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (ImGui::SmallButton("Fit")) reset_view();
    ImGui::SameLine();
    if (ImGui::SmallButton("1:1")) {
        zoom_ = 1.f;
        pan_ = ImVec2(0, 0);
    }

    if (!tex_left && !tex_right) {
        ImGui::Dummy(ImVec2(0, 8));
        ImGui::TextDisabled("%s", job.status == JobStatus::Failed ? "Nothing to show."
                                                                  : "Loading...");
        ImGui::End();
        return;
    }
    if (view_w <= 0 || view_h <= 0) {
        ImGui::End();
        return;
    }
    if (both && settings_.ui.view_mode == 3) rebuild_difference(job);

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 32 || avail.y < 32) {
        ImGui::End();
        return;
    }
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(origin, ImVec2(origin.x + avail.x, origin.y + avail.y),
                      IM_COL32(24, 24, 27, 255));

    // Fit-to-pane scale, optional zoom and pan.
    const float fit = std::min(avail.x / float(view_w), avail.y / float(view_h));
    const float scale = zoom_ <= 0.f ? fit : zoom_;
    const ImVec2 img_size(view_w * scale, view_h * scale);
    const ImVec2 img_pos(origin.x + (avail.x - img_size.x) * 0.5f + pan_.x,
                         origin.y + (avail.y - img_size.y) * 0.5f + pan_.y);
    const ImVec2 img_end(img_pos.x + img_size.x, img_pos.y + img_size.y);

    // Interaction: wheel zooms around the cursor, drag pans, the divider
    // drags the A/B split, double-click resets.
    ImGui::InvisibleButton("preview", avail, ImGuiButtonFlags_MouseButtonLeft);
    const bool hovered = ImGui::IsItemHovered();
    const bool splitting = both && settings_.ui.view_mode == 0;
    const float split_x = img_pos.x + img_size.x * split_;
    const bool near_split =
        hovered && splitting && std::fabs(ImGui::GetIO().MousePos.x - split_x) < 8.f;
    if (ImGui::IsItemActivated() && near_split) dragging_split_ = true;
    if (!ImGui::IsItemActive()) dragging_split_ = false;
    if (ImGui::IsItemActive()) {
        if (dragging_split_) {
            split_ = std::clamp((ImGui::GetIO().MousePos.x - img_pos.x) / img_size.x, 0.f, 1.f);
        } else {
            pan_.x += ImGui::GetIO().MouseDelta.x;
            pan_.y += ImGui::GetIO().MouseDelta.y;
        }
    }
    if (near_split || dragging_split_) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    if (hovered && ImGui::GetIO().MouseWheel != 0.f) {
        const float old_scale = scale;
        float target = old_scale * (ImGui::GetIO().MouseWheel > 0 ? 1.25f : 0.8f);
        target = std::clamp(target, fit * 0.25f, 32.f);
        // Keep the pixel under the cursor fixed while zooming.
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        pan_.x = mouse.x - (mouse.x - pan_.x - (origin.x + (avail.x - view_w * target) * 0.5f)) *
                     (target / old_scale) - (origin.x + (avail.x - view_w * target) * 0.5f);
        pan_.y = mouse.y - (mouse.y - pan_.y - (origin.y + (avail.y - view_h * target) * 0.5f)) *
                     (target / old_scale) - (origin.y + (avail.y - view_h * target) * 0.5f);
        zoom_ = target;
    }
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) reset_view();

    dl->PushClipRect(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), true);
    const int mode = both ? settings_.ui.view_mode : (tex_right ? 2 : 1);
    unsigned tex_diff = 0;
    {
        std::lock_guard lk(job.m);
        tex_diff = job.tex_diff;
    }
    if (mode == 3 && tex_diff) {
        dl->AddImage(ImTextureID(std::intptr_t(tex_diff)), img_pos, img_end);
    } else if (mode == 1 && tex_left) {
        dl->AddImage(ImTextureID(std::intptr_t(tex_left)), img_pos, img_end);
    } else if (mode == 2 && tex_right) {
        dl->AddImage(ImTextureID(std::intptr_t(tex_right)), img_pos, img_end);
    } else {
        if (tex_right) dl->AddImage(ImTextureID(std::intptr_t(tex_right)), img_pos, img_end);
        if (tex_left) {
            const float cx = std::clamp(split_x, img_pos.x, img_end.x);
            dl->PushClipRect(img_pos, ImVec2(cx, img_end.y), true);
            dl->AddImage(ImTextureID(std::intptr_t(tex_left)), img_pos, img_end);
            dl->PopClipRect();
        }
        if (both) {
            dl->AddLine(ImVec2(split_x, std::max(img_pos.y, origin.y)),
                        ImVec2(split_x, std::min(img_end.y, origin.y + avail.y)),
                        near_split || dragging_split_ ? IM_COL32(255, 255, 255, 230)
                                                      : IM_COL32(255, 255, 255, 130),
                        2.f);
            dl->AddText(ImVec2(std::max(img_pos.x, origin.x) + 6, origin.y + 4),
                        IM_COL32(255, 255, 255, 160), left_label);
            dl->AddText(ImVec2(std::min(split_x + 8, origin.x + avail.x - 90), origin.y + 4),
                        IM_COL32(255, 255, 255, 160), right_label);
        }
    }

    // A zoom readout, because "why does this look soft" is almost always "it
    // is being shown at 40%", and nothing on screen used to say so.
    char zoom_text[64];
    std::snprintf(zoom_text, sizeof(zoom_text), "%d x %d  at %.0f%%", view_w, view_h,
                  double(scale) * 100.0);
    const ImVec2 text_size = ImGui::CalcTextSize(zoom_text);
    const ImVec2 badge(origin.x + avail.x - text_size.x - 12, origin.y + avail.y - text_size.y - 8);
    dl->AddRectFilled(ImVec2(badge.x - 6, badge.y - 3),
                      ImVec2(badge.x + text_size.x + 6, badge.y + text_size.y + 3),
                      IM_COL32(0, 0, 0, 140), 4.f);
    dl->AddText(badge, IM_COL32(230, 230, 235, 200), zoom_text);
    dl->PopClipRect();
    ImGui::End();
}

// -------------------------------------------------------------------- log
void App::draw_log() {
    ImGui::Begin("Log");
    if (ImGui::SmallButton("Copy")) {
        std::string all;
        {
            std::lock_guard lk(log_m_);
            for (const auto& line : log_lines_) {
                all += line;
                all += '\n';
            }
        }
        ImGui::SetClipboardText(all.c_str());
    }
    ImGui::SetItemTooltip("Copy every line, for a bug report");
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
        std::lock_guard lk(log_m_);
        log_lines_.clear();
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Follow", &settings_.ui.log_autoscroll)) push_options();
    ImGui::Separator();

    ImGui::BeginChild("log-lines", ImVec2(0, 0), ImGuiChildFlags_None,
                      ImGuiWindowFlags_HorizontalScrollbar);
    {
        std::lock_guard lk(log_m_);
        for (const auto& line : log_lines_) ImGui::TextUnformatted(line.c_str());
    }
    if (settings_.ui.log_autoscroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4)
        ImGui::SetScrollHereY(1.f);
    ImGui::EndChild();
    ImGui::End();
}

// ------------------------------------------------------------------ modals
void App::draw_export_popup() {
    if (open_export_popup_) {
        ImGui::OpenPopup("Export as PNG");
        open_export_popup_ = false;
    }
    if (!ImGui::BeginPopupModal("Export as PNG", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return;

    auto job = selected_;
    int base_w = 0, base_h = 0;
    if (job) {
        std::lock_guard lk(job->m);
        base_w = job->has_info ? job->info.width : job->recon_w;
        base_h = job->has_info ? job->info.height : job->recon_h;
    }

    ImGui::TextWrapped(
        "The compressed representation is continuous, so a larger size is rendered "
        "from the gaussians rather than by enlarging a decoded image.");
    ImGui::Separator();

    const float presets[] = {0.5f, 1.f, 2.f, 4.f};
    const char* names[] = {"0.5x", "1x", "2x", "4x"};
    for (int i = 0; i < 4; ++i) {
        if (i) ImGui::SameLine();
        if (ImGui::RadioButton(names[i], export_scale_ == presets[i])) export_scale_ = presets[i];
    }
    ImGui::SetNextItemWidth(200);
    ImGui::SliderFloat("Scale", &export_scale_, kMinDecodeScale, kMaxDecodeScale, "%.2fx",
                       ImGuiSliderFlags_Logarithmic);
    export_scale_ = std::clamp(export_scale_, kMinDecodeScale, kMaxDecodeScale);

    if (base_w > 0)
        ImGui::TextDisabled("Output: %d x %d", std::max(1, int(base_w * export_scale_ + 0.5f)),
                            std::max(1, int(base_h * export_scale_ + 0.5f)));

    ImGui::Separator();
    if (ImGui::Button("Choose file and export", ImVec2(200, 0))) {
        ImGui::CloseCurrentPopup();
        export_selected_as_png(export_scale_);
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(100, 0))) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

void App::draw_quit_popup(bool& want_close) {
    if (open_quit_popup_) {
        ImGui::OpenPopup("Still working");
        open_quit_popup_ = false;
    }
    if (!ImGui::BeginPopupModal("Still working", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return;
    ImGui::TextWrapped("An image is still being compressed. Closing now throws that work away.");
    ImGui::Separator();
    if (ImGui::Button("Keep working", ImVec2(140, 0))) ImGui::CloseCurrentPopup();
    ImGui::SameLine();
    if (ImGui::Button("Close anyway", ImVec2(140, 0))) {
        confirmed_quit_ = true;
        want_close = true;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

std::string App::diagnostics_report() const {
    char buf[512];
    std::string out;
    const auto line = [&](const char* fmt, auto... args) {
        std::snprintf(buf, sizeof(buf), fmt, args...);
        out += buf;
        out += '\n';
    };
    line("gsic %s", GSIC_VERSION);
    line("build          %d-bit, %s", int(sizeof(void*) * 8),
         kIs32Bit ? "32-bit limits" : "64-bit limits");
    line("cpu kernels    %s", kernels().name);
    line("threads        %u", std::thread::hardware_concurrency());
    line("image limits   %lld pixels, %d per side, %d gaussians",
         static_cast<long long>(kMaxPixels), kMaxDimension, kMaxGaussians);
    if (gpu_ok_)
        line("gpu            available, and it agreed with the CPU on the self-check");
    else
        line("gpu            NOT USED: %s", gpu_reason_.c_str());
    const auto* renderer = glGetString(GL_RENDERER);
    const auto* version = glGetString(GL_VERSION);
    line("opengl         %s on %s", version ? reinterpret_cast<const char*>(version) : "?",
         renderer ? reinterpret_cast<const char*>(renderer) : "?");
    return out;
}

void App::draw_diagnostics() {
    if (!show_diagnostics_) return;
    ImGui::SetNextWindowSize(ImVec2(620, 340), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Diagnostics", &show_diagnostics_)) {
        ImGui::End();
        return;
    }
    ImGui::TextWrapped(
        "What this machine will do, and what to paste into a bug report. These facts "
        "vary per machine and none of them are visible from the outside.");
    ImGui::Separator();
    const std::string report = diagnostics_report();
    ImGui::TextUnformatted(report.c_str());
    ImGui::Separator();
    if (ImGui::Button("Copy report")) ImGui::SetClipboardText(report.c_str());
    ImGui::SameLine();
    if (ImGui::Button("Run speed test")) run_speed_test();
    ImGui::SetItemTooltip("Compresses a 1024x1024 test image with the current settings\n"
                          "and reports what it cost, in the queue.");
    ImGui::End();
}

void App::draw_about() {
    if (!show_about_) return;
    ImGui::SetNextWindowSize(ImVec2(520, 0), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("About gsic", &show_about_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }
    ImGui::Text("gsic %s", GSIC_VERSION);
    ImGui::TextDisabled("Gaussian splat image compressor");
    ImGui::Separator();
    ImGui::TextWrapped(
        "Compresses an image by fitting a cloud of anisotropic 2D gaussians to it, and "
        "stores the result as a .gsi. Because that representation is continuous rather "
        "than a fixed grid of pixels, a .gsi can be decoded at any size.");
    ImGui::Separator();
    ImGui::TextDisabled("Based on Image-GS (NYU ICL / Intel / AMD).");
    ImGui::TextDisabled("MIT licensed. Uses Dear ImGui, GLFW, glad, stb, zstd and "
                        "nativefiledialog-extended.");
    ImGui::TextDisabled("This application makes no network connections.");
    if (!settings_file_.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("Settings: %s", settings_file_.string().c_str());
    }
    ImGui::End();
}

// ------------------------------------------------------------------- menu
void App::draw_menu(bool& want_close) {
    if (!ImGui::BeginMainMenuBar()) return;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Add Images...", "Ctrl+O")) open_images_dialog();
        if (ImGui::MenuItem("Open .gsi...", "Ctrl+Shift+O")) open_gsi_dialog();
        if (ImGui::MenuItem("Compare Two Images...")) compare_images_dialog();
        ImGui::Separator();
        const bool has_selection = selected_ != nullptr;
        const bool done = has_selection && selected_->status == JobStatus::Done;
        if (ImGui::MenuItem("Export as PNG...", "Ctrl+E", false,
                            done && selected_->kind != JobKind::Compare))
            open_export_popup_ = true;
        if (ImGui::MenuItem("Show in Folder", nullptr, false, has_selection)) reveal_selected();
        ImGui::Separator();
        if (ImGui::MenuItem("Run Again", "Ctrl+R", false,
                            has_selection && selected_->status != JobStatus::Running &&
                                selected_->status != JobStatus::Queued))
            retry_selected();
        if (ImGui::MenuItem("Run Everything Again", nullptr, false, pipeline_.job_count() > 0))
            run_everything_again();
        if (ImGui::MenuItem("Remove from Queue", "Del", false, has_selection)) remove_selected();
        ImGui::Separator();
        if (ImGui::MenuItem("Exit", "Alt+F4")) want_close = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        const bool has_selection = selected_ != nullptr;
        if (ImGui::MenuItem("Fit to window", "F", false, has_selection)) reset_view();
        if (ImGui::MenuItem("Actual size", "1", false, has_selection)) {
            zoom_ = 1.f;
            pan_ = ImVec2(0, 0);
        }
        ImGui::Separator();
        const char* modes[] = {"Split", "Left only", "Right only", "Difference"};
        for (int i = 0; i < 4; ++i)
            if (ImGui::MenuItem(modes[i], nullptr, settings_.ui.view_mode == i)) {
                settings_.ui.view_mode = i;
                push_options();
            }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Diagnostics...")) show_diagnostics_ = true;
        if (ImGui::MenuItem("About gsic...")) show_about_ = true;
        ImGui::EndMenu();
    }

    // A live summary on the right of the bar, so progress is visible whichever
    // panel has the user's attention.
    if (pipeline_.busy()) {
        const auto jobs = pipeline_.jobs();
        const int left = int(std::count_if(jobs.begin(), jobs.end(), [](auto& j) {
            const JobStatus st = j->status;
            return st == JobStatus::Running || st == JobStatus::Queued;
        }));
        char text[64];
        std::snprintf(text, sizeof(text), "working, %d left", left);
        const float w = ImGui::CalcTextSize(text).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - w - 16);
        ImGui::TextColored(ImVec4(0.36f, 0.62f, 1.f, 1), "%s", text);
    }
    ImGui::EndMainMenuBar();
}

void App::handle_shortcuts(bool& want_close) {
    // Not while a text field has the keyboard, or "1" typed into the output
    // folder box would resize the picture; and not while a modal is up, where
    // Delete belongs to the dialog and not to the queue behind it.
    if (ImGui::GetIO().WantTextInput) return;
    if (ImGui::GetTopMostPopupModal() != nullptr) return;
    const auto pressed = [](ImGuiKeyChord chord) { return ImGui::IsKeyChordPressed(chord); };
    if (pressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_O)) open_gsi_dialog();
    else if (pressed(ImGuiMod_Ctrl | ImGuiKey_O)) open_images_dialog();
    else if (pressed(ImGuiMod_Ctrl | ImGuiKey_E)) {
        if (selected_ && selected_->status == JobStatus::Done &&
            selected_->kind != JobKind::Compare)
            open_export_popup_ = true;
    } else if (pressed(ImGuiMod_Ctrl | ImGuiKey_R)) retry_selected();
    else if (pressed(ImGuiMod_Ctrl | ImGuiKey_Q)) want_close = true;
    else if (pressed(ImGuiKey_Delete)) remove_selected();
    else if (pressed(ImGuiKey_Escape)) {
        if (selected_) pipeline_.cancel_job(selected_);
    } else if (pressed(ImGuiKey_F)) reset_view();
    else if (pressed(ImGuiKey_1)) {
        zoom_ = 1.f;
        pan_ = ImVec2(0, 0);
    } else if (pressed(ImGuiKey_DownArrow) || pressed(ImGuiKey_UpArrow)) {
        const auto jobs = pipeline_.jobs();
        if (!jobs.empty()) {
            const int at = pipeline_.index_of(selected_);
            const int step = ImGui::IsKeyChordPressed(ImGuiKey_DownArrow) ? 1 : -1;
            const int next = std::clamp(at < 0 ? 0 : at + step, 0, int(jobs.size()) - 1);
            select(jobs[size_t(next)]);
        }
    }
}

void App::draw_ui(bool& want_close) {
    // Anything freed during the previous frame is safe to delete now that the
    // frame naming it has been submitted.
    flush_released_textures();
    draw_menu(want_close);

    const ImGuiID dockspace = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
    // Build the layout until it takes. Retrying is deliberate: attempting
    // once and giving up would leave the window permanently empty if the
    // split ever failed, whereas this recovers on the next frame.
    const ImGuiDockNode* root = ImGui::DockBuilderGetNode(dockspace);
    if (!root || !root->IsSplitNode()) setup_dock(dockspace);

    draw_queue();
    draw_settings();
    draw_details();
    draw_preview();
    draw_log();
    draw_export_popup();
    draw_quit_popup(want_close);
    draw_diagnostics();
    draw_about();
    handle_shortcuts(want_close);

    // The settings file is written once things have settled rather than on
    // every change, so dragging a slider is not a burst of disk writes.
    if (settings_dirty_ && ImGui::GetTime() - settings_dirty_at_ > 1.0) persist_settings();
}

// ------------------------------------------------------------ screenshot
//
// The back buffer, as a .png. OpenGL's origin is the bottom-left corner and
// every image format in use puts it at the top-left, so the rows are reversed
// on the way out; forgetting that produces an upside-down picture that looks
// like a driver fault.
bool capture_framebuffer(GLFWwindow* window, const fs::path& out) {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    if (w < 8 || h < 8) return false;
    std::vector<unsigned char> px(size_t(w) * size_t(h) * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());

    Image img(w, h, 3);
    for (int y = 0; y < h; ++y) {
        const unsigned char* row = px.data() + size_t(h - 1 - y) * size_t(w) * 3;
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                img.plane(c)[size_t(y) * size_t(w) + size_t(x)] = float(row[x * 3 + c]) / 255.f;
    }
    return img.save_png(out);
}

// -------------------------------------------------------------- selftest
// Checks what a user would actually see. The interesting failure mode is an
// app that opens to an empty window, and in that state the ImGui window
// objects still exist, are still docked, and still report sensible sizes:
// they are simply never painted. So structural checks pass while the app is
// visibly broken, and the only trustworthy signal is the rendered image.
//
// Call after rendering and before the buffer swap, so the back buffer holds
// the frame the user would have seen.
std::string App::check_layout() const {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    if (!vp || vp->Size.x < 64.f || vp->Size.y < 64.f) return "viewport has no usable size";

    for (const char* name : {"Queue", "Settings", "Preview", "Log"}) {
        ImGuiWindow* w = ImGui::FindWindowByName(name);
        if (!w) return std::string("panel '") + name + "' does not exist";
        if (w->Size.x < 32.f || w->Size.y < 32.f)
            return std::string("panel '") + name + "' has no usable size";
    }
    // Details shares a tab strip with Settings, so it is not the visible one
    // on the frame this runs. Its existence is still worth asserting: it is
    // where the file's own properties are shown, and a layout that failed to
    // dock it would leave that panel unreachable.
    if (!ImGui::FindWindowByName("Details")) return "panel 'Details' does not exist";

    int fb_w = 0, fb_h = 0;
    glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    if (fb_w < 64 || fb_h < 64) return "framebuffer has no usable size";

    // Read the frame back and measure how much of it is something other than
    // the background the window is cleared to. A working layout paints panels
    // over most of the window; an empty one leaves nearly all of it clear.
    std::vector<unsigned char> px(size_t(fb_w) * fb_h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, fb_w, fb_h, GL_RGB, GL_UNSIGNED_BYTE, px.data());

    const unsigned char clear_r = 20, clear_g = 20, clear_b = 23;  // matches glClearColor
    std::int64_t painted = 0;
    const std::int64_t total = std::int64_t(fb_w) * fb_h;
    for (std::int64_t i = 0; i < total; ++i) {
        const int dr = int(px[i * 3 + 0]) - clear_r;
        const int dg = int(px[i * 3 + 1]) - clear_g;
        const int db = int(px[i * 3 + 2]) - clear_b;
        if (std::abs(dr) > 3 || std::abs(dg) > 3 || std::abs(db) > 3) ++painted;
    }
    const double fraction = double(painted) / double(total);
    // Everything the selftest reports goes to stderr, which is unbuffered.
    // On stdout it is block-buffered when redirected into a test harness, so
    // a crash anywhere later discards the entire history of how far the run
    // got. This is exactly the position a hosted macOS runner left us in:
    // a three-second failure with no output whatsoever and nothing to go on.
    std::fprintf(stderr, "selftest: %.1f%% of the window painted, %dx%d framebuffer\n",
                 100.0 * fraction, fb_w, fb_h);
    if (fraction < 0.5) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "only %.1f%% of the window was painted, so the interface did not render "
                      "(an empty window looks like this)",
                      100.0 * fraction);
        return buf;
    }
    return {};
}

// The other half of the selftest, and the half that was missing: compressing
// an image, and then opening the file that produced.
//
// The pipeline suite already drives this logic without a window, which covers
// the code. It does not cover this binary -- built with these flags, holding a
// real GL context, having run the GPU self-check against whatever hardware is
// in the machine, choosing a backend on that basis. That combination is what
// gets installed, and a reviewer's first two actions on it are to add an image
// and then to open the file it wrote. So the shipped executable does exactly
// that to itself before anyone ships it.
std::string App::check_compression() {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "gsic-selftest";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    if (ec) return "could not create a scratch directory for the compression check";

    const fs::path src = dir / "selftest.png";
    {
        Image img(96, 96, 3);
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x) {
                img.plane(0)[size_t(y) * img.w + x] = float(x) / float(img.w);
                img.plane(1)[size_t(y) * img.w + x] = float(y) / float(img.h);
                img.plane(2)[size_t(y) * img.w + x] = ((x / 16 + y / 16) % 2) ? 0.85f : 0.15f;
            }
        if (!img.save_png(src)) return "could not write the scratch image";
    }

    // Whatever backend this machine settled on, through the same settings path
    // the interface uses.
    EncodeOptions o = settings_.encode;
    o.preset = 3;
    o.pixels_per_gaussian = 100;
    o.steps = 200;
    o.save_next_to_input = true;
    o.export_png = false;
    pipeline_.set_options(o);

    if (pipeline_.add_files({src}) != 1) return "the queue refused a plain PNG";
    if (!pipeline_.wait_until_idle(std::chrono::seconds(60)))
        return "the image was queued and never compressed";

    auto job = pipeline_.job(pipeline_.job_count() - 1);
    if (!job) return "the queued job disappeared";
    if (job->status != JobStatus::Done) {
        std::lock_guard lk(job->m);
        return "compressing a plain PNG failed: " +
               (job->error.empty() ? std::string("no reason given") : job->error);
    }

    fs::path out;
    double reported_psnr = 0;
    const char* backend = "?";
    std::string fallback;
    {
        std::lock_guard lk(job->m);
        out = job->output_path;
        reported_psnr = job->stats.psnr;
        backend = job->stats.backend_used;
        fallback = job->stats.gpu_fallback_reason;
    }
    if (out.empty() || !fs::exists(out)) return "the encode reported success but wrote no file";

    std::vector<std::uint8_t> bytes;
    {
        std::ifstream f(out, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::string err;
    if (!decode_gsi(bytes, &err))
        return "the file this build just wrote cannot be read back: " + err;

    std::fprintf(stderr, "selftest: compressed 96x96 -> %zu bytes, %.2f dB, backend %s\n", bytes.size(),
                reported_psnr, backend);
    if (!fallback.empty())
        std::fprintf(stderr, "selftest: note - the GPU %s, so the CPU did the work\n", fallback.c_str());

    // And now the half a user reaches for next: open the .gsi that was just
    // written, the way double-clicking it in a file browser does. This is the
    // path the file association has always advertised and which, until now,
    // ended in the application telling the user their own file was not an
    // image.
    const int before = pipeline_.job_count();
    if (pipeline_.add_files({out}) != 1) return "the queue refused the .gsi it had just written";
    if (!pipeline_.wait_until_idle(std::chrono::seconds(60)))
        return "the .gsi was opened and never decoded";
    auto view = pipeline_.job(before);
    if (!view) return "the decode job disappeared";
    if (view->kind != JobKind::Decode)
        return "a .gsi was queued as something other than a file to look at";
    if (view->status != JobStatus::Done) {
        std::lock_guard lk(view->m);
        return "opening the .gsi this build just wrote failed: " +
               (view->error.empty() ? std::string("no reason given") : view->error);
    }
    {
        std::lock_guard lk(view->m);
        if (!view->has_info || view->info.width != 96 || view->info.height != 96)
            return "the decoded .gsi did not report the size it was written at";
        if (view->recon_rgba.size() != size_t(view->recon_w) * size_t(view->recon_h) * 4)
            return "the decoded .gsi produced no picture to show";
        std::fprintf(stderr, "selftest: reopened the .gsi -> %dx%d, %d gaussians, %.3f bpp\n",
                     view->info.width, view->info.height, view->info.gaussians, view->info.bpp);
    }

    fs::remove_all(dir, ec);
    return {};
}

// -------------------------------------------------------------------- run
int App::run(const std::vector<fs::path>& initial_files, const AppOptions& options) {
    const bool borderless = options.borderless;
    g_app = this;

    // Preferences first, so the window opens at the size it was last left and
    // the encode settings are the ones the user chose, not the defaults. The
    // scripted modes deliberately start from the defaults: a test or a capture
    // that behaves differently depending on what is on the machine running it
    // is not testing or capturing what it claims to.
    const bool scripted = options.selftest || !options.screenshot.empty();
    settings_file_ = settings_file_path();
    if (!scripted && !settings_file_.empty()) load_settings(settings_, settings_file_);
    clamp_settings(settings_);
    std::snprintf(out_dir_, sizeof(out_dir_), "%s", settings_.encode.out_dir.c_str());

    // Both of the failures below used to return 1 with nothing written
    // anywhere. Under CTest that produced a red test, zero output, and no way
    // to tell "this machine has no display" from "the window comes up empty" --
    // which is precisely the distinction the whole selftest exists to make.
    if (!glfwInit()) {
        const char* desc = nullptr;
        glfwGetError(&desc);
        std::fprintf(stderr, "gsic: GLFW could not start%s%s\n", desc ? ": " : "",
                     desc ? desc : " (no display or window server?)");
        return options.selftest ? kSelftestNoDisplay : 1;
    }
#if defined(__APPLE__)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    const char* glsl_version = "#version 150";
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    const char* glsl_version = "#version 330";
#endif
    float scale = 1.f;
    int win_w = 0, win_h = 0;
    GLFWmonitor* mon = glfwGetPrimaryMonitor();
    if (mon) {
        float sy = 1.f;
        glfwGetMonitorContentScale(mon, &scale, &sy);
    }
    if (borderless && mon) {
        // Client area exactly fills the monitor, kept above the taskbar. The
        // title bar is left in place but pushed off screen: GLFW window
        // positions refer to the client area, so placing it at (0, 0) puts
        // the decoration above the top edge. Dropping the decoration instead
        // leaves the docking layout unable to build.
        const GLFWvidmode* mode = glfwGetVideoMode(mon);
        win_w = mode->width;
        win_h = mode->height;
        glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    } else if (!scripted && settings_.ui.window_w > 0 && settings_.ui.window_h > 0) {
        win_w = settings_.ui.window_w;
        win_h = settings_.ui.window_h;
    } else {
        win_w = int(1360 * scale);
        win_h = int(860 * scale);
    }
    window_ = glfwCreateWindow(win_w, win_h,
                               "gsic - Gaussian Splat Image Compressor", nullptr, nullptr);
    if (!window_) {
        // No OpenGL 3.3 core context. On a real machine that means very old
        // hardware or a driver that has not been installed; on a hosted build
        // runner it means there is no desktop session to put a window in.
        // Neither is the interface failing to render, so the selftest reports
        // a skip and explains itself rather than claiming a defect.
        const char* desc = nullptr;
        glfwGetError(&desc);
        std::fprintf(stderr,
                     "gsic: could not create a window with an OpenGL 3.3 core context%s%s\n",
                     desc ? ": " : "", desc ? desc : "");
        glfwTerminate();
        return options.selftest ? kSelftestNoDisplay : 1;
    }
    if (!borderless && !scripted && settings_.ui.window_maximized) glfwMaximizeWindow(window_);
    // The compute context must be created on the main thread (the worker
    // only makes it current later), and it must come BEFORE the UI context
    // becomes current: gpu_init unbinds whatever context is current when it
    // finishes.
    {
        std::string why;
        gpu_ok_ = gpu_init(&why);
        gpu_reason_ = why;
        gpu_status_ = gpu_ok_ ? "GPU + CPU" : "CPU only (" + why + ")";
    }

    if (borderless) glfwSetWindowPos(window_, 0, 0);

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    if (options.selftest) {
        // A trail of where the run reached, and what it is running on. When a
        // build runner fails here there is no way to attach a debugger, so the
        // log has to carry enough to work from -- and which OpenGL
        // implementation answered is the single most useful fact about a
        // machine this project has learned to ask for.
        std::fprintf(stderr, "selftest: window created, %dx%d requested\n", win_w, win_h);
    }
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        // A window that exists but whose OpenGL entry points will not resolve.
        // Reported rather than returned silently, because the alternative is a
        // process that exits with a code and no explanation for it.
        std::fprintf(stderr, "gsic: could not load OpenGL functions for the window\n");
        glfwDestroyWindow(window_);
        glfwTerminate();
        return 1;
    }
    if (options.selftest) {
        const auto* renderer = glGetString(GL_RENDERER);
        const auto* version = glGetString(GL_VERSION);
        std::fprintf(stderr, "selftest: GL %s on %s\n",
                     version ? reinterpret_cast<const char*>(version) : "?",
                     renderer ? reinterpret_cast<const char*>(renderer) : "?");
    }
    glfwSetDropCallback(window_, &App::drop_callback);
    NFD_Init();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = nullptr;   // fixed layout; no ini clutter next to the exe

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.f;
    style.FrameRounding = 4.f;
    style.GrabRounding = 4.f;
    style.TabRounding = 4.f;
    style.WindowPadding = ImVec2(10, 8);
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.11f, 0.11f, 0.12f, 1.f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.16f, 0.16f, 0.20f, 1.f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.24f, 0.32f, 0.50f, 0.55f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.38f, 0.62f, 0.70f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.22f, 0.28f, 0.42f, 1.f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.36f, 0.55f, 1.f);
    style.ScaleAllSizes(scale);

    // A system font beats the embedded bitmap font at any DPI.
    const char* font_candidates[] = {
#if defined(_WIN32)
        "C:/Windows/Fonts/segoeui.ttf",
#elif defined(__APPLE__)
        "/System/Library/Fonts/SFNS.ttf", "/Library/Fonts/Arial.ttf",
#else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
#endif
    };
    for (const char* f : font_candidates)
        if (fs::exists(f) && io.Fonts->AddFontFromFileTTF(f, 17.f * scale)) break;

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    push_options();
    settings_dirty_ = false;   // nothing has actually changed yet
    pipeline_.start([this](const std::string& line) { log(line); });
    log("gsic " GSIC_VERSION " ready. Drop an image to compress it, or a .gsi to open it.");
    if (!initial_files.empty()) add_files(initial_files);

    int frames = 0;
    std::string selftest_error;
    while (!glfwWindowShouldClose(window_)) {
        // Idle-friendly: wait for events when nothing is encoding. The two
        // scripted modes need frames to keep arriving regardless, because
        // their exit condition is counted in frames.
        if (pipeline_.busy() || options.selftest || !options.screenshot.empty())
            glfwPollEvents();
        else
            glfwWaitEventsTimeout(0.25);

        // The close request is taken off GLFW and decided here instead, so
        // that a click on the window's close button can put a question on
        // screen rather than ending the process. Anything that arrives after
        // this point is seen on the next iteration.
        bool want_close = glfwWindowShouldClose(window_) != 0;
        glfwSetWindowShouldClose(window_, 0);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        draw_ui(want_close);
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window_, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.08f, 0.09f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Judge the frame before it is swapped away, then compress something,
        // then leave. The layout check has to happen while a rendered frame is
        // still in the back buffer; the compression check does not, but it
        // belongs here too so that both run against a fully started
        // application rather than a partly built one.
        ++frames;
        if (options.selftest && frames >= 30) {
            selftest_error = check_layout();
            if (selftest_error.empty()) selftest_error = check_compression();
            want_close = true;
            confirmed_quit_ = true;
        }
        // A capture waits for the queue to settle, so the picture shows a
        // finished result instead of a progress bar. The frame floor covers
        // the case where nothing was queued at all, and the idle run has to
        // last several frames because the frame in the back buffer was drawn
        // before this check: capturing the instant the worker finishes catches
        // a picture of the queue still working.
        idle_frames_ = pipeline_.busy() ? 0 : idle_frames_ + 1;
        if (!options.screenshot.empty() && frames >= 45 && idle_frames_ >= 4) {
            const bool ok = capture_framebuffer(window_, options.screenshot);
            std::fprintf(stderr, "screenshot: %s %s\n", ok ? "wrote" : "could not write",
                         options.screenshot.string().c_str());
            want_close = true;
            confirmed_quit_ = true;
        }
        glfwSwapBuffers(window_);

        // Work in progress is not thrown away without asking. Closing the
        // window mid-encode used to discard several minutes of computation
        // with no warning and nothing written.
        if (want_close) {
            if (!confirmed_quit_ && pipeline_.busy())
                open_quit_popup_ = true;
            else
                glfwSetWindowShouldClose(window_, 1);
        }
    }

    // Remember where the window was left, unless it was in one of the modes
    // that is not a normal window.
    if (!borderless && !scripted) {
        int w = 0, h = 0;
        glfwGetWindowSize(window_, &w, &h);
        settings_.ui.window_maximized = glfwGetWindowAttrib(window_, GLFW_MAXIMIZED) != 0;
        if (!settings_.ui.window_maximized && w > 0 && h > 0) {
            settings_.ui.window_w = w;
            settings_.ui.window_h = h;
        }
        settings_dirty_ = true;
        persist_settings();
    }

    // The worker holds the GPU context while it encodes, so it has to be gone
    // before anything below tears that context down.
    pipeline_.stop();
    for (const auto& j : pipeline_.jobs()) release_textures(*j);
    flush_released_textures();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    NFD_Quit();
    gpu_shutdown();
    glfwDestroyWindow(window_);
    glfwTerminate();

    if (options.selftest) {
        if (selftest_error.empty()) {
            std::fprintf(stderr, "selftest: ok (%d frames rendered, image compressed and "
                                 "reopened, compute: %s)\n",
                        frames, gpu_status_.c_str());
            return 0;
        }
        std::fprintf(stderr, "selftest: FAILED: %s\n", selftest_error.c_str());
        return 1;
    }
    return 0;
}

} // namespace

int run_app(const std::vector<fs::path>& initial_files, const AppOptions& options) {
    App app;
    return app.run(initial_files, options);
}

} // namespace gsic
