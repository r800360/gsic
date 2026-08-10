// gsic GUI: image queue on the left, original-vs-compressed preview in the
// middle, log at the bottom. One background worker encodes queued images
// sequentially (each encode already uses every core / the GPU).
#include "app.h"

#include "core/codec.h"
#include "core/gpu.h"
#include "core/image.h"
#include "core/renderer.h"
#include "core/trainer.h"

#include <glad/glad.h>
// glad before GLFW.
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <nfd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace gsic {
namespace {

// ------------------------------------------------------------------- job
enum class JobStatus { Queued, Running, Done, Failed, Cancelled };

struct Job {
    fs::path input;
    std::atomic<JobStatus> status{JobStatus::Queued};
    std::atomic<float> progress{0.f};
    std::atomic<bool> cancel{false};

    // Everything below is guarded by `m`.
    std::mutex m;
    std::string error;
    int w = 0, h = 0, channels = 0;
    std::vector<std::uint8_t> orig_rgba, recon_rgba;   // interleaved, w*h*4
    bool orig_dirty = false, recon_dirty = false;
    EncodeStats stats{};
    bool has_stats = false;
    fs::path output_path;

    // UI-thread only.
    unsigned tex_orig = 0, tex_recon = 0;
    int tex_w = 0, tex_h = 0;
};

// Planar float image -> RGBA8 (gray and gray+alpha spread over RGB).
std::vector<std::uint8_t> to_rgba(const Image& img) {
    std::vector<std::uint8_t> out(size_t(img.w) * img.h * 4);
    const std::int64_t n = img.pixels();
    auto tob = [](float v) { return std::uint8_t(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f); };
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

std::string human_size(std::int64_t bytes) {
    char buf[32];
    if (bytes >= 1024 * 1024)
        std::snprintf(buf, sizeof(buf), "%.2f MB", double(bytes) / (1024.0 * 1024.0));
    else
        std::snprintf(buf, sizeof(buf), "%.1f KB", double(bytes) / 1024.0);
    return buf;
}

// ------------------------------------------------------------------- app
class App {
public:
    int run(const std::vector<fs::path>& initial_files);

private:
    // Worker side.
    void worker_loop();
    void encode_job(const std::shared_ptr<Job>& job);
    void log(const std::string& line);

    // UI side.
    void add_files(const std::vector<fs::path>& paths);
    void open_add_dialog();
    void draw_ui();
    void draw_queue();
    void draw_settings();
    void draw_preview();
    void draw_log();
    void setup_dock(ImGuiID dockspace);
    void upload_textures(Job& job);
    static void drop_callback(GLFWwindow* win, int count, const char** paths);

    GLFWwindow* window_ = nullptr;

    std::mutex jobs_m_;
    std::vector<std::shared_ptr<Job>> jobs_;
    int selected_ = -1;

    std::mutex log_m_;
    std::deque<std::string> log_lines_;

    std::condition_variable work_cv_;
    std::mutex work_m_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> worker_busy_{false};

    // Settings (UI thread writes; worker reads a copy per job).
    int preset_ = 1;              // 0 fast, 1 balanced, 2 high, 3 custom
    int ppg_ = 400;
    int steps_ = 4000;
    int bits_index_ = 0;          // 16, 12, 10, 8
    int backend_ = 0;             // 0 auto, 1 cpu, 2 gpu
    bool save_next_to_input_ = true;
    char out_dir_[512] = "";
    bool export_png_ = false;
    std::string gpu_status_;
    bool gpu_ok_ = false;

    // Preview state.
    float split_ = 0.5f;
    float zoom_ = 0.f;            // 0 = fit
    ImVec2 pan_{0, 0};
};

App* g_app = nullptr;

void App::log(const std::string& line) {
    std::lock_guard lk(log_m_);
    log_lines_.push_back(line);
    if (log_lines_.size() > 500) log_lines_.pop_front();
}

// ----------------------------------------------------------------- worker
void App::worker_loop() {
    while (!shutdown_) {
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
        }
        {
            std::lock_guard jl(jobs_m_);
            for (auto& j : jobs_)
                if (j->status == JobStatus::Queued) {
                    next = j;
                    break;
                }
        }
        if (!next) continue;
        worker_busy_ = true;
        encode_job(next);
        worker_busy_ = false;
    }
}

void App::encode_job(const std::shared_ptr<Job>& job) {
    job->status = JobStatus::Running;
    job->progress = 0.f;

    std::string err;
    auto img = Image::load(job->input, &err);
    if (!img) {
        std::lock_guard lk(job->m);
        job->error = err;
        job->status = JobStatus::Failed;
        log(job->input.filename().string() + ": " + err);
        return;
    }
    {
        auto rgba = to_rgba(*img);
        std::lock_guard lk(job->m);
        job->w = img->w;
        job->h = img->h;
        job->channels = img->c;
        job->orig_rgba = std::move(rgba);
        job->orig_dirty = true;
    }
    log(job->input.filename().string() + ": " + std::to_string(img->w) + "x" +
        std::to_string(img->h) + ", " + std::to_string(img->c) + " channel(s)");

    // Snapshot of the current settings.
    EncodeSettings s;
    switch (preset_) {
        case 0: s.pixels_per_gaussian = 450; s.max_steps = 1200; break;
        case 1: s.pixels_per_gaussian = 300; s.max_steps = 3000; break;
        case 2: s.pixels_per_gaussian = 190; s.max_steps = 8000; break;
        default: s.pixels_per_gaussian = ppg_; s.max_steps = steps_; break;
    }
    // Quality levels scale the default per-attribute allocation together;
    // position always keeps the most bits (see QuantSpec).
    static const QuantSpec kQuality[] = {
        {16, 12, 10, 12},   // best
        {14, 11, 9, 11},    // smaller
        {12, 10, 8, 10},    // smallest
    };
    s.quant = kQuality[std::clamp(bits_index_, 0, 2)];
    s.backend = backend_ == 1 ? Backend::Cpu : (backend_ == 2 ? Backend::Gpu : Backend::Auto);

    auto progress = [&](const EncodeProgress& p) {
        job->progress = float(p.step) / float(p.max_steps);
        if (p.preview) {
            auto rgba = to_rgba(*p.preview);
            std::lock_guard lk(job->m);
            // Warmup previews are half size; the texture upload handles it.
            job->w = p.preview->w;
            job->h = p.preview->h;
            job->recon_rgba = std::move(rgba);
            job->recon_dirty = true;
        }
    };
    auto result = encode_image(*img, s, progress, &job->cancel);

    if (job->cancel) {
        job->status = JobStatus::Cancelled;
        log(job->input.filename().string() + ": cancelled");
        return;
    }

    // Write outputs.
    fs::path out = save_next_to_input_ ? job->input : fs::path(out_dir_) / job->input.filename();
    out.replace_extension(".gsi");
    std::error_code ec;
    if (!save_next_to_input_) fs::create_directories(fs::path(out_dir_), ec);
    bool write_ok = false;
    {
        std::ofstream f(out, std::ios::binary);
        if (f) {
            f.write(reinterpret_cast<const char*>(result.file.data()),
                    std::streamsize(result.file.size()));
            write_ok = bool(f);
        }
    }
    if (write_ok && export_png_) {
        fs::path png = out;
        png.replace_extension(".decoded.png");
        result.reconstruction.save_png(png);
    }

    {
        auto rgba = to_rgba(result.reconstruction);
        std::lock_guard lk(job->m);
        job->w = result.reconstruction.w;
        job->h = result.reconstruction.h;
        job->recon_rgba = std::move(rgba);
        job->recon_dirty = true;
        job->stats = result.stats;
        job->has_stats = true;
        job->output_path = out;
        if (!write_ok) {
            job->error = "cannot write " + out.string();
            job->status = JobStatus::Failed;
        } else {
            job->status = JobStatus::Done;
        }
    }
    job->progress = 1.f;
    if (write_ok) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s -> %s  (%s, %.1f%% of raw, PSNR %.2f dB, %.1fs, %s)",
                      job->input.filename().string().c_str(),
                      out.filename().string().c_str(), human_size(result.stats.file_bytes).c_str(),
                      100.0 * double(result.stats.file_bytes) /
                          double(std::max<std::int64_t>(1, result.stats.source_bytes)),
                      result.stats.psnr, result.stats.encode_seconds, result.stats.backend_used);
        log(buf);
    } else {
        log("failed to write " + out.string());
    }
}

// --------------------------------------------------------------------- UI
void App::drop_callback(GLFWwindow*, int count, const char** paths) {
    std::vector<fs::path> files;
    for (int i = 0; i < count; ++i) files.emplace_back(paths[i]);
    g_app->add_files(files);
}

void App::add_files(const std::vector<fs::path>& paths) {
    static const char* exts[] = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif"};
    int added = 0;
    std::lock_guard lk(jobs_m_);
    for (const auto& p : paths) {
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return char(std::tolower(c)); });
        const bool ok = std::any_of(std::begin(exts), std::end(exts),
                                    [&](const char* e) { return ext == e; });
        if (!ok) continue;
        auto job = std::make_shared<Job>();
        job->input = p;
        jobs_.push_back(std::move(job));
        ++added;
    }
    if (added > 0 && selected_ < 0) selected_ = 0;
    if (added > 0) work_cv_.notify_all();
}

void App::upload_textures(Job& job) {
    std::lock_guard lk(job.m);
    auto upload = [&](unsigned& tex, std::vector<std::uint8_t>& rgba, bool& dirty) {
        if (!dirty || rgba.empty()) return;
        if (!tex) glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, job.w, job.h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     rgba.data());
        dirty = false;
    };
    // Original keeps full resolution; reconstruction may change size during
    // the warmup stage, so both re-upload whole images when dirty.
    if (job.orig_dirty) {
        job.tex_w = job.w;
        job.tex_h = job.h;
    }
    upload(job.tex_orig, job.orig_rgba, job.orig_dirty);
    upload(job.tex_recon, job.recon_rgba, job.recon_dirty);
}

void App::setup_dock(ImGuiID dockspace) {
    ImGui::DockBuilderRemoveNode(dockspace);
    ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace, ImGui::GetMainViewport()->WorkSize);
    ImGuiID left, center, left_bottom, bottom;
    ImGui::DockBuilderSplitNode(dockspace, ImGuiDir_Left, 0.27f, &left, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.20f, &bottom, &center);
    ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, 0.45f, &left_bottom, &left);
    ImGui::DockBuilderDockWindow("Queue", left);
    ImGui::DockBuilderDockWindow("Settings", left_bottom);
    ImGui::DockBuilderDockWindow("Preview", center);
    ImGui::DockBuilderDockWindow("Log", bottom);
    ImGui::DockBuilderFinish(dockspace);
}

void App::open_add_dialog() {
    const nfdu8filteritem_t filters[] = {{"Images", "png,jpg,jpeg,bmp,tga,gif"}};
    const nfdpathset_t* paths = nullptr;
    nfdopendialogu8args_t args{};
    args.filterList = filters;
    args.filterCount = 1;
    if (NFD_OpenDialogMultipleU8_With(&paths, &args) != NFD_OKAY) return;
    nfdpathsetsize_t count = 0;
    NFD_PathSet_GetCount(paths, &count);
    std::vector<fs::path> files;
    for (nfdpathsetsize_t i = 0; i < count; ++i) {
        nfdu8char_t* p = nullptr;
        if (NFD_PathSet_GetPathU8(paths, i, &p) == NFD_OKAY) {
            files.emplace_back(reinterpret_cast<char*>(p));
            NFD_PathSet_FreePathU8(p);
        }
    }
    NFD_PathSet_Free(paths);
    add_files(files);
}

void App::draw_queue() {
    ImGui::Begin("Queue");
    if (ImGui::Button("Add Images...")) open_add_dialog();
    ImGui::SameLine();
    {
        std::lock_guard lk(jobs_m_);
        const bool any_active = std::any_of(jobs_.begin(), jobs_.end(), [](auto& j) {
            const JobStatus st = j->status;
            return st == JobStatus::Running || st == JobStatus::Queued;
        });
        if (any_active) {
            if (ImGui::Button("Cancel All"))
                for (auto& j : jobs_)
                    if (j->status == JobStatus::Running || j->status == JobStatus::Queued)
                        j->cancel = true;
        } else {
            if (ImGui::Button("Clear Finished")) {
                jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(),
                                           [](auto& j) { return j->status != JobStatus::Queued; }),
                            jobs_.end());
                selected_ = jobs_.empty() ? -1 : std::min(selected_, int(jobs_.size()) - 1);
            }
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(drag & drop works too)");
    ImGui::Separator();

    std::lock_guard lk(jobs_m_);
    for (int i = 0; i < int(jobs_.size()); ++i) {
        Job& job = *jobs_[i];
        ImGui::PushID(i);
        const JobStatus st = job.status;
        ImVec4 color = st == JobStatus::Done       ? ImVec4(0.35f, 0.80f, 0.42f, 1)
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
        if (ImGui::Selectable(job.input.filename().string().c_str(), selected_ == i,
                              ImGuiSelectableFlags_AllowOverlap))
            selected_ = i;
        if (st == JobStatus::Running) {
            ImGui::SameLine();
            ImGui::ProgressBar(job.progress, ImVec2(90, 0));
        } else if (st == JobStatus::Done) {
            std::lock_guard jl(job.m);
            if (job.has_stats) {
                ImGui::SameLine();
                ImGui::TextDisabled("%s | %.1f dB", human_size(job.stats.file_bytes).c_str(),
                                    job.stats.psnr);
            }
        }
        ImGui::PopID();
    }
    ImGui::End();
}

void App::draw_settings() {
    ImGui::Begin("Settings");
    const char* presets[] = {"Fast", "Balanced", "High quality", "Custom"};
    ImGui::Combo("Preset", &preset_, presets, 4);
    if (preset_ == 3) {
        ImGui::SliderInt("Pixels / gaussian", &ppg_, 100, 1200, "%d",
                         ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("Steps", &steps_, 500, 15000, "%d", ImGuiSliderFlags_Logarithmic);
    }
    const char* bit_names[] = {"Best", "Smaller", "Smallest"};
    ImGui::Combo("Precision", &bits_index_, bit_names, 3);
    const char* backends[] = {"Auto (GPU if available)", "CPU", "GPU"};
    ImGui::Combo("Backend", &backend_, backends, 3);
    ImGui::Separator();
    ImGui::Checkbox("Save next to input", &save_next_to_input_);
    if (!save_next_to_input_) {
        ImGui::InputText("##outdir", out_dir_, sizeof(out_dir_));
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            nfdu8char_t* p = nullptr;
            nfdpickfolderu8args_t args{};
            if (NFD_PickFolderU8_With(&p, &args) == NFD_OKAY && p) {
                std::snprintf(out_dir_, sizeof(out_dir_), "%s", reinterpret_cast<char*>(p));
                NFD_FreePathU8(p);
            }
        }
    }
    ImGui::Checkbox("Also export decoded .png", &export_png_);
    ImGui::Separator();
    ImGui::TextDisabled("Compute: %s", gpu_ok_ ? "GPU + CPU" : gpu_status_.c_str());
    ImGui::End();
}

void App::draw_preview() {
    ImGui::Begin("Preview");
    std::shared_ptr<Job> job;
    {
        std::lock_guard lk(jobs_m_);
        if (selected_ >= 0 && selected_ < int(jobs_.size())) job = jobs_[selected_];
    }
    if (!job) {
        ImGui::Dummy(ImVec2(0, 8));
        ImGui::TextDisabled("Add images to the queue to get started.");
        ImGui::End();
        return;
    }
    upload_textures(*job);

    // Header line with stats.
    {
        std::lock_guard lk(job->m);
        if (job->has_stats) {
            ImGui::Text("%s   %s -> %s (%.1f%%)   PSNR %.2f dB   SSIM %.4f   %.1fs [%s]",
                        job->input.filename().string().c_str(),
                        human_size(job->stats.source_bytes).c_str(),
                        human_size(job->stats.file_bytes).c_str(),
                        100.0 * double(job->stats.file_bytes) /
                            double(std::max<std::int64_t>(1, job->stats.source_bytes)),
                        job->stats.psnr, job->stats.ssim, job->stats.encode_seconds,
                        job->stats.backend_used);
        } else if (!job->error.empty()) {
            ImGui::TextColored(ImVec4(0.95f, 0.42f, 0.38f, 1), "%s", job->error.c_str());
        } else {
            ImGui::Text("%s   %dx%d", job->input.filename().string().c_str(), job->tex_w,
                        job->tex_h);
            if (job->status == JobStatus::Running) {
                ImGui::SameLine();
                ImGui::ProgressBar(job->progress, ImVec2(160, 0));
            }
        }
    }
    if (!job->tex_orig) {
        ImGui::TextDisabled("Loading...");
        ImGui::End();
        return;
    }

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
    const float fit = std::min(avail.x / job->tex_w, avail.y / job->tex_h);
    const float scale = zoom_ <= 0.f ? fit : zoom_;
    const ImVec2 img_size(job->tex_w * scale, job->tex_h * scale);
    ImVec2 img_pos(origin.x + (avail.x - img_size.x) * 0.5f + pan_.x,
                   origin.y + (avail.y - img_size.y) * 0.5f + pan_.y);
    const ImVec2 img_end(img_pos.x + img_size.x, img_pos.y + img_size.y);

    // Interaction: wheel zooms around the cursor, drag pans, the divider
    // drags the A/B split, double-click resets.
    ImGui::InvisibleButton("preview", avail, ImGuiButtonFlags_MouseButtonLeft);
    const bool hovered = ImGui::IsItemHovered();
    const float split_x = img_pos.x + img_size.x * split_;
    const bool near_split = hovered && std::fabs(ImGui::GetIO().MousePos.x - split_x) < 8.f;
    static bool dragging_split = false;
    if (ImGui::IsItemActivated() && near_split) dragging_split = true;
    if (!ImGui::IsItemActive()) dragging_split = false;
    if (ImGui::IsItemActive()) {
        if (dragging_split) {
            split_ = std::clamp((ImGui::GetIO().MousePos.x - img_pos.x) / img_size.x, 0.f, 1.f);
        } else {
            pan_.x += ImGui::GetIO().MouseDelta.x;
            pan_.y += ImGui::GetIO().MouseDelta.y;
        }
    }
    if (hovered && ImGui::GetIO().MouseWheel != 0.f) {
        const float old_scale = scale;
        float target = old_scale * (ImGui::GetIO().MouseWheel > 0 ? 1.25f : 0.8f);
        target = std::clamp(target, fit * 0.25f, 32.f);
        // Keep the pixel under the cursor fixed while zooming.
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        pan_.x = mouse.x - (mouse.x - pan_.x - (origin.x + (avail.x - job->tex_w * target) * 0.5f)) *
                     (target / old_scale) - (origin.x + (avail.x - job->tex_w * target) * 0.5f);
        pan_.y = mouse.y - (mouse.y - pan_.y - (origin.y + (avail.y - job->tex_h * target) * 0.5f)) *
                     (target / old_scale) - (origin.y + (avail.y - job->tex_h * target) * 0.5f);
        zoom_ = target;
    }
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        zoom_ = 0.f;
        pan_ = ImVec2(0, 0);
        split_ = 0.5f;
    }

    dl->PushClipRect(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), true);
    const bool has_recon = job->tex_recon != 0;
    if (has_recon)
        dl->AddImage(ImTextureID(std::intptr_t(job->tex_recon)), img_pos, img_end);
    // Original occupies the left side of the divider.
    {
        const float cx = std::clamp(split_x, img_pos.x, img_end.x);
        dl->PushClipRect(img_pos, ImVec2(cx, img_end.y), true);
        dl->AddImage(ImTextureID(std::intptr_t(job->tex_orig)), img_pos, img_end);
        dl->PopClipRect();
    }
    if (has_recon) {
        dl->AddLine(ImVec2(split_x, std::max(img_pos.y, origin.y)),
                    ImVec2(split_x, std::min(img_end.y, origin.y + avail.y)),
                    near_split || dragging_split ? IM_COL32(255, 255, 255, 230)
                                                 : IM_COL32(255, 255, 255, 130),
                    2.f);
        dl->AddText(ImVec2(std::max(img_pos.x, origin.x) + 6, origin.y + 4),
                    IM_COL32(255, 255, 255, 160), "original");
        dl->AddText(ImVec2(std::min(split_x + 8, origin.x + avail.x - 90), origin.y + 4),
                    IM_COL32(255, 255, 255, 160), "compressed");
    }
    dl->PopClipRect();
    ImGui::End();
}

void App::draw_log() {
    ImGui::Begin("Log");
    std::lock_guard lk(log_m_);
    for (const auto& line : log_lines_) ImGui::TextUnformatted(line.c_str());
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4) ImGui::SetScrollHereY(1.f);
    ImGui::End();
}

void App::draw_ui() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Add Images...")) open_add_dialog();
            if (ImGui::MenuItem("Exit")) glfwSetWindowShouldClose(window_, 1);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            ImGui::MenuItem("gsic 1.0 - gaussian splat image compressor", nullptr, false, false);
            ImGui::MenuItem("Based on Image-GS (NYU ICL / Intel / AMD)", nullptr, false, false);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    const ImGuiID dockspace = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
    const ImGuiDockNode* root = ImGui::DockBuilderGetNode(dockspace);
    if (!root || !root->IsSplitNode()) setup_dock(dockspace);

    draw_queue();
    draw_settings();
    draw_preview();
    draw_log();
}

// -------------------------------------------------------------------- run
int App::run(const std::vector<fs::path>& initial_files) {
    g_app = this;
    if (!glfwInit()) return 1;
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
    if (GLFWmonitor* mon = glfwGetPrimaryMonitor()) {
        float sy = 1.f;
        glfwGetMonitorContentScale(mon, &scale, &sy);
    }
    window_ = glfwCreateWindow(int(1360 * scale), int(860 * scale),
                               "gsic - Gaussian Splat Image Compressor", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        return 1;
    }
    // The compute context must be created on the main thread (the worker
    // only makes it current later), and it must come BEFORE the UI context
    // becomes current: gpu_init unbinds whatever context is current when it
    // finishes.
    {
        std::string why;
        gpu_ok_ = gpu_init(&why);
        gpu_status_ = gpu_ok_ ? "GPU + CPU" : "CPU only (" + why + ")";
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return 1;
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

    std::thread worker([this] { worker_loop(); });
    if (!initial_files.empty()) add_files(initial_files);

    while (!glfwWindowShouldClose(window_)) {
        // Idle-friendly: wait for events when nothing is encoding.
        if (worker_busy_)
            glfwPollEvents();
        else
            glfwWaitEventsTimeout(0.25);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        draw_ui();
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window_, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.08f, 0.09f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window_);
    }

    shutdown_ = true;
    {
        std::lock_guard lk(jobs_m_);
        for (auto& j : jobs_) j->cancel = true;
    }
    work_cv_.notify_all();
    worker.join();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    NFD_Quit();
    gpu_shutdown();
    glfwDestroyWindow(window_);
    glfwTerminate();
    return 0;
}

} // namespace

int run_app(const std::vector<fs::path>& initial_files) {
    App app;
    return app.run(initial_files);
}

} // namespace gsic
