#include "settings.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <system_error>

namespace fs = std::filesystem;

namespace gsic {
namespace {

// The format version, written first and checked on read. It is not used to
// migrate anything yet; it exists so that a future change can tell an old file
// from a corrupt one instead of guessing.
constexpr int kSettingsVersion = 1;

std::string_view trim(std::string_view s) {
    const auto space = [](char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
    };
    while (!s.empty() && space(s.front())) s.remove_prefix(1);
    while (!s.empty() && space(s.back())) s.remove_suffix(1);
    return s;
}

bool to_int(std::string_view s, int& out) {
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc{} && r.ptr == s.data() + s.size();
}

bool to_double(std::string_view s, double& out) {
    // from_chars for floating point is not available everywhere this builds,
    // so the value goes through strtod on a null-terminated copy.
    const std::string copy(s);
    char* end = nullptr;
    const double v = std::strtod(copy.c_str(), &end);
    if (!end || *end != '\0' || end == copy.c_str()) return false;
    out = v;
    return true;
}

bool to_bool(std::string_view s, bool& out) {
    if (s == "1" || s == "true" || s == "yes") { out = true; return true; }
    if (s == "0" || s == "false" || s == "no") { out = false; return true; }
    return false;
}

fs::path home_directory() {
#if defined(_WIN32)
    if (const char* p = std::getenv("LOCALAPPDATA")) return fs::path(p);
    if (const char* p = std::getenv("USERPROFILE")) return fs::path(p);
#endif
    if (const char* p = std::getenv("HOME")) return fs::path(p);
    return {};
}

}  // namespace

void clamp_settings(AppSettings& s) {
    EncodeOptions& e = s.encode;
    e.preset = std::clamp(e.preset, 0, 3);
    e.precision = std::clamp(e.precision, 0, 3);
    e.backend = std::clamp(e.backend, 0, 2);
    e.pixels_per_gaussian =
        std::clamp(e.pixels_per_gaussian, kMinPixelsPerGaussian, kMaxPixelsPerGaussian);
    e.steps = std::clamp(e.steps, kMinCustomSteps, kMaxCustomSteps);
    e.num_gaussians = std::clamp(e.num_gaussians, 0, kMaxGaussians);
    const auto bits = [](int b) { return std::clamp(b, 4, 16); };
    e.custom_quant = QuantSpec{bits(e.custom_quant.pos), bits(e.custom_quant.scale),
                               bits(e.custom_quant.rot), bits(e.custom_quant.feat)};
    // A saved budget outside the slider's range would leave the interface
    // showing a value it cannot represent, and a negative one would mean "no
    // limit" by accident rather than on purpose.
    if (!(e.time_budget_seconds >= 0.0)) e.time_budget_seconds = kDefaultTimeBudgetSeconds;
    e.time_budget_seconds = std::min(e.time_budget_seconds, double(kMaxTimeBudgetSeconds));

    UiPrefs& u = s.ui;
    u.view_mode = std::clamp(u.view_mode, 0, 3);
    if (!(u.difference_gain >= kMinDifferenceGain)) u.difference_gain = kMinDifferenceGain;
    u.difference_gain = std::clamp(u.difference_gain, kMinDifferenceGain, kMaxDifferenceGain);
    // A window smaller than this cannot show the layout, and one larger than
    // any plausible desktop is a sign the file was written by something else.
    if (u.window_w != 0 || u.window_h != 0) {
        u.window_w = std::clamp(u.window_w, 640, 16384);
        u.window_h = std::clamp(u.window_h, 480, 16384);
    }
}

std::string serialize_settings(const AppSettings& s) {
    std::string out;
    out.reserve(768);
    char buf[1024];
    const auto line = [&](const char* fmt, auto... args) {
        std::snprintf(buf, sizeof(buf), fmt, args...);
        out += buf;
        out += '\n';
    };
    out += "# gsic settings. Delete this file to return to the defaults.\n";
    line("version = %d", kSettingsVersion);
    const EncodeOptions& e = s.encode;
    line("preset = %d", e.preset);
    line("pixels_per_gaussian = %d", e.pixels_per_gaussian);
    line("steps = %d", e.steps);
    line("num_gaussians = %d", e.num_gaussians);
    line("precision = %d", e.precision);
    line("bits_pos = %d", e.custom_quant.pos);
    line("bits_scale = %d", e.custom_quant.scale);
    line("bits_rot = %d", e.custom_quant.rot);
    line("bits_color = %d", e.custom_quant.feat);
    line("seed = %u", e.seed);
    line("backend = %d", e.backend);
    line("save_next_to_input = %d", e.save_next_to_input ? 1 : 0);
    line("export_png = %d", e.export_png ? 1 : 0);
    line("time_budget_seconds = %.3f", e.time_budget_seconds);
    // Last, and taken verbatim to the end of the line, so a path containing an
    // '=' survives the round trip. A newline in a path would break the format;
    // no file system in play allows one.
    out += "out_dir = ";
    for (char c : e.out_dir)
        if (c != '\n' && c != '\r') out += c;
    out += '\n';

    const UiPrefs& u = s.ui;
    line("view_mode = %d", u.view_mode);
    line("difference_gain = %.3f", double(u.difference_gain));
    line("log_autoscroll = %d", u.log_autoscroll ? 1 : 0);
    line("window_w = %d", u.window_w);
    line("window_h = %d", u.window_h);
    line("window_maximized = %d", u.window_maximized ? 1 : 0);
    return out;
}

bool parse_settings(std::string_view text, AppSettings& out) {
    int recognised = 0;
    size_t pos = 0;
    while (pos <= text.size()) {
        const size_t nl = text.find('\n', pos);
        std::string_view raw = text.substr(pos, nl == std::string_view::npos ? nl : nl - pos);
        pos = nl == std::string_view::npos ? text.size() + 1 : nl + 1;

        std::string_view line = trim(raw);
        if (line.empty() || line.front() == '#' || line.front() == ';') continue;
        const size_t eq = line.find('=');
        if (eq == std::string_view::npos) continue;
        const std::string_view key = trim(line.substr(0, eq));
        const std::string_view value = trim(line.substr(eq + 1));
        if (key.empty()) continue;

        EncodeOptions& e = out.encode;
        UiPrefs& u = out.ui;
        int iv = 0;
        double dv = 0;
        bool bv = false;
        const auto set_int = [&](int& field) {
            if (to_int(value, iv)) { field = iv; ++recognised; }
        };
        const auto set_bool = [&](bool& field) {
            if (to_bool(value, bv)) { field = bv; ++recognised; }
        };

        if (key == "version") { ++recognised; }
        else if (key == "preset") set_int(e.preset);
        else if (key == "pixels_per_gaussian") set_int(e.pixels_per_gaussian);
        else if (key == "steps") set_int(e.steps);
        else if (key == "num_gaussians") set_int(e.num_gaussians);
        else if (key == "precision") set_int(e.precision);
        else if (key == "bits_pos") set_int(e.custom_quant.pos);
        else if (key == "bits_scale") set_int(e.custom_quant.scale);
        else if (key == "bits_rot") set_int(e.custom_quant.rot);
        else if (key == "bits_color") set_int(e.custom_quant.feat);
        else if (key == "seed") {
            if (to_int(value, iv) && iv >= 0) { e.seed = unsigned(iv); ++recognised; }
        }
        else if (key == "backend") set_int(e.backend);
        else if (key == "save_next_to_input") set_bool(e.save_next_to_input);
        else if (key == "export_png") set_bool(e.export_png);
        else if (key == "time_budget_seconds") {
            if (to_double(value, dv)) { e.time_budget_seconds = dv; ++recognised; }
        }
        else if (key == "out_dir") { e.out_dir = std::string(value); ++recognised; }
        else if (key == "view_mode") set_int(u.view_mode);
        else if (key == "difference_gain") {
            if (to_double(value, dv)) { u.difference_gain = float(dv); ++recognised; }
        }
        else if (key == "log_autoscroll") set_bool(u.log_autoscroll);
        else if (key == "window_w") set_int(u.window_w);
        else if (key == "window_h") set_int(u.window_h);
        else if (key == "window_maximized") set_bool(u.window_maximized);
        // Anything else is from a newer version, or is not ours. Ignored.
    }
    clamp_settings(out);
    return recognised > 0;
}

fs::path settings_file_path() {
    const fs::path home = home_directory();
    if (home.empty()) return {};
#if defined(_WIN32)
    return home / "gsic" / "settings.ini";
#elif defined(__APPLE__)
    return home / "Library" / "Application Support" / "gsic" / "settings.ini";
#else
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"))
        if (*xdg) return fs::path(xdg) / "gsic" / "settings.ini";
    return home / ".config" / "gsic" / "settings.ini";
#endif
}

bool save_settings(const AppSettings& settings, const fs::path& file) {
    if (file.empty()) return false;
    std::error_code ec;
    if (file.has_parent_path()) fs::create_directories(file.parent_path(), ec);

    // Through a temporary and then into place, so a crash or a full disk
    // during the write leaves the previous settings intact rather than a
    // half-written file the next launch has to cope with.
    fs::path tmp = file;
    tmp += ".partial";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        const std::string text = serialize_settings(settings);
        f.write(text.data(), std::streamsize(text.size()));
        f.flush();
        if (!f) { fs::remove(tmp, ec); return false; }
    }
    fs::remove(file, ec);
    fs::rename(tmp, file, ec);
    if (ec) { fs::remove(tmp, ec); return false; }
    return true;
}

bool load_settings(AppSettings& settings, const fs::path& file) {
    if (file.empty()) return false;
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    const std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    return parse_settings(text, settings);
}

} // namespace gsic
