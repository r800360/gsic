#pragma once

// What the application remembers between launches.
//
// It used to remember nothing. Every launch reset the preset, the output
// folder, the backend and the time limit, so anyone who had settled on a way
// of working re-entered it every single time -- and the output folder in
// particular is several clicks through a folder picker. For an application
// someone installs and keeps, that is not a missing nicety, it is the
// difference between a tool and a demo.
//
// The file is plain "key = value" text, one setting per line, so a person can
// read it, fix it, or delete it. Parsing is deliberately forgiving: an
// unrecognised key is ignored, a malformed line is skipped, and every value is
// clamped to the range the interface offers. A settings file is the one input
// a user is invited to edit by hand, and it must not be possible to make the
// application misbehave with it.

#include "pipeline.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace gsic {

// Choices that affect what is on screen rather than what is computed.
struct UiPrefs {
    // 0 split, 1 original only, 2 result only, 3 difference.
    int view_mode = 0;
    // How hard the difference view amplifies. A compressed image differs from
    // its source by a few levels per channel, which is invisible at 1x.
    float difference_gain = 8.f;
    bool log_autoscroll = true;
    // 0 means "decide from the monitor", which is what a first launch does.
    int window_w = 0, window_h = 0;
    bool window_maximized = false;
};

struct AppSettings {
    EncodeOptions encode;
    UiPrefs ui;
};

// Ranges the interface offers, shared by the sliders and by the clamping the
// parser applies, so a hand-edited file cannot ask for something the interface
// could not have produced.
inline constexpr int kMinPixelsPerGaussian = 100;
inline constexpr int kMaxPixelsPerGaussian = 1200;
inline constexpr int kMinCustomSteps = 100;
inline constexpr int kMaxCustomSteps = 20000;
inline constexpr float kMinDifferenceGain = 1.f;
inline constexpr float kMaxDifferenceGain = 64.f;

// Brings every field into range. Applied on load, and safe to call on
// anything, including a default-constructed value.
void clamp_settings(AppSettings& settings);

std::string serialize_settings(const AppSettings& settings);

// Fills `out` from `text`, leaving fields the text does not mention at
// whatever they already were. Returns false when the text contained no
// recognisable setting at all, which is how a truncated or foreign file is
// told apart from an empty one.
bool parse_settings(std::string_view text, AppSettings& out);

// Per-user configuration file. Empty when this machine offers nowhere to put
// one, in which case the application simply does not persist anything rather
// than writing next to the executable -- which under a packaged install is
// either refused or silently virtualized.
std::filesystem::path settings_file_path();

// Both are quiet about failure: settings are a convenience, and an application
// that will not start because it could not read its preferences is worse than
// one that starts with the defaults.
bool save_settings(const AppSettings& settings, const std::filesystem::path& file);
bool load_settings(AppSettings& settings, const std::filesystem::path& file);

} // namespace gsic
