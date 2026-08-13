#pragma once

#include <filesystem>
#include <vector>

namespace gsic {

// Exit code --selftest uses for "there is no window server here, so there was
// nothing to test", as opposed to "the interface is broken". CTest is told
// about it through SKIP_RETURN_CODE, so a machine that cannot open a window
// reports a skip rather than a failure -- and, just as importantly, says so
// out loud instead of being quietly filtered out of the run by the CI
// configuration, where nobody would ever notice it had stopped running.
inline constexpr int kSelftestNoDisplay = 77;

struct AppOptions {
    // Fills the primary monitor with the client area and floats the window
    // above the taskbar. Used to capture the Store screenshots at exactly
    // the monitor resolution; not something an installed copy ever needs.
    bool borderless = false;
    // Starts the real window, GL context and interface, renders a few
    // frames, checks that the layout came up, then exits with a status code.
    // Nonzero means the interface failed to build, which is the failure the
    // user would see as an empty window.
    bool selftest = false;
    // Writes what the window looks like to this .png and exits. Any files
    // given on the command line are processed first, so the capture shows a
    // finished result rather than an empty queue.
    //
    // This exists because the Store listing needs pictures of the interface,
    // and the alternative -- a person with a screen-capture tool, cropping by
    // hand -- produces a different image every time and cannot be part of a
    // build. It is also the only way to look at the interface from a script.
    std::filesystem::path screenshot;
};

// Runs the GUI application; returns the process exit code. Paths given on
// the command line (e.g. "Open with") are queued at startup.
int run_app(const std::vector<std::filesystem::path>& initial_files, const AppOptions& options);

} // namespace gsic
