#pragma once

#include <filesystem>
#include <vector>

namespace gsic {

// Runs the GUI application; returns the process exit code. Paths given on
// the command line (e.g. "Open with") are queued at startup.
int run_app(const std::vector<std::filesystem::path>& initial_files);

} // namespace gsic
