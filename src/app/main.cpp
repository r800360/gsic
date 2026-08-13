#include "app.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// On hybrid laptops, ask the driver for the discrete GPU.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

// This is a GUI-subsystem binary, so it starts with no standard streams: on
// Windows anything it prints goes nowhere. That is right for the application
// and wrong for --selftest, whose entire output is the explanation of what
// failed. Without it a test harness sees an exit code and no reason for it,
// which is the same position the Store report left us in.
//
// Order matters here. When the caller redirected us to a file or a pipe -- how
// CTest and every CI runner invoke this -- that handle is already inherited
// and must be used; reopening onto CONOUT$ regardless would send the report to
// a console nobody is reading and leave the redirection empty. So an inherited
// handle wins, and borrowing the parent's console is only the fallback for
// someone running the flag by hand from a prompt.
void connect_standard_streams() {
    const bool had_stdout = GetStdHandle(STD_OUTPUT_HANDLE) != nullptr;
    const bool had_stderr = GetStdHandle(STD_ERROR_HANDLE) != nullptr;
    if (had_stdout && had_stderr) return;
    if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
    FILE* f = nullptr;
    if (!had_stdout) freopen_s(&f, "CONOUT$", "w", stdout);
    if (!had_stderr) freopen_s(&f, "CONOUT$", "w", stderr);
}
#else
void connect_standard_streams() {}
#endif

int main(int argc, char** argv) {
    std::vector<std::filesystem::path> files;
    gsic::AppOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--borderless")
            options.borderless = true;
        else if (arg == "--selftest")
            options.selftest = true;
        else if (arg == "--screenshot" && i + 1 < argc)
            options.screenshot = argv[++i];
        else
            files.emplace_back(arg);
    }
    if (options.selftest || !options.screenshot.empty()) connect_standard_streams();
    return gsic::run_app(files, options);
}
