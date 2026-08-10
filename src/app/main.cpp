#include "app.h"

#include <filesystem>
#include <vector>

// On hybrid laptops, ask the driver for the discrete GPU.
#if defined(_WIN32)
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

int main(int argc, char** argv) {
    std::vector<std::filesystem::path> files;
    for (int i = 1; i < argc; ++i) files.emplace_back(argv[i]);
    return gsic::run_app(files);
}
