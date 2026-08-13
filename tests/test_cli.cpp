// End-to-end tests that drive the real gsic executable.
//
// The unit tests call the library directly, so they never touch argument
// parsing, file handling, exit codes, or the messages a user actually sees.
// This runs the shipped binary the way a person would and checks both the
// happy paths and the ways it can be misused.
//
// The path to the executable is passed in by CTest via $<TARGET_FILE:gsic>.

#include "core/image.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;
using gsic::Image;

static int failures = 0;
static std::string g_exe;
static fs::path g_dir;

static void fail(const std::string& what) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
}

// Runs the tool and returns its exit code. Output is discarded; these tests
// assert on observable effects rather than on message wording, which should
// be free to change.
//
// std::system does not return an exit code on POSIX. It returns an encoded
// wait status, in which the exit code occupies the high byte: a process that
// exits with 1 comes back as 256, and 2 as 512. Returning that raw made every
// failure-path assertion here wrong on Linux and macOS while passing on
// Windows, where the value really is the exit code. Worth being precise about
// what it looked like, because the symptom was misleading: the suite reported
// "truncated file was not rejected", which reads like a decoder defect. The
// decoder was correct and rejecting them; the test could not read the answer.
static int run(const std::string& args) {
    std::string cmd = "\"\"" + g_exe + "\" " + args + "\"";
#ifndef _WIN32
    cmd = "\"" + g_exe + "\" " + args;
#endif
    cmd += " > " + (g_dir / "out.txt").string() + " 2>&1";
    const int status = std::system(cmd.c_str());
#ifdef _WIN32
    return status;
#else
    if (status == -1) return -1;                       // could not run a shell at all
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);   // crashed; never expected
    return status;
#endif
}

static std::string quoted(const fs::path& p) { return "\"" + p.string() + "\""; }

static void expect(int got, int want, const std::string& what) {
    if (got != want) {
        char buf[240];
        std::snprintf(buf, sizeof(buf), "%s: exit code %d, expected %d", what.c_str(), got, want);
        fail(buf);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::puts("usage: gsic_cli_tests <path-to-gsic>");
        return 2;
    }
    g_exe = argv[1];
    g_dir = fs::temp_directory_path() / "gsic_cli_test";
    std::error_code ec;
    fs::remove_all(g_dir, ec);
    fs::create_directories(g_dir, ec);

    // A small, quick source image.
    const fs::path src = g_dir / "src.png";
    {
        Image img(128, 128, 3);
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x) {
                img.plane(0)[size_t(y) * img.w + x] = float(x) / img.w;
                img.plane(1)[size_t(y) * img.w + x] = float(y) / img.h;
                img.plane(2)[size_t(y) * img.w + x] = ((x / 16 + y / 16) % 2) ? 0.8f : 0.2f;
            }
        if (!img.save_png(src)) { fail("could not write the test image"); return 1; }
    }

    const fs::path gsi = g_dir / "src.gsi";
    const fs::path png = g_dir / "decoded.png";

    // --- happy path: compress, and the file must appear and be plausible.
    expect(run("compress " + quoted(src) + " --steps 200 --cpu -o " + quoted(g_dir)), 0,
           "compress");
    if (!fs::exists(gsi)) {
        fail("compress did not produce src.gsi");
    } else {
        const auto size = fs::file_size(gsi);
        if (size < 64 || size > 200000) {
            fail("compressed file has an implausible size: " + std::to_string(size));
        }
        std::printf("  compress -> %llu bytes\n", static_cast<unsigned long long>(size));
    }

    // --- info must accept what compress produced.
    expect(run("info " + quoted(gsi)), 0, "info");

    // --- decompress, and the result must match the source dimensions.
    expect(run("decompress " + quoted(gsi) + " -o " + quoted(png)), 0, "decompress");
    if (!fs::exists(png)) {
        fail("decompress did not produce a png");
    } else {
        auto decoded = Image::load(png);
        if (!decoded) fail("decompress produced a file that cannot be read");
        else if (decoded->w != 128 || decoded->h != 128)
            fail("decoded image has the wrong size");
        else std::printf("  decompress -> %dx%d\n", decoded->w, decoded->h);
    }

    // --- decoding at a different scale.
    const fs::path big = g_dir / "big.png";
    expect(run("decompress " + quoted(gsi) + " --scale 2 -o " + quoted(big)), 0, "decompress 2x");
    if (fs::exists(big)) {
        auto decoded = Image::load(big);
        if (!decoded || decoded->w != 256 || decoded->h != 256)
            fail("2x decode did not double the resolution");
        else std::printf("  decompress --scale 2 -> %dx%d\n", decoded->w, decoded->h);
    }

    // --- a scale nobody meant to type is refused, rather than attempted. The
    // range check lives in the core alongside the arithmetic, so this and the
    // window agree about which scales exist.
    expect(run("decompress " + quoted(gsi) + " --scale 5000 -o " + quoted(g_dir / "huge.png")), 1,
           "absurd scale");
    if (fs::exists(g_dir / "huge.png")) fail("an out-of-range scale still wrote a file");
    expect(run("decompress " + quoted(gsi) + " --scale 0 -o " + quoted(g_dir / "zero.png")), 1,
           "zero scale");

    // --- compare must run on two real images.
    expect(run("compare " + quoted(src) + " " + quoted(png)), 0, "compare");

    // --- batch: several inputs in one invocation, one output each.
    const fs::path b1 = g_dir / "b1.png", b2 = g_dir / "b2.png";
    fs::copy_file(src, b1, fs::copy_options::overwrite_existing, ec);
    fs::copy_file(src, b2, fs::copy_options::overwrite_existing, ec);
    expect(run("compress " + quoted(b1) + " " + quoted(b2) + " --steps 150 --cpu -o " +
               quoted(g_dir)),
           0, "batch compress");
    if (!fs::exists(g_dir / "b1.gsi") || !fs::exists(g_dir / "b2.gsi"))
        fail("batch compress did not produce a file for every input");

    // --- misuse must fail cleanly rather than crash or silently succeed.
    expect(run("compress " + quoted(g_dir / "does-not-exist.png") + " -o " + quoted(g_dir)), 1,
           "missing input");
    expect(run("info " + quoted(g_dir / "does-not-exist.gsi")), 1, "missing .gsi");
    expect(run("frobnicate"), 2, "unknown command");
    expect(run("compress " + quoted(src) + " --bits 99 -o " + quoted(g_dir)), 2, "out of range bits");
    expect(run("compress"), 2, "compress with no input");

    // --- a corrupt file must be rejected, not read past its end.
    const fs::path bad = g_dir / "corrupt.gsi";
    {
        std::vector<char> bytes;
        FILE* in = std::fopen(gsi.string().c_str(), "rb");
        if (in) {
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) bytes.insert(bytes.end(), buf, buf + n);
            std::fclose(in);
        }
        for (size_t i = 40; i < bytes.size(); i += 17) bytes[i] = char(~bytes[i]);
        FILE* outf = std::fopen(bad.string().c_str(), "wb");
        if (outf) { std::fwrite(bytes.data(), 1, bytes.size(), outf); std::fclose(outf); }
    }
    expect(run("info " + quoted(bad)), 1, "corrupt .gsi");
    // Truncation at every prefix length must also be refused.
    for (int frac : {1, 10, 50, 90}) {
        const fs::path trunc = g_dir / "trunc.gsi";
        const auto full = fs::file_size(gsi);
        const auto keep = full * frac / 100;
        std::vector<char> bytes(keep);
        FILE* in = std::fopen(gsi.string().c_str(), "rb");
        // The short read is checked rather than assumed. Writing whatever the
        // vector happened to contain would make the prefix under test depend
        // on uninitialized memory, and a test whose input is not reproducible
        // cannot say anything about the parser it is aimed at.
        size_t got = 0;
        if (in) { got = std::fread(bytes.data(), 1, keep, in); std::fclose(in); }
        if (got != keep) {
            fail("could not read the first " + std::to_string(keep) + " bytes of the .gsi");
            continue;
        }
        FILE* outf = std::fopen(trunc.string().c_str(), "wb");
        if (outf) { std::fwrite(bytes.data(), 1, bytes.size(), outf); std::fclose(outf); }
        const int code = run("info " + quoted(trunc));
        if (code != 1) fail("truncated file at " + std::to_string(frac) + "% was not rejected");
    }

    fs::remove_all(g_dir, ec);
    if (failures == 0) {
        std::puts("cli end-to-end passed");
        return 0;
    }
    std::printf("%d cli failure(s)\n", failures);
    return 1;
}
