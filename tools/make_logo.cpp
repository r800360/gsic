// Deterministic generator for the application icon and Microsoft Store assets.
//
// The mark is literally what the codec computes: three anisotropic 2D
// Gaussians accumulated additively over a dark plate, the same model as
// Equation (1) in the report. Everything here is fixed constants and exact
// double-precision math with no randomness and no platform drawing API, so
// the bytes are identical on every machine and every run. Rendering happens
// at 16x linear supersampling and is box filtered down, which is what keeps
// the edges crisp at 16 pixels instead of relying on a library's antialiasing.
//
// Run `cmake --build <dir> --target logo` to regenerate the assets.

#include "core/image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <vector>

#include <stb_image_write.h>

namespace fs = std::filesystem;
using gsic::Image;

namespace {

constexpr int kSupersample = 16;

// One Gaussian of the mark. Position and scale are fractions of the shorter
// canvas side, so the mark is resolution independent.
struct Blob {
    double cx, cy;        // centre
    double sx, sy;        // standard deviations along the blob's own axes
    double rot_deg;       // rotation of those axes
    double r, g, b;       // linear colour, accumulated additively
};

// Palette matches the website accent colours. The blobs are kept small
// enough and far enough apart that three distinct lobes are still readable
// at 16 pixels; overlapping them heavily just produces a white smudge.
constexpr Blob kBlobs[] = {
    {0.366, 0.422, 0.150, 0.116, -32.0, 0.400, 0.490, 1.000},  // indigo
    {0.634, 0.422, 0.150, 0.116,  32.0, 1.000, 0.420, 0.620},  // pink
    {0.500, 0.656, 0.150, 0.116,   0.0, 0.280, 0.910, 0.780},  // teal
};

// Plate colour and corner rounding, as a fraction of the shorter side.
constexpr double kPlateR = 0.070, kPlateG = 0.070, kPlateB = 0.098;
constexpr double kCornerRadius = 0.215;
// How much brighter a region gets per extra overlapping splat.
constexpr double kOverlapBoost = 0.52;

// A Gaussian evaluated directly is fog: its visible extent runs to three
// standard deviations while only the innermost one carries real weight, so
// the mark reads as a smudge at small sizes. Drawing each Gaussian at a fixed
// iso-level instead gives its footprint ellipse, which is the same object the
// renderer bounds when it bins a splat, and has a definite edge. kIsoWidth is
// the half width of the transition either side of that level; supersampling
// does the antialiasing, so it only needs to be wide enough to avoid a
// mechanical looking edge.
constexpr double kIsoLevel = 0.46;
constexpr double kIsoWidth = 0.10;

double smoothstep(double edge0, double edge1, double x) {
    const double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Signed distance to a rounded rectangle centred at the origin, negative
// inside. Used as a hard mask; supersampling turns it into a clean edge.
double rounded_rect_sdf(double px, double py, double half_w, double half_h, double r) {
    const double qx = std::abs(px) - (half_w - r);
    const double qy = std::abs(py) - (half_h - r);
    const double ax = std::max(qx, 0.0), ay = std::max(qy, 0.0);
    return std::sqrt(ax * ax + ay * ay) + std::min(std::max(qx, qy), 0.0) - r;
}

// Each blob's inverse covariance in pixel units, for a mark of `unit` pixels
// centred at (cx, cy). Shared by the tile renderer and the listing art
// renderer so the mark is the same object at every size.
struct Prepared { double px, py, a, b, c, r, g, bl; };

std::vector<Prepared> prepare_blobs(double unit, double cx, double cy) {
    std::vector<Prepared> blobs;
    for (const Blob& s : kBlobs) {
        const double th = s.rot_deg * std::numbers::pi / 180.0;
        const double co = std::cos(th), si = std::sin(th);
        const double ix = 1.0 / (s.sx * unit), iy = 1.0 / (s.sy * unit);
        const double ix2 = ix * ix, iy2 = iy * iy;
        blobs.push_back({cx + (s.cx - 0.5) * unit,
                         cy + (s.cy - 0.5) * unit,
                         ix2 * co * co + iy2 * si * si,
                         (ix2 - iy2) * si * co,
                         ix2 * si * si + iy2 * co * co,
                         s.r, s.g, s.b});
    }
    return blobs;
}

// Shades one sample: the plate, with whatever splats cover it accumulated on
// top. Factored out so the tile and the listing art cannot drift apart in
// how the mark itself is coloured.
void shade(double px, double py, const std::vector<Prepared>& blobs, const double plate[3],
           double out_rgb[3]) {
    double sum[3] = {0, 0, 0};
    double coverage = 0.0;
    for (const Prepared& p : blobs) {
        const double dx = px - p.px, dy = py - p.py;
        const double sigma = 0.5 * (p.a * dx * dx + p.c * dy * dy) + p.b * dx * dy;
        const double cov =
            smoothstep(kIsoLevel - kIsoWidth, kIsoLevel + kIsoWidth, std::exp(-sigma));
        if (cov <= 0.0) continue;
        sum[0] += cov * p.r;
        sum[1] += cov * p.g;
        sum[2] += cov * p.bl;
        coverage += cov;
    }
    for (int ch = 0; ch < 3; ++ch) out_rgb[ch] = plate[ch];
    if (coverage > 1e-9) {
        const double a = std::min(coverage, 1.0);
        const double boost = 1.0 + kOverlapBoost * std::max(coverage - 1.0, 0.0);
        for (int ch = 0; ch < 3; ++ch) {
            const double base = sum[ch] / coverage;
            out_rgb[ch] = out_rgb[ch] * (1.0 - a) + base * boost * a;
        }
    }
}

// Renders the mark at w x h with supersampling. `inset` shrinks the plate so
// a wide tile can hold a square mark with margin.
Image render_mark(int w, int h) {
    Image out(w, h, 4);
    const int ss = kSupersample;
    const double unit = std::min(w, h);          // the mark's reference size
    const double cx = w * 0.5, cy = h * 0.5;
    // The plate fills the whole canvas, so a wide tile is a wide plate rather
    // than a square one floating in transparency. The mark itself is always
    // sized from the shorter side, so it stays identical across shapes.
    const double half_w = w * 0.5, half_h = h * 0.5;
    const double radius = unit * kCornerRadius;

    const std::vector<Prepared> blobs = prepare_blobs(unit, cx, cy);

    const double inv_samples = 1.0 / (ss * ss);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double acc[4] = {0, 0, 0, 0};
            for (int sy = 0; sy < ss; ++sy) {
                for (int sx = 0; sx < ss; ++sx) {
                    // Sample at subpixel centres.
                    const double px = x + (sx + 0.5) / ss;
                    const double py = y + (sy + 0.5) / ss;

                    if (rounded_rect_sdf(px - cx, py - cy, half_w, half_h, radius) > 0.0)
                        continue;

                    // Coverage-weighted colour, normalized by total coverage:
                    // a lone splat keeps full saturation instead of being
                    // dragged toward the plate, while a region covered by two
                    // lands on the blend of their hues rather than on white.
                    const double plate[3] = {kPlateR, kPlateG, kPlateB};
                    double rgb[3];
                    shade(px, py, blobs, plate, rgb);
                    for (int ch = 0; ch < 3; ++ch) acc[ch] += std::min(rgb[ch], 1.0);
                    acc[3] += 1.0;
                }
            }
            for (int ch = 0; ch < 4; ++ch)
                out.plane(ch)[std::size_t(y) * w + x] = float(acc[ch] * inv_samples);
        }
    }
    // Undo the premultiplication introduced by masked accumulation so partly
    // covered edge pixels keep full colour against their own alpha.
    for (std::int64_t i = 0, n = out.pixels(); i < n; ++i) {
        const float a = out.plane(3)[i];
        if (a > 1e-4f)
            for (int ch = 0; ch < 3; ++ch) out.plane(ch)[i] /= a;
    }
    return out;
}


// ---------------------------------------------------------------- listing art
//
// The promotional images the Store composites into its own layouts: poster
// art, box art and the wide hero banner. These are not tiles, so they are
// full bleed with square corners -- a rounded plate inside a Store layout
// reads as a mistake -- and the mark sits at a chosen size and position
// rather than filling the canvas.
//
// There is no text in any of them. The hero slot forbids the product title
// outright, and for the other two a title would have to be baked in per
// locale, which is exactly what the Store's own overlay does better.
constexpr double kGlowR = 0.145, kGlowG = 0.155, kGlowB = 0.235;
constexpr double kGlowStrength = 0.85;

// `mark_scale` is the mark's reference size as a fraction of the shorter
// side; `cx_frac`/`cy_frac` place its centre in the canvas.
Image render_art(int w, int h, double mark_scale, double cx_frac, double cy_frac,
                 int ss = 4) {
    Image out(w, h, 4);
    const double unit = std::min(w, h) * mark_scale;
    const double cx = w * cx_frac, cy = h * cy_frac;
    const std::vector<Prepared> blobs = prepare_blobs(unit, cx, cy);

    // A flat field at this size looks dead, so the plate lifts slightly
    // toward the mark. One wide Gaussian, same model as everything else
    // here, which keeps the asset deterministic and dependency free.
    const double glow_sigma = unit * 0.95;
    const double inv_two_sigma2 = 1.0 / (2.0 * glow_sigma * glow_sigma);
    const double inv_samples = 1.0 / (ss * ss);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double acc[4] = {0, 0, 0, 0};
            for (int sy = 0; sy < ss; ++sy) {
                for (int sx = 0; sx < ss; ++sx) {
                    const double px = x + (sx + 0.5) / ss;
                    const double py = y + (sy + 0.5) / ss;

                    const double dx = px - cx, dy = py - cy;
                    const double g = kGlowStrength *
                                     std::exp(-(dx * dx + dy * dy) * inv_two_sigma2);
                    const double plate[3] = {kPlateR + (kGlowR - kPlateR) * g,
                                             kPlateG + (kGlowG - kPlateG) * g,
                                             kPlateB + (kGlowB - kPlateB) * g};
                    double rgb[3];
                    shade(px, py, blobs, plate, rgb);
                    for (int ch = 0; ch < 3; ++ch) acc[ch] += std::min(rgb[ch], 1.0);
                    acc[3] += 1.0;
                }
            }
            for (int ch = 0; ch < 4; ++ch)
                out.plane(ch)[std::size_t(y) * w + x] = float(acc[ch] * inv_samples);
        }
    }
    return out;
}

std::vector<std::uint8_t> to_rgba(const Image& img) {
    std::vector<std::uint8_t> px(std::size_t(img.w) * img.h * 4);
    for (std::int64_t i = 0, n = img.pixels(); i < n; ++i)
        for (int c = 0; c < 4; ++c)
            px[std::size_t(i) * 4 + c] =
                std::uint8_t(std::clamp(img.plane(c)[i], 0.f, 1.f) * 255.f + 0.5f);
    return px;
}

bool write_png(const Image& img, const fs::path& path) {
    const auto px = to_rgba(img);
    return stbi_write_png(path.string().c_str(), img.w, img.h, 4, px.data(), img.w * 4) != 0;
}

// Collects the bytes stb hands back when encoding to memory.
void append_bytes(void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<std::uint8_t>*>(ctx);
    const auto* p = static_cast<const std::uint8_t*>(data);
    v->insert(v->end(), p, p + size);
}

// A Windows .ico holding PNG-compressed frames, which every Windows version
// since Vista reads. Assembling it here avoids depending on an image tool.
bool write_ico(const std::vector<Image>& frames, const fs::path& path) {
    struct Encoded { std::vector<std::uint8_t> bytes; int size; };
    std::vector<Encoded> encoded;
    for (const Image& f : frames) {
        const auto px = to_rgba(f);
        std::vector<std::uint8_t> bytes;
        if (!stbi_write_png_to_func(append_bytes, &bytes, f.w, f.h, 4, px.data(), f.w * 4))
            return false;
        encoded.push_back({std::move(bytes), f.w});
    }

    std::vector<std::uint8_t> out;
    const auto u16 = [&](unsigned v) { out.push_back(std::uint8_t(v)); out.push_back(std::uint8_t(v >> 8)); };
    const auto u32 = [&](unsigned v) { for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i))); };

    u16(0);                                  // reserved
    u16(1);                                  // type: icon
    u16(unsigned(encoded.size()));
    unsigned offset = unsigned(6 + 16 * encoded.size());
    for (const auto& e : encoded) {
        out.push_back(std::uint8_t(e.size == 256 ? 0 : e.size));  // 0 means 256
        out.push_back(std::uint8_t(e.size == 256 ? 0 : e.size));
        out.push_back(0);                    // palette size
        out.push_back(0);                    // reserved
        u16(1);                              // colour planes
        u16(32);                             // bits per pixel
        u32(unsigned(e.bytes.size()));
        u32(offset);
        offset += unsigned(e.bytes.size());
    }
    for (const auto& e : encoded) out.insert(out.end(), e.bytes.begin(), e.bytes.end());

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
    return bool(f);
}

} // namespace

int main(int argc, char** argv) {
    const fs::path root = argc > 1 ? argv[1] : ".";
    const fs::path assets = root / "packaging" / "msix" / "Assets";
    std::error_code ec;
    fs::create_directories(assets, ec);

    // Square assets. Each is rendered at its own size rather than downscaled
    // from one master, so the corner radius and blob edges stay exact.
    struct Target { int w, h; const char* name; };
    const Target squares[] = {
        {44, 44, "Square44x44Logo.png"},
        {50, 50, "StoreLogo.png"},
        {150, 150, "Square150x150Logo.png"},
        {256, 256, "icon.png"},
        {300, 300, "StoreTile300x300.png"},   // Store listing tile
    };
    for (const Target& t : squares) {
        const Image img = render_mark(t.w, t.h);
        if (!write_png(img, assets / t.name)) {
            std::fprintf(stderr, "failed to write %s\n", t.name);
            return 1;
        }
        std::printf("  %-24s %dx%d\n", t.name, t.w, t.h);
    }

    const Image wide = render_mark(310, 150);
    if (!write_png(wide, assets / "Wide310x150Logo.png")) return 1;
    std::printf("  %-24s 310x150\n", "Wide310x150Logo.png");

    // GitHub Pages only serves files under docs/, so the site needs its own
    // copy. Emitting it here keeps it from drifting out of sync with the
    // Store assets.
    const fs::path site = root / "docs";
    if (fs::exists(site)) {
        if (!write_png(render_mark(256, 256), site / "icon.png")) return 1;
        std::printf("  %-24s 256x256 (website)\n", "docs/icon.png");
    }

    // ------------------------------------------------------- Store listing
    // Uploaded in Partner Center rather than shipped in the package: the
    // logo slots the listing offers, and the three tile sizes that override
    // the package's own logos. Kept in one folder beside the screenshots so
    // the set a submission needs is in one place.
    const fs::path store = root / "packaging" / "store";
    fs::create_directories(store, ec);

    struct Art { int w, h; double scale, cx, cy; const char* name; };
    const Art listing[] = {
        // 1:1 box art. Used in several layouts, and the main logo when no
        // poster art is given, so it gets the most generous mark.
        {2160, 2160, 0.81, 0.50, 0.50, "boxart-1x1-2160.png"},
        // 9:16 poster art, the main logo on Windows 10/11. The mark sits
        // slightly high: the Store draws the title underneath it.
        {1440, 2160, 0.88, 0.50, 0.44, "poster-9x16-1440x2160.png"},
        // 16:9 hero banner across the top of the listing. The title and the
        // install button are overlaid on the left, so the mark goes right of
        // centre and leaves that side quiet.
        {3840, 2160, 0.72, 0.66, 0.50, "superhero-16x9-3840x2160.png"},
    };
    for (const Art& a : listing) {
        const Image img = render_art(a.w, a.h, a.scale, a.cx, a.cy);
        if (!write_png(img, store / a.name)) {
            std::fprintf(stderr, "failed to write %s\n", a.name);
            return 1;
        }
        std::printf("  %-32s %dx%d\n", a.name, a.w, a.h);
    }

    const Target tiles[] = {
        {300, 300, "tile-1x1-300.png"},
        {150, 150, "tile-1x1-150.png"},
        {71, 71, "tile-1x1-71.png"},
    };
    for (const Target& t : tiles) {
        if (!write_png(render_mark(t.w, t.h), store / t.name)) return 1;
        std::printf("  %-32s %dx%d\n", t.name, t.w, t.h);
    }

    // Executable icon: the sizes Windows actually picks between.
    std::vector<Image> frames;
    for (int s : {16, 24, 32, 48, 64, 128, 256}) frames.push_back(render_mark(s, s));
    const fs::path ico = root / "packaging" / "gsic.ico";
    if (!write_ico(frames, ico)) {
        std::fprintf(stderr, "failed to write %s\n", ico.string().c_str());
        return 1;
    }
    std::printf("  %-24s 16..256\n", "gsic.ico");
    return 0;
}
