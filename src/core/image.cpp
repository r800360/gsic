#include "image.h"

#include "gaussians.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

namespace gsic {

// Refuse oversized input before allocating anything. The caps are tighter on
// 32-bit builds, where the working set has to fit in a much smaller address
// space (see gaussians.h).
static bool size_ok(int w, int h, int c, std::string* error) {
    if (w <= 0 || h <= 0 || c < 1 || c > kMaxChannels) {
        if (error) *error = "unsupported image format";
        return false;
    }
    if (w > kMaxDimension || h > kMaxDimension ||
        std::int64_t(w) * std::int64_t(h) > kMaxPixels) {
        if (error) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "image is too large for this build (%dx%d; limit is %lld pixels "
                          "and %d per side on a %d-bit build)",
                          w, h, static_cast<long long>(kMaxPixels), kMaxDimension,
                          kIs32Bit ? 32 : 64);
            *error = buf;
        }
        return false;
    }
    return true;
}

std::optional<Image> Image::load(const std::filesystem::path& path, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<Image> {
        if (error) *error = msg;
        return std::nullopt;
    };
    FILE* f = nullptr;
#ifdef _WIN32
    if (_wfopen_s(&f, path.c_str(), L"rb") != 0) f = nullptr;
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) return fail("cannot open file");

    int w = 0, h = 0, c = 0;
    Image img;
    if (stbi_is_16_bit_from_file(f)) {
        std::uint16_t* px = stbi_load_from_file_16(f, &w, &h, &c, 0);
        std::fclose(f);
        if (!px) return fail(stbi_failure_reason());
        std::string why;
        if (!size_ok(w, h, c, &why)) { stbi_image_free(px); return fail(why); }
        img = Image(w, h, c);
        constexpr float s = 1.f / 65535.f;
        for (int ch = 0; ch < c; ++ch) {
            float* dst = img.plane(ch);
            const std::uint16_t* src = px + ch;
            for (std::int64_t i = 0, n = img.pixels(); i < n; ++i) dst[i] = float(src[i * c]) * s;
        }
        stbi_image_free(px);
    } else {
        std::uint8_t* px = stbi_load_from_file(f, &w, &h, &c, 0);
        std::fclose(f);
        if (!px) return fail(stbi_failure_reason());
        std::string why;
        if (!size_ok(w, h, c, &why)) { stbi_image_free(px); return fail(why); }
        img = Image(w, h, c);
        constexpr float s = 1.f / 255.f;
        for (int ch = 0; ch < c; ++ch) {
            float* dst = img.plane(ch);
            const std::uint8_t* src = px + ch;
            for (std::int64_t i = 0, n = img.pixels(); i < n; ++i) dst[i] = float(src[i * c]) * s;
        }
        stbi_image_free(px);
    }
    return img;
}

bool Image::save_png(const std::filesystem::path& path) const {
    if (empty()) return false;
    std::vector<std::uint8_t> out(size_t(w) * h * c);
    for (int ch = 0; ch < c; ++ch) {
        const float* src = plane(ch);
        std::uint8_t* dst = out.data() + ch;
        for (std::int64_t i = 0, n = pixels(); i < n; ++i)
            dst[i * c] = std::uint8_t(std::clamp(src[i], 0.f, 1.f) * 255.f + 0.5f);
    }
    const std::string p = path.string();
    return stbi_write_png(p.c_str(), w, h, c, out.data(), w * c) != 0;
}

Image Image::downscaled_half() const {
    Image out((w + 1) / 2, (h + 1) / 2, c);
    for (int ch = 0; ch < c; ++ch) {
        const float* src = plane(ch);
        float* dst = out.plane(ch);
        for (int y = 0; y < out.h; ++y) {
            const int y0 = y * 2, y1 = std::min(y0 + 1, h - 1);
            for (int x = 0; x < out.w; ++x) {
                const int x0 = x * 2, x1 = std::min(x0 + 1, w - 1);
                dst[size_t(y) * out.w + x] =
                    0.25f * (src[size_t(y0) * w + x0] + src[size_t(y0) * w + x1] +
                             src[size_t(y1) * w + x0] + src[size_t(y1) * w + x1]);
            }
        }
    }
    return out;
}

float Image::sample_bilinear(int channel, float x, float y) const {
    x = std::clamp(x - 0.5f, 0.f, float(w - 1));
    y = std::clamp(y - 0.5f, 0.f, float(h - 1));
    int x0 = int(x), y0 = int(y);
    int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    float fx = x - x0, fy = y - y0;
    const float* p = plane(channel);
    float a = p[size_t(y0) * w + x0] * (1 - fx) + p[size_t(y0) * w + x1] * fx;
    float b = p[size_t(y1) * w + x0] * (1 - fx) + p[size_t(y1) * w + x1] * fx;
    return a * (1 - fy) + b * fy;
}

} // namespace gsic
