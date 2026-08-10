#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gsic {

// Planar float image: `c` channels of h*w floats in [0, 1], stored channel
// after channel. Planar layout keeps the hot per-channel loops contiguous.
struct Image {
    int w = 0, h = 0, c = 0;
    std::vector<float> data;

    Image() = default;
    Image(int w_, int h_, int c_) : w(w_), h(h_), c(c_), data(size_t(w_) * h_ * c_, 0.f) {}

    float*       plane(int i)       { return data.data() + size_t(i) * w * h; }
    const float* plane(int i) const { return data.data() + size_t(i) * w * h; }
    std::int64_t pixels() const { return std::int64_t(w) * h; }
    bool empty() const { return data.empty(); }

    // Loads any format stb_image supports (PNG, JPG, BMP, TGA, ...).
    // 16-bit PNGs keep their precision until quantization. Returns an error
    // message on failure.
    static std::optional<Image> load(const std::filesystem::path& path, std::string* error = nullptr);

    bool save_png(const std::filesystem::path& path) const;

    // Bilinear sample of one channel at pixel-space coordinates.
    float sample_bilinear(int channel, float x, float y) const;

    // 2x2 box-filtered half-resolution copy (used by the coarse warmup).
    Image downscaled_half() const;
};

} // namespace gsic
