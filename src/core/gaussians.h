#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsic {

// Address space budget, and the reason these limits depend on the build.
//
// A 32-bit process has under 2 GB of usable address space. Encoding peaks at
// roughly 50 bytes per pixel (the target, the gradient planes, and the
// transient error maps used when growing the cloud), so the ceiling on image
// size is genuinely lower there. Just as important, `size_t` is 32 bits, so
// an expression like size_t(w) * h * c silently wraps well before that.
//
// These are therefore hard limits, enforced when loading an image and when
// parsing a file, so a 32-bit build refuses oversized input with a clear
// message instead of overflowing or exhausting memory. The gaussian cap in
// particular is a security limit: the count comes from an untrusted file and
// drives an allocation.
constexpr bool kIs32Bit = sizeof(void*) == 4;
constexpr int          kMaxDimension = kIs32Bit ? 16384 : 32768;
constexpr std::int64_t kMaxPixels    = kIs32Bit ? 16'000'000 : 268'435'456;
constexpr int          kMaxGaussians = kIs32Bit ? 1'500'000 : 20'000'000;

// Limits shared by the trainer, codec and file-format validation.
constexpr int   kMaxChannels   = 4;
constexpr float kMinScalePx    = 0.3f;   // smallest gaussian std-dev, pixels
// Gaussians are truncated where sigma = 0.5 d^T Conic d reaches the cutoff
// (bounding radius = sqrt(2 * cutoff) std-devs). Splatting cost scales
// linearly with the cutoff; 3.5 measures within ~0.1 dB of 4.5 while
// rendering ~20% fewer pixels.
constexpr float kSigmaCutoff   = 3.5f;
constexpr float kRadiusFactor  = 2.6458f;  // sqrt(2 * kSigmaCutoff)
// Rendered alpha is exp(-sigma) - kAlphaBias, clamped at zero: continuous at
// the cutoff ellipse, which removes popping during optimization and keeps
// gradients honest at the boundary.
constexpr float kAlphaBias     = 0.030197f;  // exp(-kSigmaCutoff)
// Color clamp during optimization. Colors may go negative (subtractive
// detail) but an unbounded outlier would stretch the quantization range for
// every gaussian in the file, ruining decoded quality.
constexpr float kColorLo       = -2.f;
constexpr float kColorHi       = 3.f;
// Position clamp, in normalized image coordinates. Gaussians may sit just
// outside the image so their tails still cover the border; the codec
// quantizes over this same range, so every bit spent on the margin is a bit
// not spent on the image. 3% ~= 60px at 2K, which comfortably exceeds the
// largest allowed footprint.
constexpr float kPosLo         = -0.03f;
constexpr float kPosHi         = 1.03f;

// A set of anisotropic 2D gaussians, struct-of-arrays for SIMD-friendly access.
//
// Per gaussian: position (normalized [0,1] image coordinates), inverse scale
// (1 / std-dev in pixels; optimizing the inverse is more stable, following
// Image-GS), rotation angle, and `channels` color features. Colors are
// unbounded: the renderer accumulates alpha-weighted sums, so negative values
// let gaussians subtract detail.
struct GaussianCloud {
    int channels = 3;
    std::vector<float> pos_x, pos_y;     // normalized [0, 1]
    std::vector<float> sinv_x, sinv_y;   // inverse scale, pixels^-1
    std::vector<float> rot;              // radians
    std::vector<float> color;            // [n * channels], interleaved

    int count() const { return int(pos_x.size()); }

    void resize(int n) {
        pos_x.resize(n); pos_y.resize(n);
        sinv_x.resize(n); sinv_y.resize(n);
        rot.resize(n);
        color.resize(size_t(n) * channels);
    }
};

} // namespace gsic
