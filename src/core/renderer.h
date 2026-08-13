#pragma once

#include "codec.h"
#include "gaussians.h"
#include "image.h"
#include "kernels.h"

#include <optional>
#include <string>
#include <vector>

namespace gsic {

// Projects gaussians to screen (conics + bounding boxes), bins them into
// 16x16 pixel tiles, and runs the forward accumulation. Owns all the
// per-step scratch buffers so repeated steps allocate nothing.
class Renderer {
public:
    void set_size(int w, int h, int channels);

    // Recomputes projection + tile bins for the current cloud parameters.
    void project_and_bin(const GaussianCloud& cloud);

    // Renders into the given planes (each w*h floats). Requires
    // project_and_bin() with the same cloud first.
    void render(const GaussianCloud& cloud, float* const planes[4]);

    // Fills the projection/binning fields of a KernelCtx (shared with the
    // trainer's backward pass).
    void fill_ctx(KernelCtx& ctx, const GaussianCloud& cloud) const;

    int tiles_x() const { return tiles_x_; }
    int tiles_y() const { return tiles_y_; }
    int tile_count() const { return tiles_x_ * tiles_y_; }

    // Convenience: decode a cloud into a new image.
    static Image render_image(const GaussianCloud& cloud, int w, int h);

private:
    int w_ = 0, h_ = 0, channels_ = 0;
    int tiles_x_ = 0, tiles_y_ = 0;

    // Projection results, [n].
    std::vector<float> cx_, cy_, conic_a_, conic_b_, conic_c_;
    std::vector<std::int32_t> bbox_;      // [n * 4]
    std::vector<std::int32_t> tile_rect_; // [n * 4] tile-space rect, half-open

    // Binning.
    std::vector<std::uint32_t> tile_offsets_; // [tiles + 1]
    std::vector<std::uint32_t> tile_items_;
    std::vector<std::uint32_t> fill_cursor_;
};

// A parsed .gsi prepared for output at a size other than the one it was
// stored at. Because the compressed representation is continuous rather than a
// fixed raster, a scaled decode re-evaluates the gaussians at the requested
// resolution instead of resampling an already-decoded image.
struct ScaledDecode {
    GaussianCloud cloud;
    int w = 0, h = 0;
    // What the scale actually became. Equal to the request unless it was
    // clamped; the caller can then say so rather than silently producing a
    // different size than was asked for.
    float scale = 1.f;
};

// Largest and smallest decode scale offered. The upper bound is a policy
// choice rather than a format limit -- the build's pixel ceiling is checked
// separately and is what actually protects the allocation -- but a scale
// spinner with no top invites an accidental 100x that looks like a hang.
inline constexpr float kMinDecodeScale = 0.05f;
inline constexpr float kMaxDecodeScale = 8.f;

// Prepares `file` for rendering at `scale` times its stored resolution.
// Returns nullopt with a reason when the result would exceed this build's
// image limits, so an out-of-range request is refused before anything is
// allocated rather than during.
std::optional<ScaledDecode> scale_gsi(const GsiFile& file, float scale,
                                      std::string* error = nullptr);

// Decodes a parsed .gsi to an image at `scale` times its stored resolution.
// Returns an empty image (and sets `error`) when the scale is out of range.
Image render_gsi(const GsiFile& file, float scale = 1.f, std::string* error = nullptr);

} // namespace gsic
