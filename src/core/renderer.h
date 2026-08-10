#pragma once

#include "gaussians.h"
#include "image.h"
#include "kernels.h"

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

} // namespace gsic
