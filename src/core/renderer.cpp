#include "renderer.h"

#include "profiling.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gsic {

void Renderer::set_size(int w, int h, int channels) {
    w_ = w;
    h_ = h;
    channels_ = channels;
    tiles_x_ = (w + kTileSize - 1) / kTileSize;
    tiles_y_ = (h + kTileSize - 1) / kTileSize;
    tile_offsets_.assign(size_t(tile_count()) + 1, 0);
    fill_cursor_.assign(tile_count(), 0);
}

void Renderer::project_and_bin(const GaussianCloud& cloud) {
    GSIC_ZONE("project_and_bin");
    const int n = cloud.count();
    cx_.resize(n); cy_.resize(n);
    conic_a_.resize(n); conic_b_.resize(n); conic_c_.resize(n);
    bbox_.resize(size_t(n) * 4);
    tile_rect_.resize(size_t(n) * 4);

    // Projection: conic = R diag(u^2, v^2) R^T (inverse covariance), radius
    // from the largest std-dev max(1/u, 1/v).
    ThreadPool::instance().parallel_for(n, 4096, [&](std::int64_t b, std::int64_t e) {
        // Gaussian counts are capped well inside int range (gaussians.h), so
        // narrowing the work-range bounds here keeps every index below int
        // sized rather than silently truncating on 32-bit builds.
        for (int g = int(b); g < int(e); ++g) {
            const float u = cloud.sinv_x[g], v = cloud.sinv_y[g];
            const float co = std::cos(cloud.rot[g]), si = std::sin(cloud.rot[g]);
            const float u2 = u * u, v2 = v * v;
            cx_[g] = cloud.pos_x[g] * float(w_);
            cy_[g] = cloud.pos_y[g] * float(h_);
            conic_a_[g] = u2 * co * co + v2 * si * si;
            conic_b_[g] = (u2 - v2) * si * co;
            conic_c_[g] = u2 * si * si + v2 * co * co;
            const float radius = kRadiusFactor / std::min(u, v);

            int x0 = std::clamp(int(std::floor(cx_[g] - radius)), 0, w_);
            int x1 = std::clamp(int(std::ceil(cx_[g] + radius)) + 1, 0, w_);
            int y0 = std::clamp(int(std::floor(cy_[g] - radius)), 0, h_);
            int y1 = std::clamp(int(std::ceil(cy_[g] + radius)) + 1, 0, h_);
            bbox_[g * 4 + 0] = x0; bbox_[g * 4 + 1] = y0;
            bbox_[g * 4 + 2] = x1; bbox_[g * 4 + 3] = y1;
            tile_rect_[g * 4 + 0] = x0 / kTileSize;
            tile_rect_[g * 4 + 1] = y0 / kTileSize;
            tile_rect_[g * 4 + 2] = (std::max(x1, x0 + 1) - 1) / kTileSize + 1;
            tile_rect_[g * 4 + 3] = (std::max(y1, y0 + 1) - 1) / kTileSize + 1;
            if (x0 >= x1 || y0 >= y1) {  // off-screen: cover no tiles
                tile_rect_[g * 4 + 2] = tile_rect_[g * 4 + 0];
                tile_rect_[g * 4 + 3] = tile_rect_[g * 4 + 1];
            }
        }
    });

    // Counting sort into tiles: count, prefix-sum, fill.
    {
        GSIC_ZONE("bin_tiles");
        std::fill(tile_offsets_.begin(), tile_offsets_.end(), 0u);
        for (int g = 0; g < n; ++g) {
            const std::int32_t* r = &tile_rect_[size_t(g) * 4];
            for (int ty = r[1]; ty < r[3]; ++ty)
                for (int tx = r[0]; tx < r[2]; ++tx)
                    ++tile_offsets_[size_t(ty) * tiles_x_ + tx + 1];
        }
        for (size_t t = 1; t < tile_offsets_.size(); ++t) tile_offsets_[t] += tile_offsets_[t - 1];
        tile_items_.resize(tile_offsets_.back());
        std::copy(tile_offsets_.begin(), tile_offsets_.end() - 1, fill_cursor_.begin());
        for (int g = 0; g < n; ++g) {
            const std::int32_t* r = &tile_rect_[size_t(g) * 4];
            for (int ty = r[1]; ty < r[3]; ++ty)
                for (int tx = r[0]; tx < r[2]; ++tx)
                    tile_items_[fill_cursor_[size_t(ty) * tiles_x_ + tx]++] = std::uint32_t(g);
        }
    }
}

void Renderer::fill_ctx(KernelCtx& ctx, const GaussianCloud& cloud) const {
    ctx.w = w_;
    ctx.h = h_;
    ctx.channels = channels_;
    ctx.tiles_x = tiles_x_;
    ctx.tiles_y = tiles_y_;
    ctx.n = cloud.count();
    ctx.color = cloud.color.data();
    ctx.sinv_x = cloud.sinv_x.data();
    ctx.sinv_y = cloud.sinv_y.data();
    ctx.rot = cloud.rot.data();
    ctx.cx = cx_.data();
    ctx.cy = cy_.data();
    ctx.conic_a = conic_a_.data();
    ctx.conic_b = conic_b_.data();
    ctx.conic_c = conic_c_.data();
    ctx.bbox = bbox_.data();
    ctx.tile_offsets = tile_offsets_.data();
    ctx.tile_items = tile_items_.data();
}

void Renderer::render(const GaussianCloud& cloud, float* const planes[4]) {
    GSIC_ZONE("render");
    KernelCtx ctx;
    fill_ctx(ctx, cloud);
    for (int c = 0; c < channels_; ++c) ctx.render[c] = planes[c];
    const Kernels& kn = kernels();
    ThreadPool::instance().parallel_for(tile_count(), 8, [&](std::int64_t b, std::int64_t e) {
        kn.forward_tiles(ctx, int(b), int(e));
    });
}

Image Renderer::render_image(const GaussianCloud& cloud, int w, int h) {
    Renderer r;
    r.set_size(w, h, cloud.channels);
    r.project_and_bin(cloud);
    Image out(w, h, cloud.channels);
    float* planes[4] = {};
    for (int c = 0; c < cloud.channels; ++c) planes[c] = out.plane(c);
    r.render(cloud, planes);
    return out;
}

// ------------------------------------------------------------ scaled decode
//
// One implementation, shared by the command line tool and the desktop app.
// It used to live only inside the CLI's decompress command, which meant the
// window had no way to offer the same thing and the range checking existed in
// exactly one caller. A scale arriving from a text box is untrusted input in
// the same sense a file is: 100x on a 4K image asks for 16 gigapixels, and the
// honest answer is a message, not an allocation failure.
std::optional<ScaledDecode> scale_gsi(const GsiFile& file, float scale, std::string* error) {
    const auto fail = [&](const std::string& msg) -> std::optional<ScaledDecode> {
        if (error) *error = msg;
        return std::nullopt;
    };
    if (file.width <= 0 || file.height <= 0) return fail("the file has no usable size");
    if (!(scale > 0.f) || !std::isfinite(scale)) return fail("the scale must be a positive number");
    if (scale < kMinDecodeScale || scale > kMaxDecodeScale) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "the scale must be between %.2fx and %.0fx",
                      double(kMinDecodeScale), double(kMaxDecodeScale));
        return fail(buf);
    }

    ScaledDecode out;
    out.scale = scale;
    out.w = std::max(1, int(float(file.width) * scale + 0.5f));
    out.h = std::max(1, int(float(file.height) * scale + 0.5f));
    if (out.w > kMaxDimension || out.h > kMaxDimension ||
        std::int64_t(out.w) * std::int64_t(out.h) > kMaxPixels) {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "%.2fx would decode to %dx%d, past this build's limit of %lld pixels "
                      "and %d per side",
                      double(scale), out.w, out.h, static_cast<long long>(kMaxPixels),
                      kMaxDimension);
        return fail(buf);
    }

    out.cloud = file.cloud;
    if (out.w != file.width || out.h != file.height) {
        // Gaussian sizes are stored in pixels, so they have to grow with the
        // output or a 2x decode renders the same picture with half-size
        // splats. Positions are normalized and need no adjustment.
        const float ratio = float(out.w) / float(file.width);
        for (auto& v : out.cloud.sinv_x) v /= ratio;
        for (auto& v : out.cloud.sinv_y) v /= ratio;
    }
    return out;
}

Image render_gsi(const GsiFile& file, float scale, std::string* error) {
    auto scaled = scale_gsi(file, scale, error);
    if (!scaled) return {};
    return Renderer::render_image(scaled->cloud, scaled->w, scaled->h);
}

} // namespace gsic
