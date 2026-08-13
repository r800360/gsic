#pragma once

#include "gaussians.h"
#include "image.h"

#include <optional>
#include <string>

namespace gsic {

// Creates the hidden OpenGL context the GPU paths share, and verifies that
// this machine's GPU computes the same answers as the CPU before allowing it
// to be used. Safe to call repeatedly. Returns false with a reason when there
// is no capable GPU or when it failed the check, in which case every caller
// falls back to the CPU automatically.
//
// Call this once, at startup, from a thread that outlives every encode --
// in practice the main thread. It is not merely a convention: the context is
// carried by a window whose message queue belongs to its creating thread, so
// a context created on a worker becomes unusable, and blocks rather than
// fails, once that worker exits. gpu_render and the GPU training backend
// therefore never create it on demand; if this was not called they report no
// GPU and the CPU does the work.
bool gpu_init(std::string* why_not);
void gpu_shutdown();

// Decodes a cloud on the GPU: one forward render, no optimization. Returns
// nullopt when the GPU is unavailable, so callers can fall back to
// Renderer::render_image.
std::optional<Image> gpu_render(const GaussianCloud& cloud, int w, int h,
                                std::string* error = nullptr);

} // namespace gsic
