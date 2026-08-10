#pragma once

#include "gaussians.h"
#include "image.h"

#include <optional>
#include <string>

namespace gsic {

// Creates the hidden OpenGL context the GPU paths share. Must be called from
// the main thread (a GLFW requirement); safe to call repeatedly. Returns
// false with a reason when no capable GPU is present, in which case every
// caller falls back to the CPU automatically.
bool gpu_init(std::string* why_not);
void gpu_shutdown();

// Decodes a cloud on the GPU: one forward render, no optimization. Returns
// nullopt when the GPU is unavailable, so callers can fall back to
// Renderer::render_image.
std::optional<Image> gpu_render(const GaussianCloud& cloud, int w, int h,
                                std::string* error = nullptr);

} // namespace gsic
