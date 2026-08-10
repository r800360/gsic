#pragma once

#include "image.h"

namespace gsic {

// Peak signal-to-noise ratio in dB over all channels, inputs clamped to [0,1].
double psnr(const Image& a, const Image& b);

// Mean SSIM (11x11 gaussian window, standard constants), averaged over
// channels. Used for reporting only, not as a training loss.
double ssim(const Image& a, const Image& b);

} // namespace gsic
