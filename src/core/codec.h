#pragma once

#include "gaussians.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace gsic {

// Bits per stored component. Positions/scales/rotation use one value each per
// axis; every color channel uses `feat` bits.
//
// The defaults are not uniform because the attributes are not equally
// sensitive. Position error displaces a whole gaussian, so at 2K a 12-bit
// coordinate already costs ~0.5px of placement; rotation only reorients an
// ellipse and 10 bits is below the visible threshold. Colors sit in between:
// their errors add across the ~10 gaussians covering a pixel, so they need
// more precision than a single value's magnitude suggests.
struct QuantSpec {
    int pos = 16;
    int scale = 12;
    int rot = 10;
    int feat = 12;

    bool valid() const {
        auto ok = [](int b) { return b >= 4 && b <= 16; };
        return ok(pos) && ok(scale) && ok(rot) && ok(feat);
    }
};

struct GsiFile {
    GaussianCloud cloud;
    int width = 0, height = 0;
    QuantSpec quant;
};

// Compressed byte size a cloud of n gaussians will produce before entropy
// coding (useful for showing expected size in the UI).
std::int64_t payload_bits(int n, int channels, const QuantSpec& q);

// Serializes to the .gsi format: quantized parameter planes (scale in log
// domain, rotation wrapped to [0, pi)) packed tightly and zstd-compressed.
//
// Guarantees that whatever it returns non-empty, decode_gsi accepts. Anything
// it cannot represent that way -- an empty cloud, dimensions past the build's
// limits, parameters the optimizer left non-finite -- comes back as an empty
// vector instead of bytes that fail in a decoder somewhere else.
std::vector<std::uint8_t> encode_gsi(const GaussianCloud& cloud, int width, int height,
                                     const QuantSpec& quant);

// Strictly validated parse; returns nullopt (and an error message) on any
// malformed input. Safe to run on untrusted files.
std::optional<GsiFile> decode_gsi(std::span<const std::uint8_t> bytes, std::string* error = nullptr);

// In-place quantize/dequantize of cloud parameters, exactly matching what the
// file roundtrip produces. Called before final quality measurement so the
// reported PSNR reflects the decoded file. False means the roundtrip did not
// survive, in which case the cloud is left untouched and the caller must not
// treat the reported quality as describing the file.
bool quantize_cloud(GaussianCloud& cloud, const QuantSpec& quant);

} // namespace gsic
