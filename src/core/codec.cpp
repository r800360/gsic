#include "codec.h"

#include <zstd.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <numbers>

namespace gsic {

namespace {

constexpr char kMagic[4] = {'G', 'S', 'I', '1'};
constexpr std::uint8_t kVersion = 1;
constexpr std::uint8_t kFlagZstd = 1;
// Dimension and gaussian caps come from gaussians.h because they depend on
// the build's address space. Both values here arrive from an untrusted file
// and drive allocations, so they are checked before anything is reserved.
constexpr float kPi = std::numbers::pi_v<float>;

// ----------------------------------------------------------- bit packing
class BitWriter {
public:
    explicit BitWriter(std::vector<std::uint8_t>& out) : out_(out) {}
    void put(std::uint32_t value, int bits) {
        acc_ |= std::uint64_t(value) << fill_;
        fill_ += bits;
        while (fill_ >= 8) {
            out_.push_back(std::uint8_t(acc_));
            acc_ >>= 8;
            fill_ -= 8;
        }
    }
    void flush() {
        if (fill_ > 0) out_.push_back(std::uint8_t(acc_));
        acc_ = 0;
        fill_ = 0;
    }
private:
    std::vector<std::uint8_t>& out_;
    std::uint64_t acc_ = 0;
    int fill_ = 0;
};

class BitReader {
public:
    BitReader(const std::uint8_t* data, size_t size) : data_(data), size_(size) {}
    std::uint32_t get(int bits) {
        while (fill_ < bits) {
            acc_ |= std::uint64_t(pos_ < size_ ? data_[pos_] : 0) << fill_;
            ++pos_;
            fill_ += 8;
        }
        std::uint32_t v = std::uint32_t(acc_ & ((1ull << bits) - 1));
        acc_ >>= bits;
        fill_ -= bits;
        return v;
    }
private:
    const std::uint8_t* data_;
    size_t size_, pos_ = 0;
    std::uint64_t acc_ = 0;
    int fill_ = 0;
};

// ------------------------------------------------------- scalar quantizers
inline std::uint32_t quantize(float v, float lo, float hi, std::uint32_t levels) {
    const float t = (std::clamp(v, lo, hi) - lo) / (hi - lo);
    return std::uint32_t(t * float(levels) + 0.5f);
}

inline float dequantize(std::uint32_t q, float lo, float hi, std::uint32_t levels) {
    return lo + (hi - lo) * (float(q) / float(levels));
}

inline float wrap_rotation(float r) {
    // Gaussians are symmetric under a half turn, so store theta in [0, pi).
    r = std::fmod(r, kPi);
    return r < 0.f ? r + kPi : r;
}

struct Writer {
    std::vector<std::uint8_t> bytes;
    void u8(std::uint8_t v) { bytes.push_back(v); }
    void u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) bytes.push_back(std::uint8_t(v >> (8 * i)));
    }
    void f32(float v) { u32(std::bit_cast<std::uint32_t>(v)); }
};

struct Reader {
    const std::uint8_t* p;
    size_t left;
    bool ok = true;
    std::uint8_t u8() { if (left < 1) { ok = false; return 0; } --left; return *p++; }
    std::uint32_t u32() {
        if (left < 4) { ok = false; return 0; }
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= std::uint32_t(p[i]) << (8 * i);
        p += 4; left -= 4;
        return v;
    }
    float f32() { return std::bit_cast<float>(u32()); }
};

void min_max(const float* v, size_t n, float& lo, float& hi) {
    lo = v[0]; hi = v[0];
    for (size_t i = 1; i < n; ++i) { lo = std::min(lo, v[i]); hi = std::max(hi, v[i]); }
    if (!(hi > lo)) hi = lo + 1e-6f;  // degenerate range: keep the divide safe
}

} // namespace

std::int64_t payload_bits(int n, int channels, const QuantSpec& q) {
    return std::int64_t(n) * (2 * q.pos + 2 * q.scale + q.rot + channels * q.feat);
}

std::vector<std::uint8_t> encode_gsi(const GaussianCloud& cloud, int width, int height,
                                     const QuantSpec& quant) {
    const int n = cloud.count();
    const int c = cloud.channels;

    // Scales quantize in log space: their distribution is heavily skewed
    // toward small inverse scales, and log spacing matches perceptual impact.
    std::vector<float> log_sx(n), log_sy(n);
    for (int i = 0; i < n; ++i) log_sx[i] = std::log(cloud.sinv_x[i]);
    for (int i = 0; i < n; ++i) log_sy[i] = std::log(cloud.sinv_y[i]);
    float s_lo, s_hi, tmp_lo, tmp_hi;
    min_max(log_sx.data(), n, s_lo, s_hi);
    min_max(log_sy.data(), n, tmp_lo, tmp_hi);
    s_lo = std::min(s_lo, tmp_lo);
    s_hi = std::max(s_hi, tmp_hi);

    float f_lo[kMaxChannels] = {}, f_hi[kMaxChannels] = {};
    std::vector<float> feat_plane(n);
    Writer w;
    w.bytes.reserve(64 + size_t(payload_bits(n, c, quant) / 8) + 16);
    w.bytes.insert(w.bytes.end(), kMagic, kMagic + 4);
    w.u8(kVersion);
    w.u8(std::uint8_t(c));
    w.u8(kFlagZstd);
    w.u8(0);
    w.u8(std::uint8_t(quant.pos));
    w.u8(std::uint8_t(quant.scale));
    w.u8(std::uint8_t(quant.rot));
    w.u8(std::uint8_t(quant.feat));
    w.u32(std::uint32_t(width));
    w.u32(std::uint32_t(height));
    w.u32(std::uint32_t(n));
    w.f32(s_lo);
    w.f32(s_hi);
    for (int ch = 0; ch < c; ++ch) {
        for (int i = 0; i < n; ++i) feat_plane[i] = cloud.color[size_t(i) * c + ch];
        min_max(feat_plane.data(), n, f_lo[ch], f_hi[ch]);
        w.f32(f_lo[ch]);
        w.f32(f_hi[ch]);
    }

    // Quantize into tightly bit-packed attribute planes (planar layout helps
    // zstd model each attribute's statistics separately).
    std::vector<std::uint8_t> raw;
    raw.reserve(size_t(payload_bits(n, c, quant) / 8) + 8);
    BitWriter bw(raw);
    const std::uint32_t pos_levels = (1u << quant.pos) - 1;
    const std::uint32_t pos_mask = pos_levels;
    const std::uint32_t scale_levels = (1u << quant.scale) - 1;
    const std::uint32_t rot_levels = (1u << quant.rot) - 1;
    const std::uint32_t feat_levels = (1u << quant.feat) - 1;
    // Positions are delta-coded. The trainer keeps the cloud in Morton
    // order, so consecutive gaussians are spatial neighbours and their
    // deltas cluster near zero, which is exactly what zstd can model. The
    // deltas are stored modulo 2^bits so decoding is an exact prefix sum.
    {
        std::uint32_t prev_x = 0, prev_y = 0;
        for (int i = 0; i < n; ++i) {
            const std::uint32_t q = quantize(cloud.pos_x[i], kPosLo, kPosHi, pos_levels);
            bw.put((q - prev_x) & pos_mask, quant.pos);
            prev_x = q;
        }
        for (int i = 0; i < n; ++i) {
            const std::uint32_t q = quantize(cloud.pos_y[i], kPosLo, kPosHi, pos_levels);
            bw.put((q - prev_y) & pos_mask, quant.pos);
            prev_y = q;
        }
    }
    for (int i = 0; i < n; ++i) bw.put(quantize(log_sx[i], s_lo, s_hi, scale_levels), quant.scale);
    for (int i = 0; i < n; ++i) bw.put(quantize(log_sy[i], s_lo, s_hi, scale_levels), quant.scale);
    for (int i = 0; i < n; ++i)
        bw.put(quantize(wrap_rotation(cloud.rot[i]), 0.f, kPi, rot_levels), quant.rot);
    for (int ch = 0; ch < c; ++ch)
        for (int i = 0; i < n; ++i)
            bw.put(quantize(cloud.color[size_t(i) * c + ch], f_lo[ch], f_hi[ch], feat_levels),
                   quant.feat);
    bw.flush();

    std::vector<std::uint8_t> compressed(ZSTD_compressBound(raw.size()));
    const size_t csize = ZSTD_compress(compressed.data(), compressed.size(),
                                       raw.data(), raw.size(), 19);
    compressed.resize(ZSTD_isError(csize) ? 0 : csize);

    w.u32(std::uint32_t(raw.size()));
    w.u32(std::uint32_t(compressed.size()));
    w.bytes.insert(w.bytes.end(), compressed.begin(), compressed.end());
    return std::move(w.bytes);
}

std::optional<GsiFile> decode_gsi(std::span<const std::uint8_t> bytes, std::string* error) {
    auto fail = [&](const char* msg) -> std::optional<GsiFile> {
        if (error) *error = msg;
        return std::nullopt;
    };
    Reader r{bytes.data(), bytes.size()};
    char magic[4];
    for (char& m : magic) m = char(r.u8());
    if (!r.ok || std::memcmp(magic, kMagic, 4) != 0) return fail("not a .gsi file");
    if (r.u8() != kVersion) return fail("unsupported version");
    const int c = r.u8();
    const std::uint8_t flags = r.u8();
    r.u8();
    QuantSpec q;
    q.pos = r.u8(); q.scale = r.u8(); q.rot = r.u8(); q.feat = r.u8();
    const std::uint32_t width = r.u32();
    const std::uint32_t height = r.u32();
    const std::uint32_t n = r.u32();
    if (!r.ok) return fail("truncated header");
    if (c < 1 || c > kMaxChannels) return fail("bad channel count");
    if (!q.valid()) return fail("bad bit depths");
    if (width == 0 || height == 0 || width > std::uint32_t(kMaxDimension) ||
        height > std::uint32_t(kMaxDimension) ||
        std::int64_t(width) * std::int64_t(height) > kMaxPixels)
        return fail("dimensions too large for this build");
    if (n == 0 || n > std::uint32_t(kMaxGaussians))
        return fail("gaussian count too large for this build");

    const float s_lo = r.f32(), s_hi = r.f32();
    float f_lo[kMaxChannels], f_hi[kMaxChannels];
    for (int ch = 0; ch < c; ++ch) { f_lo[ch] = r.f32(); f_hi[ch] = r.f32(); }
    const std::uint32_t raw_size = r.u32();
    const std::uint32_t comp_size = r.u32();
    if (!r.ok) return fail("truncated header");
    if (!std::isfinite(s_lo) || !std::isfinite(s_hi)) return fail("bad scale range");
    for (int ch = 0; ch < c; ++ch)
        if (!std::isfinite(f_lo[ch]) || !std::isfinite(f_hi[ch])) return fail("bad color range");

    const std::uint64_t expect_bits = payload_bits(int(n), c, q);
    if (raw_size != (expect_bits + 7) / 8) return fail("payload size mismatch");
    if (comp_size > r.left) return fail("truncated payload");

    std::vector<std::uint8_t> raw(raw_size);
    if (flags & kFlagZstd) {
        const size_t got = ZSTD_decompress(raw.data(), raw.size(), r.p, comp_size);
        if (ZSTD_isError(got) || got != raw_size) return fail("corrupt payload");
    } else {
        if (comp_size != raw_size) return fail("payload size mismatch");
        std::memcpy(raw.data(), r.p, raw_size);
    }

    GsiFile file;
    file.width = int(width);
    file.height = int(height);
    file.quant = q;
    GaussianCloud& cloud = file.cloud;
    cloud.channels = c;
    cloud.resize(int(n));

    BitReader br(raw.data(), raw.size());
    const std::uint32_t pos_levels = (1u << q.pos) - 1;
    const std::uint32_t pos_mask = pos_levels;
    const std::uint32_t scale_levels = (1u << q.scale) - 1;
    const std::uint32_t rot_levels = (1u << q.rot) - 1;
    const std::uint32_t feat_levels = (1u << q.feat) - 1;
    {
        std::uint32_t acc = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            acc = (acc + br.get(q.pos)) & pos_mask;
            cloud.pos_x[i] = dequantize(acc, kPosLo, kPosHi, pos_levels);
        }
        acc = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            acc = (acc + br.get(q.pos)) & pos_mask;
            cloud.pos_y[i] = dequantize(acc, kPosLo, kPosHi, pos_levels);
        }
    }
    for (std::uint32_t i = 0; i < n; ++i)
        cloud.sinv_x[i] = std::exp(dequantize(br.get(q.scale), s_lo, s_hi, scale_levels));
    for (std::uint32_t i = 0; i < n; ++i)
        cloud.sinv_y[i] = std::exp(dequantize(br.get(q.scale), s_lo, s_hi, scale_levels));
    for (std::uint32_t i = 0; i < n; ++i)
        cloud.rot[i] = dequantize(br.get(q.rot), 0.f, kPi, rot_levels);
    for (int ch = 0; ch < c; ++ch)
        for (std::uint32_t i = 0; i < n; ++i)
            cloud.color[size_t(i) * c + ch] =
                dequantize(br.get(q.feat), f_lo[ch], f_hi[ch], feat_levels);
    return file;
}

void quantize_cloud(GaussianCloud& cloud, const QuantSpec& quant) {
    // Roundtrip through the exact encode/decode transforms.
    auto bytes = encode_gsi(cloud, 1, 1, quant);
    auto file = decode_gsi(bytes);
    if (file) cloud = std::move(file->cloud);
}

} // namespace gsic
