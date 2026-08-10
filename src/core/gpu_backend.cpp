// OpenGL 4.3 compute backend. Mirrors the CPU pipeline one-to-one:
//   project -> bin (count/scan/fill) -> forward -> loss grad -> reduce
//   -> backward -> adam
// entirely on the GPU; the only per-step readback is the 8-byte loss value.
// Requires GL 4.3 (2012+ hardware on Windows/Linux); macOS has no compute
// support in OpenGL, so it always uses the CPU path there.
#include "gpu.h"
#include "trainer.h"

#if defined(GSIC_ENABLE_GPU)

#include "gaussians.h"
#include "profiling.h"

#include <glad/glad.h>
// glad must come first.
#include <GLFW/glfw3.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace gsic {

namespace {

// ------------------------------------------------------------------ shaders
// Common prelude: parameter/gradient buffers are laid out as five geometry
// groups of n floats (pos_x, pos_y, sinv_x, sinv_y, rot) followed by n*C
// colors, matching the CPU trainer.
constexpr const char* kPrelude = R"GLSL(#version 430
layout(local_size_x = LOCAL_X, local_size_y = LOCAL_Y) in;
uniform int uN;          // gaussian count
uniform int uW;          // image width
uniform int uH;          // image height
uniform int uC;          // channels
uniform int uTilesX;
uniform int uTilesY;
const int TILE = 16;
const float SIGMA_CUTOFF = 3.5;
const float RADIUS_FACTOR = 2.6458;
const float ALPHA_BIAS = 0.030197;   // exp(-SIGMA_CUTOFF); keeps alpha continuous
)GLSL";

constexpr const char* kProject = R"GLSL(
layout(std430, binding = 0) buffer Params { float params[]; };
layout(std430, binding = 1) buffer Proj { float proj[]; };     // 5 groups of n
layout(std430, binding = 2) buffer Bbox { ivec4 bbox[]; };
layout(std430, binding = 3) buffer TileCount { int tile_count[]; };
void main() {
    int g = int(gl_GlobalInvocationID.x);
    if (g >= uN) return;
    float u = params[2 * uN + g], v = params[3 * uN + g];
    float co = cos(params[4 * uN + g]), si = sin(params[4 * uN + g]);
    float u2 = u * u, v2 = v * v;
    float cx = params[g] * float(uW);
    float cy = params[uN + g] * float(uH);
    proj[g] = cx;
    proj[uN + g] = cy;
    proj[2 * uN + g] = u2 * co * co + v2 * si * si;
    proj[3 * uN + g] = (u2 - v2) * si * co;
    proj[4 * uN + g] = u2 * si * si + v2 * co * co;
    float radius = RADIUS_FACTOR / min(u, v);
    int x0 = clamp(int(floor(cx - radius)), 0, uW);
    int x1 = clamp(int(ceil(cx + radius)) + 1, 0, uW);
    int y0 = clamp(int(floor(cy - radius)), 0, uH);
    int y1 = clamp(int(ceil(cy + radius)) + 1, 0, uH);
    bbox[g] = ivec4(x0, y0, x1, y1);
    if (x0 >= x1 || y0 >= y1) return;
    int tx0 = x0 / TILE, tx1 = (x1 - 1) / TILE + 1;
    int ty0 = y0 / TILE, ty1 = (y1 - 1) / TILE + 1;
    for (int ty = ty0; ty < ty1; ++ty)
        for (int tx = tx0; tx < tx1; ++tx)
            atomicAdd(tile_count[ty * uTilesX + tx], 1);
}
)GLSL";

// Exclusive scan of tile counts into offsets; one workgroup, chunked so any
// tile count works. Also copies offsets into the fill cursors.
constexpr const char* kScan = R"GLSL(
layout(std430, binding = 0) buffer TileCount { int tile_count[]; };
layout(std430, binding = 1) buffer TileOffset { int tile_offset[]; };  // T + 1
layout(std430, binding = 2) buffer Cursor { int cursor[]; };
shared int chunk_sum[LOCAL_X];
void main() {
    int T = uTilesX * uTilesY;
    int tid = int(gl_LocalInvocationID.x);
    int chunk = (T + LOCAL_X - 1) / LOCAL_X;
    int begin = tid * chunk, end = min(begin + chunk, T);
    int s = 0;
    for (int i = begin; i < end; ++i) s += tile_count[i];
    chunk_sum[tid] = s;
    barrier();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < LOCAL_X; ++i) { int t = chunk_sum[i]; chunk_sum[i] = acc; acc += t; }
        tile_offset[T] = acc;
    }
    barrier();
    int acc = chunk_sum[tid];
    for (int i = begin; i < end; ++i) {
        tile_offset[i] = acc;
        cursor[i] = acc;
        acc += tile_count[i];
    }
}
)GLSL";

constexpr const char* kFill = R"GLSL(
layout(std430, binding = 0) buffer Bbox { ivec4 bbox[]; };
layout(std430, binding = 1) buffer Cursor { int cursor[]; };
layout(std430, binding = 2) buffer Items { int items[]; };
uniform int uItemCapacity;
layout(std430, binding = 3) buffer Overflow { int overflow; };
void main() {
    int g = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 32768u);
    if (g >= uN) return;
    ivec4 b = bbox[g];
    if (b.x >= b.z || b.y >= b.w) return;
    int tx0 = b.x / TILE, tx1 = (b.z - 1) / TILE + 1;
    int ty0 = b.y / TILE, ty1 = (b.w - 1) / TILE + 1;
    for (int ty = ty0; ty < ty1; ++ty)
        for (int tx = tx0; tx < tx1; ++tx) {
            int idx = atomicAdd(cursor[ty * uTilesX + tx], 1);
            if (idx < uItemCapacity) items[idx] = g;
            else atomicMax(overflow, 1);
        }
}
)GLSL";

// One 16x16 workgroup per tile; gaussians stream through shared memory in
// batches of 256 while every thread accumulates its own pixel.
constexpr const char* kForward = R"GLSL(
layout(std430, binding = 0) buffer Proj { float proj[]; };
layout(std430, binding = 1) buffer Params { float params[]; };  // colors at 5n
layout(std430, binding = 2) buffer TileOffset { int tile_offset[]; };
layout(std430, binding = 3) buffer Items { int items[]; };
layout(std430, binding = 4) buffer RenderImg { float render_img[]; };  // C planes
shared vec2 s_xy[256];
shared vec3 s_conic[256];
shared vec4 s_color[256];
void main() {
    int tile = int(gl_WorkGroupID.y) * uTilesX + int(gl_WorkGroupID.x);
    int px = int(gl_GlobalInvocationID.x);
    int py = int(gl_GlobalInvocationID.y);
    bool inside = px < uW && py < uH;
    float fx = float(px) + 0.5, fy = float(py) + 0.5;
    int tid = int(gl_LocalInvocationIndex);
    int begin = tile_offset[tile], end = tile_offset[tile + 1];
    vec4 acc = vec4(0.0);
    for (int base = begin; base < end; base += 256) {
        int load = base + tid;
        if (load < end) {
            int g = items[load];
            s_xy[tid] = vec2(proj[g], proj[uN + g]);
            s_conic[tid] = vec3(proj[2 * uN + g], proj[3 * uN + g], proj[4 * uN + g]);
            vec4 col = vec4(0.0);
            for (int c = 0; c < uC; ++c) col[c] = params[5 * uN + g * uC + c];
            s_color[tid] = col;
        }
        barrier();
        int count = min(256, end - base);
        for (int i = 0; i < count; ++i) {
            vec2 d = vec2(fx, fy) - s_xy[i];
            vec3 k = s_conic[i];
            float sigma = 0.5 * (k.x * d.x * d.x + k.z * d.y * d.y) + k.y * d.x * d.y;
            if (sigma < SIGMA_CUTOFF) acc += (exp(-sigma) - ALPHA_BIAS) * s_color[i];
        }
        barrier();
    }
    if (inside) {
        int pix = py * uW + px;
        for (int c = 0; c < uC; ++c) render_img[c * uW * uH + pix] = acc[c];
    }
}
)GLSL";

// grad = inv_norm * (w1 * sign(d) + 2 * w2 * d); per-workgroup L1/L2 partial
// sums (float atomics are not core GL, so partials reduce in a second pass).
constexpr const char* kLossGrad = R"GLSL(
layout(std430, binding = 0) buffer RenderImg { float render_img[]; };
layout(std430, binding = 1) buffer TargetImg { float target_img[]; };
layout(std430, binding = 2) buffer GradImg { float grad_img[]; };
layout(std430, binding = 3) buffer Partials { vec2 partials[]; };
uniform float uWL1;
uniform float uWL2;
uniform float uInvNorm;
shared vec2 s_sum[LOCAL_X];
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    vec2 sum = vec2(0.0);
    if (x < uW && y < uH) {
        int pix = y * uW + x;
        for (int c = 0; c < uC; ++c) {
            int i = c * uW * uH + pix;
            float d = render_img[i] - target_img[i];
            sum += vec2(abs(d), d * d);
            grad_img[i] = uInvNorm * ((d >= 0.0 ? uWL1 : -uWL1) + 2.0 * uWL2 * d);
        }
    }
    int tid = int(gl_LocalInvocationID.x);
    s_sum[tid] = sum;
    barrier();
    for (int s = LOCAL_X / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        barrier();
    }
    if (tid == 0)
        partials[int(gl_WorkGroupID.y) * int(gl_NumWorkGroups.x) + int(gl_WorkGroupID.x)] =
            s_sum[0];
}
)GLSL";

constexpr const char* kReduce = R"GLSL(
layout(std430, binding = 0) buffer Partials { vec2 partials[]; };
layout(std430, binding = 1) buffer LossOut { vec2 loss_out; };
uniform int uCount;
shared vec2 s_sum[LOCAL_X];
void main() {
    int tid = int(gl_LocalInvocationID.x);
    vec2 sum = vec2(0.0);
    for (int i = tid; i < uCount; i += LOCAL_X) sum += partials[i];
    s_sum[tid] = sum;
    barrier();
    for (int s = LOCAL_X / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        barrier();
    }
    if (tid == 0) loss_out = s_sum[0];
}
)GLSL";

// One 64-thread workgroup per gaussian; threads stride the bounding box and
// reduce in shared memory. Per row the geometry gradient needs only the
// moment sums S0..S2 of t = dL/dsigma (see kernels.inl); on the GPU we keep
// the direct per-pixel form, which is just as cheap with one thread per
// strided pixel.
constexpr const char* kBackward = R"GLSL(
layout(std430, binding = 0) buffer Params { float params[]; };  // colors at 5n
layout(std430, binding = 1) buffer Proj { float proj[]; };
layout(std430, binding = 2) buffer Bbox { ivec4 bbox[]; };
layout(std430, binding = 3) buffer GradImg { float grad_img[]; };
layout(std430, binding = 4) buffer Grads { float grads[]; };
shared float red[LOCAL_X];
float reduce_add(float v) {
    int tid = int(gl_LocalInvocationID.x);
    red[tid] = v;
    barrier();
    for (int s = LOCAL_X / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}
void main() {
    int g = int(gl_WorkGroupID.x + gl_WorkGroupID.y * 32768u);
    if (g >= uN) return;
    ivec4 b = bbox[g];
    float cx = proj[g], cy = proj[uN + g];
    float A = proj[2 * uN + g], B = proj[3 * uN + g], C = proj[4 * uN + g];
    vec4 col = vec4(0.0);
    for (int c = 0; c < uC; ++c) col[c] = params[5 * uN + g * uC + c];

    float dA = 0.0, dB = 0.0, dC = 0.0, dcx = 0.0, dcy = 0.0;
    vec4 dcol = vec4(0.0);
    int bw = b.z - b.x, bh = b.w - b.y;
    int area = bw * bh;
    for (int i = int(gl_LocalInvocationID.x); i < area; i += LOCAL_X) {
        int x = b.x + i % bw, y = b.y + i / bw;
        float dx = float(x) + 0.5 - cx, dy = float(y) + 0.5 - cy;
        float sigma = 0.5 * (A * dx * dx + C * dy * dy) + B * dx * dy;
        if (sigma >= SIGMA_CUTOFF) continue;
        float e = exp(-sigma);
        int pix = y * uW + x;
        float w = 0.0;
        for (int c = 0; c < uC; ++c) {
            float gg = grad_img[c * uW * uH + pix];
            dcol[c] += (e - ALPHA_BIAS) * gg;
            w += col[c] * gg;
        }
        float t = -e * w;   // dLoss/dsigma
        dA += t * 0.5 * dx * dx;
        dB += t * dx * dy;
        dC += t * 0.5 * dy * dy;
        dcx -= t * (A * dx + B * dy);
        dcy -= t * (B * dx + C * dy);
    }
    dA = reduce_add(dA); dB = reduce_add(dB); dC = reduce_add(dC);
    dcx = reduce_add(dcx); dcy = reduce_add(dcy);
    for (int c = 0; c < uC; ++c) dcol[c] = reduce_add(dcol[c]);
    if (gl_LocalInvocationID.x != 0u) return;

    float u = params[2 * uN + g], v = params[3 * uN + g];
    float th = params[4 * uN + g];
    float co = cos(th), si = sin(th);
    float u2v2 = u * u - v * v;
    float s2t = 2.0 * si * co, c2t = co * co - si * si;
    grads[g] = dcx * float(uW);
    grads[uN + g] = dcy * float(uH);
    grads[2 * uN + g] = 2.0 * u * (dA * co * co + dB * si * co + dC * si * si);
    grads[3 * uN + g] = 2.0 * v * (dA * si * si - dB * si * co + dC * co * co);
    grads[4 * uN + g] = u2v2 * (-dA * s2t + dB * c2t + dC * s2t);
    for (int c = 0; c < uC; ++c) grads[5 * uN + g * uC + c] = dcol[c];
}
)GLSL";

// Group layout mirrors the CPU: [pos_x | pos_y | sinv_x | sinv_y | rot | color].
constexpr const char* kAdam = R"GLSL(
layout(std430, binding = 0) buffer Params { float params[]; };
layout(std430, binding = 1) buffer Grads { float grads[]; };
layout(std430, binding = 2) buffer AdamM { float am[]; };
layout(std430, binding = 3) buffer AdamV { float av[]; };
uniform float uLr[6];       // per group, already multiplied by schedule
uniform vec2 uClampPos;
uniform vec2 uClampScale;
uniform vec2 uClampColor;
uniform vec2 uBias;         // 1 - beta^t
void main() {
    int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 32768u) * 4;
    int total = 5 * uN + uN * uC;
    for (int k = i; k < min(i + 4, total); ++k) {
        int group = min(k / uN, 5);
        float g = grads[k];
        float m = 0.9 * am[k] + 0.1 * g;
        float v = 0.999 * av[k] + 0.001 * g * g;
        am[k] = m;
        av[k] = v;
        float p = params[k] - uLr[group] * (m / uBias.x) / (sqrt(v / uBias.y) + 1e-8);
        if (group < 2) p = clamp(p, uClampPos.x, uClampPos.y);
        else if (group < 4) p = clamp(p, uClampScale.x, uClampScale.y);
        else if (group > 4) p = clamp(p, uClampColor.x, uClampColor.y);
        params[k] = p;
    }
}
)GLSL";

// ------------------------------------------------------------------ context
struct GpuContext {
    GLFWwindow* window = nullptr;
    bool ready = false;
    std::string fail_reason = "not initialized";
    std::mutex use_mutex;   // one training job at a time
    GLuint prog_project = 0, prog_scan = 0, prog_fill = 0, prog_forward = 0;
    GLuint prog_lossgrad = 0, prog_reduce = 0, prog_backward = 0, prog_adam = 0;
};

GpuContext& ctx() {
    static GpuContext c;
    return c;
}

GLuint compile_program(const char* body, int local_x, int local_y, std::string* err) {
    std::string src = std::string(kPrelude) + body;
    const auto replace_all = [&](const std::string& from, const std::string& to) {
        for (size_t p = src.find(from); p != std::string::npos; p = src.find(from, p + to.size()))
            src.replace(p, from.size(), to);
    };
    replace_all("LOCAL_X", std::to_string(local_x));
    replace_all("LOCAL_Y", std::to_string(local_y));

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* cstr = src.c_str();
    glShaderSource(shader, 1, &cstr, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        if (err) *err = log;
        glDeleteShader(shader);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);
    glDeleteShader(shader);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        if (err) *err = log;
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

// ------------------------------------------------------------------ backend
class GpuBackend final : public ITrainBackend {
public:
    GpuBackend(std::unique_lock<std::mutex> lock) : lock_(std::move(lock)) {}

    ~GpuBackend() override {
        glfwMakeContextCurrent(ctx().window);
        for (GLuint b : buffers_)
            if (b) glDeleteBuffers(1, &b);
        glfwMakeContextCurrent(nullptr);
    }

    const char* name() const override { return "gpu"; }

    bool prepare(const Image& target, GaussianCloud& cloud, const EncodeSettings& s) override {
        GSIC_ZONE("gpu_prepare");
        glfwMakeContextCurrent(ctx().window);
        while (glGetError() != GL_NO_ERROR) {}  // drain stale errors
        target_ = &target;
        cloud_ = &cloud;
        settings_ = s;
        w_ = target.w; h_ = target.h; c_ = target.c;
        n_ = cloud.count();
        tiles_x_ = (w_ + 15) / 16;
        tiles_y_ = (h_ + 15) / 16;
        // Same scale cap as the CPU backend (see trainer.cpp): bounded
        // footprints keep per-step cost proportional to the pixel count.
        sinv_lo_ = 1.f / std::min(s.max_scale_px, 0.5f * float(std::max(w_, h_)));
        sinv_hi_ = 1.f / kMinScalePx;

        const GLsizeiptr plane_bytes = GLsizeiptr(w_) * h_ * c_ * 4;
        const GLsizeiptr param_count = GLsizeiptr(n_) * 5 + GLsizeiptr(n_) * c_;
        item_capacity_ = std::max<GLsizeiptr>(GLsizeiptr(n_) * 16, 1 << 20);

        // (Re)allocate everything; parameters upload in trainer layout.
        std::vector<float> packed(param_count);
        pack_params(cloud, packed);
        alloc(kParams, param_count * 4, packed.data());
        alloc(kGrads, param_count * 4);
        // Adam moments survive a pyramid level change (same reasoning as the
        // CPU backend): reallocating would zero them and throw away the
        // optimizer's learned step sizes.
        if (adam_bytes_ != param_count * 4) {
            alloc(kAdamM, param_count * 4);   // alloc() zero-fills
            alloc(kAdamV, param_count * 4);
            adam_bytes_ = param_count * 4;
        }
        alloc(kProj, GLsizeiptr(n_) * 5 * 4);
        alloc(kBbox, GLsizeiptr(n_) * 16);
        alloc(kTileCount, GLsizeiptr(tiles_x_) * tiles_y_ * 4);
        alloc(kTileOffset, (GLsizeiptr(tiles_x_) * tiles_y_ + 1) * 4);
        alloc(kCursor, GLsizeiptr(tiles_x_) * tiles_y_ * 4);
        alloc(kItems, item_capacity_ * 4);
        alloc(kOverflow, 4);
        alloc(kTarget, plane_bytes, target.data.data());
        alloc(kRender, plane_bytes);
        alloc(kGradImg, plane_bytes);
        loss_groups_x_ = (w_ + 255) / 256;
        alloc(kPartials, GLsizeiptr(loss_groups_x_) * h_ * 8);
        alloc(kLossOut, 8);
        const GLenum err = glGetError();
        glfwMakeContextCurrent(nullptr);
        if (err != GL_NO_ERROR)
            std::fprintf(stderr, "gsic: GPU buffer setup failed (GL error 0x%x); using CPU\n",
                         err);
        return err == GL_NO_ERROR;
    }

    float step(int t, float lr_mult) override {
        GSIC_ZONE("gpu_step");
        glfwMakeContextCurrent(ctx().window);
        auto& c = ctx();
        const auto barrier = [] { glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); };

        // Project + count tiles.
        glUseProgram(c.prog_project);
        set_common(c.prog_project);
        const GLuint zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kTileCount]);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kOverflow]);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        bind({kParams, kProj, kBbox, kTileCount});
        glDispatchCompute(GLuint((n_ + 255) / 256), 1, 1);
        barrier();

        // Scan + fill.
        glUseProgram(c.prog_scan);
        set_common(c.prog_scan);
        bind({kTileCount, kTileOffset, kCursor});
        glDispatchCompute(1, 1, 1);
        barrier();
        glUseProgram(c.prog_fill);
        set_common(c.prog_fill);
        glUniform1i(glGetUniformLocation(c.prog_fill, "uItemCapacity"), int(item_capacity_));
        bind({kBbox, kCursor, kItems, kOverflow});
        dispatch_1d(n_, 256);
        barrier();

        // Forward.
        glUseProgram(c.prog_forward);
        set_common(c.prog_forward);
        bind({kProj, kParams, kTileOffset, kItems, kRender});
        glDispatchCompute(GLuint(tiles_x_), GLuint(tiles_y_), 1);
        barrier();

        // Loss gradient + reduction.
        glUseProgram(c.prog_lossgrad);
        set_common(c.prog_lossgrad);
        glUniform1f(glGetUniformLocation(c.prog_lossgrad, "uWL1"), settings_.w_l1);
        glUniform1f(glGetUniformLocation(c.prog_lossgrad, "uWL2"), settings_.w_l2);
        glUniform1f(glGetUniformLocation(c.prog_lossgrad, "uInvNorm"),
                    1.f / (float(w_) * float(h_) * float(c_)));
        bind({kRender, kTarget, kGradImg, kPartials});
        glDispatchCompute(GLuint(loss_groups_x_), GLuint(h_), 1);
        barrier();
        glUseProgram(c.prog_reduce);
        set_common(c.prog_reduce);
        glUniform1i(glGetUniformLocation(c.prog_reduce, "uCount"), loss_groups_x_ * h_);
        bind({kPartials, kLossOut});
        glDispatchCompute(1, 1, 1);
        barrier();

        // Backward.
        glUseProgram(c.prog_backward);
        set_common(c.prog_backward);
        bind({kParams, kProj, kBbox, kGradImg, kGrads});
        dispatch_1d(n_, 1);   // one workgroup per gaussian
        barrier();

        // Adam.
        glUseProgram(c.prog_adam);
        set_common(c.prog_adam);
        const float lrs[6] = {settings_.lr_pos * lr_mult, settings_.lr_pos * lr_mult,
                              settings_.lr_scale * lr_mult, settings_.lr_scale * lr_mult,
                              settings_.lr_rot * lr_mult, settings_.lr_feat * lr_mult};
        glUniform1fv(glGetUniformLocation(c.prog_adam, "uLr"), 6, lrs);
        glUniform2f(glGetUniformLocation(c.prog_adam, "uClampPos"), kPosLo, kPosHi);
        glUniform2f(glGetUniformLocation(c.prog_adam, "uClampScale"), sinv_lo_, sinv_hi_);
        glUniform2f(glGetUniformLocation(c.prog_adam, "uClampColor"), kColorLo, kColorHi);
        glUniform2f(glGetUniformLocation(c.prog_adam, "uBias"),
                    1.f - std::pow(0.9f, float(t)), 1.f - std::pow(0.999f, float(t)));
        bind({kParams, kGrads, kAdamM, kAdamV});
        const std::int64_t total = (std::int64_t(n_) * (5 + c_) + 3) / 4;
        dispatch_1d(int(total), 256);
        barrier();

        // Loss readback (8 bytes; steps are serially dependent anyway).
        float loss[2] = {};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kLossOut]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8, loss);

        // Item overflow: grow capacity for subsequent steps.
        int overflow = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kOverflow]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &overflow);
        if (overflow) {
            item_capacity_ *= 2;
            alloc(kItems, item_capacity_ * 4);
        }
        glfwMakeContextCurrent(nullptr);
        return loss[0] / (float(w_) * float(h_) * float(c_));
    }

    void snapshot(Image& out) override {
        glfwMakeContextCurrent(ctx().window);
        out = Image(w_, h_, c_);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kRender]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(out.data.size()) * 4,
                           out.data.data());
        glfwMakeContextCurrent(nullptr);
    }

    void sync_cloud(GaussianCloud& cloud) override {
        glfwMakeContextCurrent(ctx().window);
        const int n = n_;
        std::vector<float> packed(size_t(n) * 5 + size_t(n) * c_);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kParams]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(packed.size()) * 4,
                           packed.data());
        std::memcpy(cloud.pos_x.data(), packed.data(), n * 4);
        std::memcpy(cloud.pos_y.data(), packed.data() + n, n * 4);
        std::memcpy(cloud.sinv_x.data(), packed.data() + 2 * size_t(n), n * 4);
        std::memcpy(cloud.sinv_y.data(), packed.data() + 3 * size_t(n), n * 4);
        std::memcpy(cloud.rot.data(), packed.data() + 4 * size_t(n), n * 4);
        std::memcpy(cloud.color.data(), packed.data() + 5 * size_t(n), size_t(n) * c_ * 4);
        glfwMakeContextCurrent(nullptr);
    }

private:
    enum Buffer {
        kParams, kGrads, kAdamM, kAdamV, kProj, kBbox,
        kTileCount, kTileOffset, kCursor, kItems, kOverflow,
        kTarget, kRender, kGradImg, kPartials, kLossOut, kBufferCount
    };

    void pack_params(const GaussianCloud& cloud, std::vector<float>& packed) {
        const int n = cloud.count();
        std::memcpy(packed.data(), cloud.pos_x.data(), n * 4);
        std::memcpy(packed.data() + n, cloud.pos_y.data(), n * 4);
        std::memcpy(packed.data() + 2 * size_t(n), cloud.sinv_x.data(), n * 4);
        std::memcpy(packed.data() + 3 * size_t(n), cloud.sinv_y.data(), n * 4);
        std::memcpy(packed.data() + 4 * size_t(n), cloud.rot.data(), n * 4);
        std::memcpy(packed.data() + 5 * size_t(n), cloud.color.data(), size_t(n) * cloud.channels * 4);
    }

    void alloc(Buffer which, GLsizeiptr bytes, const void* data = nullptr) {
        if (!buffers_[which]) glGenBuffers(1, &buffers_[which]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[which]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, GL_DYNAMIC_DRAW);
        if (!data) {
            const GLuint zero = 0;
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER,
                              GL_UNSIGNED_INT, &zero);
        }
    }

    void bind(std::initializer_list<Buffer> list) {
        GLuint slot = 0;
        for (Buffer b : list) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot++, buffers_[b]);
    }

    void set_common(GLuint prog) {
        glUniform1i(glGetUniformLocation(prog, "uN"), n_);
        glUniform1i(glGetUniformLocation(prog, "uW"), w_);
        glUniform1i(glGetUniformLocation(prog, "uH"), h_);
        glUniform1i(glGetUniformLocation(prog, "uC"), c_);
        glUniform1i(glGetUniformLocation(prog, "uTilesX"), tiles_x_);
        glUniform1i(glGetUniformLocation(prog, "uTilesY"), tiles_y_);
    }

    void dispatch_1d(int groups, int local) {
        const int total = (groups + local - 1) / local;
        const int gx = std::min(total, 32768);
        const int gy = (total + gx - 1) / gx;
        glDispatchCompute(GLuint(gx), GLuint(gy), 1);
    }

    std::unique_lock<std::mutex> lock_;
    const Image* target_ = nullptr;
    GaussianCloud* cloud_ = nullptr;
    EncodeSettings settings_;
    std::array<GLuint, kBufferCount> buffers_{};
    int w_ = 0, h_ = 0, c_ = 0, n_ = 0, tiles_x_ = 0, tiles_y_ = 0;
    int loss_groups_x_ = 0;
    GLsizeiptr item_capacity_ = 0;
    GLsizeiptr adam_bytes_ = 0;
    float sinv_lo_ = 0.f, sinv_hi_ = 1.f;
};

} // namespace

bool gpu_init(std::string* why_not) {
    auto& c = ctx();
    if (c.ready) return true;
#if defined(__APPLE__)
    c.fail_reason = "macOS has no OpenGL compute support";
    if (why_not) *why_not = c.fail_reason;
    return false;
#else
    if (!glfwInit()) {
        c.fail_reason = "GLFW init failed";
        if (why_not) *why_not = c.fail_reason;
        return false;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    c.window = glfwCreateWindow(64, 64, "gsic-compute", nullptr, nullptr);
    glfwDefaultWindowHints();
    if (!c.window) {
        c.fail_reason = "no OpenGL 4.3 capable GPU/driver";
        if (why_not) *why_not = c.fail_reason;
        return false;
    }
    glfwMakeContextCurrent(c.window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        c.fail_reason = "failed to load OpenGL functions";
        glfwDestroyWindow(c.window);
        c.window = nullptr;
        glfwMakeContextCurrent(nullptr);
        if (why_not) *why_not = c.fail_reason;
        return false;
    }

    struct Prog { GLuint* slot; const char* src; int lx, ly; const char* name; };
    const Prog progs[] = {
        {&c.prog_project, kProject, 256, 1, "project"},
        {&c.prog_scan, kScan, 256, 1, "scan"},
        {&c.prog_fill, kFill, 256, 1, "fill"},
        {&c.prog_forward, kForward, 16, 16, "forward"},
        {&c.prog_lossgrad, kLossGrad, 256, 1, "lossgrad"},
        {&c.prog_reduce, kReduce, 256, 1, "reduce"},
        {&c.prog_backward, kBackward, 64, 1, "backward"},
        {&c.prog_adam, kAdam, 256, 1, "adam"},
    };
    for (const auto& p : progs) {
        std::string err;
        *p.slot = compile_program(p.src, p.lx, p.ly, &err);
        if (!*p.slot) {
            c.fail_reason = std::string("shader '") + p.name + "' failed: " + err;
            glfwMakeContextCurrent(nullptr);
            if (why_not) *why_not = c.fail_reason;
            return false;
        }
    }
    glfwMakeContextCurrent(nullptr);
    c.ready = true;
    return true;
#endif
}

void gpu_shutdown() {
    auto& c = ctx();
    if (c.window) {
        glfwDestroyWindow(c.window);
        c.window = nullptr;
    }
    c.ready = false;
    c.fail_reason = "shut down";
}

std::optional<Image> gpu_render(const GaussianCloud& cloud, int w, int h, std::string* error) {
    GSIC_ZONE("gpu_render");
    auto& c = ctx();
    if (!c.ready && !gpu_init(error)) return std::nullopt;
    std::unique_lock lock(c.use_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        if (error) *error = "GPU busy";
        return std::nullopt;
    }
    const int n = cloud.count();
    const int ch = cloud.channels;
    if (n <= 0 || w <= 0 || h <= 0) return std::nullopt;

    glfwMakeContextCurrent(c.window);
    while (glGetError() != GL_NO_ERROR) {}
    const int tiles_x = (w + 15) / 16, tiles_y = (h + 15) / 16;

    // Same buffer set the trainer uses, minus everything gradient related.
    enum { bParams, bProj, bBbox, bTileCount, bTileOffset, bCursor, bItems, bOverflow,
           bRender, bCount };
    GLuint buf[bCount] = {};
    glGenBuffers(bCount, buf);
    const auto fill = [&](int which, GLsizeiptr bytes, const void* data) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[which]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, GL_DYNAMIC_DRAW);
        if (!data) {
            const GLuint zero = 0;
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER,
                              GL_UNSIGNED_INT, &zero);
        }
    };

    std::vector<float> packed(size_t(n) * 5 + size_t(n) * ch);
    std::memcpy(packed.data(), cloud.pos_x.data(), size_t(n) * 4);
    std::memcpy(packed.data() + n, cloud.pos_y.data(), size_t(n) * 4);
    std::memcpy(packed.data() + 2 * size_t(n), cloud.sinv_x.data(), size_t(n) * 4);
    std::memcpy(packed.data() + 3 * size_t(n), cloud.sinv_y.data(), size_t(n) * 4);
    std::memcpy(packed.data() + 4 * size_t(n), cloud.rot.data(), size_t(n) * 4);
    std::memcpy(packed.data() + 5 * size_t(n), cloud.color.data(), size_t(n) * ch * 4);

    GLsizeiptr item_capacity = std::max<GLsizeiptr>(GLsizeiptr(n) * 16, 1 << 20);
    fill(bParams, GLsizeiptr(packed.size()) * 4, packed.data());
    fill(bProj, GLsizeiptr(n) * 5 * 4, nullptr);
    fill(bBbox, GLsizeiptr(n) * 16, nullptr);
    fill(bTileCount, GLsizeiptr(tiles_x) * tiles_y * 4, nullptr);
    fill(bTileOffset, (GLsizeiptr(tiles_x) * tiles_y + 1) * 4, nullptr);
    fill(bCursor, GLsizeiptr(tiles_x) * tiles_y * 4, nullptr);
    fill(bItems, item_capacity * 4, nullptr);
    fill(bOverflow, 4, nullptr);
    fill(bRender, GLsizeiptr(w) * h * ch * 4, nullptr);

    const auto set_common = [&](GLuint prog) {
        glUniform1i(glGetUniformLocation(prog, "uN"), n);
        glUniform1i(glGetUniformLocation(prog, "uW"), w);
        glUniform1i(glGetUniformLocation(prog, "uH"), h);
        glUniform1i(glGetUniformLocation(prog, "uC"), ch);
        glUniform1i(glGetUniformLocation(prog, "uTilesX"), tiles_x);
        glUniform1i(glGetUniformLocation(prog, "uTilesY"), tiles_y);
    };
    const auto bind = [&](std::initializer_list<int> list) {
        GLuint slot = 0;
        for (int b : list) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot++, buf[b]);
    };
    const auto barrier = [] { glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); };
    const auto dispatch_1d = [](int count, int local) {
        const int total = (count + local - 1) / local;
        const int gx = std::min(total, 32768);
        glDispatchCompute(GLuint(gx), GLuint((total + gx - 1) / gx), 1);
    };

    glUseProgram(c.prog_project);
    set_common(c.prog_project);
    bind({bParams, bProj, bBbox, bTileCount});
    dispatch_1d(n, 256);
    barrier();

    glUseProgram(c.prog_scan);
    set_common(c.prog_scan);
    bind({bTileCount, bTileOffset, bCursor});
    glDispatchCompute(1, 1, 1);
    barrier();

    glUseProgram(c.prog_fill);
    set_common(c.prog_fill);
    glUniform1i(glGetUniformLocation(c.prog_fill, "uItemCapacity"), int(item_capacity));
    bind({bBbox, bCursor, bItems, bOverflow});
    dispatch_1d(n, 256);
    barrier();

    glUseProgram(c.prog_forward);
    set_common(c.prog_forward);
    bind({bProj, bParams, bTileOffset, bItems, bRender});
    glDispatchCompute(GLuint(tiles_x), GLuint(tiles_y), 1);
    barrier();

    Image out(w, h, ch);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[bRender]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(out.data.size()) * 4,
                       out.data.data());
    int overflow = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[bOverflow]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &overflow);
    const GLenum err = glGetError();
    glDeleteBuffers(bCount, buf);
    glfwMakeContextCurrent(nullptr);

    // An item overflow means some gaussians were dropped from tiles, so the
    // image would be subtly wrong; fall back rather than return bad pixels.
    if (err != GL_NO_ERROR || overflow) {
        if (error) *error = overflow ? "tile overflow" : "GL error during render";
        return std::nullopt;
    }
    return out;
}

std::unique_ptr<ITrainBackend> make_gpu_backend(std::string* why_not) {
    auto& c = ctx();
    if (!c.ready && !gpu_init(why_not)) return nullptr;
    std::unique_lock lock(c.use_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        if (why_not) *why_not = "GPU busy with another job";
        return nullptr;
    }
    return std::make_unique<GpuBackend>(std::move(lock));
}

} // namespace gsic

#endif // GSIC_ENABLE_GPU
