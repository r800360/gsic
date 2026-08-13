# How it works

[← back to the README](../README.md)

An image is represented as a sum of anisotropic 2D Gaussians. Each Gaussian has
a position, scale, rotation, and color.

A pixel receives the weighted contribution of each Gaussian covering it:

```text
out(p) = Σᵢ αᵢ(p) · colorᵢ

αᵢ = max(exp(-σᵢ) - bias, 0)

σᵢ = ½ dᵀ Conicᵢ d
d  = p - centerᵢ
```

Encoding optimizes these parameters against the source image using Adam and an
L1 reconstruction loss.

Initial Gaussian positions are sampled according to image gradient magnitude.
During optimization, more Gaussians are added in regions where reconstruction
error remains high. Detailed regions therefore receive more representation
capacity than flat regions.

The `.gsi` format stores quantized Gaussian parameters and entropy-codes the
result with Zstandard.

Because the stored representation is a set of shapes rather than a grid of
pixels, decoding is not tied to the resolution the file was written at:
`--scale` (and **Export as PNG…** in the app) evaluates the Gaussians at the
requested output size instead of resizing an already-decoded image.

---

## Source layout

```text
src/core/       core implementation, no UI dependencies
  image         image loading/saving and planar float pixels
  gaussians     Gaussian parameter storage and shared constants
  renderer      projection, tile binning, forward rendering, scaled decode
  kernels       performance-critical CPU kernels
  trainer       initialization, optimization schedule, backend fallback
  codec         .gsi quantization, bit packing, and parsing
  metrics       PSNR and SSIM
  format        the one place a byte count or a duration becomes text
  gpu_backend   OpenGL 4.3 compute backend and its startup self-check
  thread_pool   fork-join parallel_for

src/cli/        command-line front end
src/app/
  pipeline      the queue and the work it runs, no UI dependencies
  settings      what the application remembers between launches
  app           ImGui desktop application: the window around the pipeline
tests/          the six suites
```

`src/app/pipeline.cpp` holds one queue with three kinds of entry — compress an
image, open a `.gsi`, compare two images — sharing one worker, one status model
and one preview representation. That split is deliberate: it is the code path a
user exercises when they add a file and expect something to happen, and while
it was tangled up with the interface there was no way to test it. It now has no
window, no GL context and no ImGui, so the `app` suite drives exactly what a
person drives.

---

## Performance

Measured on a 16-core Intel Core Ultra 7 255H with an RTX 4050 laptop GPU using
a 2048×2048 RGB image and the `balanced` preset:

| Backend | Encode | Decode |
| --- | ---: | ---: |
| GPU, OpenGL compute | 22 s | 21 ms |
| CPU, AVX2, 16 threads | 34 s | 21 ms |

Decoding uses the CPU forward renderer by default. Unlike encoding, it does not
involve optimization.

The largest performance improvements came from the following changes.

### Tile-binned rendering

The image is divided into 16×16 tiles and each Gaussian is assigned only to the
tiles it overlaps.

Each tile processes a small local Gaussian list before writing its output once.
This improves cache locality and avoids repeatedly scanning the entire Gaussian
set for every pixel.

### Fused forward and loss evaluation

During training, the full-resolution rendered image does not need to be
materialized.

The forward render and loss calculation are fused, avoiding extra full-image
memory traffic. This improved training performance by roughly 25% in testing.

### Unnormalized splat accumulation

Image-GS uses a normalized top-K formulation.

`gsic` instead uses an unnormalized Gaussian sum. This removes the dependency
between neighboring Gaussian gradients and allows the backward pass to operate
independently per Gaussian, without atomics or top-K sorting.

The same formulation is used by both the CPU and GPU backends.

### Analytic row moments

For pixels on a fixed row, the five geometric Gaussian gradients can be
expressed using three accumulated moment sums.

This reduces the amount of work inside the innermost backward-pass loop.

### Exact row intervals

The Gaussian exponent is quadratic in `x` along a scanline.

Instead of checking every pixel inside the Gaussian's bounding rectangle, the
renderer solves for the exact interval intersecting the cutoff ellipse. This
avoids evaluating pixels that lie inside the box but outside the Gaussian
footprint.

### Runtime SIMD dispatch

CPU kernels are compiled for both a baseline SSE2 target and AVX2+FMA.

The appropriate implementation is selected using `cpuid` at startup, allowing
one binary to run efficiently across a wide range of x86 processors.

The 32-bit build explicitly uses SSE2 rather than relying on legacy x87
floating-point code generation.

The exponential used in the hot loop is approximated by a degree-4 polynomial
with roughly `1e-4` relative error.

### Learning-rate schedule

The learning rate remains constant for the first 80% of training and decays
near the end.

In testing, cosine decay from the beginning reduced final PSNR by about 0.6 dB,
while keeping the rate constant for the entire run lost about 0.2 dB.

The late decay gives Gaussians enough time to migrate across the image before
the optimization begins settling.

### Morton ordering

Gaussians are periodically sorted along a Morton/Z-order curve.

Spatially nearby Gaussians are therefore more likely to be nearby in memory,
improving cache locality in the backward pass.

The ordering also produces smaller position deltas, which helps entropy coding.

---

## The compute context belongs to one thread

`gpu_init()` is called once at startup, from a thread that outlives every
encode, and nothing else may create the context. This is not a style
preference. The context is carried by a hidden window, and on Windows that
window's message queue belongs to the thread that created it; when that thread
exits, operations on the window block rather than fail.

The context used to be created lazily by whichever thread first wanted a GPU
backend — an encoding worker. Measured: the first image compressed correctly in
8.2 s, and the second, identical, never finished. One core busy, no error, no
progress, the interface still responding to clicks.

So `gpu_render` and the GPU training backend never create the context on
demand. If nobody initialised it, they report no GPU and the CPU does the work,
which is a correct answer instead of a deadlock.
