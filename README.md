# gsic: Gaussian Splat Image Compressor

Compresses images by fitting a cloud of anisotropic 2D Gaussians to them. `gsic` is a C++23 reimplementation of [Image-GS](https://github.com/NYU-ICL/image-gs), built around a faster CPU/GPU training pipeline.

It ships as both a desktop app and command-line tool, uses GPU compute when available, and falls back to a vectorized multithreaded CPU implementation everywhere else.

On the Image-GS 45-image benchmark, `gsic` reaches **33.34 dB at 0.340 bpp**, compared with **32.99 dB at 0.366 bpp** reported by the paper. At roughly 0.32 bpp it also performs particularly well on stylized images, where it beats size-matched JPEG by around 3 dB.

A 2048×2048 image encodes in about 22 seconds on an RTX 4050 laptop GPU and decodes on the CPU in about 21 ms.

See [Comparison with the paper](#comparison-with-the-paper) for the full results.

---

## Build

You need:

- a C++23 compiler: MSVC 19.38+, GCC 13+, or Clang 17+
- CMake 3.25+
- Ninja
- [vcpkg](https://github.com/microsoft/vcpkg)

Set `VCPKG_ROOT` to your vcpkg checkout. Dependencies are declared in `vcpkg.json` and are fetched automatically during the first configure.

```bash
git clone <this-repo> gsic
cd gsic

export VCPKG_ROOT=/path/to/vcpkg
# Windows: set VCPKG_ROOT=C:\dev\vcpkg

cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release
```

Use `windows-release` or `macos-release` instead when appropriate.

Binaries are written to `build/<preset>/`:

- `gsic-gui`: desktop application
- `gsic`: command-line tool

### Presets

| Preset | Description |
| --- | --- |
| `windows-release`, `linux-release`, `macos-release` | Optimized build |
| `windows-x86-release`, `linux-x86-release` | 32-bit build |
| `windows-debug` | Unoptimized build with debug information |
| `windows-profile` | Release build with [Tracy](https://github.com/wolfpld/tracy) instrumentation |
| `windows-asan` | AddressSanitizer build |

On Windows, `build.cmd` locates Visual Studio's bundled CMake and Ninja, so they do not need to be on `PATH`:

```bat
build.cmd local-release
```

### Platform notes

- **Windows** builds statically using the `x64-windows-static` triplet and static CRT, so `gsic-gui.exe` can be copied to another machine without installing the Visual C++ redistributable. Release builds also include standard Windows version metadata. Code signing is recommended for public distribution.

- **Linux** requires X11/Wayland development headers for GLFW and D-Bus for the file dialog. On Debian/Ubuntu:

  ```bash
  sudo apt install xorg-dev libwayland-dev libxkbcommon-dev libdbus-1-dev
  ```

- **macOS** has no OpenGL compute support, so encoding uses the CPU backend. Both Apple Silicon and Intel 64-bit builds are supported. There is no 32-bit macOS build.

### 32-bit builds

32-bit builds are supported on Windows and Linux, including GPU acceleration.

The main limitation is address space. Encoding temporarily uses substantially more memory than the compressed image itself, so 32-bit builds enforce a 16-megapixel input limit. The corresponding 64-bit limit is much higher.

Image dimensions and Gaussian counts are validated before allocation, including when reading potentially malformed `.gsi` files.

Files are interchangeable between 32-bit and 64-bit builds.

---

## Using it

### Desktop app

Run `gsic-gui`, then drag images onto the window or use **Add Images…**.

Images are compressed one at a time. Each encode already makes heavy use of the available CPU cores or GPU, so running multiple encodes concurrently generally does not help.

The preview displays the original and compressed result with a draggable divider and updates while optimization is running.

- Scroll to zoom
- Drag to pan
- Double-click to reset

### Command line

```bash
# Compress one or more images
gsic compress photo.png
gsic compress *.png --preset high -o out/

# Decode to PNG
gsic decompress photo.gsi
gsic decompress photo.gsi --scale 2.0 -o photo_2x.png

# Inspect, compare, or benchmark
gsic info photo.gsi
gsic compare original.png decoded.png
gsic bench photo.png --preset fast
```

Useful options include:

- `--preset fast|balanced|high`
- `-n <count>` to set the Gaussian budget directly
- `--bits-pos`
- `--bits-scale`
- `--bits-rot`
- `--bits-color`
- `--cpu` / `--gpu` to force a backend
- `--png` to also write the decoded image

Run:

```bash
gsic --help
```

for the full list.

Because the compressed representation is continuous rather than a fixed raster, `--scale` evaluates the Gaussian representation directly at the requested output resolution instead of resizing an already-decoded image.

---

## How it works

An image is represented as a sum of anisotropic 2D Gaussians. Each Gaussian has a position, scale, rotation, and color.

A pixel receives the weighted contribution of each Gaussian covering it:

```text
out(p) = Σᵢ αᵢ(p) · colorᵢ

αᵢ = max(exp(-σᵢ) - bias, 0)

σᵢ = ½ dᵀ Conicᵢ d
d  = p - centerᵢ
```

Encoding optimizes these parameters against the source image using Adam and an L1 reconstruction loss.

Initial Gaussian positions are sampled according to image gradient magnitude. During optimization, more Gaussians are added in regions where reconstruction error remains high.

Detailed regions therefore receive more representation capacity than flat regions.

The `.gsi` format stores quantized Gaussian parameters and entropy-codes the result with Zstandard.

### Source layout

```text
src/core/       core implementation, no UI dependencies
  image         image loading/saving and planar float pixels
  gaussians     Gaussian parameter storage and shared constants
  renderer      projection, tile binning, and forward rendering
  kernels       performance-critical CPU kernels
  trainer       initialization and optimization schedule
  codec         .gsi quantization, bit packing, and parsing
  metrics       PSNR and SSIM
  gpu_backend   OpenGL 4.3 compute backend
  thread_pool   fork-join parallel_for

src/cli/        command-line front end
src/app/        ImGui desktop application
tests/          unit tests
```

---

## Performance

Measured on a 16-core Intel Core Ultra 7 255H with an RTX 4050 laptop GPU using a 2048×2048 RGB image and the `balanced` preset:

| Backend | Encode | Decode |
| --- | ---: | ---: |
| GPU, OpenGL compute | 22 s | 21 ms |
| CPU, AVX2, 16 threads | 34 s | 21 ms |

Decoding uses the CPU forward renderer by default. Unlike encoding, it does not involve optimization.

The largest performance improvements came from the following changes.

### Tile-binned rendering

The image is divided into 16×16 tiles and each Gaussian is assigned only to the tiles it overlaps.

Each tile processes a small local Gaussian list before writing its output once. This improves cache locality and avoids repeatedly scanning the entire Gaussian set for every pixel.

### Fused forward and loss evaluation

During training, the full-resolution rendered image does not need to be materialized.

The forward render and loss calculation are fused, avoiding extra full-image memory traffic. This improved training performance by roughly 25% in testing.

### Unnormalized splat accumulation

Image-GS uses a normalized top-K formulation.

`gsic` instead uses an unnormalized Gaussian sum. This removes the dependency between neighboring Gaussian gradients and allows the backward pass to operate independently per Gaussian, without atomics or top-K sorting.

The same formulation is used by both the CPU and GPU backends.

### Analytic row moments

For pixels on a fixed row, the five geometric Gaussian gradients can be expressed using three accumulated moment sums.

This reduces the amount of work inside the innermost backward-pass loop.

### Exact row intervals

The Gaussian exponent is quadratic in `x` along a scanline.

Instead of checking every pixel inside the Gaussian's bounding rectangle, the renderer solves for the exact interval intersecting the cutoff ellipse. This avoids evaluating pixels that lie inside the box but outside the Gaussian footprint.

### Runtime SIMD dispatch

CPU kernels are compiled for both a baseline SSE2 target and AVX2+FMA.

The appropriate implementation is selected using `cpuid` at startup, allowing one binary to run efficiently across a wide range of x86 processors.

The 32-bit build explicitly uses SSE2 rather than relying on legacy x87 floating-point code generation.

The exponential used in the hot loop is approximated by a degree-4 polynomial with roughly `1e-4` relative error.

### Learning-rate schedule

The learning rate remains constant for the first 80% of training and decays near the end.

In testing, cosine decay from the beginning reduced final PSNR by about 0.6 dB, while keeping the rate constant for the entire run lost about 0.2 dB.

The late decay gives Gaussians enough time to migrate across the image before the optimization begins settling.

### Morton ordering

Gaussians are periodically sorted along a Morton/Z-order curve.

Spatially nearby Gaussians are therefore more likely to be nearby in memory, improving cache locality in the backward pass.

The ordering also produces smaller position deltas, which helps entropy coding.

---

## Comparison with the paper

The Image-GS paper reports optimization timings for 2K×2K images on an NVIDIA A6000.

Section 4.4 reports:

- 10K Gaussians, 1000 steps: 18.74 s
- 50K Gaussians, 1000 steps: 26.32 s

Running `gsic` with the same resolution, Gaussian counts, step count, and optimization loop gives:

| 2K×2K, 1000 steps | Image-GS, A6000 | gsic, RTX 4050 Laptop | Speedup |
| --- | ---: | ---: | ---: |
| 10K Gaussians | 18.74 s | **6.38 s** | **2.94×** |
| 50K Gaussians | 26.32 s | **7.90 s** | **3.33×** |

The two implementations were measured on substantially different GPUs, so the table should be interpreted as an end-to-end implementation comparison rather than a hardware-normalized benchmark.

The largest difference comes from the unnormalized splat formulation, which removes the sorting and atomic operations required by the top-K formulation.

### Quality

The paper reports mean PSNR over its full 45-image dataset: nine images each from anime, art, painting, photo, and vector categories.

| Full 45-image dataset | PSNR | Rate |
| --- | ---: | ---: |
| Image-GS, paper §5 | 32.99 ± 4.49 dB | 0.366 bpp |
| gsic | **33.34 ± 4.99 dB** | **0.340 bpp** |

`gsic` is therefore 0.35 dB higher on this benchmark while using about 7% fewer bits.

Per-category PSNR:

| Category | PSNR |
| --- | ---: |
| Photo | 36.00 dB |
| Anime | 34.74 dB |
| Art | 33.57 dB |
| Vector | 31.44 dB |
| Painting | 30.94 dB |

### Limitations

**Decode performance is not directly comparable to the paper.**

The paper reports a 2K×2K forward-render time of 3.7 ms on an A6000. `gsic` takes about 20–21 ms to decode the same resolution on the CPU.

A GPU decode path is available through:

```bash
gsic decompress --gpu
```

but currently takes around 30 ms when producing a CPU-resident output image. The transfer back to system memory outweighs the faster GPU render, so CPU decoding remains the default.

The paper's reported forward-pass timing and `gsic`'s full decode-to-system-memory timing therefore measure different workloads.

The demonstration video associated with Image-GS is also not used for direct comparison because the exact optimization step count and workload are not specified.

### Scaling to large images

Training cost scales approximately linearly with the number of pixels because each optimization step processes the image.

For example, 4000 optimization steps on a 67-megapixel image involve roughly two orders of magnitude more pixel work than 4000 steps at 2K resolution.

A coarse-to-fine image pyramid was tested as a possible optimization, but it produced worse results at both 2K and 8K. The current initialization was strong enough that spending optimization steps at lower resolutions did not provide a net benefit.

---

## Profiling

Build the `windows-profile` preset and run the [Tracy profiler](https://github.com/wolfpld/tracy).

The main encode stages appear as named zones:

```text
project_and_bin
forward_loss
backward
adam
```

Profiling instrumentation is compiled out of normal release builds.

---

## Quality vs JPEG

The `balanced` preset was tested at roughly 0.32 bpp against JPEG files encoded to approximately the same size.

Positive Δ means `gsic` has higher PSNR.

| Image | Size | gsic PSNR | JPEG PSNR | Δ |
| --- | ---: | ---: | ---: | ---: |
| anime-1_2k | 163 KB | 41.56 dB | 38.35 dB | **+3.21 dB** |
| art-3_2k | 161 KB | 37.17 dB | 33.81 dB | **+3.36 dB** |
| vector-2_2k | 163 KB | 30.32 dB | 27.16 dB | **+3.16 dB** |
| photo-3_2k | 164 KB | 39.70 dB | 39.25 dB | +0.45 dB |
| painting-5_2k | 163 KB | 31.34 dB | 33.11 dB | −1.77 dB |
| Kodak, 6 images at 0.34 bpp | 16 KB | 25.70 dB avg | 25.87 dB avg | −0.17 dB |

The method performs best on images with large coherent regions and sharp edges, including illustration, vector graphics, and some types of digital art.

Highly textured photographic content is more difficult. On the Kodak images tested here, `gsic` is approximately on par with or slightly behind JPEG at the same rate.

Test images come from the Image-GS media set and the [Kodak image suite](https://r0k.us/graphics/kodak/).

---

## Correctness and safety

`ctest` covers:

- analytic gradients checked against numerical finite differences
- codec round trips
- renderer symmetry
- end-to-end compression
- GPU/CPU agreement

GPU-dependent tests are skipped automatically when no compatible GPU is available.

The `.gsi` parser validates dimensions, Gaussian counts, and encoded payload sizes before allocating memory. File-provided sizes are not used directly without range checking.

The parser test suite includes truncated files, corrupted headers, and random input buffers.

The CLI has also been tested under AddressSanitizer with a collection of corrupted `.gsi` files without detecting memory errors or crashes.

Memory is managed using standard containers throughout the core implementation. OpenGL objects are owned and released by the GPU backend.

---

## Distribution

Release binaries for Windows, Linux, and macOS are built automatically.

Pushing a version tag runs the release workflow:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow builds and tests each platform, packages the binaries, generates SHA-256 checksums, builds the technical report, and attaches the results to a GitHub Release.

`.github/workflows/ci.yml` builds and tests pushes and pull requests across all three platforms.

`.github/workflows/msix.yml` builds the Microsoft Store package.

The user-facing download page is `docs/index.html`, served through GitHub Pages. It detects the visitor's platform and presents the corresponding build.

### Icon and Store assets

Application icons and Microsoft Store assets are generated by:

```text
tools/make_logo.cpp
```

The generator uses fixed parameters to construct the Gaussian-based logo and emits the Store assets together with a multi-resolution `gsic.ico`.

To regenerate them:

```bash
cmake --build build/<preset> --target logo
```

Commit the generated files after changing the mark.

---

## License

MIT. See [LICENSE](LICENSE).

The license file also contains attribution for Image-GS and the third-party dependencies used by the project.
