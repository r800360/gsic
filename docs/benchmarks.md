# Benchmarks

[← back to the README](../README.md)

## Comparison with the paper

The Image-GS paper reports optimization timings for 2K×2K images on an NVIDIA
A6000.

Section 4.4 reports:

- 10K Gaussians, 1000 steps: 18.74 s
- 50K Gaussians, 1000 steps: 26.32 s

Running `gsic` with the same resolution, Gaussian counts, step count, and
optimization loop gives:

| 2K×2K, 1000 steps | Image-GS, A6000 | gsic, RTX 4050 Laptop | Speedup |
| --- | ---: | ---: | ---: |
| 10K Gaussians | 18.74 s | **6.38 s** | **2.94×** |
| 50K Gaussians | 26.32 s | **7.90 s** | **3.33×** |

The two implementations were measured on substantially different GPUs, so the
table should be interpreted as an end-to-end implementation comparison rather
than a hardware-normalized benchmark.

The largest difference comes from the unnormalized splat formulation, which
removes the sorting and atomic operations required by the top-K formulation.

## Quality

The paper reports mean PSNR over its full 45-image dataset: nine images each
from anime, art, painting, photo, and vector categories.

| Full 45-image dataset | PSNR | Rate |
| --- | ---: | ---: |
| Image-GS, paper §5 | 32.99 ± 4.49 dB | 0.366 bpp |
| gsic | **33.34 ± 4.99 dB** | **0.340 bpp** |

`gsic` is therefore 0.35 dB higher on this benchmark while using about 7% fewer
bits.

Per-category PSNR:

| Category | PSNR |
| --- | ---: |
| Photo | 36.00 dB |
| Anime | 34.74 dB |
| Art | 33.57 dB |
| Vector | 31.44 dB |
| Painting | 30.94 dB |

## Quality vs JPEG

The `balanced` preset was tested at roughly 0.32 bpp against JPEG files encoded
to approximately the same size.

Positive Δ means `gsic` has higher PSNR.

| Image | Size | gsic PSNR | JPEG PSNR | Δ |
| --- | ---: | ---: | ---: | ---: |
| anime-1_2k | 163 KB | 41.56 dB | 38.35 dB | **+3.21 dB** |
| art-3_2k | 161 KB | 37.17 dB | 33.81 dB | **+3.36 dB** |
| vector-2_2k | 163 KB | 30.32 dB | 27.16 dB | **+3.16 dB** |
| photo-3_2k | 164 KB | 39.70 dB | 39.25 dB | +0.45 dB |
| painting-5_2k | 163 KB | 31.34 dB | 33.11 dB | −1.77 dB |
| Kodak, 6 images at 0.34 bpp | 16 KB | 25.70 dB avg | 25.87 dB avg | −0.17 dB |

The method performs best on images with large coherent regions and sharp edges,
including illustration, vector graphics, and some types of digital art.

Highly textured photographic content is more difficult. On the Kodak images
tested here, `gsic` is approximately on par with or slightly behind JPEG at the
same rate.

Test images come from the Image-GS media set and the
[Kodak image suite](https://r0k.us/graphics/kodak/).

## Limitations

### Decode performance is not directly comparable to the paper

The paper reports a 2K×2K forward-render time of 3.7 ms on an A6000. `gsic`
takes about 20–21 ms to decode the same resolution on the CPU.

A GPU decode path is available through:

```bash
gsic decompress --gpu
```

but currently takes around 30 ms when producing a CPU-resident output image.
The transfer back to system memory outweighs the faster GPU render, so CPU
decoding remains the default.

The paper's reported forward-pass timing and `gsic`'s full
decode-to-system-memory timing therefore measure different workloads.

The demonstration video associated with Image-GS is also not used for direct
comparison because the exact optimization step count and workload are not
specified.

### Scaling to large images

Training cost scales approximately linearly with the number of pixels because
each optimization step processes the image.

For example, 4000 optimization steps on a 67-megapixel image involve roughly
two orders of magnitude more pixel work than 4000 steps at 2K resolution.

A coarse-to-fine image pyramid was tested as a possible optimization, but it
produced worse results at both 2K and 8K. The current initialization was strong
enough that spending optimization steps at lower resolutions did not provide a
net benefit.

This is what the desktop application's time limit exists for; see
[the README](../README.md#the-wait-is-bounded) for the measurements behind it.
