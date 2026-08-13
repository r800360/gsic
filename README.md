# gsic: Gaussian Splat Image Compressor

Compresses images by fitting a cloud of anisotropic 2D Gaussians to them. `gsic` is a C++23 reimplementation of [Image-GS](https://github.com/NYU-ICL/image-gs), built around a faster CPU/GPU training pipeline.

It ships as both a desktop app and a command-line tool, uses GPU compute when available, and falls back to a vectorized multithreaded CPU implementation everywhere else.

On the Image-GS 45-image benchmark, `gsic` reaches **33.34 dB at 0.340 bpp**, compared with **32.99 dB at 0.366 bpp** reported by the paper. At roughly 0.32 bpp it also performs particularly well on stylized images, where it beats size-matched JPEG by around 3 dB.

A 2048×2048 image encodes in about 22 seconds on an RTX 4050 laptop GPU and decodes on the CPU in about 21 ms.

---

## Documentation

| | |
| --- | --- |
| [Building](docs/building.md) | Requirements, presets, platform notes, packaging and release |
| [How it works](docs/architecture.md) | The method, the source layout, and where the speed comes from |
| [Benchmarks](docs/benchmarks.md) | Comparison with the paper, quality against JPEG, limitations |
| [Testing](docs/testing.md) | The six suites, what each exists to catch, and correctness guarantees |

---

## Using it

### Desktop app

Run `gsic-gui`. Drag images onto the window to compress them, or drag a `.gsi`
to open one you compressed earlier. **Add Images…** and **Open .gsi…** do the
same thing through a file dialog, and on Windows the Store package registers
`.gsi` so a compressed file opens by double-clicking it.

Images are compressed one at a time. Each encode already makes heavy use of the
available CPU cores or GPU, so running several concurrently does not help.

The preview shows the original and the compressed result with a draggable
divider, and updates while optimization is running.

- Scroll to zoom, drag to pan, double-click to reset
- Drag the divider, or switch to one side, or to a **Difference** view with an
  adjustable gain
- **Details** lists everything `gsic info` prints: dimensions, Gaussian count,
  bit depths, file size and bits per pixel
- **Export as PNG…** decodes at any scale from 0.05× to 8×, evaluating the
  Gaussians at that resolution rather than resizing a decoded image
- **Compare with original…** measures a decoded file against the picture it was
  made from, and **File ▸ Compare Two Images…** does the same for any two images
- **Help ▸ Diagnostics** reports what this machine will do and can run a speed
  test; **Copy report** puts all of it on the clipboard for a bug report

Settings, window size and view preferences are remembered between launches.

### Changing your mind

Nothing has to be added twice. Change the preset, the precision or the time
limit and press **Run again** on the item in the preview (or `Ctrl+R`, or
right-click it in the queue) — the entry runs again in place with the settings
as they are now, and overwrites its own output. **Run All Again** does the same
for everything finished, which is what a whole folder compressed at the wrong
preset needs.

When the time limit is what stopped a run, the preview says so — *"Stopped at
872 of 3000 steps to stay inside the 3 s limit"* — and offers the fix next to
it: one button that raises the limit and runs the image again. It doubles each
time it is pressed, and once that would leave the top of the slider it offers
no limit at all.

Keyboard: `Ctrl+O` add images, `Ctrl+Shift+O` open a `.gsi`, `Ctrl+E` export,
`Ctrl+R` run again, `Del` remove, `Esc` cancel, `F` fit, `1` actual size,
arrow keys to move through the queue.

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

# Report what this machine will do
gsic diagnose
```

Useful options include:

- `--preset fast|balanced|high`
- `-n <count>` to set the Gaussian budget directly
- `--bits-pos`, `--bits-scale`, `--bits-rot`, `--bits-color`
- `--cpu` / `--gpu` to force a backend
- `--png` to also write the decoded image
- `--time-budget <seconds>` to stop early rather than run the full schedule

Run `gsic --help` for the full list. Everything the command line can do is also
reachable from the desktop app.

---

## The wait is bounded

Per-step cost scales with pixel count, so a step count that feels fine on one
machine is a multi-minute wait on a slower one with a bigger image — and a
person watching a progress bar for five minutes reasonably concludes the
feature does not work.

The desktop app therefore takes a **time limit** (30 s by default, adjustable
in Settings up to 240 s, 0 for no limit) and stops the optimizer when it is
spent. The CLI has no limit by default, so scripted encodes stay reproducible;
use `--time-budget <seconds>` to opt in.

Measured on a 12 MP photo at the balanced preset, one CPU:

| Steps | PSNR | Time |
| ---: | ---: | ---: |
| 250 | 31.97 dB | 21 s |
| 750 | 35.89 dB | 43 s |
| 1500 | 37.23 dB | 102 s |
| 3000 | 38.38 dB | 244 s |

The last half of a full run costs 143 seconds and buys 1.15 dB, so the tail
this gives up is small and the wait it removes is not. The interface shows an
estimate of the remaining time, and the log says when a run was shortened.

The budget is enforced by the clock rather than by a step count worked out in
advance, because per-step cost is not constant: the cloud doubles across the
progressive additions and the Gaussians spread as they optimize. Steps 8–24 of
a 12 MP encode run about five times faster than the run average, so an early
extrapolation promised 30 seconds and delivered 152.

Live previews are throttled the same way. A preview renders the whole image, so
its cost grows with image size while a fixed interval does not: at 12 MP a
snapshot costs 132 ms against a 250 ms interval. The interval now stretches to
keep previews within a tenth of the encode. Removing that throttle and
measuring, a test encode went from 2.9 s to 133 s, with 75% of it spent drawing
progress pictures.

---

## Quick build

You need a C++23 compiler, CMake 3.25+, Ninja, and
[vcpkg](https://github.com/microsoft/vcpkg).

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release
```

Use `windows-release` or `macos-release` instead where appropriate. Binaries
land in `build/<preset>/` as `gsic-gui` and `gsic`. See
[Building](docs/building.md) for the rest.

---

## Reporting a problem

Run:

```bash
gsic diagnose
```

or open **Help ▸ Diagnostics** in the desktop app and press **Copy report**.

It reports the CPU kernels selected, thread count, whether the GPU passed its
self-check (and why not, if it did not), and what an encode actually costs on
that hardware. Paste all of it into the report — those facts vary per machine
and none of them are visible from the outside.

---

## License

MIT. See [LICENSE](LICENSE).

The license file also contains attribution for Image-GS and the third-party
dependencies used by the project.
