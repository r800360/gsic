# Testing, correctness and safety

[← back to the README](../README.md)

`ctest` runs six suites, about twenty seconds in total. They are separate
targets so a failure says which layer broke.

| Suite | Covers |
| --- | --- |
| `core` | Analytic gradients against numerical finite differences, codec round trip, scaled decode, hostile input, size limits, renderer symmetry, CPU/GPU agreement |
| `regression` | Quality and file-size baselines per content type, determinism, decode fidelity, rescaled decode |
| `app` | The desktop app's queue: compressing, opening a `.gsi`, exporting, comparing, remembered settings — driven the way a user drives them |
| `robustness` | The failure paths that only appear on other people's hardware |
| `cli` | The shipped executable: compress, decompress, info, compare, batch, and every failure path |
| `gui` | Starts the real window and GL context, reads the rendered frame back, compresses an image, then reopens the file it just wrote |

Four of these exist because of specific mistakes made while building this.

## `regression` answers "did my change help or hurt"

It encodes four procedurally generated images, each targeting a different
weakness (smooth gradients, hard edges, dense texture, mixed content), and
fails if quality drops more than 0.35 dB below a recorded baseline or the file
grows more than 5%. The corpus is generated from fixed constants rather than
committed as files, so it is byte-reproducible anywhere with nothing to
download. The tolerance covers the small differences between SSE2 and AVX2
kernels and between the two backends, both measured at under 0.25 dB.

Regenerate the baselines after a deliberate quality change, and say in the
commit why the numbers moved:

```bash
./build/<preset>/gsic_regression --record
```

## `gui` answers "does the app actually render, and does it work"

An app that opens to an empty window passes every structural check you can
write: the interface objects all exist, are docked, and report sensible sizes.
They are simply never painted. So the selftest starts the real window, renders
30 frames, reads the framebuffer back, and requires that more than half of it
differs from the clear colour.

It then compresses an image through the real pipeline and requires the
resulting file to decode — and then **opens that file the way a user would**,
checking it comes back as something to look at, at the size it was written, with
pixels to show. That last step is the path the Store package's file association
has always advertised: before it existed, double-clicking a `.gsi` in Explorer
started the application and told the user their own file "could not be read as
an image".

All of this runs in the shipped binary, on whatever hardware it finds, after
the GPU self-check has chosen a backend — which is the combination that
actually gets installed.

## `app` answers "does adding a file produce a result"

This is the feature the Store review found unusable, and before this suite
existed nothing tested it: the library underneath was covered and the window on
top was covered, and picking a file, deciding where the output goes, writing it
and reporting what happened was only ever checked by opening the app and
looking. The pipeline now lives apart from the interface
(`src/app/pipeline.cpp`) precisely so a machine can drive it, with no window and
no GL context, on every platform.

It covers the whole life of a compressed file, not just its creation:

- compress an image, throw the pipeline away, and open the `.gsi` with a fresh
  one — which is what a second launch of the application is
- the intake decision, by extension and by the four magic bytes, so a renamed
  `.gsi` still opens and a renamed `.png` is still refused with a reason
- decoding at another size, and refusing a scale that would ask for more pixels
  than the build can address
- exporting a `.png` at 1× and 2×, and reading it back
- damaged input: truncated, corrupt, empty, and an ordinary PNG wearing the
  extension — each must end in a sentence, not a crash
- measuring a decoded file against its original, and comparing two images
- changing the settings and running the same entry again: the new values have
  to reach the encoder, or the button appears to do nothing — and a run the
  clock cut short has to be distinguishable from one that simply finished, or
  the interface cannot know to offer a longer limit
- running a whole batch again at a different preset, reusing the same entries
  rather than leaving two of everything in the queue
- removing and clearing entries, which hand the job back so the interface can
  release the GL textures hanging off it
- the settings file, round-tripped and then hand-damaged: unknown keys, garbage
  values, and numbers no slider could have produced must not reach the encoder

## `robustness` answers "what happens on hardware I do not have"

A compute backend that starts fine and then goes wrong, an optimizer that ends
in a state the file format cannot represent, a destination that cannot be
written. None of these need the failing hardware to reproduce; they need a way
to inject the failure, which is what the seams in
`set_gpu_backend_factory_for_testing` and `plan_output` are for. A recovery path
that only runs on the machines where it is already broken is not a recovery
path.

---

Every one of these was validated by injecting a known regression and confirming
it fails. A test that has never failed has not been tested. For the record,
with the mid-encode fallback removed, a backend that stops being correct at step
40 of 240 produces **14.54 dB where the CPU produces 37.67 dB** — and the old
code wrote that file out without a word.

The same was done for the newer cases: routing `.gsi` files back to the image
reader, dropping the Gaussian rescale from a scaled decode, and disabling the
settings clamp each make the corresponding test fail with a message naming what
broke.

---

## Correctness and safety

`ctest` covers:

- analytic gradients checked against numerical finite differences
- codec round trips
- renderer symmetry
- end-to-end compression
- GPU/CPU agreement

GPU-dependent tests are skipped automatically when no compatible GPU is
available.

### The GPU is verified before it is trusted

Compiling the compute shaders proves a driver accepts them. It does not prove
it runs them correctly, and that gap is where the awkward failures live: the
shaders use shared-memory reductions, barriers inside loops whose bounds come
from a buffer, and atomic tile insertion, all places where implementations
differ in practice.

So at startup the application solves a fixed 64×48 miniature of the real
problem on both backends and compares them. A machine whose GPU disagrees is
not slower, it is wrong, and it spends the session on the CPU with the reason
shown in the Settings panel. The check costs a few milliseconds.

The same judgement continues during a run. A GL error or a non-finite loss
withdraws the GPU mid-encode and the image is redone on the CPU from the same
seed, which reproduces the CPU-only result byte for byte. The log says it
happened.

### Two invariants

- **The encoder never emits bytes its own decoder rejects.** Anything it cannot
  represent returns nothing and is reported as a failure, rather than writing a
  `.gsi` that nothing will open.
- **An encode that succeeded is never lost to an unwritable destination.** The
  output location is chosen and proven writable *before* the work is spent, and
  if the first choice is refused the file goes somewhere that works and the log
  says where.

Files are written to a neighbouring temporary name and moved into place, so a
write that dies partway — a full disk, a removed drive — cannot leave a
truncated file where a valid one should be. This applies to the `.gsi`, to
exported `.png` files, and to the settings file.

### Untrusted input

The `.gsi` parser validates dimensions, Gaussian counts, and encoded payload
sizes before allocating memory. File-provided sizes are not used directly
without range checking. The parser test suite includes truncated files,
corrupted headers, and random input buffers.

A decode scale is treated the same way: it arrives from a text box, and 5000×
on a 4K image asks for a hundred gigapixels, so the range and the resulting
pixel count are checked before anything is allocated. The check lives beside
the arithmetic in the core, so the command line tool and the window cannot
disagree about which scales exist.

The settings file is the one input a user is invited to edit by hand. Its
parser ignores unrecognised keys, skips malformed lines, and clamps every value
to the range the interface offers, so no hand-edited file can ask the encoder
for something the sliders could not have produced.

The CLI has also been tested under AddressSanitizer with a collection of
corrupted `.gsi` files without detecting memory errors or crashes.

Memory is managed using standard containers throughout the core implementation.
OpenGL objects are owned and released by the GPU backend; the textures the
interface uploads for previews are released when their queue entry is removed,
one frame after the last frame that referred to them.
