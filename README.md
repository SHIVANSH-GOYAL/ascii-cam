# Live ASCII Camera

A real-time webcam-to-ASCII renderer that runs entirely in your terminal. Frames
from your webcam are captured with OpenCV, converted to ASCII with vectorized
NumPy operations, and drawn straight to the terminal with ANSI escape codes.
There is no GUI window, no image preview app, and no network access at any
point.

```bash
$ python main.py
```

## Features

- Three rendering modes: monochrome, green phosphor, true 24-bit color
- Horizontally mirrored by default, like a selfie view
- Adaptive local-contrast recovery and edge sharpening
- AMOLED-style black cutoff, so pitch-black renders empty
- Adjustable contrast, brightness, gamma, and detail presets
- Stable, low-flicker rendering that only rewrites lines that changed
- Terminal resize detection with correct character-cell aspect correction
- Compact status line showing FPS, resolution, and active settings
- Space to freeze a frame while the camera keeps running

## Requirements

- Python 3.11+
- A webcam (built-in or USB)
- A terminal that supports ANSI escape sequences

## Installation

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run
python main.py
```

The project can also be installed as a package:

```bash
pip install .
ascii-cam                  # installed entry point
python -m ascii_cam        # equivalent
```

Pillow is not needed to run the app. Install it only for ramp calibration:

```bash
pip install .[calibrate]
```

## Usage

Pick a camera by index when more than one is available:

```bash
python main.py --camera=1
```

Two flags are supported:

```bash
python main.py --diagnose            # report the environment, then render 3s
python main.py --diagnose --seconds 5
```

`--camera=` is the only option that affects normal operation; everything else is a
runtime key.

### Diagnosing a machine that renders differently

`--diagnose` exists for the case where the app works on one machine and looks
wrong on another with the same hardware. It prints the terminal size, detected
host, VT-processing state, color capability, terminal window-versus-buffer
sizes, the render grid, the camera resolution the app will request, dependency
versions, and a measured frame rate. Compare the output between the two machines:

```bash
python main.py --diagnose > machine-a.txt 2>&1
python main.py --diagnose > machine-b.txt 2>&1
diff machine-a.txt machine-b.txt
```

The lines that matter most are **VT processing**, **24-bit color**, **host**, and
**camera resolution**. `TERM` is normally unset on Windows even inside Windows
Terminal, so an unset value there is expected and not a problem by itself.

`python diagnose.py` is a separate, stronger check: it replays the exact bytes
the app emits through a small terminal emulator and prints the reconstructed
screen, so it catches wrapping, bad cursor moves, and color that bleeds between
rows.

## Controls

| Key        | Action                                          |
|------------|-------------------------------------------------|
| `Q`, `Esc` | Quit (releases the camera)                      |
| `1`        | Monochrome ASCII                                 |
| `2`        | Green phosphor terminal                          |
| `3`        | True-color ASCII (24-bit ANSI foreground)        |
| `D`        | Cycle detail preset: flat → balanced → high      |
| `M`        | Toggle horizontal mirroring (on by default)      |
| `B`        | Cycle the AMOLED black cutoff                    |
| `+` / `-`  | Increase / decrease contrast                      |
| `[` / `]`  | Decrease / increase brightness                    |
| `G`        | Cycle gamma presets (0.6 - 2.2)                  |
| `F`        | Toggle the FPS readout                           |
| `Space`    | Freeze / unfreeze the displayed frame             |

The active controls are always shown on the status line at the bottom of the
screen, including the current mirror, black cutoff, and detail settings.

## Shading direction

The mapping direction is the single most important thing to get right, because a
terminal draws glyphs as *ink on a dark background*. So:

| Luminance        | Glyph                                  |
|------------------|----------------------------------------|
| Black (0.0)      | index 0 — a space, empty cell           |
| Dark gray        | sparse, low-density characters          |
| Mid gray         | medium-density characters               |
| Bright           | dense characters                        |
| White (1.0)      | last index — the densest glyph          |

The ramp is ordered **sparsest first**, so index 0 is a space and the last index
is the densest glyph. An empty cell is the only way a terminal can show black,
and maximum ink is the only way it can show white.

This was previously inverted — `black -> '@'` and `white -> ' '` — which made
every shadow render as a solid block of ink and every highlight vanish. The cause
was the ramp string being ordered densest-first while `ramp_indices` indexed it
directly with normalized luminance. Reversing the ramp fixed the direction
without reversing any image data, so the scene's brightness relationships are
preserved rather than inverted.

Two tests guard against this regressing:
`TestIntensityDirection.test_black_is_a_space` and the tripwire
`test_inverted_ramp_would_give_the_opposite`, which asserts the density
direction explicitly.

## AMOLED black

A real AMOLED panel switches its sub-pixels fully off, so pitch-black costs
nothing and reads as empty. A terminal cannot do that, but a dark background
plus *no ink at all* comes close: emitting a space instead of a faint glyph
leaves the cell visually empty.

`B` cycles the cutoff — 0.00 (off), 0.02, 0.04 (default), 0.08, 0.15 — and the
active value is shown in the status line. It is implemented as its own pipeline
step (`apply_black_cutoff`) rather than a change to the ramp:

- the ramp keeps its full dark range, so a cutoff of 0 still gives you dense
  darks when you want them;
- the threshold sits *after* contrast, brightness, and gamma, so it responds to
  those controls exactly as the glyph mapping does;
- detail above the threshold is untouched, so a face in shadow keeps its
  features while the black behind it goes empty.

Set it in config with `"black_cutoff": 0.10`.

## The character ramp

The default ramp has 27 levels and was **measured**, not guessed. Everyone
correctly notes that glyph density — not alphabet order — determines the mapping,
and it is easy to assume a published ramp is right. It usually is not:
`calibrate_ramp.py` rasterises each candidate with a real font and measures the
fraction of its character cell covered by ink, which found that **42 of the 64
candidate characters collapse into the 0.06–0.13 coverage band**. Ordering them
all would place neighbors with identical ink but very different shapes side by
side, producing churn in flat regions instead of detail.

So the ramp is chosen by *even coverage spacing*, which makes every neighbouring
pair differ by a consistent perceptual step. Regenerate or re-tune it with:

```bash
python calibrate_ramp.py              # inspect the measured coverage table
python calibrate_ramp.py --levels 32  # candidate ramp counts
python calibrate_ramp.py --font PATH  # measure with your terminal's font
```

Pillow is used by `calibrate_ramp.py` only and is **not** a runtime dependency;
the measured ordering is baked into `ascii_cam/config.py`. Set a custom ramp with
`"ramp": "..."` in the config, ordered sparse-first so index 0 is a blank.

## How it works

```
webcam ─► cv2.VideoCapture ─► mirror (cv2.flip) ─► CLAHE ─► unsharp mask
                                                                    │
                 ┌──────────────────────────────────────────────────┘
                 ▼
           luminance ──► resize (INTER_AREA) ──► normalize ──► ramp index ──► glyph
                 │                                          │
                 └──► enhance ──► resize ──► color sample ◄─┘  (true color, same cells)
                                                                    │
                                                           ANSI row strings
                                                                    │
                                                      cursor-positioned write
```

1. **Capture** (`ascii_cam/camera.py`) decodes frames on a background thread and
   hands the renderer the most recent one, so the pipeline never blocks on
   capture latency. The capture resolution is chosen from the terminal grid
   (see [Performance](#performance)) and the frame is handed back in the camera's
   native BGR order with no conversion.
2. **Convert** (`ascii_cam/converter.py`) resizes to a working image, mirrors it,
   converts to grayscale and applies CLAHE plus an unsharp mask *at the working
   resolution*, resizes to the character grid with `INTER_AREA`, applies
   contrast/brightness/gamma, applies the black cutoff, then indexes the ramp.
   All per-pixel work happens inside NumPy or OpenCV, with no Python loops over
   pixels.
3. **Render** (`ascii_cam/renderer.py`) builds one string per row and only
   repaints rows whose content changed, using `ESC[row;1H` positioning plus
   erase-to-end-of-line. The cursor is hidden while drawing and restored on
   exit, including on `Ctrl-C`.

Enhancement runs on a working image sized to the *terminal*, not the camera.
CLAHE and `GaussianBlur` are the two expensive stages and their cost scales with
pixel count, so enhancing a full camera frame for output that the resize then
discards is wasted work. Their tile grid is derived from the image size (~16-pixel
tiles) so the strength of the contrast recovery does not change with the
resolution.

Luminance chooses the glyph and RGB chooses its color, both sampled from **one**
resized image, so glyph and color can never drift apart. Dark pixels map to
sparse characters (`.`, ` `) and bright pixels to dense ones (`&`, `#`, `@`).

### Why `cv2.flip` and not a slice

`mirror()` uses `cv2.flip(arr, 1)` rather than `arr[:, ::-1]`. The slice shares
memory and looks free, but it returns a negatively-strided array, and every
OpenCV call that follows (`cvtColor`, CLAHE, `GaussianBlur`) takes a
non-contiguous path. On the development machine one `cvtColor` cost 0.07 ms
contiguous versus 2.16 ms on the view — a 30x penalty that made the whole
conversion 7.2 ms instead of 3.8 ms. `cv2.flip` copies once into a normal layout
and is the faster option overall.

It also flips the *working* image rather than the camera frame, which is exactly
equivalent: `INTER_AREA` is a symmetric box average, so resize-then-flip gives
the identical cells to flip-then-resize (asserted in
`TestCaptureSizing.test_resize_then_flip_equals_flip_then_resize`).

### Area interpolation

Downsampling uses `INTER_AREA` rather than point sampling. Averaging is not a
shortcut here: it is what makes a character's ink match its cell's mean
brightness, and it suppresses the moire that point sampling produces on fine
texture.

### True-color output

color escapes are ~15 bytes and a character is 1, so a naive true-color row is
dominated by escapes. Two things keep it down: runs of identical quantized color
emit a single escape, and colours are quantized to `color_quantize` levels per
channel (default 16, i.e. 4096 colours — indistinguishable at cell size, ~20%
less output). Set `"color_quantize": 0` in the config for full 24-bit. Measured
overhead versus monochrome is roughly 5.5x bytes per frame at the same frame
rate.

color sampling does not reduce spatial resolution: the color grid is the same
cell grid the glyphs are drawn on.

## Aspect ratio

Terminal cells are roughly twice as tall as they are wide, so resizing a square
image onto a square grid of characters would make it look twice as tall as it is.
`compute_grid()` corrects for this using a `char_aspect` ratio (default `0.5`,
i.e. cell width / cell height) and letterboxes the grid to fit the terminal. The
required grid ratio is:

    cols / rows = source_aspect / char_aspect

For a 16:9 webcam in cells that are 2:1 that is 3.56 columns per row, so a square
image ends up as 2 columns per row and looks square on screen. The displayed
image never stretches. Tune `char_aspect` in `ascii_cam/config.py` if your
terminal font uses a different cell shape.

You can also launch the app as a module:

```bash
python -m ascii_cam
```

## Supported terminals

| Terminal                     | Status |
|------------------------------|--------|
| Windows Terminal             | Full support, all three modes |
| PowerShell 7 in Windows Terminal | Full support |
| VS Code integrated terminal   | Full support |
| Windows Console Host (conhost) | Works; true color depends on Windows version |
| macOS Terminal.app, iTerm2, GNOME Terminal, Windows Subsystem for Linux | Works |

The app checks `TERM` / `COLORTERM` and falls back to monochrome when ANSI
color is unavailable.

## Performance

The default target is 24 FPS. Two things were measured rather than assumed, both
with the same camera, terminal, and mode: CPU cost before and after the
optimization pass, and the quality cost of the lower capture resolution.

### CPU, before and after

Measured with `python bench_attribution.py`, which samples `time.process_time()`
over the whole process (camera thread included) against wall time, 5-second
windows, real webcam, 80x24 terminal, monochrome, detail=balanced at 24 fps:

| Stage                       | Before     | After      | Change |
|-----------------------------|------------|------------|--------|
| Camera reader thread alone  | 42–45% core | **6.2%**  | ~7x lower |
| Full loop (all stages)      | 82–96% core | **29.9%** | ~3x lower |
| Achieved frame rate         | 23.3 fps   | **23.7 fps** | unchanged |

Where the saving came from, each measured independently:

| Change | Reason |
|--------|--------|
| No full-resolution color conversion | The reader thread converted every 640x480 frame from BGR to RGB before handing it over — 20% of a core for output the resize immediately discards. The conversion now happens on the working image, three orders of magnitude smaller. |
| Capture resolution scaled to the grid | `read()` alone cost 26% of a core at 640x480 versus 8% at 320x240. |

`read()` cost at 640x480 vs 320x240: 25.9% vs 8.1% of a core.

### Quality cost of the lower capture resolution

Measured on a synthetic scene with finger-scale stripes, a gradient, and a shadow
region, rendered to a 61x23 grid. "Retained structure" is the edge energy of the
glyph-index map — how much of the scene's structure survives quantization:

| Capture source | Edge energy | Distinct glyphs |
|----------------|-------------|-----------------|
| 640x480        | 1337        | 12              |
| **320x240**    | **1208** (90%) | 12           |
| 160x120        | 684 (51%)   | 12              |

320x240 keeps 90% of the retained structure for roughly one sixth of the capture
cost; 160x120 falls off a cliff to half, so the adaptive sizing stays at or above
320x240 and scales up for large terminals. It requests only standard resolutions
because drivers silently ignore arbitrary sizes and fall back to 640x480.

If you would rather have maximum source detail, set the capture size explicitly:

```python
camera = ThreadedCamera(index=0, width=640, height=480)
```

### Conversion cost

`python bench_loop.py`, 61x23 grid, measured inside the real loop:

| Stage               | Cost    |
|---------------------|---------|
| prepare (convert)   | ~2.8 ms |
| render              | ~0.2 ms |
| keys                | ~0.01 ms|
| terminal/grid probe | ~0.2 ms |

Conversion is well inside the 24 FPS budget, so the achieved rate is set by the
webcam rather than the pipeline — which is why CPU dropped so far without the
frame rate moving.

Two pacing details, both of which were bugs at some point:

- Frame time is measured from the *top* of the loop so the sleep subtracts the
  real processing cost. Measuring after rendering instead left the body time on
  top of the target interval and capped the rate below the requested one (21 fps
  against a 24 target).
- The FPS readout is sampled after the pacing sleep. Timing only the processing
  cost reported ~230 fps for an app that was genuinely running at 30.

Output volume per frame grows with color, which is the real cost on a slow
terminal:

Output volume per frame, which is the real cost on a slow terminal:

| Mode        | Bytes/frame |
|-------------|-------------|
| Monochrome  | ~790        |
| Phosphor    | ~1050       |
| True color | ~4400       |

True color is the only mode that can lag on a slow terminal. Press `1` or `2`
to fall back.

### Terminal font size and output dimensions

Detail is bounded by the number of character cells you can fit, so a smaller
font means more detail. Recommended for Windows Terminal:

| Font size | Typical grid (80x24 window) | Notes |
|-----------|-----------------------------|-------|
| Cascadia Mono / Consolas, **10–12 pt** | 61x23 to 80x24 | Comfortable to read |
| **8–9 pt** | 100x40 to 120x45 | Good detail; still readable on a 1080p screen |
| 6–7 pt | 150x50+ | Maximum detail, fatiguing over time |

Change it in **Settings → Defaults → Appearance → Font size** (or
`Ctrl+,` on the profile's Appearance tab). The app adapts automatically: it
re-measures the terminal every frame and recomputes the grid, so no restart is
needed. The capture resolution follows the grid too, which is why a larger
terminal also asks the camera for more pixels.

Font size is terminal configuration, not an application control, so the app does
not attempt to change it and does not modify your global terminal settings. Run
it with the usual `python main.py`.

## Troubleshooting

**"Camera 0 is unavailable"**
The camera is missing, in use by another application (Zoom, Teams, OBS), or
blocked by permissions. Check Windows **Settings → Privacy & security → Camera**
and close other apps using it. The app also lists which camera indices did open,
so run `python main.py --camera=<n>` with one of those.

**Nothing appears in the terminal, but the app is running and the camera is on**
Run `python diagnose.py`. It replays the exact bytes into a terminal emulator and
shows the reconstructed screen, plus it checks the two failure modes that hide
the picture: a whole-line erase (`ESC[2K`) emitted *after* the art is drawn, and
a trailing newline that scrolls the frame on every tick. Paste the output when
asking for help.

**The image scrolls or flickers continuously**
Something is emitting a line feed. The drawing area must end exactly where the
status line ends so the final row is not scrolled. Confirm with
`python diagnose.py` that "line feeds written" is `0`.

**Garbled colors / literal escape codes**
True color requires 24-bit ANSI support. Use mode `1` or `2`. On Windows the app
assumes true color is available, because Windows Terminal, VS Code's terminal,
and conhost since 1607 all honour it; on POSIX it checks `COLORTERM`/`TERM` and
only a 256-color terminal falls back to plain glyphs. If color genuinely is
unavailable the characters are still drawn, just without it — use
`python diagnose.py --env` to see which capability was detected.

**Pressing 3 shows no color**
That was a real bug, not a terminal limitation. The capability probe read
`TERM`/`COLORTERM` only, and Windows leaves both unset even inside Windows
Terminal, so mode 3 silently rendered monochrome. It is fixed: on Windows, ANSI
support now implies 24-bit support. Regression tests:
`test_windows_true_color_is_assumed` and
`test_renderer_uses_color_when_supported`.

**Flat areas look noisy or grainy**
That is CLAHE amplifying sensor noise at `balanced` or `high`. Press `D` until
the status line reads `flat`, or reduce the CLAHE clip limit in
`DETAIL_PRESETS` in `ascii_cam/config.py`. Ordered dithering is available but off
by default for the same reason — turn it on with `"dither": 2` if you would
rather see stable texture than smooth tones.

**Low or jumpy frame rate**
The camera warm-up takes about a second. After that, run `python bench_modes.py`
to see what each mode actually costs, and reduce the terminal window size (fewer
characters means less output) or use monochrome mode.

**Image looks stretched or squished**
The shape of the character cell depends on your terminal font. `char_aspect` in
`ascii_cam/config.py` controls the correction: lower it if the image looks too
tall, raise it if it looks too wide.

**Nothing renders when piping to a file**
The app requires a real terminal. Run it directly in a terminal, not with output
redirected.

**The image drifts upward or flickers, but only on one machine**
The status line was clamped to the real console width, so it can no longer wrap.
Before that fix, a status line longer than the console width wrapped on the final
row and scrolled the terminal once per frame — visible as steady upward drift.
The effect depends only on console width, so two machines with identical hardware
but different window sizes could look completely different. If this still
happens, compare `python main.py --diagnose` between the machines and check the
`size` and `VT processing` lines.

**Colors are wrong, or mode 3 looks monochrome**
The app reads the console's VT-processing flag and the OS build rather than
assuming True. On Windows, 24-bit SGR needs Windows 10 build 18362 or newer;
earlier builds accept VT but clamp to a 16-color palette. Run
`python main.py --diagnose` and check `VT processing`, `24-bit color`, and
`build`. If VT is off, the app now attempts to enable it at startup. When 24-bit
is genuinely unavailable, mode 3 still draws the glyphs, just without color.

**Pressing 3 shows no colour**
That was a real bug, not a terminal limitation. The capability probe read
`TERM`/`COLORTERM` only, and Windows leaves both unset even inside Windows
Terminal, so mode 3 silently rendered monochrome. It is fixed: on Windows, ANSI
support now implies 24-bit support. Regression tests:
`test_windows_true_color_is_assumed` and
`test_renderer_uses_color_when_supported`.

**Dark areas look empty and I want the detail back**
That is the black cutoff. Press `B` until the status line reads `blk:0.00`, or
raise `black_cutoff`'s neighbors in `BLACK_CUTOFF_PRESETS`. If everything looks
flat instead, press `D` for the `flat` preset or lower the clip limits in
`DETAIL_PRESETS`.

## Testing

```bash
# Unit tests (no camera needed)
python -m unittest discover -s tests -v

# End-to-end loop checks (requires a working camera)
python verify_loop.py

# Replay the output through a terminal emulator
python diagnose.py

# CPU attribution: reader thread vs full loop vs control
python bench_attribution.py

# Per-stage timing of the render loop
python bench_loop.py

# Legacy vs upgraded renderer on the same live frames
python compare_renderers.py
```

`diagnose.py` answers the *"my terminal shows nothing"* question. It runs the
real loop, records every byte sent to the terminal, and then **replays those
bytes through a small VT emulator** that reproduces what a terminal would
actually put on screen — cursor positioning, line erasing, and scrolling.

```bash
python diagnose.py              # run frames, show the reconstructed screen
python diagnose.py --env        # environment facts only
python diagnose.py --frames 20  # more frames
```

This catches bugs that a log file hides. If you redirect the output to a file
and strip the escape codes it *looks* like art is being printed, but a real
terminal may still show nothing. That is exactly how the invisible-art bug was
found, and its reconstructed screen also confirms the shading direction — bright
regions should come out as `@`, `&`, `#`, and pitch-black as blank.

### What the tests cover

190 tests, `python -m unittest discover -s tests`:

- **Shading direction** — black resolves to a space, white to the densest glyph,
  monotonic density across a gradient, agreement with an independent reference
  mapping, and a tripwire that fails loudly if the ramp is ever re-inverted.
- **AMOLED black** — values at or below the threshold become spaces, values above
  are untouched, the ramp keeps its full dark range, cutoff survives the tone
  controls, and the input array is not mutated.
- **color mode state** — the `1`/`2`/`3` keys move the active mode, repeated
  presses stay stable, and the `B` key cycles the cutoff.
- **RGB ANSI generation** — `ESC[38;2;R;G;Bm` escapes with the sampled RGB, exact
  values when quantization is off, run-length encoding of repeated color, a
  reset at the end of every color row, and no color bleeding into the status
  bar.
- **color/character alignment** — colours match an independent area average,
  alignment survives mirroring, and the shape tracks the glyph grid.
- **Terminal resizing and aspect** — the grid recomputes for each terminal size,
  the status row is never overwritten, proportions are preserved, and extreme
  aspect ratios never overshoot the terminal.
- **BGR order** — frames from OpenCV are converted correctly, and reading them
  the wrong way round is caught.
- **Capture sizing** — only standard resolutions are requested, sizing scales
  with the grid, and resize-then-flip equals flip-then-resize.

### What was verified on the real camera

Not everything is unit-testable. These were run against the live webcam and an
80x24 terminal:

- Shading direction reversed and verified by eye — bright regions now dense,
  dark regions sparse, pure black empty.
- True color emits real 24-bit escapes: 3,379 escapes with 135 distinct RGB
  values in a 6-second window against a **plain Windows `TERM`/`COLORTERM`**
  profile, which is what previously produced monochrome.
- CPU before/after as tabulated above, from `bench_attribution.py`.
- Capture resolution adapts: 320x240 on an 80x24 terminal, 800x600 on a
  200x75 grid.
- Clean exit code 0 on `Q`, all three modes switching, freeze, resize, and cursor
  restoration via `verify_loop.py`.

The image used here is a low-contrast grey synthetic source, so the CLAHE
recovery is not judged on a face or a hand — that would need a real scene.

### Measured result

On a synthetic scene containing finger-scale stripes, a soft gradient, and a dark
shadow region — the kind of content a face and hands produce — rendered to 61x23:

|                       | Ramp levels used | Retained structure |
|-----------------------|------------------|--------------------|
| Legacy 10-level ramp  | 5 of 10          | 4%                 |
| Upgraded 27-level ramp| 15 of 27         | 14%                |

Roughly **3.6x** more of the source's distinguishable structure survives the
glyph quantization.

## Project layout

```
ascii_cam/
  config.py      ASCII ramps, gamma presets, detail presets, escape sequences
  camera.py      capture wrapper, threaded reader, adaptive capture sizing
  converter.py   mirror, CLAHE, sharpen, resize, luminance, black cutoff, ramp
  renderer.py    capability probing, cursor-positioned ANSI output, color
  main.py        State, key handling, render loop, --diagnose
main.py               CLI entry point
diagnose.py           replay the output through a terminal emulator
verify_loop.py        end-to-end loop checks
bench_loop.py         per-stage timing of the render loop
bench_attribution.py  CPU cost: reader thread vs full loop vs control
bench_modes.py        fps and bytes per mode/detail preset
calibrate_ramp.py     measure glyph ink coverage to build the ramp
compare_renderers.py  legacy vs upgraded on the same live frames
tests/                unit tests
```

Every script except `calibrate_ramp.py` requires a working camera, unless it is a
unit test. `calibrate_ramp.py` needs a font file and Pillow instead.

## Contributing

```bash
pip install -r requirements.txt
pip install ".[dev]"

# Tests and lint
python -m unittest discover -s tests -v
python -m pyflakes ascii_cam main.py verify_loop.py
```

The unit suite needs no camera. `verify_loop.py`, the `bench_*.py` scripts,
`compare_renderers.py`, and `diagnose.py` need one.

Regenerate the character ramp after changing the candidate set, then paste the
result into `ascii_cam/config.py`:

```bash
pip install ".[calibrate]"
python calibrate_ramp.py --levels 32
```

## Privacy

Frames are read from the camera into memory only. Nothing is written to disk and
nothing is sent anywhere. The app never saves or uploads camera frames.

## License

MIT. See [LICENSE](LICENSE).
