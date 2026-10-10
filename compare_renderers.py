"""Compare the legacy and upgraded pipelines on the same live frames.

Captures frames once, then renders each through the legacy 10-level ramp without
enhancement and through the upgraded pipeline with the measured ramp, CLAHE,
unsharp mask, and mirroring. Because both consume identical frames, any
difference is attributable to the pipeline rather than scene motion.

    python compare_renderers.py
"""

from __future__ import annotations

from ascii_cam import Camera
from ascii_cam.converter import (
    ASCII_RAMP,
    ASCII_RAMP_PRECISE,
    render_frame,
    to_ascii,
)
from ascii_cam.config import DETAIL_PRESETS

COLS, ROWS = 78, 27


def grab(count: int = 2) -> list:
    """Capture frames from the default camera and release it."""
    cam = Camera(index=0)
    cam.open()
    try:
        for _ in range(80):
            if cam.read() is not None:
                break
        frames = [cam.read() for _ in range(count)]
    finally:
        cam.close()
    return [f for f in frames if f is not None]


def show(title: str, grid) -> None:
    """Print a grid inside a fixed border."""
    border = "+" + "-" * COLS + "+"
    print(title)
    print(border)
    for row in grid:
        print("|" + "".join(str(c) for c in row).ljust(COLS) + "|")
    print(border)
    print()


def main() -> int:
    frames = grab(2)
    if not frames:
        print("No camera frames captured.")
        return 3
    print(f"captured {len(frames)} frame(s), rendering {COLS}x{ROWS}")
    print()

    for number, frame in enumerate(frames, 1):
        border = "=" * (COLS + 2)
        print(border)
        print(f"FRAME {number}")
        print(border)
        print()

        legacy = to_ascii(
            frame, COLS, ROWS, contrast=1.2, gamma=1.0,
            ramp=ASCII_RAMP, mirror=False, color_order="bgr",
        )
        show(
            "LEGACY   10-level ramp, no CLAHE, no sharpen, no mirror",
            legacy,
        )

        _label, clip, sharp, _ss = DETAIL_PRESETS[1]
        grid, colors = render_frame(
            frame, COLS, ROWS, contrast=1.2, gamma=1.0,
            ramp=ASCII_RAMP_PRECISE, clahe_clip=clip, sharpen_amount=sharp,
            mirror=True, color_order="bgr",
        )
        show(
            f"UPGRADED {len(ASCII_RAMP_PRECISE)}-level measured ramp, "
            f"CLAHE {clip}, sharpen {sharp}, mirror on",
            grid,
        )

        legacy_levels = len({str(c) for row in legacy for c in row})
        upgraded_levels = len({str(c) for row in grid for c in row})
        print(
            "distinct glyphs used:  legacy %d/%d   upgraded %d/%d"
            % (
                legacy_levels, len(ASCII_RAMP),
                upgraded_levels, len(ASCII_RAMP_PRECISE),
            )
        )
        print()

        print("UPGRADED TRUE COLOR (first 6 rows):")
        for row in range(min(6, len(grid))):
            parts = []
            for col in range(COLS):
                r, g, b = (int(v) for v in colors[row, col])
                parts.append(f"\033[38;2;{r};{g};{b}m{grid[row, col]}")
            print("\033[0m" + "".join(parts) + "\033[0m")
        print("\033[0m")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())