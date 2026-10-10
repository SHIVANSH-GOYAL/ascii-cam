"""Measure glyph ink coverage to calibrate the ASCII ramp.

Terminal glyphs do not increase in density along alphabetical order, and a
hand-copied "generic" ramp usually misorders several characters. This tool
rasterizes each candidate with an actual monospace font and measures the
fraction of its character cell covered by ink.

    python calibrate_ramp.py                # measure with Consolas
    python calibrate_ramp.py --font PATH    # measure with another font
    python calibrate_ramp.py --levels 32    # target ramp sizes to compare

Pillow is required for calibration only. It is not a runtime dependency of the
app; the measured result is baked into ``ascii_cam/config.py``.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Candidate glyphs, roughly sparse to dense as published in the ASCII-art
# literature. The measured ordering may differ.
CANDIDATE = "  .'`^,:;Il!i~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

DEFAULT_FONT = r"C:\Windows\Fonts\consola.ttf"


def measure(ch: str, font: ImageFont.FreeTypeFont, px: int = 40) -> tuple[float, float]:
    """Return (cell coverage, advance) for one character.

    Coverage is ink pixels over total pixels of the glyph's advance box, so it
    reflects how much of one terminal cell the character darkens.
    """
    try:
        advance = float(font.getlength(ch))
    except (AttributeError, OSError):
        advance = 0.0
    width = max(1, int(round(advance)))
    height = px * 2
    cell = Image.new("L", (width, height), 0)
    ImageDraw.Draw(cell).text((0, 0), ch, font=font, fill=255)
    arr = np.asarray(cell, dtype=np.float32) / 255.0
    return float(arr.sum() / arr.size), advance


def build_ramp(measured, levels: int = 32) -> list[str]:
    """Choose ``levels`` characters spaced evenly in coverage.

    Terminal cells appear brighter in proportion to the ink they cover, so equal
    steps in coverage are equal steps in perceived brightness. Ordering every
    candidate instead would place neighbors with identical ink but very different
    shapes side by side, which reads as churn rather than detail.
    """
    ordered = [ch for _, ch in sorted(measured)]
    covs = [c for c, _ in sorted(measured)]
    if not covs:
        return ordered
    lo, hi = covs[0], covs[-1]
    if hi <= lo:
        return ordered[:1]
    picked: list[str] = []
    for i in range(levels):
        target = lo + (hi - lo) * i / (levels - 1)
        best = min(range(len(covs)), key=lambda k: (abs(covs[k] - target), covs[k]))
        if ordered[best] not in picked:
            picked.append(ordered[best])
    return picked


def gaps(ramp, coverage_of) -> list[float]:
    """Differences between successive coverage values in a ramp."""
    vals = [coverage_of(ch) for ch in ramp]
    return [round(b - a, 4) for a, b in zip(vals, vals[1:])]


def _report_font_sanity(coverages: list[float], char_count: int) -> None:
    """Warn when the font is rendering missing glyphs as boxes."""
    distinct = len(set(coverages))
    if distinct < char_count // 2:
        print("WARNING: many glyphs share a coverage value; the font may be")
        print("         rendering missing glyphs as boxes.")
    if max(coverages) - min(coverages) < 0.15:
        print("WARNING: coverage spread is small; check the font path.")


def main(argv: list[str]) -> int:
    font_path = DEFAULT_FONT
    if "--font" in argv:
        font_path = argv[argv.index("--font") + 1]

    px = 40
    font = ImageFont.truetype(font_path, px)

    print(f"font: {font_path} @ {px}px")
    print(f"{'char':>6} {'coverage':>9} {'advance':>8}")
    print("-" * 30)

    measured = sorted(
        (measure(ch, font) + (ch,) for ch in CANDIDATE)
    )
    # sorted() orders by (coverage, advance, char); keep (coverage, char) pairs
    # in the same sparse-to-dense order for the rest of the tool.
    pairs = [(cov, ch) for cov, _adv, ch in measured]
    coverage_of = {ch: cov for cov, ch in pairs}

    for cov, _adv, ch in measured:
        print(f"{repr(ch):>6} {cov:9.4f} {_adv:8.2f}")

    print()
    print(f"chars: {len(CANDIDATE)}  unique: {len(set(CANDIDATE))}")
    print(f"coverage range: {pairs[0][0]:.4f} .. {pairs[-1][0]:.4f}")
    _report_font_sanity([c for c, _ in pairs], len(CANDIDATE))

    print()
    print("ramp candidates by even coverage spacing:")
    print(f"{'levels':>6} {'len':>4} {'min gap':>8} {'mean gap':>9}")
    for n in (16, 20, 24, 32, 40, 64):
        ramp = build_ramp(pairs, n)
        g = gaps(ramp, coverage_of.get)
        print(f"{n:>6} {len(ramp):>4} {min(g):>8.4f} {sum(g) / len(g):>9.4f}")

    levels = 32
    if "--levels" in argv:
        levels = int(argv[argv.index("--levels") + 1])

    ramp = build_ramp(pairs, levels)
    print()
    print(f"CHOSEN with {levels} target levels -> {len(ramp)} characters")
    print("coverage per level (sparse -> dense):")
    for i, ch in enumerate(ramp):
        print(f"  {i:>2} {repr(ch):>5} {coverage_of[ch]:.4f}")
    print()
    print("successive coverage gaps:")
    print(gaps(ramp, coverage_of.get))
    print()
    print("ASCII_RAMP_PRECISE (sparse first):")
    print(repr("".join(ramp)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
