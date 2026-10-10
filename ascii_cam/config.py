"""Configuration constants for the ASCII camera.

Ramp ordering is the load-bearing decision in this module. A terminal draws
glyphs as ink on a dark background, so index 0 must hold the sparsest glyph
(a space) and the final index the densest. A black pixel therefore resolves to
an empty cell and a white pixel to maximum ink.

The precise ramps here were measured with ``calibrate_ramp.py`` against Consolas
(the Windows Terminal default). Counting ink coverage per character cell rather
than trusting a hand-copied list matters: 42 of the 64 candidate glyphs land in
the 0.06-0.13 coverage band, so ordering them all would place neighbors with
identical ink but very different shapes side by side, which reads as churn
rather than detail. The ramps are chosen at even coverage intervals instead.
"""

# 10-level ramp, ordered sparse to dense. Index 0 is a space.
ASCII_RAMP = " .:-=+*#%@"

# 27-level ramp measured against Consolas at 40px. Ink coverage runs from 0.0000
# (space) to 0.1837 ("@").
# Regenerate with: python calibrate_ramp.py --levels 32
ASCII_RAMP_PRECISE = " `.':,^~!/?LvzYoahUmO#0MQ&@"

# 64-level ramp for large terminals, ordered sparse to dense.
ASCII_RAMP_MAX = " .,-':,_^~;!*/+?rc|L)(vJiz}{lt[]xYnuC1fIojakhZwUXbqpdmO#80$%WBMQ&@"

# Gamma presets cycled with the "G" key.
GAMMA_PRESETS = [0.6, 0.8, 1.0, 1.2, 1.5, 2.2]

# ANSI escape sequences.
ESC = "\033"
HIDE_CURSOR = ESC + "[?25l"
SHOW_CURSOR = ESC + "[?25h"
CLEAR_SCREEN = ESC + "[2J"
HOME = ESC + "[H"

# Erase from the cursor to the end of the current line. Used after a row's text
# is written so stale trailing characters are dropped while the text just
# written is preserved.
#
# This must not be ESC[2K ("erase entire line"), which deletes the whole line
# and leaves the screen blank.
ERASE_TO_EOL = ESC + "[0K"

# Render modes.
MODE_MONOCHROME = "monochrome"
MODE_PHOSPHOR = "phosphor"
MODE_TRUE_COLOR = "truecolor"

# Pitch-black cutoff, in normalized luminance. Pixels at or below it render as
# spaces so black reads as empty rather than faint. Applied after the tone
# controls so it responds to contrast, brightness, and gamma the same way the
# glyph mapping does, and kept separate from the ramp so a cutoff of 0 still
# yields the full dark range.
BLACK_CUTOFF = 0.04

# Presets cycled with the "B" key.
BLACK_CUTOFF_PRESETS = [0.0, 0.02, 0.04, 0.08, 0.15]

# CLAHE (Contrast Limited Adaptive Histogram Equalization) restores shadow and
# highlight detail that a single global tone curve discards. The clip limit caps
# how much local contrast may be boosted, which is what keeps sensor noise from
# surfacing as grit.
DEFAULT_CLAHE_CLIP = 2.0
DEFAULT_CLAHE_TILE = (8, 8)

# Unsharp mask amount applied to the working image.
DEFAULT_SHARPEN = 0.35

# Enhancement runs at this multiple of the cell grid instead of the camera
# frame. CLAHE and GaussianBlur are the two expensive stages and their cost
# scales with pixel count; a 640x480 frame holds roughly 100x the pixels of a
# 61x23 cell grid for output that the resize then discards.
ENHANCE_SCALE = 2

# Ordered dither matrix size; 0 disables dithering.
DEFAULT_DITHER = 0
BAYER_2X2 = ((0, 2), (3, 1))

# Detail presets cycled with the "D" key:
#   (label, clahe_clip, sharpen, supersample)
DETAIL_PRESETS = [
    ("flat", 0.0, 0.0, 1),
    ("balanced", 2.0, 0.35, 2),
    ("high", 3.0, 0.6, 2),
]

# Default settings.
DEFAULT_CONFIG = {
    "mode": MODE_MONOCHROME,
    "contrast": 1.0,
    "brightness": 0.0,  # additive offset in normalized space [-1, 1]
    "gamma": 1.0,
    "show_fps": True,
    "target_fps": 24,
    "char_aspect": 0.5,  # cell width / cell height
    "margin": 0,
    "fps_smoothing": 0.9,
    "mirror": True,
    "ramp": ASCII_RAMP_PRECISE,
    "detail_index": 1,
    "dither": DEFAULT_DITHER,
    "color_quantize": 16,  # levels per channel for true-color runs; 0 = full 24-bit
    "black_cutoff": BLACK_CUTOFF,
}

# Green phosphor palette: foreground and background.
PHOSPHOR_FG = (0, 255, 65)
PHOSPHOR_BG = (0, 10, 0)

# Status line. Vitals first and control hints last, so truncating a narrow
# terminal drops the hints before it drops the live information.
STATUS_TEMPLATE = (
    "FPS:{fps:4.1f} {res} {mode} {detail} "
    "C:{contrast:.2f} B:{brightness:+.2f} G:{gamma:.2f} "
    "mir:{mirror:<3} blk:{black_cutoff:.2f} | "
    "Q quit 1 mono 2 phos 3 color D detail M mirror B black "
    "+-contrast []bright G gamma F fps Space freeze"
)