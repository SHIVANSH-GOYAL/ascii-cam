"""Vectorized image-to-ASCII conversion.

The pipeline is branch-free over whole numpy arrays, with OpenCV performing the
per-pixel work:

    frame -> resize to working size -> mirror -> grayscale -> CLAHE -> sharpen
                                                                    |
                                                    resize to cell grid -> normalize -> cutoff -> ramp -> glyph
                                                                    |
                                                    resize same cells -> color sample

Two invariants matter here:

* Glyph density and color come from a *single* resized image, so a glyph and its
  RGB sample always describe the same cell. Sampling them separately is how
  color/glyph misalignment creeps in.
* Enhancement runs at the cell grid's working resolution, not the camera's.
  CLAHE and GaussianBlur cost scales with pixel count, so enhancing a full
  camera frame for output that the resize discards is wasted work.
"""

from __future__ import annotations

import numpy as np

from .config import (
    ASCII_RAMP,
    ASCII_RAMP_PRECISE,
    BAYER_2X2,
    ENHANCE_SCALE,
)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a float luminance array in [0, 255].

    Uses the Rec. 601 luma formula, which weights green most heavily and matches
    human perception.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def mirror(rgb: np.ndarray) -> np.ndarray:
    """Flip horizontally into a contiguous array.

    ``cv2.flip`` rather than ``arr[:, ::-1]``: the slice returns a
    negatively-strided view, and every OpenCV call that follows (cvtColor, CLAHE,
    GaussianBlur) falls onto a slower non-contiguous path.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3:
        raise ValueError("mirror() expects an (H, W, 3) image")
    import cv2

    return cv2.flip(arr, 1)


def mirror_rgb(rgb: np.ndarray) -> np.ndarray:
    """Horizontal flip, exposed for callers that need it on its own."""
    return mirror(rgb)


def resize_rgb(rgb: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Resize an image to ``rows`` x ``cols`` using area interpolation.

    Area interpolation averages every pixel falling into a destination cell. It
    is the right filter here because a character represents one cell, so its ink
    should equal that cell's mean brightness. It also suppresses the aliasing
    point sampling produces on fine texture.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("cols and rows must be positive")
    if rgb.shape[0] == rows and rgb.shape[1] == cols:
        return rgb
    import cv2

    return cv2.resize(rgb, (cols, rows), interpolation=cv2.INTER_AREA)


def _to_gray(rgb: np.ndarray, color_order: str = "rgb") -> np.ndarray:
    """Convert an image to grayscale, honoring the caller's channel order."""
    import cv2

    if color_order == "bgr":
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _to_rgb(rgb: np.ndarray, color_order: str = "rgb") -> np.ndarray:
    """Convert an image to RGB, honoring the caller's channel order."""
    if color_order == "bgr":
        import cv2

        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb


_CLAHE_CACHE: dict = {}


def _clahe(clip: float, shape):
    """Build a CLAHE operator sized for the given image shape.

    Cached per (clip, tile grid) because constructing one allocates a histogram
    workspace, which would show up as per-frame garbage on a long run.

    The tile grid is derived from the image size rather than hard-coded: CLAHE
    equalizes locally within each tile, so a fixed tile count makes the physical
    tile area -- and therefore the scale of the contrast recovery -- change with
    the resolution being processed. Fixing the target tile size at ~16 pixels
    keeps behavior stable across resolutions.
    """
    if clip <= 0 or shape is None:
        return None
    import cv2

    rows, cols = int(shape[0]), int(shape[1])
    if not rows or not cols:
        return None
    tile_y = max(1, min(8, rows // 16))
    tile_x = max(1, min(8, cols // 16))
    key = (float(clip), tile_x, tile_y)
    if key not in _CLAHE_CACHE:
        _CLAHE_CACHE[key] = cv2.createCLAHE(
            clipLimit=float(clip), tileGridSize=(tile_x, tile_y)
        )
    return _CLAHE_CACHE[key]


def sharpen(gray: np.ndarray, amount: float) -> np.ndarray:
    """Apply an unsharp mask with a Gaussian kernel, preserving dtype and range."""
    if amount <= 0:
        return gray
    import cv2

    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def enhance_gray(
    gray: np.ndarray,
    clip: float = 0.0,
    sharpen_amount: float = 0.0,
    color_order: str = "rgb",
):
    """Recover local detail in shadows and highlights, then restore edges.

    CLAHE runs first so the equalized result feeds the caller's global tone
    controls, and sharpening last so the edge boost is not re-equalized away.
    Returns uint8 in [0, 255].

    ``gray`` may be an RGB frame or a grayscale array of any dtype; CLAHE accepts
    only 8-bit single channel, so the input is normalized first.
    """
    src = np.asarray(gray)
    if src.ndim == 3:
        src = _to_gray(src, color_order)
    else:
        src = np.atleast_2d(src)
        if src.ndim != 2:
            raise ValueError("enhance_gray() expects grayscale or (H, W, 3) RGB")

    if src.dtype != np.uint8:
        # Normalize by peak rather than by min-max range: min-max turns a
        # perfectly uniform image, such as a flat wall, into pure black.
        as_float = src.astype(np.float32)
        peak = float(as_float.max())
        if peak <= 1.0:
            as_float = as_float * 255.0  # already normalized to [0, 1]
        src = np.rint(np.clip(as_float, 0.0, 255.0)).astype(np.uint8)

    operator = _clahe(clip, src.shape)
    if operator is not None:
        src = operator.apply(src)
    if sharpen_amount > 0:
        src = sharpen(src, sharpen_amount)
    return np.clip(src, 0, 255).astype(np.uint8)


def _bayer(size: int):
    """Return an ordered dither matrix of the given power-of-two size."""
    if size <= 1:
        return None
    matrix = np.array(BAYER_2X2, dtype=np.float32)
    n = 1
    while n < size:
        n *= 2
        matrix = np.block(
            [
                [4 * matrix + 0, 4 * matrix + 2],
                [4 * matrix + 3, 4 * matrix + 1],
            ]
        )
    return matrix / (matrix.max() + 1.0)


def normalize(
    gray: np.ndarray,
    contrast: float = 1.0,
    brightness: float = 0.0,
    gamma: float = 1.0,
    dither: int = 0,
    dither_scale: float = 1.0,
) -> np.ndarray:
    """Map a [0, 255] grayscale array to [0, 1] with tone adjustments applied.

    Contrast is applied around mid-gray so highlights are not clipped, then an
    additive brightness offset, then gamma correction. Optional ordered dithering
    perturbs each cell so intermediate brightness can be represented as a pattern
    of neighbors instead of one flat tone.
    """
    centered = (np.asarray(gray, dtype=np.float32) / 255.0) - 0.5
    scaled = centered * contrast + brightness * 0.5
    if gamma != 1.0:
        # Clamp before the fractional power so negatives cannot become NaN.
        scaled = np.clip(scaled, 0.0, 1.0)
        norm = np.power(scaled, float(gamma))
    else:
        norm = scaled
    norm = np.clip(norm + 0.5, 0.0, 1.0)

    if dither and dither > 1 and norm.ndim == 2 and norm.size:
        matrix = _bayer(dither)
        if matrix is not None:
            rows, cols = norm.shape
            if rows >= matrix.shape[0] and cols >= matrix.shape[1]:
                reps = (rows // matrix.shape[0] + 1, cols // matrix.shape[1] + 1)
                tiled = np.tile(matrix, reps)[:rows, :cols]
                # Center the pattern on 0.5 and scale by the matrix size so the
                # perturbation spans a single ramp step rather than the whole
                # range.
                offset = (tiled - 0.5) / float(matrix.size)
                norm = np.clip(norm + offset * dither_scale, 0.0, 1.0)
    return norm


def ramp_indices(
    norm: np.ndarray, ramp: str = ASCII_RAMP_PRECISE
) -> np.ndarray:
    """Map normalized [0, 1] values to indices into the ASCII ramp.

    The ramp is ordered sparse to dense, so:

        norm 0.0 (black)  -> index 0           -> a space (empty cell)
        norm 1.0 (white)  -> index len(ramp)-1 -> the densest glyph

    This direction is required because a terminal draws ink on a dark background.
    """
    n = len(ramp) - 1
    return np.rint(np.clip(norm, 0.0, 1.0) * n).astype(np.intp)


def apply_black_cutoff(
    norm: np.ndarray, black_cutoff: float = 0.0
) -> np.ndarray:
    """Pin pitch-black cells to 0 so they resolve to a space.

    ``black_cutoff`` is a normalized-luminance threshold. Detail above it is
    untouched, and a cutoff of 0 disables the effect. The ramp is left alone, so
    a cutoff of 0 still yields the full dark range.
    """
    if black_cutoff <= 0:
        return norm
    out = np.array(norm, dtype=np.float32, copy=True)
    out[norm <= black_cutoff] = 0.0
    return out


def _chars(ramp: str) -> np.ndarray:
    return np.asarray(list(ramp))


def render_frame(
    rgb: np.ndarray,
    cols: int,
    rows: int,
    contrast: float = 1.0,
    brightness: float = 0.0,
    gamma: float = 1.0,
    ramp: str = ASCII_RAMP_PRECISE,
    clahe_clip: float = 0.0,
    sharpen_amount: float = 0.0,
    mirror: bool = False,
    dither: int = 0,
    black_cutoff: float = 0.0,
    enhance_scale: int = ENHANCE_SCALE,
    color_order: str = "rgb",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a frame into a character grid and a matching color grid.

    Returns ``(grid, colors)`` where ``grid`` is a (rows, cols) array of
    single-character strings and ``colors`` is a (rows, cols, 3) uint8 RGB array
    from the same resized image, so ``grid[r, c]`` and ``colors[r, c]`` always
    describe the same cell.

    ``mirror`` flips horizontally. ``black_cutoff`` pins pitch-black cells to
    empty. ``color_order`` is ``"bgr"`` for a frame that arrived straight from
    OpenCV and has not yet been converted.
    """
    if color_order not in ("rgb", "bgr"):
        raise ValueError("color_order must be 'rgb' or 'bgr'")

    work_cols, work_rows = cols * enhance_scale, rows * enhance_scale
    small = resize_rgb(rgb, work_cols, work_rows)

    # Flipping the working image is equivalent to flipping the source, since
    # area interpolation is a symmetric box average, and it touches ~100x fewer
    # pixels.
    if mirror:
        import cv2

        small = cv2.flip(small, 1)

    gray = enhance_gray(
        small, clip=clahe_clip, sharpen_amount=sharpen_amount, color_order=color_order
    )
    gray_cells = resize_rgb(gray, cols, rows)
    norm = normalize(
        gray_cells,
        contrast=contrast,
        brightness=brightness,
        gamma=gamma,
        dither=dither,
    )
    norm = apply_black_cutoff(norm, black_cutoff)
    grid = _chars(ramp)[ramp_indices(norm, ramp)]

    # Color comes from the same working image at the same cell grid, so glyph and
    # sample stay aligned. The RGB conversion happens on the cell grid, which is
    # three orders of magnitude smaller than a camera frame.
    colors = _to_rgb(resize_rgb(small, cols, rows), color_order)

    return grid, np.asarray(colors, dtype=np.uint8)


def to_ascii(
    rgb: np.ndarray,
    cols: int,
    rows: int,
    contrast: float = 1.0,
    brightness: float = 0.0,
    gamma: float = 1.0,
    ramp: str = ASCII_RAMP_PRECISE,
    clahe_clip: float = 0.0,
    sharpen_amount: float = 0.0,
    mirror: bool = False,
    dither: int = 0,
    black_cutoff: float = 0.0,
) -> np.ndarray:
    """Convert a frame to a (rows, cols) array of single characters."""
    grid, _ = render_frame(
        rgb, cols, rows,
        contrast=contrast, brightness=brightness, gamma=gamma, ramp=ramp,
        clahe_clip=clahe_clip, sharpen_amount=sharpen_amount, mirror=mirror,
        dither=dither, black_cutoff=black_cutoff,
    )
    return grid


def to_gray_ascii(
    gray: np.ndarray,
    cols: int,
    rows: int,
    contrast: float = 1.0,
    brightness: float = 0.0,
    gamma: float = 1.0,
    ramp: str = ASCII_RAMP,
    dither: int = 0,
) -> np.ndarray:
    """Convert an already-grayscale frame to a (rows, cols) char array."""
    small = resize_rgb(np.asarray(gray, dtype=np.uint8), cols, rows)
    norm = normalize(
        small, contrast=contrast, brightness=brightness, gamma=gamma, dither=dither
    )
    return _chars(ramp)[ramp_indices(norm, ramp)]


def sample_colors(
    rgb: np.ndarray, cols: int, rows: int, mirror: bool = False
) -> np.ndarray:
    """Return a (rows, cols, 3) uint8 array of averaged colors per cell."""
    src = mirror_rgb(rgb) if mirror else np.asarray(rgb, dtype=np.uint8)
    return resize_rgb(np.asarray(src, dtype=np.uint8), cols, rows)


def compute_grid(
    term_cols: int,
    term_rows: int,
    margin: int = 0,
    char_aspect: float = 0.5,
    source_aspect: float = 1.0,
    status_rows: int = 1,
) -> tuple[int, int]:
    """Compute the (cols, rows) grid that fits the terminal without stretching.

    Terminal cells are about twice as tall as they are wide, so ``char_aspect``
    (cell width / cell height) corrects the aspect ratio of the sampled region.
    ``status_rows`` are reserved at the bottom so the control line is never
    overwritten.
    """
    usable_cols = max(1, term_cols - 2 * max(0, margin))
    usable_rows = max(1, term_rows - 2 * max(0, margin) - max(1, status_rows))

    # The grid occupies (cols * cell_width) by (rows * cell_height) on screen, so
    # a match to the source aspect requires:
    #     (cols / rows) * char_aspect == source_aspect
    # hence cols/rows == source_aspect / char_aspect.
    eff_aspect = float(source_aspect) / (char_aspect if char_aspect else 1.0)

    cols = usable_cols
    rows = int(round(cols / eff_aspect))
    if rows > usable_rows:
        rows = usable_rows
        cols = int(round(rows * eff_aspect))
    return max(1, cols), max(1, rows)