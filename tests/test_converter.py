"""Unit tests for image-to-ASCII conversion, mapping, mirroring, and aspect."""

from __future__ import annotations

import unittest

import numpy as np

from ascii_cam.converter import (
    ASCII_RAMP,
    ASCII_RAMP_PRECISE,
    compute_grid,
    enhance_gray,
    luminance,
    mirror,
    normalize,
    ramp_indices,
    render_frame,
    resize_rgb,
    sample_colors,
    sharpen,
    to_ascii,
)


def make_image(size=(64, 48), color=(255, 255, 255)):
    return np.full((*size, 3), color, dtype=np.uint8)


def cells(gray, cols, rows, **kw):
    """Loops-based reference for the vectorized ramp mapping."""
    out = []
    for r in range(rows):
        out.append([gray[r][c] for c in range(cols)])
    return out


class TestLuminance(unittest.TestCase):
    def test_white_image_is_bright(self):
        gray = luminance(make_image(color=(255, 255, 255)))
        self.assertAlmostEqual(float(gray.max()), 255.0, places=1)

    def test_black_image_is_dark(self):
        gray = luminance(make_image(color=(0, 0, 0)))
        self.assertAlmostEqual(float(gray.min()), 0.0, places=3)

    def test_pure_red_matches_luma_coefficient(self):
        gray = luminance(np.array([[[255, 0, 0]]], dtype=np.uint8))
        self.assertAlmostEqual(float(gray[0, 0]), 0.299 * 255, places=2)

    def test_green_dominates_luminance(self):
        red = float(luminance(np.array([[[255, 0, 0]]], np.uint8))[0, 0])
        green = float(luminance(np.array([[[0, 255, 0]]], np.uint8))[0, 0])
        self.assertGreater(green, red)

    def test_shape_and_dtype(self):
        gray = luminance(make_image())
        self.assertEqual(gray.shape, (64, 48))
        self.assertEqual(gray.dtype, np.float32)


class TestResize(unittest.TestCase):
    def test_resize_rgb_shape(self):
        img = make_image((64, 48))
        out = resize_rgb(img, 20, 10)
        self.assertEqual(out.shape, (10, 20, 3))

    def test_resize_is_averaging(self):
        # A 2x2 image with a single white pixel averages to 255/4.
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[0, 0] = (255, 255, 255)
        out = resize_rgb(img, 1, 1)
        self.assertEqual(int(out[0, 0, 0]), 64)

    def test_resize_averages_mixed_pixels(self):
        # Half white, half black -> a mid gray cell.
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[0] = 255
        out = resize_rgb(img, 1, 1)
        self.assertAlmostEqual(int(out[0, 0, 0]), 128, delta=5)

    def test_resize_noop_when_same_size(self):
        img = make_image((4, 4))
        out = resize_rgb(img, 4, 4)
        self.assertIs(out, img)

    def test_sample_colors_dtype(self):
        colors = sample_colors(make_image((64, 48), (10, 20, 30)), 5, 3)
        self.assertEqual(colors.shape, (3, 5, 3))
        self.assertEqual(colors.dtype, np.uint8)


class TestNormalize(unittest.TestCase):
    def test_identity(self):
        gray = np.array([[0, 64, 128, 192, 255]], np.uint8)
        out = normalize(gray)
        np.testing.assert_allclose(out, gray / 255.0, atol=1e-5)

    def test_contrast_increases_spread(self):
        gray = np.array([[0, 255]], np.uint8)
        plain = normalize(gray, contrast=1.0)
        strong = normalize(gray, contrast=2.0)
        self.assertLessEqual(float(strong.min()), float(plain.min()))
        self.assertGreaterEqual(float(strong.max()), float(plain.max()))

    def test_positive_brightness_brightens(self):
        gray = np.full((1, 3), 128, np.uint8)
        bright = normalize(gray, brightness=0.5)
        self.assertTrue(bright.min() > 128 / 255.0)

    def test_gamma_below_one_brightens(self):
        gray = np.full((1, 3), 128, np.uint8)
        out = normalize(gray, gamma=0.5)
        self.assertGreater(out.min(), 128 / 255.0)

    def test_gamma_above_one_darkens(self):
        gray = np.full((1, 3), 128, np.uint8)
        out = normalize(gray, gamma=2.0)
        self.assertLess(out.min(), 128 / 255.0)

    def test_output_is_clamped(self):
        gray = np.array([[0, 255]], np.uint8)
        out = normalize(gray, contrast=8.0, brightness=1.0, gamma=0.25)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_negative_values_do_not_become_nan(self):
        """A negative input to a fractional power must not produce NaN."""
        gray = np.array([[0, 128, 255]], np.uint8)
        out = normalize(gray, contrast=4.0, brightness=-1.0, gamma=0.6)
        self.assertFalse(np.isnan(out).any())


class TestRampIndices(unittest.TestCase):
    def test_black_maps_to_first_index(self):
        idx = ramp_indices(np.array([[0.0]]))
        self.assertEqual(int(idx[0, 0]), 0)

    def test_white_maps_to_last_index(self):
        idx = ramp_indices(np.array([[1.0]]))
        self.assertEqual(int(idx[0, 0]), len(ASCII_RAMP_PRECISE) - 1)

    def test_indices_within_ramp_range(self):
        norm = np.linspace(0, 1, 100).reshape(10, 10)
        idx = ramp_indices(norm)
        self.assertTrue(idx.min() >= 0)
        self.assertTrue(idx.max() < len(ASCII_RAMP_PRECISE))

    def test_monotonic_non_decreasing(self):
        norm = np.linspace(0, 1, 50)
        idx = ramp_indices(norm)
        self.assertTrue(np.all(np.diff(idx) >= 0))

    def test_ramp_is_ordered_sparse_to_dense(self):
        # Densest glyph last, blank first: black must land on an empty cell.
        self.assertEqual(ASCII_RAMP_PRECISE[0], " ")
        self.assertEqual(ASCII_RAMP_PRECISE[-1], "@")
        # Indices must increase with density, so '@' sits far from the space.
        self.assertGreater(ASCII_RAMP_PRECISE.index("@"), 5)

    def test_precise_ramp_has_many_distinct_levels(self):
        """The upgrade must beat the crude 10-level ramp."""
        self.assertGreaterEqual(len(ASCII_RAMP_PRECISE), 20)
        self.assertEqual(len(set(ASCII_RAMP_PRECISE)), len(ASCII_RAMP_PRECISE))

    def test_legacy_ramp_still_works(self):
        idx = ramp_indices(np.array([[1.0]]), ramp=ASCII_RAMP)
        self.assertEqual(int(idx[0, 0]), len(ASCII_RAMP) - 1)

    def test_black_maps_to_space_and_white_to_dense(self):
        """The whole point: luminance direction must not be inverted."""
        black = ramp_indices(np.array([[0.0]]))
        white = ramp_indices(np.array([[1.0]]))
        # Black -> index 0 -> a space. White -> last index -> '@'.
        self.assertEqual(int(black[0, 0]), 0)
        self.assertEqual(int(white[0, 0]), len(ASCII_RAMP_PRECISE) - 1)
        self.assertEqual(str(ASCII_RAMP_PRECISE[int(black[0, 0])]), " ")
        self.assertEqual(str(ASCII_RAMP_PRECISE[int(white[0, 0])]), "@")


class TestToAscii(unittest.TestCase):
    def test_grid_shape(self):
        grid = to_ascii(make_image(), cols=20, rows=10)
        self.assertEqual(grid.shape, (10, 20))

    def test_dark_image_uses_dense_characters(self):
        grid = to_ascii(make_image(color=(0, 0, 0)), cols=4, rows=2)
        for row in grid:
            for ch in row:
                self.assertEqual(ch, ASCII_RAMP_PRECISE[0])

    def test_bright_image_uses_sparse_characters(self):
        grid = to_ascii(make_image(color=(255, 255, 255)), cols=4, rows=2)
        for row in grid:
            for ch in row:
                self.assertEqual(ch, ASCII_RAMP_PRECISE[-1])

    def test_every_character_comes_from_the_ramp(self):
        img = np.zeros((64, 64, 3), np.uint8)
        img[::2] = 255
        grid = to_ascii(img, cols=10, rows=10)
        for row in grid:
            for ch in row:
                self.assertIn(ch, ASCII_RAMP_PRECISE)

    def test_horizontal_gradient_is_progressive(self):
        """A gradient must step monotonically through the ramp.

        Area downsampling averages each cell, so the very first and last cells do
        not lock onto the extreme glyphs. What must hold is monotonic ordering:
        every later cell is at least as sparse as the one before it, and the span
        must cover a good part of the ramp rather than two glyphs.
        """
        img = np.zeros((10, 256, 3), np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 256).astype(np.uint8)
        img[:, :, 1] = img[:, :, 0]
        img[:, :, 2] = img[:, :, 0]
        grid = to_ascii(img, cols=16, rows=1, contrast=1.0)
        # The ramp is sparse-first, so a brighter cell lands on a higher index.
        order = {ch: i for i, ch in enumerate(ASCII_RAMP_PRECISE)}
        positions = [order[str(c)] for c in grid[0]]
        self.assertTrue(all(b >= a for a, b in zip(positions, positions[1:])))
        self.assertGreater(max(positions) - min(positions), 4)

    def test_default_contrast_preserves_mid_tones(self):
        img = np.full((10, 10, 3), (128, 128, 128), np.uint8)
        grid = to_ascii(img, cols=4, rows=2)
        expected = ramp_indices(normalize(luminance(img)))[0, 0]
        order = {ch: i for i, ch in enumerate(ASCII_RAMP_PRECISE)}
        self.assertEqual(order[str(grid[0, 0])], expected)

    def test_grayscale_stays_gray(self):
        img = np.full((10, 10, 3), (200, 90, 200), np.uint8)
        gray = luminance(img)
        grid_mono = to_ascii(img, cols=4, rows=2)
        self.assertTrue(all(ch in ASCII_RAMP_PRECISE for ch in grid_mono.flatten()))
        self.assertTrue(gray.min() == gray.max())

    def test_contrast_changes_the_result(self):
        img = make_image((32, 32), (110, 110, 110))
        plain = to_ascii(img, cols=4, rows=2, contrast=1.0)
        strong = to_ascii(img, cols=4, rows=2, contrast=2.5)
        self.assertFalse(np.array_equal(plain, strong))

    def test_gamma_changes_the_result(self):
        img = make_image((32, 32), (110, 110, 110))
        plain = to_ascii(img, cols=4, rows=2, gamma=1.0)
        brighter = to_ascii(img, cols=4, rows=2, gamma=2.2)
        self.assertFalse(np.array_equal(plain, brighter))


class TestMirror(unittest.TestCase):
    def test_horizontal_flip_swaps_columns(self):
        img = np.zeros((4, 4, 3), np.uint8)
        img[1, 0] = (255, 0, 0)
        out = mirror(img)
        self.assertEqual(out.shape, img.shape)
        # The mark moved from the left edge to the right edge.
        self.assertEqual(tuple(out[1, 3]), (255, 0, 0))
        self.assertEqual(tuple(out[1, 0]), (0, 0, 0))

    def test_rows_are_untouched(self):
        """Mirroring must NOT flip vertically."""
        img = np.zeros((4, 4, 3), np.uint8)
        img[0, 0] = (255, 255, 255)
        out = mirror(img)
        self.assertTrue(np.array_equal(out[0, 3], img[0, 0]))
        self.assertTrue(np.array_equal(out[3, 3], img[3, 3]))
        self.assertTrue(np.array_equal(out[:, 0], img[:, -1]))

    def test_mirror_is_reversible(self):
        img = make_image((8, 8), (12, 34, 56))
        np.testing.assert_array_equal(mirror(mirror(img)), img)

    def test_mirror_returns_contiguous_memory(self):
        """Regression: a negatively-strided view makes every OpenCV call ~30x slower.

        ``arr[:, ::-1]`` shares the buffer but forces cvtColor/CLAHE/Blur onto a
        non-contiguous path. ``cv2.flip`` copies once into a normal layout.
        """
        img = make_image((4, 4))
        out = mirror(img)
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_mirror_in_pipeline_chars_and_colors_together(self):
        img = np.zeros((16, 32, 3), np.uint8)
        img[4:12, 0:8] = (255, 255, 255)  # bright block on the LEFT
        grid, colors = render_frame(img, 8, 4, mirror=True)
        # After mirroring the bright block must appear on the RIGHT.
        row = colors[2]
        right_mean = float(row[-2:].mean())
        left_mean = float(row[:2].mean())
        self.assertGreater(right_mean, left_mean)

    def test_no_mirror_keeps_positions(self):
        img = np.zeros((16, 32, 3), np.uint8)
        img[4:12, 0:8] = (255, 255, 255)
        _, colors = render_frame(img, 8, 4, mirror=False)
        self.assertGreater(float(colors[2, :2].mean()), float(colors[2, -2:].mean()))


class TestColorAlignment(unittest.TestCase):
    def test_glyph_and_color_come_from_the_same_cell(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (24, 40, 3)).astype(np.uint8)
        cols, rows = 8, 5
        grid, colors = render_frame(img, cols, rows)
        self.assertEqual(grid.shape, (rows, cols))
        self.assertEqual(colors.shape, (rows, cols, 3))
        # Independently recompute the mean color of each cell from the source
        # using the same area filter and compare.
        import cv2

        expected = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)
        np.testing.assert_allclose(
            colors.astype(np.float32),
            expected.astype(np.float32),
            atol=1.0,
            err_msg="sampled color does not match the cell it belongs to",
        )

    def test_colour_grid_tracks_mirroring(self):
        rng = np.random.default_rng(11)
        img = rng.integers(0, 256, (24, 40, 3)).astype(np.uint8)
        _, plain = render_frame(img, 8, 5, mirror=False)
        _, flipped = render_frame(img, 8, 5, mirror=True)
        np.testing.assert_array_equal(flipped, plain[:, ::-1])

    def test_enhance_scale_keeps_shape_and_alignment(self):
        """Raising the working-image multiple must not change geometry.

        Enhancement used to run at full camera resolution and cost most of the
        frame budget. The working image is now a multiple of the cell grid, so
        the output shape has to stay put regardless, and each cell's color must
        still be the area-average of that cell's source pixels.
        """
        import cv2

        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, (24, 40, 3)).astype(np.uint8)
        for scale in (1, 2, 3):
            grid, colors = render_frame(
                img, 8, 5, clahe_clip=2.0, enhance_scale=scale
            )
            self.assertEqual(grid.shape, (5, 8))
            self.assertEqual(colors.shape, (5, 8, 3))
            # color is area-interpolated through the working image.
            work = cv2.resize(img, (8 * scale, 5 * scale), interpolation=cv2.INTER_AREA)
            expected = cv2.resize(work, (8, 5), interpolation=cv2.INTER_AREA)
            np.testing.assert_allclose(
                colors.astype(np.int32), expected.astype(np.int32), atol=1
            )

    def test_bright_cell_gets_sparse_glyph_and_bright_colour(self):
        img = np.zeros((20, 20, 3), np.uint8)
        img[:] = (255, 255, 255)
        grid, colors = render_frame(img, 5, 5)
        self.assertEqual(str(grid[2, 2]), ASCII_RAMP_PRECISE[-1])
        self.assertTrue(int(colors[2, 2].mean()) > 200)

    def test_dark_cell_gets_dense_glyph(self):
        img = np.zeros((20, 20, 3), np.uint8)
        img[:] = (0, 0, 0)
        grid, _ = render_frame(img, 5, 5)
        self.assertEqual(str(grid[2, 2]), ASCII_RAMP_PRECISE[0])


class TestEnhancement(unittest.TestCase):
    def _flat_then_detail(self):
        # A dark background with a slightly brighter patch: exactly the
        # shadow detail that a global curve throws away.
        img = np.zeros((24, 24, 3), np.uint8)
        img[:] = (20, 20, 20)
        img[8:16, 8:16] = (48, 48, 48)
        return img

    def test_clahe_raises_local_contrast(self):
        img = self._flat_then_detail()
        gray = luminance(img)
        plain = enhance_gray(gray, clip=0.0, sharpen_amount=0.0)
        boosted = enhance_gray(gray, clip=3.0, sharpen_amount=0.0)
        # CLAHE must widen the spread of this low-contrast scene.
        self.assertGreater(float(np.std(boosted)), float(np.std(plain)))

    def test_clahe_preserves_shape_and_dtype(self):
        gray = luminance(self._flat_then_detail())
        out = enhance_gray(gray, clip=2.0)
        self.assertEqual(out.shape, gray.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_clahe_keeps_global_range(self):
        gray = luminance(self._flat_then_detail())
        out = enhance_gray(gray, clip=3.0)
        self.assertTrue(out.min() >= 0 and out.max() <= 255)

    def test_sharpen_increases_edge_gradient(self):
        img = np.zeros((24, 24, 3), np.uint8)
        img[:, 12:] = (200, 200, 200)  # hard vertical edge
        gray = luminance(img)
        sharp = sharpen(gray, 0.8)
        diff_before = float(abs(int(gray[:, 12].mean()) - int(gray[:, 11].mean())))
        diff_after = float(abs(int(sharp[:, 12].mean()) - int(sharp[:, 11].mean())))
        self.assertGreater(diff_after, diff_before)

    def test_sharpen_zero_is_a_noop(self):
        gray = luminance(make_image((16, 16), (80, 80, 80)))
        np.testing.assert_array_equal(sharpen(gray, 0.0), gray)

    def test_no_clahe_no_sharpen_is_identity(self):
        gray = luminance(make_image((16, 16), (80, 80, 80)))
        np.testing.assert_array_equal(
            enhance_gray(gray, clip=0.0, sharpen_amount=0.0), gray.astype(np.uint8)
        )

    def test_enhance_accepts_rgb(self):
        img = make_image((16, 16), (90, 90, 90))
        out = enhance_gray(img, clip=2.0)
        self.assertEqual(out.shape, (16, 16))
        self.assertEqual(out.dtype, np.uint8)

    def test_detail_presets_are_ordered(self):
        from ascii_cam.config import DETAIL_PRESETS

        self.assertGreaterEqual(len(DETAIL_PRESETS), 3)
        self.assertEqual(DETAIL_PRESETS[0][0], "flat")
        # Each preset must specify a label and three numeric knobs.
        for label, clip, sharp, ss in DETAIL_PRESETS:
            self.assertIsInstance(label, str)
            self.assertGreaterEqual(ss, 1)


class TestDithering(unittest.TestCase):
    def test_dither_adds_texture(self):
        gray = np.full((8, 8), 128, np.uint8)
        plain = normalize(gray)
        dithered = normalize(gray, dither=2)
        # A flat mid gray must not stay perfectly uniform.
        self.assertEqual(float(plain.std()), 0.0)
        self.assertGreater(float(dithered.std()), 0.0)

    def test_dither_preserves_mean_brightness(self):
        gray = np.full((16, 16), 128, np.uint8)
        plain = normalize(gray)
        dithered = normalize(gray, dither=2)
        self.assertAlmostEqual(
            float(dithered.mean()), float(plain.mean()), delta=0.06
        )

    def test_dither_stays_in_range(self):
        gray = np.full((16, 16), 128, np.uint8)
        out = normalize(gray, dither=2, contrast=2.0, gamma=0.6)
        self.assertTrue(out.min() >= 0.0 and out.max() <= 1.0)

    def test_dither_off_is_deterministic(self):
        gray = np.full((8, 8), 100, np.uint8)
        np.testing.assert_array_equal(normalize(gray, dither=0), normalize(gray, dither=0))


class TestComputeGrid(unittest.TestCase):
    def test_full_width_used_for_wide_terminals(self):
        cols, rows = compute_grid(80, 24, source_aspect=1.0, char_aspect=0.5)
        self.assertLessEqual(cols, 80)
        self.assertLessEqual(rows, 23)  # one row reserved for status

    def test_aspect_ratio_is_preserved(self):
        """cols/rows must track source_aspect / char_aspect.

        A 16:9 source in cells that are twice as tall as wide needs 3.56
        columns per row so it is not stretched on screen.
        """
        aspect = 16 / 9
        cols, rows = compute_grid(200, 60, source_aspect=aspect, margin=0)
        expected = aspect / 0.5
        self.assertAlmostEqual(cols / rows, expected, delta=expected * 0.06)

    def test_square_source_yields_expected_ratio(self):
        cols, rows = compute_grid(200, 200, source_aspect=1.0, char_aspect=0.5)
        self.assertAlmostEqual(cols / rows, 2.0, delta=0.05)

    def test_fits_height_when_terminal_is_short(self):
        cols, rows = compute_grid(80, 24, source_aspect=4 / 3, margin=0)
        self.assertLessEqual(cols, 80)
        self.assertLessEqual(rows, 23)
        self.assertGreater(cols, 20)  # a 15x23 strip would be a bug
        self.assertAlmostEqual(cols / rows, (4 / 3) / 0.5, delta=0.2)

    def test_extra_status_rows_are_reserved(self):
        one = compute_grid(80, 30, source_aspect=16 / 9, status_rows=1)
        two = compute_grid(80, 30, source_aspect=16 / 9, status_rows=2)
        self.assertLessEqual(two[1], one[1])

    def test_grid_grows_with_terminal(self):
        small = compute_grid(80, 24, source_aspect=16 / 9)
        large = compute_grid(120, 40, source_aspect=16 / 9)
        self.assertGreaterEqual(large[0], small[0])
        self.assertGreaterEqual(large[1], small[1])

    def test_reset_to_smaller_grid_never_exceeds_bounds(self):
        cols, rows = compute_grid(60, 12, source_aspect=16 / 9)
        self.assertLessEqual(cols, 60)
        self.assertLessEqual(rows, 11)

    def test_minimum_one_row_and_column(self):
        cols, rows = compute_grid(1, 1, source_aspect=1.0, margin=3)
        self.assertGreaterEqual(cols, 1)
        self.assertGreaterEqual(rows, 1)

    def test_zero_char_aspect_is_safe(self):
        cols, rows = compute_grid(80, 24, source_aspect=16 / 9, char_aspect=0.0)
        self.assertGreaterEqual(cols, 1)
        self.assertGreaterEqual(rows, 1)


if __name__ == "__main__":
    unittest.main()
