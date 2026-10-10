"""Tests for the shading direction, true-color path, AMOLED black, and ANSI output.

These cover the four behaviours that are easy to get silently wrong: which end of
the ramp a pixel lands on, whether mode 3 actually emits color, whether the
black cutoff is its own step rather than a change to the ramp, and whether a
glyph's RGB sample describes the same cell the glyph occupies.
"""

from __future__ import annotations

import re
import sys
import unittest

import numpy as np

from ascii_cam.converter import (
    apply_black_cutoff,
    compute_grid,
    normalize,
    ramp_indices,
    render_frame,
)
from ascii_cam.config import (
    ASCII_RAMP,
    ASCII_RAMP_PRECISE,
    BLACK_CUTOFF_PRESETS,
    MODE_MONOCHROME,
    MODE_PHOSPHOR,
    MODE_TRUE_COLOR,
)
from ascii_cam.main import State, apply_key
from ascii_cam.camera import capture_size_for_grid
from ascii_cam.renderer import Renderer, supports_ansi, supports_true_color


class FakeStream:
    """Captures writes so tests can assert on the emitted escape sequences."""

    def __init__(self, term: str = "xterm-256color", colorterm: str = "truecolor"):
        self.chunks: list[str] = []
        self.flushed = 0
        self._env = {"TERM": term, "COLORTERM": colorterm}

    def write(self, text: str):
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        self.flushed += 1

    @property
    def env(self):
        return self._env

    @property
    def value(self) -> str:
        return "".join(self.chunks)


def index_in_ramp(char: str, ramp: str) -> int:
    return ramp.index(char)


def glyph_grid(
    gray: np.ndarray,
    ramp: str = ASCII_RAMP_PRECISE,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Reference implementation of the luminance -> glyph mapping."""
    norm = normalize(gray.astype(np.float32), contrast=contrast, gamma=gamma)
    chars = np.asarray(list(ramp))
    return chars[ramp_indices(norm, ramp)].reshape(gray.shape)


class TestIntensityDirection(unittest.TestCase):
    """Luminance must map to glyph density the way a terminal draws ink."""

    def test_black_is_a_space(self):
        grid, _ = render_frame(np.zeros((8, 8, 3), np.uint8), 3, 3)
        self.assertEqual(str(grid[1, 1]), " ")

    def test_white_is_the_densest_glyph(self):
        grid, _ = render_frame(
            np.full((8, 8, 3), 255, np.uint8), 3, 3
        )
        self.assertEqual(str(grid[1, 1]), ASCII_RAMP_PRECISE[-1])

    def test_monotonic_density_across_a_gradient(self):
        """Density must increase with luminance, never decrease."""
        steps = 12
        values = np.linspace(0, 255, steps).astype(np.uint8)
        frame = np.zeros((4, steps, 3), np.uint8)
        for i, v in enumerate(values):
            frame[:, i] = (int(v), int(v), int(v))
        grid, _ = render_frame(frame, steps, 1)
        # The ramp is sparse-first, so the index *is* the density.
        density = [index_in_ramp(ch, ASCII_RAMP_PRECISE) for ch in grid[0]]
        self.assertTrue(
            all(b >= a for a, b in zip(density, density[1:])),
            f"density fell as luminance rose: {density}",
        )
        self.assertEqual(density[0], 0)          # darkest cell -> empty
        self.assertEqual(density[-1], len(ASCII_RAMP_PRECISE) - 1)

    def test_gradient_uses_many_ramp_levels(self):
        """A smooth gradient must not collapse onto two or three glyphs."""
        steps = 20
        values = np.linspace(0, 255, steps).astype(np.uint8)
        frame = np.zeros((4, steps, 3), np.uint8)
        frame[:] = values.reshape(1, steps, 1)
        grid, _ = render_frame(frame, steps, 1)
        self.assertGreaterEqual(len(set(grid[0].tolist())), 8)

    def test_mid_gray_lands_mid_ramp(self):
        grid, _ = render_frame(np.full((8, 8, 3), 128, np.uint8), 3, 3)
        mid = len(ASCII_RAMP_PRECISE) // 2
        index = index_in_ramp(str(grid[1, 1]), ASCII_RAMP_PRECISE)
        self.assertAlmostEqual(index, mid, delta=4)

    def test_matches_reference_mapping(self):
        """The vectorized pipeline must equal an independent reference.

        Enhancement and the working-image scale are turned off so the pipeline is
        exactly resize -> grayscale -> normalize -> ramp, which is what the
        reference below computes independently.
        """
        rng = np.random.default_rng(2)
        gray = rng.integers(0, 256, (9, 11)).astype(np.uint8)
        frame = np.stack([gray] * 3, axis=-1)
        grid, _ = render_frame(
            frame, 11, 9, contrast=1.2, gamma=1.3, clahe_clip=0.0,
            sharpen_amount=0.0, enhance_scale=1,
        )
        np.testing.assert_array_equal(
            grid, glyph_grid(gray, contrast=1.2, gamma=1.3)
        )

    def test_inverted_ramp_would_give_the_opposite(self):
        """Asserts the endpoints directly, so a re-inversion fails loudly.

        Ordering the ramp densest-first again would swap which glyph black and
        white resolve to. Comparing the two ends is the cheapest way to catch it.
        """
        dense_first = ASCII_RAMP_PRECISE[::-1]
        self.assertEqual(dense_first[0], "@")
        self.assertEqual(dense_first[-1], " ")
        # The pipeline's own ramp resolves the opposite way.
        self.assertEqual(ASCII_RAMP_PRECISE[0], " ")
        self.assertEqual(ASCII_RAMP_PRECISE[-1], "@")
        black, _ = render_frame(np.zeros((8, 8, 3), np.uint8), 3, 3)
        white, _ = render_frame(np.full((8, 8, 3), 255, np.uint8), 3, 3)
        self.assertNotEqual(str(black[1, 1]), str(white[1, 1]))

    def test_legacy_ramp_agrees_on_endpoints(self):
        for value in (0, 255):
            frame = np.full((8, 8, 3), value, np.uint8)
            precise, _ = render_frame(frame, 3, 3, ramp=ASCII_RAMP_PRECISE)
            legacy, _ = render_frame(frame, 3, 3, ramp=ASCII_RAMP)
            self.assertEqual(
                str(precise[1, 1]), str(legacy[1, 1]), f"mismatch at {value}"
            )


class TestAMOLEDBlack(unittest.TestCase):
    """The black cutoff must be its own explicit step, not a ramp change."""

    def test_zero_cutoff_disables_the_effect(self):
        norm = np.array([[0.01, 0.05, 0.5]], dtype=np.float32)
        out = apply_black_cutoff(norm, 0.0)
        np.testing.assert_array_equal(out, norm)

    def test_values_at_or_below_threshold_become_space(self):
        norm = np.array([[0.0, 0.01, 0.04, 0.041]], dtype=np.float32)
        out = apply_black_cutoff(norm, 0.04)
        # 0.0, 0.01, 0.04 are at or below the threshold -> pinned to 0.
        self.assertEqual(out[0, 0], 0.0)
        self.assertEqual(out[0, 1], 0.0)
        self.assertEqual(out[0, 2], 0.0)
        # 0.041 is above it and must be untouched.
        self.assertAlmostEqual(float(out[0, 3]), 0.041, places=5)

    def test_bright_pixels_are_never_affected(self):
        norm = np.linspace(0.3, 1.0, 11, dtype=np.float32)
        out = apply_black_cutoff(norm, 0.15)
        np.testing.assert_allclose(out, norm)

    def test_cutoff_does_not_mutate_input(self):
        norm = np.array([[0.01, 0.9]], dtype=np.float32)
        clone = norm.copy()
        apply_black_cutoff(norm, 0.5)
        np.testing.assert_array_equal(norm, clone)

    def test_pure_black_renders_as_empty_cells_in_every_mode(self):
        black = np.zeros((8, 8, 3), np.uint8)
        grid, colors = render_frame(black, 4, 4, black_cutoff=0.15)
        for r in range(4):
            for c in range(4):
                self.assertEqual(str(grid[r, c]), " ")
        # color grid is still supplied, but the glyph is blank so no ink is
        # drawn and the cell reads as black on a dark background.
        self.assertEqual(colors.shape, (4, 4, 3))

    def test_near_black_above_threshold_keeps_a_glyph(self):
        """Dark but meaningful detail must survive."""
        faint = np.full((8, 8, 3), 40, np.uint8)
        grid, _ = render_frame(faint, 4, 4, black_cutoff=0.04)
        self.assertNotEqual(str(grid[1, 1]), " ")

    def test_ramp_still_contains_its_full_dark_range(self):
        """The ramp must not be emptied of dark glyphs to fake the cutoff."""
        self.assertGreaterEqual(len(ASCII_RAMP_PRECISE), 20)
        # The dark end of the ramp still has real glyphs in it, which is what
        # lets a screen keep dense darks when the cutoff is set to 0.
        tail = ASCII_RAMP_PRECISE[1:8]
        self.assertTrue(all(ch != " " for ch in tail))
        faint = np.full((8, 8, 3), 60, np.uint8)
        grid_off, _ = render_frame(faint, 4, 4, black_cutoff=0.0)
        grid_on, _ = render_frame(faint, 4, 4, black_cutoff=0.30)
        self.assertNotEqual(str(grid_off[1, 1]), str(grid_on[1, 1]))

    def test_cutoff_survives_tone_controls(self):
        """Cutoff is applied after contrast/brightness/gamma, so it tracks them."""
        dark = np.full((8, 8, 3), 30, np.uint8)
        plain, _ = render_frame(dark, 4, 4, black_cutoff=0.04)
        brighter, _ = render_frame(
            dark, 4, 4, black_cutoff=0.04, brightness=1.0
        )
        # Brightness lifts the pixel above the threshold, so it gains a glyph.
        self.assertNotEqual(str(plain[1, 1]), str(brighter[1, 1]))

    def test_presets_are_ordered_and_include_off(self):
        self.assertIn(0.0, BLACK_CUTOFF_PRESETS)
        self.assertEqual(BLACK_CUTOFF_PRESETS, sorted(BLACK_CUTOFF_PRESETS))


class TestModeSwitching(unittest.TestCase):
    """The 1/2/3 keys must move the active mode and the indicator must agree."""

    def test_select_monochrome(self):
        state = State()
        apply_key("1", state)
        self.assertEqual(state.mode, MODE_MONOCHROME)

    def test_select_phosphor(self):
        state = State()
        apply_key("2", state)
        self.assertEqual(state.mode, MODE_PHOSPHOR)

    def test_select_true_color(self):
        state = State()
        apply_key("3", state)
        self.assertEqual(state.mode, MODE_TRUE_COLOR)

    def test_cycling_all_three(self):
        state = State()
        seen = []
        for key in ("3", "2", "1", "3"):
            apply_key(key, state)
            seen.append(state.mode)
        self.assertEqual(
            seen, [MODE_TRUE_COLOR, MODE_PHOSPHOR, MODE_MONOCHROME, MODE_TRUE_COLOR]
        )

    def test_repeated_presses_stay_stable(self):
        state = State()
        for _ in range(5):
            apply_key("3", state)
        self.assertEqual(state.mode, MODE_TRUE_COLOR)

    def test_b_key_cycles_black_cutoff(self):
        from ascii_cam.config import BLACK_CUTOFF_PRESETS

        state = State()
        seen = {state.black_cutoff}
        for _ in range(len(BLACK_CUTOFF_PRESETS)):
            apply_key("b", state)
            seen.add(state.black_cutoff)
        self.assertEqual(len(seen), len(BLACK_CUTOFF_PRESETS))

    def test_uppercase_b_also_works(self):
        a, b = State(), State()
        apply_key("B", a)
        apply_key("b", b)
        self.assertEqual(a.black_cutoff, b.black_cutoff)


class TestTrueColorOutput(unittest.TestCase):
    """Mode 3 must emit real 24-bit color from the frame, not plain glyphs."""

    def _render(self, rows, cols, colors, color_levels: int = 16, **kw):
        stream = FakeStream()
        renderer = Renderer(out_stream=stream, color_levels=color_levels)
        grid = np.empty((rows, cols), dtype=object)
        grid[:] = "#"
        renderer.render(grid, mode=MODE_TRUE_COLOR, colors=colors, **kw)
        return stream.value

    def test_emits_24_bit_escapes_with_source_rgb(self):
        """Mode 3 must emit an ESC[38;2;R;G;Bm sequence, using real RGB values."""
        colors = np.zeros((2, 5, 3), np.uint8)
        colors[:, :, 0] = 200
        colors[:, :, 1] = 90
        colors[:, :, 2] = 40
        out = self._render(2, 5, colors)
        self.assertRegex(out, r"\033\[38;2;\d+;\d+;\d+m")

    def test_full_24bit_preserves_exact_values(self):
        """With quantization off the sampled RGB must survive unchanged."""
        colors = np.zeros((2, 5, 3), np.uint8)
        colors[:, :, 0] = 123
        colors[:, :, 1] = 7
        colors[:, :, 2] = 200
        out = self._render(2, 5, colors, color_levels=0)
        self.assertIn("\033[38;2;123;7;200m", out)

    def test_quantized_values_are_nearby(self):
        """quantization must stay close enough to keep skin tones recognisable."""
        colors = np.zeros((2, 5, 3), np.uint8)
        colors[:, :, 0] = 200  # skin-ish
        colors[:, :, 1] = 90
        colors[:, :, 2] = 40
        out = self._render(2, 5, colors)
        found = re.search(r"\033\[38;2;(\d+);(\d+);(\d+)m", out)
        self.assertIsNotNone(found)
        r, g, b = (int(found.group(i)) for i in (1, 2, 3))
        # Each channel must land within one 16-level step (~17) of the source.
        self.assertLessEqual(abs(r - 200), 17)
        self.assertLessEqual(abs(g - 90), 17)
        self.assertLessEqual(abs(b - 40), 17)

    def test_color_is_reset_at_end_of_row(self):
        """Each true-color row must close with a reset so color cannot bleed."""
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        colors = np.full((3, 5, 3), 120, np.uint8)
        grid = np.empty((3, 5), dtype=object)
        grid[:] = "#"
        row = renderer._truecolor_row(grid[1], 1, colors)
        self.assertTrue(row.endswith("\033[0m"))

    def test_no_color_bleeds_into_status_bar(self):
        """The status line must be drawn from a clean color state."""
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        colors = np.full((2, 5, 3), 30, np.uint8)
        grid = np.empty((2, 5), dtype=object)
        grid[:] = "#"
        renderer.render(
            grid, mode=MODE_TRUE_COLOR, colors=colors, fps=25.0,
            resolution=(640, 480),
        )
        value = stream.value
        status_start = value.rindex("\033[3;1H")
        # No foreground escapes may reach the status row.
        self.assertNotIn("38;2;", value[status_start:])

    def test_run_length_skips_repeated_color(self):
        """A run of one color must emit exactly one escape for the sane bytes."""
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        colors = np.full((2, 5, 3), 200, np.uint8)
        grid = np.empty((2, 5), dtype=object)
        grid[:] = "#"
        row = renderer._truecolor_row(grid[1], 1, colors)
        self.assertEqual(row.count("38;2;"), 1)
        self.assertEqual(row.count("#"), 5)
        self.assertTrue(row.endswith("\033[0m"))

    def test_windows_true_color_is_assumed(self):
        """TERM is normally unset on Windows; that must not mean mono.

        Regression test for the reported bug where pressing 3 produced
        black-and-white output because the capability probe read TERM/COLORTERM
        only and Windows leaves both unset.
        """
        stream = FakeStream(term="", colorterm="")
        if sys.platform.startswith("win"):
            self.assertTrue(supports_true_color(stream))
        else:
            self.assertTrue(supports_ansi(stream) is False)

    def test_posix_256_color_still_falls_back(self):
        stream = FakeStream(term="xterm-256color", colorterm="")
        if not sys.platform.startswith("win"):
            self.assertFalse(supports_true_color(stream))

    def test_renderer_uses_color_when_supported(self):
        stream = FakeStream(term="xterm-256color", colorterm="truecolor")
        renderer = Renderer(out_stream=stream)
        self.assertTrue(renderer._supports_truecolor)
        out = self._render(1, 3, np.full((1, 3, 3), 90, np.uint8))
        self.assertIn("38;2;", out)


class TestColorCharacterAlignment(unittest.TestCase):
    """A glyph and its RGB sample must describe the same source cell."""

    def test_colors_match_independent_area_average(self):
        import cv2

        rng = np.random.default_rng(9)
        img = rng.integers(0, 256, (30, 40, 3)).astype(np.uint8)
        cols, rows = 8, 6
        grid, colors = render_frame(img, cols, rows)
        expected = cv2.resize(
            img, (cols * 2, rows * 2), interpolation=cv2.INTER_AREA
        )
        expected = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)
        np.testing.assert_allclose(
            colors.astype(np.int32), expected.astype(np.int32), atol=1
        )

    def test_alignment_survives_mirroring(self):
        rng = np.random.default_rng(11)
        img = rng.integers(0, 256, (30, 40, 3)).astype(np.uint8)
        _, plain = render_frame(img, 8, 6, mirror=False)
        _, flipped = render_frame(img, 8, 6, mirror=True)
        np.testing.assert_array_equal(flipped, plain[:, ::-1])

    def test_color_grid_shape_tracks_char_grid(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (30, 40, 3)).astype(np.uint8)
        for cols, rows in ((5, 5), (20, 3), (3, 20)):
            grid, colors = render_frame(img, cols, rows)
            self.assertEqual(grid.shape, (rows, cols))
            self.assertEqual(colors.shape, (rows, cols, 3))

    def test_dark_cell_gets_blank_glyph_and_its_own_color(self):
        black = np.full((8, 8, 3), (5, 5, 5), np.uint8)
        grid, colors = render_frame(black, 3, 3, black_cutoff=0.15)
        self.assertEqual(str(grid[1, 1]), " ")
        # The color grid is still produced; the blank glyph means no ink, so the
        # cell reads as black on a dark background.
        self.assertTrue(int(colors[1, 1].mean()) <= 6)

    def test_colour_sampling_does_not_change_glyph_density_source(self):
        """Glyph density comes from luminance, color from RGB, same cell."""
        img = np.full((8, 8, 3), (200, 200, 200), np.uint8)
        grid, colors = render_frame(img, 3, 3)
        # Luminance 200 -> index 20 in the 27-level ramp, not the densest glyph.
        self.assertEqual(str(grid[1, 1]), ASCII_RAMP_PRECISE[20])
        self.assertTrue(int(colors[1, 1].mean()) > 180)

    def test_saturated_color_keeps_luminance_driven_glyph(self):
        """A saturated but dark color must not be drawn with a dense glyph."""
        img = np.zeros((8, 8, 3), np.uint8)
        img[:] = (180, 10, 10)  # strong red, low luminance
        grid, colors = render_frame(img, 3, 3)
        # Luminance ~57 -> mid-ramp, not the densest glyph.
        self.assertNotEqual(str(grid[1, 1]), ASCII_RAMP_PRECISE[-1])
        # Its color is still the red we asked for.
        self.assertGreater(int(colors[1, 1][0]), int(colors[1, 1][1]))


class TestResizeAndAspect(unittest.TestCase):
    """Resizing the terminal must change the grid without changing proportions."""

    def test_grid_fills_available_area(self):
        cols, rows = compute_grid(120, 40, source_aspect=16 / 9)
        self.assertLessEqual(cols, 120)
        self.assertLessEqual(rows, 39)

    def test_status_row_is_never_overwritten(self):
        for term_rows in (10, 24, 50, 100):
            _, rows = compute_grid(90, term_rows, source_aspect=4 / 3, status_rows=1)
            self.assertLessEqual(rows, term_rows - 1)

    def test_aspect_ratio_is_preserved(self):
        aspect = 16 / 9
        cols, rows = compute_grid(200, 60, source_aspect=aspect)
        self.assertAlmostEqual(cols / rows, aspect / 0.5, delta=0.15)

    def test_shrinking_terminal_shrinks_grid(self):
        big = compute_grid(120, 40, source_aspect=16 / 9)
        small = compute_grid(60, 20, source_aspect=16 / 9)
        self.assertLess(small[0], big[0])
        self.assertLess(small[1], big[1])

    def test_extreme_aspects_do_not_overshoot(self):
        for aspect in (0.2, 1.0, 8.0):
            cols, rows = compute_grid(80, 24, source_aspect=aspect)
            self.assertLessEqual(cols, 80)
            self.assertLessEqual(rows, 23)

    def test_minimum_grid(self):
        cols, rows = compute_grid(1, 2, source_aspect=1.0, margin=5)
        self.assertGreaterEqual(cols, 1)
        self.assertGreaterEqual(rows, 1)

    def test_gradient_renders_at_every_grid_size(self):
        rng = np.random.default_rng(4)
        img = rng.integers(0, 256, (60, 80, 3)).astype(np.uint8)
        for cols, rows in ((10, 10), (30, 12), (5, 30)):
            grid, colors = render_frame(img, cols, rows)
            self.assertEqual(grid.shape, (rows, cols))
            self.assertEqual(colors.shape, (rows, cols, 3))
            for ch in grid.flatten():
                self.assertIn(str(ch), ASCII_RAMP_PRECISE)


class TestColorOrder(unittest.TestCase):
    """Frames arrive BGR from OpenCV; the conversion must happen cheaply."""

    def _pair(self, rows=6, cols=8):
        """A syntactically red patch: BGR (0,0,255) and RGB (255,0,0)."""
        bgr = np.zeros((rows, cols, 3), np.uint8)
        bgr[:, :, 2] = 255
        rgb = bgr.copy()
        rgb = rgb[:, ::-1, ::-1]
        return bgr, rgb

    def test_bgr_order_is_not_treated_as_rgb(self):
        bgr, rgb = self._pair()
        grid_b, colors_b = render_frame(bgr, 8, 6, color_order="bgr")
        grid_r, colors_r = render_frame(rgb, 8, 6, color_order="rgb")
        # Luminance of a saturated red is low, so the glyphs agree...
        np.testing.assert_array_equal(grid_b, grid_r)
        # ...but the emitted color must come out red either way.
        self.assertTrue(
            all(int(colors_b[r, c][0]) > int(colors_b[r, c][1]) + 40 for r in range(6) for c in range(8))
        )

    def test_wrong_order_swaps_channels(self):
        bgr, _ = self._pair()
        _, right = render_frame(bgr, 8, 6, color_order="bgr")
        _, wrong = render_frame(bgr, 8, 6, color_order="rgb")
        # Corrected: the red channel lands at index 0 and blue is empty.
        self.assertEqual(int(right[0, 0][0]), 255)
        self.assertEqual(int(right[0, 0][2]), 0)
        # Read the wrong way round, red lands at index 2 instead.
        self.assertEqual(int(wrong[0, 0][0]), 0)
        self.assertEqual(int(wrong[0, 0][2]), 255)

    def test_luminance_is_order_independent(self):
        """Glyph density must not depend on the channel order."""
        bgr, _ = self._pair()
        rgb = bgr[:, ::-1, ::-1]
        g_b, _ = render_frame(bgr, 8, 6, color_order="bgr")
        g_r, _ = render_frame(rgb, 8, 6, color_order="rgb")
        np.testing.assert_array_equal(g_b, g_r)

    def test_invalid_order_is_rejected(self):
        with self.assertRaises(ValueError):
            render_frame(np.zeros((4, 4, 3), np.uint8), 2, 2, color_order="xyz")


class TestCaptureSizing(unittest.TestCase):
    """Capture resolution must track the grid, not be a fixed constant."""

    def test_small_grid_asks_for_only_what_it_needs(self):
        width, height = capture_size_for_grid(61, 23)
        self.assertLessEqual(width, 640)
        self.assertLessEqual(height, 480)

    def test_large_grid_gets_more_pixels(self):
        small = capture_size_for_grid(61, 23, )
        large = capture_size_for_grid(200, 75)
        self.assertGreaterEqual(large[0], small[0])
        self.assertGreaterEqual(large[1], small[1])

    def test_only_standard_sizes_are_requested(self):
        """Drivers ignore arbitrary sizes and silently fall back to 640x480."""
        from ascii_cam.camera import STANDARD_CAPTURE_SIZES

        for cols, rows in ((5, 5), (61, 23), (400, 150), (800, 300)):
            size = capture_size_for_grid(cols, rows)
            self.assertIn(size, STANDARD_CAPTURE_SIZES, (cols, size))

    def test_zero_grid_is_safe(self):
        width, height = capture_size_for_grid(0, 0)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_resize_then_flip_equals_flip_then_resize(self):
        """Justifies flipping the working image instead of the camera frame."""
        import cv2

        rng = np.random.default_rng(6)
        img = rng.integers(0, 256, (40, 60, 3)).astype(np.uint8)
        a = cv2.flip(
            cv2.resize(img, (8, 6), interpolation=cv2.INTER_AREA), 1
        )
        b = cv2.resize(
            cv2.flip(img, 1), (8, 6), interpolation=cv2.INTER_AREA
        )
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
