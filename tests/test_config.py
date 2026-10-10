"""Unit tests for configuration defaults, presets, and settings validation."""

from __future__ import annotations

import unittest

from ascii_cam.config import (
    ASCII_RAMP,
    ASCII_RAMP_PRECISE,
    DEFAULT_CONFIG,
    DETAIL_PRESETS,
    GAMMA_PRESETS,
    MODE_MONOCHROME,
    MODE_PHOSPHOR,
    MODE_TRUE_COLOR,
    PHOSPHOR_BG,
    PHOSPHOR_FG,
    STATUS_TEMPLATE,
)


class TestAsciiRamp(unittest.TestCase):
    def test_ramp_is_non_empty(self):
        self.assertTrue(ASCII_RAMP)

    def test_ramp_is_ordered_sparse_to_dense(self):
        """Ramp index 0 must be the sparsest glyph and the last the densest.

        A terminal draws ink on a dark background, so a space (no ink) is the
        only way to render black and the densest glyph is the only way to render
        white. Ordering the other way round inverts every dark and bright area
        in the scene.
        """
        self.assertEqual(ASCII_RAMP[0], " ")
        self.assertEqual(ASCII_RAMP[-1], "@")
        self.assertEqual(ASCII_RAMP_PRECISE[0], " ")
        self.assertEqual(ASCII_RAMP_PRECISE[-1], "@")

    def test_ramp_has_enough_levels_for_detail(self):
        self.assertGreaterEqual(len(ASCII_RAMP), 8)

    def test_ramp_characters_are_unique(self):
        self.assertEqual(len(set(ASCII_RAMP)), len(ASCII_RAMP))


class TestGammaPresets(unittest.TestCase):
    def test_presets_are_sorted_ascending(self):
        self.assertEqual(GAMMA_PRESETS, sorted(GAMMA_PRESETS))

    def test_default_gamma_is_in_presets(self):
        self.assertIn(DEFAULT_CONFIG["gamma"], GAMMA_PRESETS)

    def test_presets_include_neutral(self):
        self.assertIn(1.0, GAMMA_PRESETS)

    def test_default_index_points_at_neutral_gamma(self):
        from ascii_cam.main import State

        state = State()
        state.cycle_gamma()
        # Cycling once must move to a different, valid preset value.
        self.assertIn(state.gamma, GAMMA_PRESETS)
        self.assertNotEqual(state.gamma, 1.0)

    def test_cycle_wraps_around(self):
        from ascii_cam.main import State

        state = State()
        self.assertEqual(state._gamma_index, 2)  # 1.0
        for _ in range(len(GAMMA_PRESETS)):
            state.cycle_gamma()
        self.assertEqual(state._gamma_index, 2)
        self.assertEqual(state.gamma, 1.0)


class TestSidecarConstants(unittest.TestCase):
    def test_true_color_default(self):
        from ascii_cam.main import State

        state = State({"mode": MODE_TRUE_COLOR})
        self.assertEqual(state.mode, MODE_TRUE_COLOR)

    def test_true_color_marker(self):
        from ascii_cam.main import State

        state = State({"gamma": 0.6, "mode": "unexpected-mode"})
        self.assertEqual(state.mode, DEFAULT_CONFIG["mode"])
        self.assertEqual(state.gamma, 0.6)


class TestModes(unittest.TestCase):
    def test_distinct_mode_names(self):
        self.assertEqual(
            len({MODE_MONOCHROME, MODE_PHOSPHOR, MODE_TRUE_COLOR}), 3
        )

    def test_default_mode_is_monochrome(self):
        self.assertEqual(DEFAULT_CONFIG["mode"], MODE_MONOCHROME)


class TestPhosphorPalette(unittest.TestCase):
    def test_fg_is_green_dominant(self):
        r, g, b = PHOSPHOR_FG
        self.assertGreater(g, r)
        self.assertGreater(g, b)

    def test_bg_is_dark(self):
        r, g, b = PHOSPHOR_BG
        self.assertLessEqual(max(r, g, b), 32)

    def test_channels_in_range(self):
        for color in (PHOSPHOR_FG, PHOSPHOR_BG):
            for channel in color:
                self.assertTrue(0 <= channel <= 255)


class TestStatusTemplate(unittest.TestCase):
    def _fmt(self, **over):
        values = dict(
            fps=24.0, res="640x480", mode=MODE_MONOCHROME, detail="balanced",
            contrast=1.0, brightness=0.0, gamma=1.0, mirror="on",
            black_cutoff=0.04,
        )
        values.update(over)
        return STATUS_TEMPLATE.format(**values)

    def test_template_renders_without_error(self):
        text = self._fmt()
        self.assertIn("FPS", text)
        self.assertIn("640x480", text)
        self.assertIn("monochrome", text)

    def test_template_includes_controls(self):
        text = self._fmt()
        for hint in ("Q/Esc", "1 mono", "3 color", "D detail", "M mirror",
                     "B black", "contrast", "G gamma", "F fps", "Space", "freeze"):
            self.assertIn(hint, text)

    def test_template_mentions_every_control_key(self):
        text = self._fmt()
        for key in ("1", "2", "3", "D", "M", "B", "G", "F"):
            self.assertIn(key, text)

    def test_template_shows_mirror_state(self):
        self.assertIn("mirror: on", self._fmt(mirror="on"))
        self.assertIn("mirror: off", self._fmt(mirror="off"))

    def test_template_shows_black_cutoff(self):
        text = self._fmt(black_cutoff=0.15)
        self.assertIn("black: 0.15", text)

    def test_status_is_compact(self):
        """The status line must not crowd out the image on a narrow terminal."""
        self.assertLessEqual(len(self._fmt()), 260)


class TestDefaultConfig(unittest.TestCase):
    def test_required_keys_present(self):
        for key in ("mode", "contrast", "brightness", "gamma", "show_fps", "target_fps"):
            self.assertIn(key, DEFAULT_CONFIG)

    def test_fps_target_is_reasonable(self):
        self.assertGreaterEqual(DEFAULT_CONFIG["target_fps"], 20)
        self.assertLessEqual(DEFAULT_CONFIG["target_fps"], 30)

    def test_char_aspect_is_less_than_one(self):
        # A terminal cell is taller than wide, so width/height < 1.
        self.assertLess(DEFAULT_CONFIG["char_aspect"], 1.0)

    def test_default_contrast_is_neutral(self):
        self.assertEqual(DEFAULT_CONFIG["contrast"], 1.0)

    def test_mirror_is_on_by_default(self):
        """Requirement: mirrored by default, like a selfie view."""
        self.assertTrue(DEFAULT_CONFIG["mirror"])

    def test_default_ramp_is_the_precise_one(self):
        self.assertEqual(DEFAULT_CONFIG["ramp"], ASCII_RAMP_PRECISE)

    def test_detail_default_is_valid(self):
        index = DEFAULT_CONFIG["detail_index"]
        self.assertTrue(0 <= index < len(DETAIL_PRESETS))


if __name__ == "__main__":
    unittest.main()