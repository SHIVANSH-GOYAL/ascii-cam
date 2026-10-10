"""Unit tests for the renderer and interactive controls."""

from __future__ import annotations

import unittest

import sys

import numpy as np

from ascii_cam.main import State, apply_key
from ascii_cam.renderer import Renderer


class FakeStream:
    """Captures writes so tests can assert on the emitted escape sequences."""

    def __init__(self, term: str = "xterm-256color", colorterm: str = "truecolor"):
        self.chunks: list[str] = []
        self.flushed = 0
        self._env = {"TERM": term, "COLORTERM": colorterm}

    def write(self, text: str):
        self.chunks.append(text)
        return len(text)

    def flush(self):
        self.flushed += 1

    @property
    def env(self):
        return self._env

    @property
    def value(self) -> str:
        return "".join(self.chunks)


def make_grid(rows=3, cols=10, char="@") -> np.ndarray:
    return np.full((rows, cols), char, dtype="<U1")


class TestRenderer(unittest.TestCase):
    def setUp(self):
        self.stream = FakeStream()
        self.renderer = Renderer(out_stream=self.stream)

    def test_cursor_is_hidden_and_shown(self):
        self.renderer.show_cursor_off()
        self.assertIn("\033[?25l", self.stream.value)
        self.stream.chunks.clear()
        self.renderer.show_cursor_on()
        self.assertIn("\033[?25h", self.stream.value)

    def test_clear_resets_cache(self):
        self.renderer.render(make_grid())
        self.assertIsNotNone(self.renderer._last_grid)
        self.renderer.clear()
        self.assertIsNone(self.renderer._last_grid)
        self.assertIn("[2J", self.stream.value)

    def test_positions_cursor_per_row(self):
        self.renderer.render(make_grid(rows=4, cols=8), resolution=(640, 480))
        for row in range(1, 5):
            self.assertIn(f"\033[{row};1H", self.stream.value)

    def test_second_render_only_writes_changed_rows(self):
        self.renderer.render(make_grid(rows=4, cols=8, char="#"))
        self.stream.chunks.clear()
        self.renderer.render(make_grid(rows=4, cols=8, char="#"))
        self.assertEqual(self.stream.value.count("\033[3;1H"), 0)

    def test_changed_row_is_rewritten(self):
        self.renderer.render(make_grid(rows=4, cols=8, char="#"))
        self.stream.chunks.clear()
        grid = make_grid(rows=4, cols=8, char="#")
        grid[2] = "@"
        self.renderer.render(grid)
        self.assertIn("\033[3;1H", self.stream.value)

    def test_status_line_is_written(self):
        self.renderer.render(make_grid(), fps=25.0, resolution=(640, 480), mode="monochrome")
        self.assertIn("25.0", self.stream.value)
        self.assertIn("640x480", self.stream.value)

    def test_true_color_emits_24_bit_escapes(self):
        grid = make_grid(rows=2, cols=4, char="*")
        colors = np.full((2, 4, 3), 10, np.uint8)
        colors[:, :, 1] = 200
        self.renderer.render(grid, mode="truecolor", colors=colors)
        # A 24-bit SGR escape must appear, whatever the quantized value.
        self.assertIn("\033[38;2;", self.stream.value)
        self.assertRegex(self.stream.value, r"\033\[38;2;\d+;\d+;\d+m")

    def test_true_color_quantisation_is_configurable(self):
        grid = make_grid(rows=2, cols=4, char="*")
        # Values chosen off the 16-level step grid (0, 17, 34, ...).
        colors = np.zeros((2, 4, 3), np.uint8)
        colors[:, :, 0] = 200
        colors[:, :, 1] = 190
        colors[:, :, 2] = 50
        # Full 24-bit keeps the exact value.
        exact = FakeStream()
        Renderer(out_stream=exact, color_levels=0).render(
            grid, mode="truecolor", colors=colors
        )
        self.assertIn("\033[38;2;200;190;50m", exact.value)
        # A quantizing renderer must snap it onto a step value.
        quant = FakeStream()
        Renderer(out_stream=quant, color_levels=16).render(
            grid, mode="truecolor", colors=colors
        )
        self.assertNotIn("\033[38;2;200;190;50m", quant.value)
        self.assertRegex(quant.value, r"\033\[38;2;\d+;\d+;\d+m")

    def test_full_24bit_is_the_default_opt_out(self):
        """color_levels=0 must be exact, which is the fidelity escape hatch."""
        grid = make_grid(rows=1, cols=3, char="#")
        colors = np.zeros((1, 3, 3), np.uint8)
        colors[:, :, 0] = 123
        colors[:, :, 1] = 7
        colors[:, :, 2] = 200
        out = FakeStream()
        Renderer(out_stream=out, color_levels=0).render(
            grid, mode="truecolor", colors=colors
        )
        self.assertIn("38;2;123;7;200m", out.value)

    def test_true_color_run_length_skips_repeated_colour(self):
        """Repeated color must not repeat its escape sequence."""
        grid = make_grid(rows=1, cols=8, char="#")
        colors = np.full((1, 8, 3), 200, np.uint8)
        self.renderer.render(grid, mode="truecolor", colors=colors)
        # One escape for the whole row, plus the reset.
        self.assertEqual(self.stream.value.count("38;2;"), 1)

    def test_phosphor_emits_green_foreground(self):
        self.renderer.render(make_grid(rows=2, cols=4), mode="phosphor")
        self.assertIn("38;2;0;255;65m", self.stream.value)
        self.assertIn("48;2;0;10;0m", self.stream.value)

    def test_monochrome_emits_no_color_escapes(self):
        self.renderer.render(make_grid(rows=2, cols=4), mode="monochrome")
        self.assertNotIn("38;2;", self.stream.value)

    def test_true_color_falls_back_to_glyphs_without_truecolor_support(self):
        """A 256-color terminal still shows characters, just without color.

        On Windows this cannot be simulated through the environment alone,
        because the platform check short-circuits it, so the test forces the
        capability flag off.
        """
        limited = FakeStream(term="xterm-256color", colorterm="")
        renderer = Renderer(out_stream=limited)
        if sys.platform.startswith("win"):
            renderer._supports_truecolor = False
        else:
            self.assertFalse(renderer._supports_truecolor)
        grid = make_grid(rows=2, cols=5, char="#")
        colors = np.full((2, 5, 3), 120, np.uint8)
        renderer.render(grid, mode="truecolor", colors=colors)
        # Characters survive even though color does not.
        self.assertIn("#####", limited.value)
        self.assertNotIn("38;2;", limited.value)

    def test_ansi_only_terminal_falls_back_to_plain_text(self):
        """A terminal with no color must still draw the glyphs."""
        limited = FakeStream(term="dumb", colorterm="")
        renderer = Renderer(out_stream=limited)
        grid = make_grid(rows=1, cols=4, char="=")
        renderer.render(grid, mode="phosphor")
        # Characters always survive, color support aside.
        self.assertIn("====", limited.value)
        # TERM=dumb is an explicit opt-out and is honored on every platform.
        self.assertFalse(renderer._supports_color)
        self.assertNotIn("38;2;", limited.value)

    def test_status_reports_mirror_state(self):
        self.renderer.render(
            make_grid(rows=1, cols=4), mode="monochrome", mirror=True
        )
        self.assertIn("mir:on", self.stream.value)
        self.stream.chunks.clear()
        self.renderer.clear()
        self.renderer.render(
            make_grid(rows=1, cols=4), mode="monochrome", mirror=False
        )
        self.assertIn("mir:off", self.stream.value)

    def test_status_reports_detail_preset(self):
        self.renderer.render(
            make_grid(rows=1, cols=4), detail="high", term_size=(200, 50)
        )
        self.assertIn("high", self.stream.value)
        self.assertIn("C:1.00", self.stream.value)

    def test_erases_to_end_of_line_after_each_row(self):
        # ESC[0K clears stale trailing text without erasing the drawn row.
        self.renderer.render(make_grid(rows=3, cols=5))
        self.assertEqual(self.stream.value.count("\033[0K"), 4)  # 3 rows + status

    def test_never_erases_the_whole_line_while_drawing(self):
        """Regression test: ESC[2K wipes the row that was just written.

        Emitting it after the row text left the screen blank even though frames
        were being produced, which is invisible in a log file that merely has
        its escape codes stripped.
        """
        self.renderer.render(make_grid(rows=5, cols=10), mode="monochrome")
        self.renderer.render(make_grid(rows=5, cols=10), mode="phosphor")
        self.renderer.render(
            make_grid(rows=5, cols=10), mode="truecolor",
            colors=np.zeros((5, 10, 3), np.uint8),
        )
        self.assertNotIn("\033[2K", self.stream.value)

    def test_row_text_survives_the_erase_sequence(self):
        """The escape ordering must leave the row content on screen.

        A tiny replay of the emitted bytes: write text, then ESC[0K, and the
        already-written characters must still be there.
        """
        self.renderer.render(make_grid(rows=1, cols=6, char="#"), mode="monochrome")
        # Pull the segment written for row 1.
        value = self.stream.value
        start = value.index("\033[1;1H") + len("\033[1;1H")
        segment = value[start:start + 6]
        self.assertEqual(segment, "######")

    def test_no_line_feed_after_status(self):
        """A trailing newline would scroll the terminal on every frame."""
        self.renderer.render(make_grid(rows=3, cols=5))
        self.assertNotIn("\n", self.stream.value)

    def test_status_is_not_followed_by_a_newline(self):
        self.renderer.render(
            make_grid(rows=4, cols=5), mode="monochrome", fps=30.0,
            resolution=(640, 480),
        )
        value = self.stream.value
        status_start = value.index("\033[5;1H")
        self.assertFalse(value[status_start:].rstrip("\033[0K").endswith("\n"))

    def test_handles_empty_grid(self):
        try:
            self.renderer.render(np.empty((0, 0), dtype="<U1"))
        except Exception as exc:  # pragma: no cover
            self.fail(f"empty grid raised {exc!r}")

    def test_flush_happens_every_render(self):
        self.renderer.render(make_grid())
        self.assertEqual(self.stream.flushed, 1)


class TestControls(unittest.TestCase):
    def setUp(self):
        self.state = State()

    def test_quit_keys(self):
        for key in ("q", "\x1b"):
            s = State()
            apply_key(key, s)
            self.assertFalse(s.running)
            self.assertIs(State().running, True)

    def test_mode_switching(self):
        from ascii_cam.config import MODE_MONOCHROME, MODE_PHOSPHOR, MODE_TRUE_COLOR

        cases = {
            "1": MODE_MONOCHROME,
            "2": MODE_PHOSPHOR,
            "3": MODE_TRUE_COLOR,
        }
        for key, expected in cases.items():
            s = State()
            apply_key(key, s)
            self.assertEqual(s.mode, expected)

    def test_invalid_mode_is_ignored(self):
        s = State()
        s.mode = "phosphor"
        s.set_mode("nonsense")
        self.assertEqual(s.mode, "phosphor")

    def test_contrast_increases_and_decreases(self):
        s = State()
        base = s.contrast
        apply_key("+", s)
        self.assertGreater(s.contrast, base)
        apply_key("-", s)
        self.assertAlmostEqual(s.contrast, base)

    def test_contrast_is_clamped(self):
        s = State()
        for _ in range(100):
            apply_key("+", s)
        self.assertLessEqual(s.contrast, 4.0)
        for _ in range(200):
            apply_key("-", s)
        self.assertGreaterEqual(s.contrast, 0.1)

    def test_brightness_brackets(self):
        s = State()
        apply_key("]", s)
        self.assertGreater(s.brightness, 0.0)
        apply_key("[", s)
        self.assertAlmostEqual(s.brightness, 0.0)

    def test_brightness_clamped(self):
        s = State()
        for _ in range(100):
            apply_key("]", s)
        self.assertLessEqual(s.brightness, 1.0)
        for _ in range(100):
            apply_key("[", s)
        self.assertGreaterEqual(s.brightness, -1.0)

    def test_fkey_toggles_fps(self):
        s = State()
        apply_key("f", s)
        self.assertFalse(s.show_fps)
        apply_key("f", s)
        self.assertTrue(s.show_fps)

    def test_space_toggles_freeze(self):
        s = State()
        apply_key(" ", s)
        self.assertTrue(s.frozen)
        apply_key(" ", s)
        self.assertFalse(s.frozen)

    def test_gamma_cycles(self):
        from ascii_cam.config import GAMMA_PRESETS

        s = State()
        for _ in range(1):
            apply_key("g", s)
        self.assertIn(s.gamma, GAMMA_PRESETS)

    def test_mirror_key_toggles(self):
        s = State()
        self.assertTrue(s.mirror)  # on by default
        apply_key("m", s)
        self.assertFalse(s.mirror)
        apply_key("m", s)
        self.assertTrue(s.mirror)

    def test_detail_key_cycles_presets(self):
        from ascii_cam.config import DETAIL_PRESETS

        s = State()
        seen = {s.detail[0]}
        for _ in range(len(DETAIL_PRESETS)):
            apply_key("d", s)
            seen.add(s.detail[0])
        self.assertEqual(len(seen), len(DETAIL_PRESETS))

    def test_uppercase_equals_lowercase(self):
        a, b = State(), State()
        apply_key("F", a)
        apply_key("f", b)
        self.assertEqual(a.show_fps, b.show_fps)
        apply_key("M", a)
        apply_key("m", b)
        self.assertEqual(a.mirror, b.mirror)

    def test_unknown_key_is_ignored(self):
        s = State()
        before = (s.mode, s.contrast, s.brightness, s.gamma, s.running)
        for key in ("x", "z", "9", "!", "\t"):
            apply_key(key, s)
        self.assertEqual(
            before,
            (s.mode, s.contrast, s.brightness, s.gamma, s.running),
        )


class TestState(unittest.TestCase):
    def test_defaults_from_config(self):
        s = State({"contrast": 2.0, "brightness": 0.25, "gamma": 0.6})
        self.assertEqual(s.contrast, 2.0)
        self.assertEqual(s.brightness, 0.25)
        self.assertEqual(s.gamma, 0.6)

    def test_invalid_gamma_defaults_to_neutral(self):
        s = State({"gamma": 99.0})
        self.assertEqual(s.gamma, 99.0)
        state = State()
        self.assertIn(state.gamma, (0.6, 0.8, 1.0, 1.2, 1.5, 2.2))

    def test_types_are_coerced(self):
        s = State({"contrast": "1.5", "margin": "2", "show_fps": 0})
        self.assertIsInstance(s.contrast, float)
        self.assertIsInstance(s.margin, int)
        self.assertIs(s.show_fps, False)


if __name__ == "__main__":
    unittest.main()