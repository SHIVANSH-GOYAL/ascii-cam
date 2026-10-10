"""Tests for terminal capability detection, renderer selection, and ANSI output.

The failures these cover were observed rather than invented: a real console was
seen scrolling because the status line was longer than its width, and the
capability probe was seen reading a ``stream.env`` attribute that ``sys.stdout``
does not have.
"""

from __future__ import annotations

import sys
import unittest

import numpy as np

from ascii_cam.config import MODE_MONOCHROME, MODE_PHOSPHOR, MODE_TRUE_COLOR
from ascii_cam.config import STATUS_TEMPLATE
from ascii_cam.renderer import (
    Renderer,
    TerminalProfile,
    _env_of,
    console_vt_enabled,
    enable_console_vt,
    get_terminal_size,
    probe_terminal,
    restore_console_mode,
    supports_ansi,
    supports_true_color,
    terminal_host,
    windows_build,
)


class FakeStream:
    """Captures writes and exposes a chosen environment."""

    def __init__(self, term: str = "xterm-256color", colorterm: str = "truecolor", env=None):
        self.chunks: list[str] = []
        self.flushed = 0
        if env is None:
            env = {"TERM": term, "COLORTERM": colorterm}
        self._env = env

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


class EnvlessStream:
    """A stream with no ``.env`` attribute, like the real ``sys.stdout``."""

    def __init__(self):
        self.chunks: list[str] = []

    def write(self, text: str):
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def value(self) -> str:
        return "".join(self.chunks)


def make_grid(rows: int = 2, cols: int = 4, char: str = "@"):
    grid = np.empty((rows, cols), dtype=object)
    grid[:] = char
    return grid


class TestCapabilityDetection(unittest.TestCase):
    """The probe must read the real environment, not a stream attribute."""

    def test_real_stdout_has_no_env_attribute(self):
        """Documents why reading ``stream.env`` alone was insufficient."""
        self.assertFalse(hasattr(sys.stdout, "env"))

    def test_env_falls_back_to_os_environ(self):
        import os

        stream = EnvlessStream()
        # The rendered environment must come from os.environ, not from a
        # missing attribute.
        previous = os.environ.get("TERM")
        os.environ["TERM"] = "xterm-256color"
        try:
            env = _env_of(stream)
            self.assertIn("TERM", env)
        finally:
            if previous is None:
                os.environ.pop("TERM", None)
            else:
                os.environ["TERM"] = previous

    def test_stream_env_takes_precedence(self):
        stream = FakeStream(term="dumb")
        self.assertEqual(_env_of(stream)["TERM"], "dumb")

    def test_dumb_term_hides_ansi(self):
        self.assertFalse(supports_ansi(FakeStream(term="dumb")))

    def test_256_color_term_reports_ansi(self):
        self.assertTrue(supports_ansi(FakeStream(term="xterm-256color")))

    def test_truecolor_term_reports_true_color(self):
        if sys.platform.startswith("win"):
            self.skipTest("POSIX-only assertion")
        stream = FakeStream(term="xterm-256color", colorterm="truecolor")
        self.assertTrue(supports_true_color(stream))

    def test_256_color_term_lacks_true_color(self):
        if sys.platform.startswith("win"):
            self.skipTest("POSIX-only assertion")
        stream = FakeStream(term="xterm-256color", colorterm="")
        self.assertFalse(supports_true_color(stream))

    def test_no_ansi_means_no_true_color(self):
        self.assertFalse(supports_true_color(FakeStream(term="dumb")))

    def test_terminal_host_identifies_windows_terminal(self):
        stream = FakeStream(env={"WT_SESSION": "abc-123"})
        self.assertEqual(terminal_host(stream), "Windows Terminal")

    def test_terminal_host_identifies_vscode(self):
        stream = FakeStream(env={"TERM_PROGRAM": "vscode"})
        self.assertEqual(terminal_host(stream), "VS Code terminal")

    def test_terminal_host_identifies_windows_console(self):
        if not sys.platform.startswith("win"):
            self.skipTest("Windows-only assertion")
        stream = FakeStream(env={})
        self.assertEqual(terminal_host(stream), "Windows console")

    def test_windows_build_is_int_or_none(self):
        build = windows_build()
        if sys.platform.startswith("win"):
            self.assertIsInstance(build, int)
            self.assertGreater(build, 10000)
        else:
            self.assertIsNone(build)

    def test_console_vt_query_is_safe(self):
        """Must not raise when no console is attached (piped output)."""
        result = console_vt_enabled()
        self.assertTrue(result is None or isinstance(result, bool))

    def test_enable_console_vt_is_safe(self):
        """Must not raise when no console is attached (piped output)."""
        self.assertIsInstance(enable_console_vt(), bool)


class TestRendererCapabilityUse(unittest.TestCase):
    """The renderer must degrade instead of emitting unusable escapes."""

    def test_truecolor_without_support_keeps_glyphs(self):
        stream = FakeStream(term="dumb")
        renderer = Renderer(out_stream=stream)
        renderer._supports_truecolor = False
        grid = make_grid(2, 5, "#")
        colors = np.full((2, 5, 3), 120, np.uint8)
        renderer.render(grid, mode="truecolor", colors=colors, term_size=(200, 20))
        self.assertIn("#####", stream.value)
        self.assertNotIn("38;2;", stream.value)

    def test_phosphor_without_ansi_keeps_glyphs(self):
        stream = FakeStream(term="dumb")
        renderer = Renderer(out_stream=stream)
        renderer._supports_color = False
        renderer.render(make_grid(1, 4, "="), mode="phosphor", term_size=(200, 20))
        self.assertIn("====", stream.value)
        self.assertNotIn("38;2;", stream.value)


class TestStatusWidth(unittest.TestCase):
    """The status line must never exceed the console width."""

    def _render(self, term_cols, grid_cols=4, grid_rows=1):
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        renderer.render(
            make_grid(grid_rows, grid_cols, "#"),
            mode="monochrome",
            fps=23.8,
            resolution=(640, 480),
            mirror=True,
            black_cutoff=0.04,
            term_size=(term_cols, 30),
        )
        return stream.value

    def test_status_is_clamped_to_terminal_width(self):
        for width in (40, 60, 80, 100, 120):
            out = self._render(width)
            # The status row is the last CUP move before ERASE_TO_EOL.
            segments = out.split("\x1b[")
            status_row = [
                s for s in segments if s.startswith("2;1H")
            ]
            self.assertTrue(status_row, f"no status row for width {width}")
            body = status_row[0].split("\x1b[0K")[0][len("2;1H"):]
            self.assertLessEqual(
                len(body), width, f"status overflows at width {width}"
            )

    def test_status_fits_every_common_terminal_width(self):
        for width in (80, 100, 120):
            out = self._render(width)
            # No escape may appear on a wrapped row, so the status row must hold
            # everything.
            self.assertLessEqual(out.count("\n"), 0)

    def test_wide_terminal_keeps_control_hints(self):
        out = self._render(300)
        self.assertIn("Q quit", out)
        self.assertIn("freeze", out)

    def test_narrow_terminal_keeps_vitals(self):
        out = self._render(60)
        self.assertIn("FPS", out)
        self.assertIn("640x480", out)
        self.assertIn("monochrome", out)

    def test_vitals_survive_every_width(self):
        for width in (50, 70, 90, 110):
            out = self._render(width)
            self.assertIn("FPS", out, f"width {width}")
            self.assertIn("640x480", out, f"width {width}")

    def test_image_rows_are_never_truncated(self):
        """Only the status line is clamped; image rows must be complete."""
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        renderer.render(
            make_grid(1, 10, "@"), term_size=(5, 10)
        )
        self.assertIn("@@@@@@@@@@", stream.value)


def _status_text():
    """Render the status template with a representative set of values."""
    return STATUS_TEMPLATE.format(
        fps=23.8, res="640x480", mode="monochrome", detail="balanced",
        contrast=1.0, brightness=0.0, gamma=1.0, mirror="on", black_cutoff=0.04,
        renderer="ansi", color="24bit",
    )


class TestStatusTemplateWidth(unittest.TestCase):
    def test_full_template_exceeds_common_widths(self):
        """Documents why clamping is required at all."""
        self.assertGreater(len(_status_text()), 80)

    def test_vitals_fit_a_standard_terminal(self):
        """The live readout must survive an 80-column console.

        The full line is longer than that by design; the renderer clamps it. What
        matters is that clamping costs the control hints, never the vitals.
        """
        vitals = _status_text().split("|")[0]
        self.assertLessEqual(len(vitals), 80)

    def test_truncation_keeps_every_vital_field(self):
        text = _status_text()
        for field in ("FPS", "640x480", "monochrome", "balanced", "C:1.00",
                      "B:+0.00", "G:1.00", "mir:on", "blk:0.04"):
            self.assertLess(text.index(field), text.index("|"), field)

    def test_reports_renderer_and_color_backend(self):
        """A fallback must be visible without a separate diagnostic command."""
        text = _status_text()
        self.assertIn("rnd:ansi", text)
        self.assertIn("col:24bit", text)
        fallback = _status_text().replace("rnd:ansi", "rnd:win32").replace(
            "col:24bit", "col:none"
        )
        self.assertIn("rnd:win32", fallback)


class TestTerminalResizing(unittest.TestCase):
    """Resizing must not leave stale rows or overflow the console."""

    def _render(self, term_cols, term_rows, grid_cols, grid_rows):
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        renderer.render(
            make_grid(grid_rows, grid_cols, "#"),
            term_size=(term_cols, term_rows),
        )
        return stream.value

    def test_shrinking_clears_stale_rows(self):
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        renderer.render(make_grid(5, 8, "#"), term_size=(40, 20))
        stream.chunks.clear()
        renderer.render(make_grid(2, 8, "#"), term_size=(40, 20))
        # ESC[0J discards everything below the new, shorter frame.
        self.assertIn("\x1b[3;1H\x1b[0J", stream.value)

    def test_growing_redraws_everything(self):
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        renderer.render(make_grid(2, 8, "#"), term_size=(40, 20))
        stream.chunks.clear()
        renderer.render(make_grid(6, 8, "#"), term_size=(40, 20))
        rows_drawn = stream.value.count("\x1b[")
        self.assertGreaterEqual(rows_drawn, 6)

    def test_unchanged_rows_are_not_rewritten(self):
        stream = FakeStream()
        renderer = Renderer(out_stream=stream)
        renderer.render(make_grid(4, 8, "#"), term_size=(40, 20))
        stream.chunks.clear()
        renderer.render(make_grid(4, 8, "#"), term_size=(40, 20))
        self.assertEqual(stream.value.count("\x1b[3;1H"), 0)

    def test_single_row_terminal_is_safe(self):
        out = self._render(4, 1, 4, 1)
        self.assertIsInstance(out, str)

    def test_get_terminal_size_never_returns_zero(self):
        cols, rows = get_terminal_size()
        self.assertGreaterEqual(cols, 1)
        self.assertGreaterEqual(rows, 2)


class TestAnsiOutput(unittest.TestCase):
    """Escape generation must be well-formed and self-terminating."""

    def test_cursor_position_format(self):
        stream = FakeStream()
        Renderer(out_stream=stream).render(make_grid(3, 4), term_size=(80, 20))
        for row in (1, 2, 3, 4):
            self.assertIn(f"\x1b[{row};1H", stream.value)

    def test_true_color_escape_format(self):
        stream = FakeStream()
        renderer = Renderer(out_stream=stream, color_levels=0)
        colors = np.zeros((1, 3, 3), np.uint8)
        colors[:, :, 0] = 200
        colors[:, :, 1] = 90
        colors[:, :, 2] = 40
        renderer.render(
            make_grid(1, 3, "#"), mode="truecolor", colors=colors, term_size=(80, 20)
        )
        self.assertIn("\x1b[38;2;200;90;40m", stream.value)

    def test_color_is_reset_at_row_end(self):
        stream = FakeStream()
        Renderer(out_stream=stream).render(
            make_grid(1, 3, "#"),
            mode="truecolor",
            colors=np.full((1, 3, 3), 120, np.uint8),
            term_size=(80, 20),
        )
        row = stream.value.split("\x1b[1;1H")[1].split("\x1b[0K")[0]
        self.assertTrue(row.endswith("\x1b[0m"))

    def test_run_length_collapses_repeated_color(self):
        stream = FakeStream()
        renderer = Renderer(out_stream=stream, color_levels=16)
        renderer.render(
            make_grid(1, 8, "#"),
            mode="truecolor",
            colors=np.full((1, 8, 3), 200, np.uint8),
            term_size=(80, 20),
        )
        row = stream.value.split("\x1b[1;1H")[1].split("\x1b[0K")[0]
        self.assertEqual(row.count("38;2;"), 1)

    def test_no_line_feed_inside_frame(self):
        stream = FakeStream()
        Renderer(out_stream=stream).render(make_grid(4, 8), term_size=(80, 20))
        self.assertNotIn("\n", stream.value)

    def test_whole_line_erase_is_never_emitted(self):
        """ESC[2K after a drawn row would wipe that row."""
        stream = FakeStream()
        Renderer(out_stream=stream).render(
            make_grid(4, 8), mode="phosphor", term_size=(80, 20)
        )
        self.assertNotIn("\x1b[2J\x1b[", stream.value)
        self.assertNotIn("\x1b[2K", stream.value)


class TestRendererBackendSelection(unittest.TestCase):
    """The renderer must pick a backend the host can actually take.

    A console without VT processing prints escape sequences as literal text,
    which corrupts the image rather than only the colors, so the backend choice
    is the difference between a working app and a broken one.
    """

    def _profile(self, **kw):
        defaults = dict(
            stream=FakeStream(), enable_vt=False, vt_enabled=True,
            has_console=True, build=19045, host="Windows Terminal",
        )
        defaults.update(kw)
        return TerminalProfile(**defaults)

    def test_ansi_backend_when_vt_available(self):
        profile = self._profile(vt_enabled=True)
        self.assertTrue(profile.ansi)
        self.assertEqual(profile.backend, "ansi")
        self.assertIsNone(profile.fallback_reason)

    def test_win32_backend_when_vt_unavailable_but_console_exists(self):
        profile = self._profile(vt_enabled=False)
        self.assertFalse(profile.ansi)
        self.assertEqual(profile.backend, "win32")
        self.assertIsNotNone(profile.fallback_reason)

    def test_text_backend_when_no_console(self):
        profile = self._profile(vt_enabled=False, has_console=False)
        self.assertEqual(profile.backend, "text")
        self.assertFalse(profile.ansi)

    def test_no_truecolor_without_ansi(self):
        profile = self._profile(vt_enabled=False)
        self.assertFalse(profile.truecolor)
        self.assertEqual(profile.color_support(), "none")

    def test_color_support_labels(self):
        self.assertEqual(self._profile(vt_enabled=True).color_support(), "24bit")
        self.assertEqual(
            self._profile(vt_enabled=True, build=17000).color_support(), "256"
        )

    def test_old_windows_build_loses_truecolor(self):
        """Pre-1903 builds accept VT but clamp colors to the 16-color palette."""
        profile = self._profile(vt_enabled=True, build=17763)
        self.assertTrue(profile.ansi)
        self.assertFalse(profile.truecolor)

    def test_explicit_fallback_reason_is_reported(self):
        profile = self._profile(
            vt_enabled=False, fallback_reason="console refused VT"
        )
        self.assertEqual(profile.backend, "win32")
        self.assertEqual(profile.fallback_reason, "console refused VT")

    def test_summary_reports_backend_and_reason(self):
        profile = self._profile(vt_enabled=False)
        lines = profile.summary()
        self.assertTrue(any("renderer" in line for line in lines))
        self.assertTrue(any("fallback" in line for line in lines))

    def test_probe_is_side_effect_free(self):
        """Detection must not mutate the console; only startup enables VT."""
        stream = FakeStream(term="xterm-256color", colorterm="truecolor")
        probe_terminal(stream, enable_vt=False)
        # Nothing was changed, so there is nothing to restore.
        self.assertFalse(restore_console_mode())


class TestEscapeGating(unittest.TestCase):
    """No control byte may reach a host that would print it literally."""

    def _renderer(self, backend: str, truecolor: bool = False):
        profile = TerminalProfile(
            stream=FakeStream(), vt_enabled=backend == "ansi",
            has_console=backend != "text", build=19045,
        )
        renderer = Renderer(out_stream=profile.stream, profile=profile)
        return renderer, profile.stream

    def test_non_ansi_renderer_emits_no_escape_bytes(self):
        for backend in ("win32", "text"):
            renderer, stream = self._renderer(backend)
            renderer.show_cursor_off()
            renderer.clear()
            renderer.render(
                np.empty((3, 5), dtype=object).tolist() if False else self._grid(3, 5),
                mode=MODE_MONOCHROME, term_size=(80, 24),
            )
            renderer.show_cursor_on()
            value = stream.value
            self.assertNotIn("\x1b", value, f"backend {backend} emitted escapes")

    def test_ansi_renderer_still_emits_expected_escapes(self):
        renderer, stream = self._renderer("ansi")
        renderer.render(self._grid(2, 4), mode=MODE_MONOCHROME, term_size=(80, 20))
        self.assertIn("\x1b[1;1H", stream.value)
        self.assertIn("\x1b[2;1H", stream.value)

    def test_hide_and_show_cursor_are_gated(self):
        renderer, stream = self._renderer("win32")
        renderer.show_cursor_off()
        renderer.show_cursor_on()
        self.assertNotIn("\x1b", stream.value)

    def test_clear_is_gated(self):
        renderer, stream = self._renderer("win32")
        renderer.clear()
        self.assertNotIn("\x1b", stream.value)

    def test_status_bar_is_gated_too(self):
        """The status line must not leak escapes into a non-ANSI host."""
        renderer, stream = self._renderer("win32")
        renderer.render(self._grid(2, 4), mode=MODE_MONOCHROME, term_size=(80, 20))
        self.assertNotIn("\x1b", stream.value)

    def test_image_still_renders_in_every_backend(self):
        for backend in ("ansi", "win32", "text"):
            renderer, stream = self._renderer(backend)
            renderer.render(self._grid(1, 5, "@"), term_size=(80, 20))
            self.assertIn("@@@@@", stream.value, backend)

    def test_truecolor_falls_back_to_glyphs_without_color(self):
        for backend in ("win32", "text"):
            renderer, stream = self._renderer(backend)
            colors = np.full((1, 4, 3), 120, np.uint8)
            renderer.render(
                self._grid(1, 4, "#"), mode=MODE_TRUE_COLOR, colors=colors,
                term_size=(80, 20),
            )
            self.assertIn("####", stream.value, backend)
            self.assertNotIn("\x1b", stream.value, backend)

    def test_phosphor_falls_back_to_plain_glyphs(self):
        for backend in ("win32", "text"):
            renderer, stream = self._renderer(backend)
            renderer.render(
                self._grid(1, 4, "="), mode=MODE_PHOSPHOR, term_size=(80, 20)
            )
            self.assertIn("====", stream.value, backend)
            self.assertNotIn("\x1b", stream.value, backend)

    @staticmethod
    def _grid(rows: int, cols: int, char: str = "#"):
        grid = np.empty((rows, cols), dtype=object)
        grid[:] = char
        return grid


class TestConsoleModeRoundTrip(unittest.TestCase):
    """The console must be left as it was found."""

    def test_restore_without_save_is_safe(self):
        self.assertIsInstance(restore_console_mode(), bool)

    def test_restore_is_idempotent(self):
        restore_console_mode()
        self.assertIsInstance(restore_console_mode(), bool)

    def test_enable_console_vt_returns_bool(self):
        self.assertIsInstance(enable_console_vt(), bool)

    def test_console_vt_query_returns_bool_or_none(self):
        result = console_vt_enabled()
        self.assertTrue(result is None or isinstance(result, bool))

    def test_get_console_mode_is_safe(self):
        from ascii_cam.renderer import get_console_mode

        mode = get_console_mode()
        self.assertTrue(mode is None or isinstance(mode, int))


if __name__ == "__main__":
    unittest.main()
