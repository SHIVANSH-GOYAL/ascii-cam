"""Terminal rendering for the live ASCII camera.

Rows are drawn with absolute cursor positioning and only rewritten when their
content changes, which keeps output small and avoids the flicker a full
clear-and-redraw produces. The cursor is hidden while rendering; callers restore
it on every exit path via ``show_cursor_on``.
"""

from __future__ import annotations

import sys

import numpy as np

from .config import (
    CLEAR_SCREEN,
    ERASE_TO_EOL,
    ESC,
    HOME,
    HIDE_CURSOR,
    MODE_MONOCHROME,
    MODE_PHOSPHOR,
    MODE_TRUE_COLOR,
    PHOSPHOR_BG,
    PHOSPHOR_FG,
    SHOW_CURSOR,
    STATUS_TEMPLATE,
)

_COLOR_ON = "\033[38;2;"
_COLOR_OFF = "\033[0m"
_PHOSPHOR_ON = "\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm" % (
    *PHOSPHOR_FG,
    *PHOSPHOR_BG,
)


def get_terminal_size() -> tuple[int, int]:
    """Return (cols, rows) of the terminal, falling back to 80x24."""
    import shutil

    cols, rows = shutil.get_terminal_size((80, 24))
    return max(1, cols), max(1, rows)


def supports_ansi(stream) -> bool:
    """Whether SGR color escapes are likely to be honored."""
    env = getattr(stream, "env", None)
    term = env.get("TERM", "") if env else ""
    if term in ("", "dumb"):
        # Windows Terminal and conhost since 1607 both handle ANSI, so an unset
        # TERM does not disable it there. Elsewhere it does.
        return sys.platform.startswith("win")
    if "truecolor" in env.get("COLORTERM", "").lower():
        return True
    return "256" in term or "color" in term.lower() or term.startswith("xterm")


def supports_true_color(stream) -> bool:
    """Whether 24-bit foreground escapes will be honored.

    On POSIX, ``COLORTERM``/``TERM`` is the reliable signal and a 256-color
    terminal really does drop the extra bits.

    On Windows, ``TERM`` is normally unset even inside Windows Terminal, so a
    check built on it alone would disable true color for every Windows user.
    Windows Terminal, the VS Code terminal, and conhost since 1607 all accept
    24-bit SGR, so ANSI support implies 24-bit support there.
    """
    if not supports_ansi(stream):
        return False
    if sys.platform.startswith("win"):
        return True

    env = getattr(stream, "env", None)
    colorterm = (env.get("COLORTERM", "") if env else "").lower()
    term = (env.get("TERM", "") if env else "").lower()
    return "truecolor" in colorterm or "24bit" in colorterm or "direct" in term


class Renderer:
    """Render ASCII frames using cursor positioning to minimize output."""

    def __init__(self, out_stream=None, color_levels: int = 16):
        self.out = out_stream if out_stream is not None else sys.stdout
        self._hidden_cursor = False
        self._last_grid: list[str] | None = None
        self._supports_color = supports_ansi(self.out)
        self._supports_truecolor = supports_true_color(self.out)
        self._color_levels = int(color_levels)

    # -- lifecycle ---------------------------------------------------------
    def show_cursor_off(self) -> None:
        """Hide the cursor. No-op once hidden."""
        if not self._hidden_cursor:
            self.out.write(HIDE_CURSOR)
            self.out.flush()
            self._hidden_cursor = True

    def show_cursor_on(self) -> None:
        """Restore the cursor and reset attributes. No-op once shown."""
        if self._hidden_cursor:
            self.out.write(SHOW_CURSOR + _COLOR_OFF)
            self.out.flush()
            self._hidden_cursor = False

    def clear(self) -> None:
        """Clear the screen and invalidate the row cache."""
        self.out.write(CLEAR_SCREEN + HOME)
        self.out.flush()
        self._last_grid = None

    # -- rendering ---------------------------------------------------------
    def render(
        self,
        grid,
        mode: str = MODE_MONOCHROME,
        fps: float = 0.0,
        resolution: tuple[int, int] | None = None,
        contrast: float = 1.0,
        brightness: float = 0.0,
        gamma: float = 1.0,
        colors=None,
        detail: str = "balanced",
        mirror: bool = False,
        black_cutoff: float = 0.0,
    ) -> None:
        """Render a (rows, cols) grid of single-character strings.

        ``colors`` must be a matching (rows, cols, 3) uint8 RGB array for true
        color; it is ignored otherwise. Only rows whose content differs from the
        previous frame are rewritten.
        """
        rows = len(grid)
        lines = [self._build_row(grid[r], r, mode, colors) for r in range(rows)]

        if self._last_grid is not None and len(self._last_grid) == rows:
            changed = [i for i in range(rows) if self._last_grid[i] != lines[i]]
        else:
            changed = list(range(rows))

        out = self.out
        write = out.write
        for i in changed:
            # ERASE_TO_EOL, not ESC[2K: the latter erases the entire line and
            # would wipe out the row just written.
            write(f"{ESC}[{i + 1};1H")
            write(lines[i])
            write(ERASE_TO_EOL)

        # Drop any rows left over from a taller frame after a resize.
        if self._last_grid is not None and len(self._last_grid) > rows:
            write(f"{ESC}[{rows + 1};1H{ESC}[0J")

        write(f"{ESC}[{rows + 1};1H")
        write(
            self._status(
                mode, fps, resolution, contrast, brightness, gamma, detail, mirror,
                black_cutoff,
            )
        )
        # Omitting a trailing newline keeps the status row as the last line of
        # the drawing area; a line feed there would scroll the terminal one line
        # on every frame.
        write(ERASE_TO_EOL)
        out.flush()

        self._last_grid = lines

    def _build_row(self, row, index: int, mode: str, colors) -> str:
        """Build the display string for one row, including color escapes."""
        if mode == MODE_TRUE_COLOR and colors is not None:
            if self._supports_truecolor:
                return self._truecolor_row(row, index, colors)
            # Terminal cannot take 24-bit: keep the glyphs, drop the color.
            return "".join(row.tolist())
        if mode == MODE_PHOSPHOR and self._supports_color:
            return _PHOSPHOR_ON + "".join(row.tolist()) + _COLOR_OFF
        return "".join(row.tolist())

    def _truecolor_row(self, row, index: int, colors) -> str:
        """Build a true-color row, emitting an escape only when color changes.

        A color escape is roughly 15 bytes against 1 for a glyph, so on an
        unoptimized row the escapes dominate. Cells across skin and walls often
        share a quantized color, which collapses into a run.
        """
        chars = row.tolist()
        if self._color_levels > 0:
            step = 255.0 / max(1, self._color_levels - 1)
            quant = np.rint(np.asarray(colors[index], dtype=np.float32) / step)
            quant = np.clip(quant, 0, self._color_levels - 1).astype(np.uint8)
            scaled = np.rint(quant * (255.0 / max(1, self._color_levels - 1)))
            cells = scaled.astype(np.uint8)
        else:
            cells = colors[index]

        rgb = cells.reshape(-1, 3)
        keys = (
            (rgb[:, 0].astype(np.int64) << 16)
            | (rgb[:, 1].astype(np.int64) << 8)
            | rgb[:, 2]
        ).tolist()

        parts: list[str] = []
        append = parts.append
        last = None
        for ch, key in zip(chars, keys):
            if key != last:
                append(f"{_COLOR_ON}{(key >> 16) & 255};{(key >> 8) & 255};{key & 255}m")
                last = key
            append(ch)
        append(_COLOR_OFF)
        return "".join(parts)

    def _status(
        self,
        mode: str,
        fps: float,
        resolution: tuple[int, int] | None,
        contrast: float,
        brightness: float,
        gamma: float,
        detail: str = "balanced",
        mirror: bool = False,
        black_cutoff: float = 0.0,
    ) -> str:
        """Format the status line."""
        return STATUS_TEMPLATE.format(
            fps=fps,
            res=f"{resolution[0]}x{resolution[1]}" if resolution else "--",
            mode=mode,
            detail=detail,
            contrast=contrast,
            brightness=brightness,
            gamma=gamma,
            mirror="on" if mirror else "off",
            black_cutoff=black_cutoff,
        )