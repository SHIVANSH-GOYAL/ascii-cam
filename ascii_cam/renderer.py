"""Terminal rendering for the live ASCII camera.

Rows are drawn with absolute cursor positioning and only rewritten when their
content changes, which keeps output small and avoids the flicker a full
clear-and-redraw produces. The cursor is hidden while rendering; callers restore
it on every exit path via ``show_cursor_on``.

Two details here are correctness requirements rather than tuning:

* Every drawn line must fit the terminal width. A line longer than the width
  wraps, and wrapping on the last row scrolls the console, which repeats once per
  frame. The status line is truncated to the reported width for that reason.
* Terminal capability is probed from the real environment and the console mode,
  because a stream such as ``sys.stdout`` exposes neither. Guessing wrong means
  emitting escapes a host prints literally.
"""

from __future__ import annotations

import os
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
    # A grid row count of 0 would place the status line on row 1 and overwrite
    # the image, so the lower bound must be at least 2.
    return max(1, cols), max(2, rows)


def _env_of(stream) -> dict:
    """Environment for capability probing.

    ``sys.stdout`` and friends have no ``.env``; without this fallback the
    capability check would see an empty environment on every real run and report
    defaults. Pass a test double with ``.env`` to probe a specific environment.
    """
    env = getattr(stream, "env", None)
    if env is None:
        try:
            return os.environ
        except Exception:  # pragma: no cover - defensive
            return {}
    return env


def _has_console() -> bool:
    """True when the process has a console screen buffer attached."""
    if os.name != "nt":
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    handle = kernel32.CreateFileW(
        ctypes.create_unicode_buffer("CONOUT$"),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        0,
        None,
    )
    if handle in (0, -1):  # INVALID_HANDLE_VALUE
        return False
    kernel32.CloseHandle(handle)
    return True


def console_vt_enabled() -> bool | None:
    """Whether the console has virtual-terminal processing enabled.

    Returns ``None`` when there is no console or the query fails, so callers can
    tell "not a console" apart from "console says no".
    """
    if os.name != "nt":
        return None
    import ctypes

    kernel32 = ctypes.windll.kernel32
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    handle = kernel32.CreateFileW(
        ctypes.create_unicode_buffer("CONOUT$"),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        0,
        None,
    )
    if handle in (0, -1):
        return None
    try:
        mode = ctypes.c_uint32()
        ok = kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        if not ok:
            return None
        return bool(mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:  # pragma: no cover - defensive
        return None
    finally:
        kernel32.CloseHandle(handle)


def enable_console_vt() -> bool:
    """Best-effort switch on virtual-terminal processing for this console.

    Some hosts leave it off, in which case SGR escapes reach the console as
    literal text. Returns True when it is enabled after the attempt.
    """
    if os.name != "nt":
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    handle = kernel32.CreateFileW(
        ctypes.create_unicode_buffer("CONOUT$"),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        0,
        None,
    )
    if handle in (0, -1):
        return False
    try:
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception:  # pragma: no cover - defensive
        return False
    finally:
        kernel32.CloseHandle(handle)


def terminal_host(stream=None) -> str:
    """Identify the terminal host in use, for reporting and capability choices."""
    env = _env_of(stream)
    if env.get("WT_SESSION"):
        return "Windows Terminal"
    if (env.get("TERM_PROGRAM", "") or "").lower() == "vscode":
        return "VS Code terminal"
    if os.name == "nt":
        return "Windows console"
    if env.get("TERM"):
        return env["TERM"]
    return "unknown"


# Windows console gained 24-bit SGR in the 1903 release (build 18362). Earlier
# Windows 10 builds accept VT sequences but clamp colors to the 16-color palette.
_WINDOWS_TRUECOLOR_MIN_BUILD = 18362


def windows_build() -> int | None:
    """Current Windows build number, or ``None`` off Windows or on failure."""
    if os.name != "nt":
        return None
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
        )
        try:
            return int(winreg.QueryValueEx(key, "CurrentBuild")[0])
        finally:
            winreg.CloseKey(key)
    except Exception:  # pragma: no cover - defensive
        return None


def supports_ansi(stream) -> bool:
    """Whether SGR escape sequences will be honored by this terminal.

    ``TERM=dumb`` is an explicit opt-out and is honored everywhere: it means "do
    not emit escapes", which is different from ``TERM`` being unset. Windows
    Terminal and conhost since 1607 both handle ANSI, so an *unset* TERM does not
    disable it there.
    """
    env = _env_of(stream)
    term = env.get("TERM", "")
    if term == "dumb":
        return False
    if term == "":
        return os.name == "nt"
    if "truecolor" in env.get("COLORTERM", "").lower():
        return True
    return "256" in term or "color" in term.lower() or term.startswith("xterm")


def supports_true_color(stream, assume_windows_vt: bool = True) -> bool:
    """Whether 24-bit foreground escapes will be honored.

    On POSIX, ``COLORTERM``/``TERM`` is the reliable signal and a 256-color
    terminal really does drop the extra bits.

    On Windows, ``TERM`` is normally unset even inside Windows Terminal, so a
    check built on it alone would disable true color for every Windows user.
    Windows Terminal, the VS Code terminal, and conhost since Windows 10 1903 all
    accept 24-bit SGR; earlier Windows 10 builds accept VT but clamp colors to
    the 16-color palette, so the build number is consulted there.
    """
    if not supports_ansi(stream):
        return False
    if os.name != "nt":
        env = _env_of(stream)
        colorterm = env.get("COLORTERM", "").lower()
        term = env.get("TERM", "").lower()
        return "truecolor" in colorterm or "24bit" in colorterm or "direct" in term

    build = windows_build()
    if build is not None and build < _WINDOWS_TRUECOLOR_MIN_BUILD:
        return False

    if assume_windows_vt:
        mode = console_vt_enabled()
        # An unknown mode is not evidence of failure; assume the common case.
        if mode is False:
            return False
    return True


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
        term_size: tuple[int, int] | None = None,
    ) -> None:
        """Render a (rows, cols) grid of single-character strings.

        ``colors`` must be a matching (rows, cols, 3) uint8 RGB array for true
        color; it is ignored otherwise. Only rows whose content differs from the
        previous frame are rewritten.

        ``term_size`` is the real terminal size and is used to clamp the status
        line. When omitted, the grid's own width is used, which is always at most
        the terminal width and therefore still safe.
        """
        rows = len(grid)
        lines = [self._build_row(grid[r], r, mode, colors) for r in range(rows)]

        if self._last_grid is not None and len(self._last_grid) == rows:
            changed = [i for i in range(rows) if self._last_grid[i] != lines[i]]
        else:
            changed = list(range(rows))

        # The status row sits below the image, so it must fit the width or it
        # wraps and, on the final console row, scrolls the screen each frame.
        width = self._status_width(grid, term_size)

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
                black_cutoff, width,
            )[:width]
        )
        # Omitting a trailing newline keeps the status row as the last line of
        # the drawing area; a line feed there would scroll the terminal one line
        # on every frame.
        write(ERASE_TO_EOL)
        out.flush()

        self._last_grid = lines

    @staticmethod
    def _status_width(grid, term_size) -> int:
        """Terminal width to clamp the status line to.

        Prefers the caller's measured size. When that is unavailable it asks the
        terminal directly, so a letterboxed grid narrower than the console does
        not needlessly truncate the status line. The grid width is the last
        resort and is always safe because the grid cannot exceed the console.
        """
        if term_size:
            return max(1, int(term_size[0]))
        try:
            cols, _ = get_terminal_size()
            return max(1, int(cols))
        except Exception:  # pragma: no cover - defensive
            first = grid[0] if len(grid) else []
            return max(1, len(first))

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
        width: int = 0,
    ) -> str:
        """Format the status line.

        Vitals come first and the control hints last, so truncating a narrow
        terminal loses the hints before it loses the information.
        """
        text = STATUS_TEMPLATE.format(
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
        if width:
            return text[:width]
        return text