"""Terminal rendering and capability probing for the live ASCII camera.

Three things happen here, in order of importance:

1. **Probe the host before emitting anything.** A Windows console without
   ``ENABLE_VIRTUAL_TERMINAL_PROCESSING`` prints every escape sequence as
   literal text, which corrupts the whole image rather than just the colors.
   So the renderer never emits a control byte until it knows the host accepts
   one.

2. **Try to enable terminal processing.** On Windows that is a single
   ``SetConsoleMode`` call, which needs no installation and no user action. The
   original console mode is saved and restored on exit.

3. **Fall back to a backend that needs no escapes.** When VT cannot be turned
   on, cursor positioning and line erasing go through the Win32 console API
   instead, so the picture still updates in place. Only if there is no console
   at all does the renderer degrade to plain line output.

Rows are drawn only when their content changes, which keeps output small and
avoids the flicker a full clear-and-redraw produces.
"""

from __future__ import annotations

import os
import sys
import time

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

# Console mode bits used by the Win32 backend.
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_ENABLE_PROCESSED_OUTPUT = 0x0001

# Windows console gained 24-bit SGR in the 1903 release (build 18362). Earlier
# Windows 10 builds accept VT but clamp colors to the 16-color palette.
_WINDOWS_TRUECOLOR_MIN_BUILD = 18362


# ---------------------------------------------------------------------------
# Win32 console helpers. These are the only escape-free way to position the
# cursor and clear a region, and they are available on every supported Windows.
# ---------------------------------------------------------------------------
def _open_console(read_write: bool = True):
    """Open a handle to the console screen buffer, or ``None`` on failure."""
    if os.name != "nt":
        return None
    import ctypes

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    access = (GENERIC_READ | GENERIC_WRITE) if read_write else GENERIC_READ
    handle = ctypes.windll.kernel32.CreateFileW(
        ctypes.create_unicode_buffer("CONOUT$"),
        access,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        0,
        None,
    )
    if handle in (0, -1):  # INVALID_HANDLE_VALUE
        return None
    return handle


def _close_console(handle) -> None:
    if handle:
        import ctypes

        ctypes.windll.kernel32.CloseHandle(handle)


def get_console_mode(handle=None) -> int | None:
    """Read the output console mode, or ``None`` if unavailable."""
    opened = handle is None
    if opened:
        handle = _open_console(read_write=False)
        if handle is None:
            return None
    try:
        import ctypes

        mode = ctypes.c_uint32()
        ok = ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        return mode.value if ok else None
    except Exception:  # pragma: no cover - defensive
        return None
    finally:
        if opened:
            _close_console(handle)


def set_console_mode(handle, mode: int) -> bool:
    """Write a console mode. Returns True when the call succeeded."""
    try:
        import ctypes

        return bool(
            ctypes.windll.kernel32.SetConsoleMode(handle, int(mode))
        )
    except Exception:  # pragma: no cover - defensive
        return False


def console_vt_enabled() -> bool | None:
    """Whether the console has virtual-terminal processing enabled.

    Returns ``None`` when there is no console or the query fails, so callers can
    tell "not a console" apart from "console says no".
    """
    if os.name != "nt":
        return None
    mode = get_console_mode()
    return None if mode is None else bool(mode & _ENABLE_VIRTUAL_TERMINAL_PROCESSING)


def _has_console() -> bool:
    """True when the process has a console screen buffer attached."""
    return _open_console(read_write=False) is not None


# The console mode before the app touched it, so it can be restored on exit.
_SAVED_CONSOLE_MODE: int | None = None


def enable_console_vt(save_original: bool = True) -> bool:
    """Best-effort switch on virtual-terminal processing for this console.

    Some hosts leave it off, in which case SGR escapes reach the console as
    literal text. Enabling it needs no installation and no user action.

    Returns True when it is enabled after the attempt. When ``save_original`` is
    set, the pre-existing mode is recorded so :func:`restore_console_mode` can
    put it back.
    """
    global _SAVED_CONSOLE_MODE
    if os.name != "nt":
        return False
    handle = _open_console(read_write=True)
    if handle is None:
        return False
    try:
        mode = get_console_mode(handle)
        if mode is None:
            return False
        if save_original and _SAVED_CONSOLE_MODE is None:
            _SAVED_CONSOLE_MODE = mode
        if mode & _ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True
        return set_console_mode(handle, mode | _ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    finally:
        _close_console(handle)


def restore_console_mode() -> bool:
    """Put back the console mode recorded at startup.

    Safe to call when nothing was saved, and safe to call more than once.
    """
    global _SAVED_CONSOLE_MODE
    saved = _SAVED_CONSOLE_MODE
    if saved is None:
        return False
    _SAVED_CONSOLE_MODE = None
    if os.name != "nt":
        return False
    handle = _open_console(read_write=True)
    if handle is None:
        return False
    try:
        return set_console_mode(handle, saved)
    finally:
        _close_console(handle)


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


def supports_ansi(stream=None, assume_windows_vt: bool = False) -> bool:
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
        if os.name != "nt":
            return False
        # On Windows an unset TERM proves nothing either way; the console mode is
        # the only reliable signal unless the caller vouches for the host.
        return bool(assume_windows_vt) or console_vt_enabled() is not False
    if "truecolor" in env.get("COLORTERM", "").lower():
        return True
    return "256" in term or "color" in term.lower() or term.startswith("xterm")


def supports_true_color(
    stream=None, assume_windows_vt: bool = False, build: int | None = None
) -> bool:
    """Whether 24-bit foreground escapes will be honored.

    On POSIX, ``COLORTERM``/``TERM`` is the reliable signal and a 256-color
    terminal really does drop the extra bits.

    On Windows, ``TERM`` is normally unset even inside Windows Terminal, so a
    check built on it alone would disable true color for every Windows user.
    Windows Terminal, the VS Code terminal, and conhost since Windows 10 1903 all
    accept 24-bit SGR; earlier Windows 10 builds accept VT but clamp colors to
    the 16-color palette, so the build number is consulted there.

    ``build`` overrides the detected OS build, which lets a caller probe a
    different machine's profile without reading the registry.
    """
    if not supports_ansi(stream, assume_windows_vt=assume_windows_vt):
        return False
    if os.name != "nt":
        env = _env_of(stream)
        colorterm = env.get("COLORTERM", "").lower()
        term = env.get("TERM", "").lower()
        return "truecolor" in colorterm or "24bit" in colorterm or "direct" in term

    detected = build if build is not None else windows_build()
    if detected is not None and detected < _WINDOWS_TRUECOLOR_MIN_BUILD:
        return False
    if not assume_windows_vt and console_vt_enabled() is False:
        return False
    return True


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------
class TerminalProfile:
    """Everything the renderer needs to know about the host, measured once.

    ``backend`` is one of:

    ``"ansi"``   escapes are honored; render exactly as designed.
    ``"win32"``  VT is unavailable but a console exists; position the cursor and
                 clear regions through the Win32 API, with no escape bytes at
                 all. Colors are off.
    ``"text"``   no console at all (piped output); plain line output only.
    """

    def __init__(
        self,
        stream=None,
        enable_vt: bool = False,
        vt_enabled=None,
        has_console=None,
        build=None,
        host=None,
        fallback_reason: str | None = None,
        saved_console_mode: int | None = None,
    ):
        self.stream = stream
        self.host = host if host is not None else terminal_host(stream)
        self.build = build if build is not None else windows_build()
        self.has_console = (
            has_console if has_console is not None else _has_console()
        )
        self.vt_attempted = bool(enable_vt)
        self.saved_console_mode = saved_console_mode

        if vt_enabled is not None:
            self.vt_enabled = vt_enabled
        else:
            self.vt_enabled = console_vt_enabled()

        if self.vt_enabled is False and enable_vt:
            # The one call that fixes a host that prints escapes literally.
            self.vt_enabled = enable_console_vt()
            if self.vt_enabled and saved_console_mode is None:
                self.saved_console_mode = _SAVED_CONSOLE_MODE

        self.ansi = self._resolve_ansi(fallback_reason)
        self.fallback_reason = fallback_reason
        self.truecolor = self._resolve_truecolor()

        if self.ansi:
            self.backend = "ansi"
        elif self.has_console:
            self.backend = "win32"
            if not self.fallback_reason:
                self.fallback_reason = (
                    "virtual-terminal processing unavailable; using Win32 "
                    "console positioning with no color"
                )
        else:
            self.backend = "text"
            if not self.fallback_reason:
                self.fallback_reason = "no console attached (output is redirected)"

    def _resolve_ansi(self, override) -> bool:
        """Decide whether escape sequences are safe to emit.

        A measured ``vt_enabled is False`` is direct evidence and outranks the
        environment: ``TERM`` is normally unset on Windows even in hosts that do
        honor escapes, and conversely a POSIX ``TERM`` says nothing about a
        Windows console. A caller-supplied reason always wins.
        """
        if override is not None:
            return False
        if self.vt_enabled is False:
            return False
        return supports_ansi(self.stream, assume_windows_vt=bool(self.vt_enabled))

    def _resolve_truecolor(self) -> bool:
        if not self.ansi:
            return False
        return supports_true_color(
            self.stream, assume_windows_vt=bool(self.vt_enabled), build=self.build
        )

    def color_support(self) -> str:
        """Short label for the status line."""
        if not self.ansi:
            return "none"
        return "24bit" if self.truecolor else "256"

    def summary(self) -> list[str]:
        """Human-readable lines describing this profile."""
        vt = {True: "yes", False: "no", None: "unknown"}[self.vt_enabled]
        lines = [
            f"renderer   : {self.backend}",
            f"host       : {self.host}",
            f"color      : {self.color_support()}",
            f"VT enabled : {vt}"
            + (" (enabled at startup)" if self.vt_attempted and self.vt_enabled else ""),
        ]
        if self.build is not None:
            lines.append(f"os build   : {self.build}")
        if self.fallback_reason:
            lines.append(f"fallback   : {self.fallback_reason}")
        return lines

    def restore(self) -> bool:
        """Undo any console mode change this profile made."""
        return restore_console_mode()


def probe_terminal(stream=None, enable_vt: bool = False) -> TerminalProfile:
    """Detect the host, optionally enabling VT, and choose a renderer backend.

    ``enable_vt`` is what mutates console state, so it is left off for
    read-only probing and switched on by the application at startup.
    """
    return TerminalProfile(stream, enable_vt=enable_vt)


def get_terminal_size() -> tuple[int, int]:
    """Return (cols, rows) of the terminal, falling back to 80x24."""
    import shutil

    cols, rows = shutil.get_terminal_size((80, 24))
    # A grid row count of 0 would place the status line on row 1 and overwrite
    # the image, so the lower bound must be at least 2.
    return max(1, cols), max(2, rows)


# ---------------------------------------------------------------------------
# Win32 cursor primitives, used only when escapes are not safe.
# ---------------------------------------------------------------------------
def _win32_move(row: int, col: int) -> None:
    """Move the console cursor without emitting an escape sequence."""
    handle = _open_console(read_write=True)
    if handle is None:
        return
    try:
        import ctypes

        class Coord(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        ctypes.windll.kernel32.SetConsoleCursorPosition(
            handle, Coord(int(col), int(row))
        )
    except Exception:  # pragma: no cover - defensive
        pass
    finally:
        _close_console(handle)


def _win32_clear_to_end_of_line(row: int, col: int, width: int) -> None:
    """Blank from (row, col) to the end of the row, without escapes."""
    handle = _open_console(read_write=True)
    if handle is None:
        return
    try:
        import ctypes

        class Coord(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        cells = max(0, width - col)
        if cells:
            written = ctypes.c_ulong()
            ctypes.windll.kernel32.FillConsoleOutputCharacterW(
                handle,
                ctypes.c_wchar(" "),
                cells,
                Coord(int(col), int(row)),
                ctypes.byref(written),
            )
    except Exception:  # pragma: no cover - defensive
        pass
    finally:
        _close_console(handle)


def _win32_clear_screen(width: int, height: int) -> None:
    """Blank the whole visible area, without escapes."""
    handle = _open_console(read_write=True)
    if handle is None:
        return
    try:
        import ctypes

        class Coord(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        written = ctypes.c_ulong()
        ctypes.windll.kernel32.FillConsoleOutputCharacterW(
            handle,
            ctypes.c_wchar(" "),
            int(width * height),
            Coord(0, 0),
            ctypes.byref(written),
        )
        ctypes.windll.kernel32.SetConsoleCursorPosition(handle, Coord(0, 0))
    except Exception:  # pragma: no cover - defensive
        pass
    finally:
        _close_console(handle)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class Renderer:
    """Render ASCII frames, choosing the backend the host can actually take.

    Every control sequence goes through :meth:`_ansi`, which is a no-op when the
    host would print it literally. That is the difference between a console with
    a corrupted image and a console that simply shows no color.
    """

    def __init__(self, out_stream=None, color_levels: int = 16, profile=None):
        self.out = out_stream if out_stream is not None else sys.stdout
        self._hidden_cursor = False
        self._last_grid: list[str] | None = None
        self._color_levels = int(color_levels)
        # Detection only: never mutate console state as a side effect of
        # constructing a renderer, which keeps tests free of side effects.
        self.profile = profile or probe_terminal(self.out, enable_vt=False)
        self._supports_color = self.profile.ansi
        self._supports_truecolor = self.profile.truecolor
        self._supports_escapes = self.profile.ansi

    # -- terminal control ---------------------------------------------------
    def _ansi(self, sequence: str) -> None:
        """Emit an escape sequence only when the host honors it.

        Emitting one the host prints literally corrupts the whole frame, so
        this is the single gate through which all control output passes.
        """
        if self._supports_escapes:
            self.out.write(sequence)

    def _move_to(self, row: int, col: int = 0) -> None:
        """Position the cursor, using escapes or the Win32 API as appropriate."""
        if self.profile.backend == "ansi":
            self.out.write(f"{ESC}[{row + 1};{col + 1}H")
            return
        if self.profile.backend == "win32":
            _win32_move(row, col)

    def _erase_to_end_of_line(self, row: int, col: int, width: int) -> None:
        """Clear from the cursor to the end of the row.

        ANSI uses ESC[0K, never ESC[2K: the latter erases the entire line and
        would wipe out the row just written.
        """
        if self.profile.backend == "ansi":
            self.out.write(ERASE_TO_EOL)
            return
        if self.profile.backend == "win32":
            _win32_clear_to_end_of_line(row, col, width)

    # -- lifecycle ---------------------------------------------------------
    def show_cursor_off(self) -> None:
        """Hide the cursor. No-op once hidden."""
        if not self._hidden_cursor:
            self._ansi(HIDE_CURSOR)
            self.out.flush()
            self._hidden_cursor = True

    def show_cursor_on(self) -> None:
        """Restore the cursor and reset attributes. No-op once shown."""
        if self._hidden_cursor:
            self._ansi(SHOW_CURSOR + _COLOR_OFF)
            self.out.flush()
            self._hidden_cursor = False

    def clear(self) -> None:
        """Clear the screen and invalidate the row cache."""
        width, height = get_terminal_size()
        if self.profile.backend == "ansi":
            self.out.write(CLEAR_SCREEN + HOME)
        elif self.profile.backend == "win32":
            _win32_clear_screen(width, height)
        else:
            # No console and no cursor control: nothing useful to clear.
            self.out.write("\r\n" * height)
        self.out.flush()
        self._last_grid = None

    # -- startup diagnostic -------------------------------------------------
    def show_startup_summary(self, seconds: float = 2.5) -> None:
        """Report the chosen renderer inside the terminal, then clear it.

        Held long enough to read rather than being wiped by the first frame, and
        skippable with any key so it never blocks a user who wants to look now.
        """
        lines = self.profile.summary()
        width, height = get_terminal_size()
        rows = max(1, len(lines))
        if rows + 1 > height:
            rows = max(1, height - 1)
            lines = lines[:rows]

        if self.profile.backend == "ansi":
            self.out.write(CLEAR_SCREEN + HOME)
        elif self.profile.backend == "win32":
            _win32_clear_screen(width, height)
        for i, line in enumerate(lines):
            self._move_to(i, 0)
            self.out.write(line[:width])
            self._erase_to_end_of_line(i, len(line[:width]), width)
        if self.profile.backend == "text":
            self.out.write("\r\n" * rows)
        self.out.flush()

        if seconds > 0:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if _key_available() is not None:
                    break
                time.sleep(0.05)

        # Hand the area over to the image.
        if self.profile.backend == "ansi":
            self.out.write(CLEAR_SCREEN + HOME)
        elif self.profile.backend == "win32":
            _win32_clear_screen(width, height)
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

        for i in changed:
            self._move_to(i, 0)
            self.out.write(lines[i])
            self._erase_to_end_of_line(i, len(lines[i]), width)

        # Drop any rows left over from a taller frame after a resize.
        if self._last_grid is not None and len(self._last_grid) > rows:
            self._move_to(rows, 0)
            if self.profile.backend == "ansi":
                self.out.write(ESC + "[0J")
            else:
                _win32_clear_to_end_of_line(rows, 0, width * 8)

        # The status line is drawn on its own row below the image so it can
        # never be mixed into the picture.
        self._move_to(rows, 0)
        self.out.write(
            self._status(
                mode, fps, resolution, contrast, brightness, gamma, detail, mirror,
                black_cutoff, width,
            )[:width]
        )
        self._erase_to_end_of_line(rows, width, width)
        self.out.flush()

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
            renderer=self.profile.backend,
            color=self.profile.color_support(),
        )
        if width:
            return text[:width]
        return text


def _key_available() -> str | None:
    """Non-blocking key check used to let the startup banner be skipped."""
    try:
        if os.name == "nt":
            import msvcrt

            return "\0" if msvcrt.kbhit() else None
        import select

        return "\0" if select.select([sys.stdin], [], [], 0)[0] else None
    except Exception:  # pragma: no cover - defensive
        return None