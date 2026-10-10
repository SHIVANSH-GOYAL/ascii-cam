"""Diagnose why ASCII output may be invisible in a real terminal.

Runs the real live loop, captures every byte it sends, and replays those bytes
through a minimal terminal emulator that reproduces what a terminal would show --
cursor positioning, line erasing, and scrolling on line feed.

    python diagnose.py             # run frames and show the reconstructed screen
    python diagnose.py --env       # environment facts only
    python diagnose.py --frames 20 # render more frames

Read the reconstructed screen at the end. Missing, misaligned, or jittering art
means the bug is in the bytes emitted, not in the terminal.
"""

from __future__ import annotations

import os
import shutil
import sys

import ascii_cam.main as m
from ascii_cam import ThreadedCamera
from ascii_cam.config import DEFAULT_CONFIG
from ascii_cam.renderer import Renderer, supports_ansi


class Recorder:
    """Stream that records writes instead of printing them."""

    def __init__(self):
        self.data: list[str] = []

    def write(self, text: str) -> int:
        self.data.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def value(self) -> str:
        return "".join(self.data)

    @property
    def env(self):
        return os.environ


class Screen:
    """Character grid that applies the escape sequences this app emits."""

    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.grid = [[" "] * cols for _ in range(rows)]
        self.y = self.x = 0

    def _clamp(self) -> None:
        self.y = max(0, min(self.rows - 1, self.y))
        self.x = max(0, min(self.cols - 1, self.x))

    def put(self, ch: str) -> None:
        if ch == "\n":
            self.y += 1
            if self.y >= self.rows:
                self.grid.pop(0)
                self.grid.append([" "] * self.cols)
                self.y = self.rows - 1
            return
        if ch == "\r":
            self.x = 0
            return
        self._clamp()
        self.grid[self.y][self.x] = ch
        self.x += 1
        if self.x >= self.cols:
            self.x = self.cols - 1

    def text(self) -> list[str]:
        return ["".join(row) for row in self.grid]


def replay(data: str, cols: int, rows: int) -> Screen:
    """Apply a byte stream to a Screen, emulating the important sequences."""
    screen = Screen(cols, rows)
    i, n = 0, len(data)
    while i < n:
        ch = data[i]
        if ch == "\033" and i + 1 < n and data[i + 1] == "[":
            j = i + 2
            params = ""
            # Parameter bytes only: digits, ';', and the '?' private marker. The
            # final letter is not part of the parameters.
            while j < n and (data[j].isdigit() or data[j] in ";?"):
                params += data[j]
                j += 1
            if j < n:
                final = data[j]
                if final == "H":  # cursor position
                    parts = params.split(";") if params else []
                    row = int(parts[0]) - 1 if parts and parts[0].isdigit() else 0
                    col = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
                    screen.y, screen.x = row, col
                elif final == "J":  # erase in display
                    mode = int(params) if params.isdigit() else 0
                    if mode == 0:
                        for c in range(screen.x, screen.cols):
                            screen.grid[screen.y][c] = " "
                        for r in range(screen.y + 1, screen.rows):
                            screen.grid[r] = [" "] * screen.cols
                    elif mode == 2:
                        screen.grid = [[" "] * screen.cols for _ in range(screen.rows)]
                elif final == "K":  # erase in line
                    mode = int(params) if params.isdigit() else 0
                    if mode == 0:
                        for c in range(screen.x, screen.cols):
                            screen.grid[screen.y][c] = " "
                    elif mode == 2:
                        screen.grid[screen.y] = [" "] * screen.cols
                i = j + 1
                continue
            i += 2
            continue
        screen.put(ch)
        i += 1
    return screen


class _EnvView:
    """Exposes ``os.environ`` the way the renderer expects to read it."""

    @property
    def env(self):
        return os.environ


def environment_report() -> None:
    """Print terminal facts the renderer bases its decisions on."""
    cols, rows = shutil.get_terminal_size((80, 24))
    print("=== ENVIRONMENT ===")
    print(f"terminal size: {cols} cols x {rows} rows")
    print(f"TERM       = {os.environ.get('TERM', '(unset)')!r}")
    print(f"COLORTERM  = {os.environ.get('COLORTERM', '(unset)')!r}")
    print(f"isatty     = {sys.stdout.isatty()}")
    print(f"supports_ansi = {supports_ansi(_EnvView())}")
    print(f"platform   = {sys.platform}")


def main() -> int:
    environment_report()

    if "--env" in sys.argv:
        return 0

    frames = 12
    if "--frames" in sys.argv:
        frames = int(sys.argv[sys.argv.index("--frames") + 1])

    print()
    print(f"=== RUNNING {frames} FRAMES OF THE REAL LOOP ===")
    recorder = Recorder()
    renderer = Renderer(out_stream=recorder)
    camera = ThreadedCamera(index=0)
    counter = {"n": 0}

    def keys():
        counter["n"] += 1
        return "q" if counter["n"] > frames else None

    original = m.read_key
    m.read_key = keys
    try:
        rc = m.run(camera=camera, renderer=renderer, config=dict(DEFAULT_CONFIG))
    finally:
        m.read_key = original

    data = recorder.value
    print(f"exit code: {rc}")
    print(f"bytes emitted: {len(data)}")

    cols, rows = shutil.get_terminal_size((80, 24))
    print(f"emulator size: {cols} x {rows}")

    # A line feed inside the drawing area scrolls the terminal each frame.
    feeds = data.count("\n")
    print(f"line feeds written: {feeds}")
    if feeds:
        print("  WARNING: line feeds scroll the terminal while rendering")

    # ESC[2K erases the whole line and wipes out the row just drawn.
    whole_line_erases = data.count("\033[2K")
    print(f"whole-line erases: {whole_line_erases}")
    if whole_line_erases:
        print("  WARNING: ESC[2K erases rows after they are drawn")

    screen = replay(data, cols, rows)
    print()
    print("=== RECONSTRUCTED SCREEN (what your terminal shows) ===")
    for line in screen.text():
        print("|" + line.rstrip() + "|")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())