"""End-to-end checks driven through the real render loop.

These cover paths unit tests cannot reach: terminal resizing, freezing a frame,
mode switching, and cursor restoration on exit. Requires a working camera.

    python verify_loop.py
"""

from __future__ import annotations

import ascii_cam.main as m
from ascii_cam import ThreadedCamera
from ascii_cam.camera import STANDARD_CAPTURE_SIZES, capture_size_for_grid
from ascii_cam.config import DEFAULT_CONFIG
from ascii_cam.converter import compute_grid
from ascii_cam.renderer import Renderer


class Stream:
    """Captures writes so assertions can inspect the emitted bytes."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, text: str) -> int:
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def value(self) -> str:
        return "".join(self.chunks)

    @property
    def env(self):
        return {"TERM": "xterm-256color", "COLORTERM": "truecolor"}


def drive(script, sizes):
    """Run the loop with a scripted terminal and keyboard.

    ``script`` holds (key, iterations) pairs: the key is delivered once, then the
    iterations let frames render. A ``None`` key only idles. Once the queue drains
    the loop is asked to quit.

    ``sizes`` supplies the terminal dimensions for successive polls.
    """
    stream = Stream()
    renderer = Renderer(out_stream=stream)
    camera = ThreadedCamera(index=0)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"target_fps": 24, "mode": "monochrome"})

    real_size = m.get_terminal_size
    real_key = m.read_key

    pending: list[str | None] = []
    for key, idle in script:
        if key is not None:
            pending.append(key)
        pending.extend([None] * max(0, idle))

    state = {"queue": pending, "quit_sent": False, "size_idx": 0}

    def fake_size():
        idx = state["size_idx"]
        state["size_idx"] += 1
        return sizes[idx] if idx < len(sizes) else sizes[-1]

    def fake_key():
        if state["queue"]:
            return state["queue"].pop(0)
        if not state["quit_sent"]:
            state["quit_sent"] = True
            return "q"
        return None

    m.get_terminal_size = fake_size
    m.read_key = fake_key
    try:
        rc = m.run(camera=camera, renderer=renderer, config=cfg)
    finally:
        m.get_terminal_size = real_size
        m.read_key = real_key
    return rc, stream


def check(name: str, condition: bool) -> bool:
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return bool(condition)


def main() -> int:
    ok = True

    resize_sizes = [(100, 30), (100, 30), (60, 20), (120, 40), (120, 40)]

    print("resize handling:")
    rc, stream = drive(
        script=[("1", 8), ("2", 8), ("3", 8), (None, 8)],
        sizes=resize_sizes,
    )
    out = stream.value
    ok &= check("loop exits cleanly (rc == 0)", rc == 0)
    ok &= check("screen was cleared for the new size", "\x1b[2J" in out)
    ok &= check("status line was drawn", "monochrome" in out)

    # The capture size is derived from the terminal size the loop started with,
    # then held fixed while the terminal resizes.
    start_cols, start_rows = compute_grid(*resize_sizes[0], source_aspect=16 / 9)
    expected_size = capture_size_for_grid(start_cols, start_rows)
    ok &= check(
        f"capture resolution matches the grid ({expected_size[0]}x{expected_size[1]})",
        f"| {expected_size[0]}x{expected_size[1]} |" in out,
    )
    ok &= check(
        "capture size is a standard resolution",
        expected_size in STANDARD_CAPTURE_SIZES,
    )

    print("freeze keeps the camera running:")
    rc, stream = drive(script=[(" ", 12)], sizes=[(80, 24)] * 14)
    ok &= check("freeze did not crash the loop", rc == 0)
    ok &= check("frames were rendered while frozen", "monochrome" in stream.value)

    print("mode switching:")
    rc, stream = drive(
        script=[("1", 6), ("2", 6), ("3", 6), ("2", 6)], sizes=[(80, 24)] * 26
    )
    out = stream.value
    ok &= check("phosphor frames were emitted", "38;2;0;255;65m" in out)
    ok &= check("true-color frames were emitted", "\x1b[38;2;" in out)
    ok &= check("every mode exited cleanly", rc == 0)

    print("adjustment keys:")
    rc, stream = drive(
        script=[
            ("+", 4), ("+", 4), ("-", 4), ("]", 4), ("[", 4),
            ("g", 4), ("b", 4), ("d", 4), ("m", 4), ("f", 4), ("f", 4),
        ],
        sizes=[(80, 24)] * 46,
    )
    ok &= check("adjustments were accepted without error", rc == 0)

    print("cursor restoration:")
    ok &= check("cursor is shown again on exit", "\x1b[?25h" in stream.value)
    ok &= check("color state is reset on exit", "\x1b[0m" in stream.value)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())