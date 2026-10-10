"""Measure achieved frame rate and output volume per mode and detail preset.

Runs the real loop against the real camera with a byte-counting renderer and
reports frames per second and bytes per frame for each combination. Requires a
working camera.

    python bench_modes.py                      # all combinations
    MODE=truecolor DI=2 python bench_modes.py  # one combination
"""

from __future__ import annotations

import os
import time

import ascii_cam.main as m
from ascii_cam import Renderer, ThreadedCamera
from ascii_cam.config import DEFAULT_CONFIG, DETAIL_PRESETS

MODE = os.environ.get("MODE", "monochrome")
DETAIL = int(os.environ.get("DI", "1"))
FRAMES = int(os.environ.get("FRAMES", "120"))


class Counter:
    """Discards output while counting bytes written."""

    def __init__(self) -> None:
        self.bytes = 0

    def write(self, text: str) -> int:
        self.bytes += len(text)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def env(self):
        return {"TERM": "xterm-256color", "COLORTERM": "truecolor"}


def main() -> int:
    stream = Counter()
    renderer = Renderer(out_stream=stream)
    camera = ThreadedCamera(index=0)
    config = dict(DEFAULT_CONFIG)
    config.update({"mode": MODE, "detail_index": DETAIL})

    count = {"n": 0}

    def keys():
        count["n"] += 1
        return "q" if count["n"] > FRAMES else None

    real_key = m.read_key
    m.read_key = keys
    started = time.perf_counter()
    try:
        rc = m.run(camera=camera, renderer=renderer, config=config)
    finally:
        m.read_key = real_key

    elapsed = time.perf_counter() - started
    frames = count["n"] - 1
    label = DETAIL_PRESETS[DETAIL][0]
    per_frame = stream.bytes / max(1, frames)
    throughput = per_frame * frames / elapsed / 1024
    print(
        "%-11s %-9s %4d frames %6.1f fps  %7d B/frame (%.0f KB/s)  rc=%d"
        % (MODE, label, frames, frames / elapsed, per_frame, throughput, rc)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())