"""Time every stage of the render loop with the real loop running.

Unlike a micro-benchmark, this wraps each stage of the actual ``run()`` loop in a
timer, so work that is not obvious from the source shows up here. Requires a
working camera.

    python bench_loop.py
"""

from __future__ import annotations

import statistics
import time

import ascii_cam.main as m
from ascii_cam.camera import ThreadedCamera
from ascii_cam.config import DEFAULT_CONFIG
from ascii_cam.renderer import Renderer


class NullStream:
    """Discards output while exposing the environment the renderer reads."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def env(self):
        return {"TERM": "xterm"}


def main() -> int:
    base = Renderer(out_stream=NullStream())
    samples: dict[str, list[float]] = {}

    # (label, owner, attribute) for each stage to time.
    targets = [
        ("render", base, "render"),
        ("prepare", m, "_prepare_frame"),
        ("grid", m, "_grid_size"),
        ("keys", m, "_drain_keys"),
        ("read", m.Camera, "read"),
    ]
    originals = {label: getattr(owner, attr) for label, owner, attr in targets}

    def make_wrapper(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                samples.setdefault(label, []).append(time.perf_counter() - start)

        return wrapper

    for label, owner, attr in targets:
        setattr(owner, attr, make_wrapper(originals[label]))

    calls = {"n": 0}

    def keys():
        calls["n"] += 1
        return "q" if calls["n"] > 60 else None

    real_key = m.read_key
    m.read_key = keys
    camera = ThreadedCamera(index=0)
    wall0 = time.perf_counter()
    try:
        rc = m.run(camera=camera, renderer=base, config=dict(DEFAULT_CONFIG))
    finally:
        m.read_key = real_key
        for label, owner, attr in targets:
            setattr(owner, attr, originals[label])

    wall = time.perf_counter() - wall0
    frames = calls["n"] - 1
    # Camera warm-up sits inside the wall clock, so the interval is taken from
    # the loop's own frame count.
    interval = wall / max(1, frames)

    print(f"exit code           : {rc}")
    print(f"frames              : {frames}")
    print(f"wall clock fps      : {frames / wall:.1f}  (includes camera warm-up)")
    print(f"mean frame interval : {interval * 1000:.2f} ms")
    print()
    print("per-stage cost:")
    total = 0.0
    for name, values in samples.items():
        mean = statistics.mean(values) * 1000
        total += mean
        p95 = sorted(values)[int(0.95 * len(values))] * 1000
        print(
            f"  {name:<9} {mean:6.3f} ms  median "
            f"{statistics.median(values) * 1000:6.3f}  p95 {p95:6.3f}"
        )
    print(f"  {'sum':<9} {total:6.3f} ms")
    print(f"  {'pacing':<9} {max(0.0, interval * 1000 - total):6.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())