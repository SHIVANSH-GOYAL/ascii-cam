"""Attribute CPU cost to each stage of the pipeline.

Measures three workloads separately so a "uses too much CPU" report can be
attributed rather than guessed at:

  A: the camera reader thread alone
  B: the full loop (capture + convert + render)
  C: a control busy loop, to make the harness's own cost visible

Requires a working camera.

    python bench_attribution.py
"""

from __future__ import annotations

import time

from ascii_cam import ThreadedCamera
from ascii_cam.converter import compute_grid
from ascii_cam.main import State, _prepare_frame
from ascii_cam.renderer import Renderer, get_terminal_size

SECONDS = 5.0


class NullStream:
    """Discards output while exposing the environment the renderer reads."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def env(self):
        return {"TERM": "xterm"}


def _cpu_fraction(start_wall: float, start_cpu: float) -> tuple[float, float]:
    """Return (wall seconds, fraction of one core) since the given marks."""
    wall = time.perf_counter() - start_wall
    cpu = time.process_time() - start_cpu
    return wall, (cpu / wall if wall else 0.0)


def reader_only() -> None:
    """Measure the reader thread without a spin-loop confound.

    ``ThreadedCamera.read()`` returns a cached pointer almost for free, so
    polling it in a tight loop would measure the harness rather than the camera.
    Polling at the frame rate isolates the thread's real cost.
    """
    camera = ThreadedCamera(index=0)
    camera.open()
    time.sleep(1.0)
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    delivered = 0
    while time.perf_counter() - wall0 < SECONDS:
        if camera.read() is not None:
            delivered += 1
        time.sleep(1.0 / 30.0)
    wall, fraction = _cpu_fraction(wall0, cpu0)
    print(
        "A reader thread only   %6.1f%% of one core  (%7.1f frames/s)"
        % (100 * fraction, delivered / wall)
    )
    camera.close()


def full_loop(target_fps: float = 24.0) -> None:
    """Measure the whole loop at a fixed target frame rate."""
    camera = ThreadedCamera(index=0)
    camera.open()
    time.sleep(1.0)
    state = State({"mode": "monochrome", "detail_index": 1, "mirror": True})
    renderer = Renderer(out_stream=NullStream())
    cols, rows = compute_grid(*get_terminal_size(), source_aspect=4 / 3)
    interval = 1.0 / target_fps

    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    frames = 0
    while time.perf_counter() - wall0 < SECONDS:
        frame = camera.read()
        if frame is None:
            continue
        frame_start = time.perf_counter()
        grid, _colors = _prepare_frame(frame, state, cols, rows)
        renderer.render(
            grid, mode="monochrome", fps=22.0, resolution=(320, 240)
        )
        frames += 1
        remaining = interval - (time.perf_counter() - frame_start)
        if remaining > 0:
            time.sleep(remaining)
    wall, fraction = _cpu_fraction(wall0, cpu0)
    print(
        "B full loop at %.0f fps   %6.1f%% of one core  (%7.1f fps)"
        % (target_fps, 100 * fraction, frames / wall)
    )
    camera.close()


def control() -> None:
    """A busy loop, so the harness's own cost is visible."""
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    while time.perf_counter() - wall0 < 2.0:
        pass
    _wall, fraction = _cpu_fraction(wall0, cpu0)
    print("C control busy loop     %6.1f%% of one core" % (100 * fraction))


def main() -> int:
    print(f"CPU attribution ({SECONDS:g}-second samples, real camera):")
    reader_only()
    full_loop(24)
    control()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())