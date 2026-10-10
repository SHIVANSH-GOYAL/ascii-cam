"""Entry point and main loop for the live ASCII camera."""

from __future__ import annotations

import contextlib
import select
import sys
import time

from .camera import (
    Camera,
    CameraError,
    ThreadedCamera,
    capture_size_for_grid,
    list_cameras,
)
from .config import (
    BLACK_CUTOFF_PRESETS,
    DEFAULT_CONFIG,
    DETAIL_PRESETS,
    ESC,
    GAMMA_PRESETS,
    MODE_MONOCHROME,
    MODE_PHOSPHOR,
    MODE_TRUE_COLOR,
)
from .converter import compute_grid, render_frame
from .renderer import Renderer, get_terminal_size

RESET_COLOR = "\033[0m"
NO_CAMERA = (
    "\033[31mNo camera frame received.\033[0m Check that a webcam is connected "
    "and that no other application is using it."
)
KEY_TO_MODE = {"1": MODE_MONOCHROME, "2": MODE_PHOSPHOR, "3": MODE_TRUE_COLOR}
VALID_MODES = (MODE_MONOCHROME, MODE_PHOSPHOR, MODE_TRUE_COLOR)


class State:
    """Mutable runtime settings driven by keyboard input."""

    def __init__(self, config: dict | None = None):
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.mode = cfg["mode"] if cfg["mode"] in VALID_MODES else DEFAULT_CONFIG["mode"]
        self.contrast = float(cfg["contrast"])
        self.brightness = float(cfg["brightness"])
        self.gamma = float(cfg["gamma"])
        self.show_fps = bool(cfg["show_fps"])
        self.target_fps = float(cfg["target_fps"])
        self.margin = int(cfg["margin"])
        self.char_aspect = float(cfg["char_aspect"])
        self.mirror = bool(cfg["mirror"])
        self.ramp = cfg["ramp"]
        self.color_quantize = int(cfg.get("color_quantize", 16))
        self.dither = int(cfg.get("dither", 0))
        self.black_cutoff = float(cfg.get("black_cutoff", BLACK_CUTOFF_PRESETS[2]))
        self._black_cutoff_index = self._nearest_preset_index(self.black_cutoff)
        self.detail_index = self._clamp_index(cfg.get("detail_index", 1), DETAIL_PRESETS)
        self.running = True
        self.frozen = False
        self._gamma_index = (
            GAMMA_PRESETS.index(self.gamma) if self.gamma in GAMMA_PRESETS else 2
        )

    # -- adjusters ---------------------------------------------------------
    @staticmethod
    def _clamp_index(value: int, presets) -> int:
        return max(0, min(len(presets) - 1, int(value)))

    @staticmethod
    def _nearest_preset_index(value: float) -> int:
        return min(
            range(len(BLACK_CUTOFF_PRESETS)),
            key=lambda i: abs(BLACK_CUTOFF_PRESETS[i] - value),
        )

    def cycle_black_cutoff(self) -> None:
        self._black_cutoff_index = (self._black_cutoff_index + 1) % len(
            BLACK_CUTOFF_PRESETS
        )
        self.black_cutoff = BLACK_CUTOFF_PRESETS[self._black_cutoff_index]

    def cycle_gamma(self) -> None:
        self._gamma_index = (self._gamma_index + 1) % len(GAMMA_PRESETS)
        self.gamma = GAMMA_PRESETS[self._gamma_index]

    def cycle_detail(self) -> None:
        self.detail_index = (self.detail_index + 1) % len(DETAIL_PRESETS)

    def adjust_contrast(self, delta: float) -> None:
        self.contrast = max(0.1, min(4.0, round(self.contrast + delta, 3)))

    def adjust_brightness(self, delta: float) -> None:
        self.brightness = max(-1.0, min(1.0, round(self.brightness + delta, 3)))

    def set_mode(self, mode: str) -> None:
        if mode in VALID_MODES:
            self.mode = mode

    @property
    def detail(self) -> tuple[str, float, float, int]:
        """(label, clahe_clip, sharpen, supersample) for the active preset."""
        return DETAIL_PRESETS[self.detail_index]


def read_key() -> str | None:
    """Return a normalized control key, or ``None`` if no key is pending.

    Uses ``msvcrt`` on Windows and ``select`` on POSIX. Failures degrade to
    ``None`` so a terminal that cannot report keys cannot crash the loop.
    """
    try:
        if sys.platform.startswith("win"):
            import msvcrt

            if not msvcrt.kbhit():
                return None
            raw = msvcrt.getch()
            if raw in (b"\x00", b"\xe0"):
                # Discard the second byte of an extended key so it is not read
                # as a control.
                msvcrt.getch()
                return None
            char = raw.decode("utf-8", errors="replace").lower()
        else:
            if not select.select([sys.stdin], [], [], 0)[0]:
                return None
            char = sys.stdin.read(1).lower()
    except Exception:
        return None
    return "esc" if char == "\x1b" else char


def apply_key(key: str, state: State) -> None:
    """Apply a single key press to the runtime state.

    Accepts both the raw escape character and its "esc" name.
    """
    if not key:
        return
    key = key.lower()
    if key in ("q", "esc", "\x1b", "\x03"):
        state.running = False
    elif key in KEY_TO_MODE:
        state.set_mode(KEY_TO_MODE[key])
    elif key == "f":
        state.show_fps = not state.show_fps
    elif key == "m":
        state.mirror = not state.mirror
    elif key == "b":
        state.cycle_black_cutoff()
    elif key == "d":
        state.cycle_detail()
    elif key == " ":
        state.frozen = not state.frozen
    elif key == "g":
        state.cycle_gamma()
    elif key in ("+", "="):
        state.adjust_contrast(0.1)
    elif key in ("-", "_"):
        state.adjust_contrast(-0.1)
    elif key == "[":
        state.adjust_brightness(-0.05)
    elif key == "]":
        state.adjust_brightness(0.05)


def _drain_keys(state: State) -> None:
    """Consume every pending key so fast repeats are not lost.

    Stops at the first quit so buffered keystrokes typed ahead of it are not
    applied.
    """
    while state.running:
        key = read_key()
        if key is None:
            return
        apply_key(key, state)


def _grid_size(state: State, resolution) -> tuple[int, int]:
    """Recompute the grid from the current terminal size."""
    term_cols, term_rows = get_terminal_size()
    source_aspect = (resolution[0] / resolution[1]) if resolution else (16 / 9)
    return compute_grid(
        term_cols,
        term_rows,
        margin=state.margin,
        char_aspect=state.char_aspect,
        source_aspect=source_aspect,
    )


def _prepare_frame(frame, state: State, cols: int, rows: int):
    """Convert a frame to a grid and its matching color grid.

    Glyph and color come from one call so they cannot drift apart. The frame is
    in the camera's native BGR order, so the channel conversion happens on the
    small working image rather than the full frame.
    """
    _label, clahe_clip, sharpen_amount, _supersample = state.detail
    return render_frame(
        frame, cols, rows,
        contrast=state.contrast,
        brightness=state.brightness,
        gamma=state.gamma,
        ramp=state.ramp,
        clahe_clip=clahe_clip,
        sharpen_amount=sharpen_amount,
        mirror=state.mirror,
        dither=state.dither,
        black_cutoff=state.black_cutoff,
        color_order="bgr",
    )


def _wait_for_first_frame(camera: Camera, renderer: Renderer, timeout: float = 3.0):
    """Block until the first frame arrives, showing progress meanwhile.

    Most webcams need a few hundred milliseconds of warm-up before decoding the
    first frame, so a short wait is the common case rather than a failure.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = camera.read()
        if frame is not None:
            return frame
        with contextlib.suppress(Exception):
            renderer.out.write(f"{ESC}[2J{ESC}[HWaiting for camera...")
            renderer.out.flush()
        time.sleep(0.05)
    return None


def run(
    camera: Camera | None = None,
    renderer: Renderer | None = None,
    config: dict | None = None,
) -> int:
    """Run the capture loop. Returns a process exit code."""
    state = State(config)
    renderer = renderer or Renderer(color_levels=state.color_quantize)
    owned = camera is None
    camera = camera or ThreadedCamera(index=0)

    cols = rows = 0
    frame = None
    resolution = None
    fps = 0.0
    frame_interval = 1.0 / state.target_fps if state.target_fps > 0 else 0.0
    missing_frames = 0
    show_error = False

    renderer.clear()
    renderer.show_cursor_off()

    # Size the capture to the terminal before opening the device, so the camera
    # supplies roughly the pixels the grid can use.
    if cols == 0:
        term_cols, term_rows = get_terminal_size()
        cols, rows = compute_grid(
            term_cols, term_rows, margin=state.margin,
            char_aspect=state.char_aspect, source_aspect=16 / 9,
        )
    if hasattr(camera, "width"):
        camera.width, camera.height = capture_size_for_grid(cols, rows)

    try:
        if owned or not camera.is_open:
            camera.open()
        resolution = camera.actual_resolution
        frame = _wait_for_first_frame(camera, renderer)

        while state.running:
            # Frame time is measured from the top of the loop so the pacing sleep
            # below subtracts real processing cost rather than adding to the
            # target interval.
            frame_start = time.perf_counter()
            _drain_keys(state)
            if not state.running:
                break

            new_size = _grid_size(state, resolution)
            if new_size != (cols, rows):
                cols, rows = new_size
                renderer.clear()

            if not state.frozen:
                new_frame = camera.read()
                if new_frame is None:
                    missing_frames += 1
                    if missing_frames > 60:
                        missing_frames = 0
                        show_error = True
                    time.sleep(0.05)
                else:
                    missing_frames = 0
                    show_error = False
                    frame = new_frame
                    resolution = camera.actual_resolution or (
                        frame.shape[1], frame.shape[0]
                    )

            if frame is None:
                _render_message(renderer, NO_CAMERA, rows or 1, cols or 40)
                continue

            grid, colors = _prepare_frame(frame, state, cols, rows)
            renderer.render(
                grid,
                mode=state.mode,
                fps=fps if state.show_fps else 0.0,
                resolution=resolution,
                contrast=state.contrast,
                brightness=state.brightness,
                gamma=state.gamma,
                colors=colors,
                detail=state.detail[0],
                mirror=state.mirror,
                black_cutoff=state.black_cutoff,
            )
            if show_error:
                _render_message(renderer, NO_CAMERA, rows, cols)

            now = time.perf_counter()
            remaining = frame_interval - (now - frame_start)
            if remaining > 0:
                time.sleep(remaining)

            # Measured across the pacing sleep, so the readout reflects the rate
            # on screen rather than the compute-only rate.
            dt = time.perf_counter() - frame_start
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps == 0.0 else fps * 0.9 + instant * 0.1
    except CameraError as exc:
        renderer.show_cursor_on()
        _report_camera_error(exc)
        return 3
    except KeyboardInterrupt:
        pass
    finally:
        if owned:
            camera.close()
        renderer.show_cursor_on()
        with contextlib.suppress(Exception):
            renderer.out.write(RESET_COLOR)
            renderer.out.flush()
    return 0


def _render_message(renderer: Renderer, message: str, rows: int, cols: int) -> None:
    """Show a plain-text status message at the top of the screen."""
    with contextlib.suppress(Exception):
        renderer.out.write(f"{ESC}[1;1H{ESC}[2K{message}\r\n")
        renderer.out.flush()


def _report_camera_error(exc: CameraError) -> None:
    """Report a camera failure on stderr, listing indices that did open."""
    _error(str(exc))
    available = list_cameras()
    if available:
        indices = ", ".join(str(i) for i in available)
        _error(f"Available camera indices: {indices}")
        _error(f"Try: python main.py --camera={available[0]}")


def _error(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")


def _parse_camera_index(argv: list[str]) -> int:
    """Read the camera index from ``--camera=N``; defaults to 0."""
    index = 0
    for arg in argv:
        if arg.startswith("--camera="):
            with contextlib.suppress(ValueError):
                index = int(arg.split("=", 1)[1])
    return index


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    index = _parse_camera_index(argv)
    camera = ThreadedCamera(index=index)
    try:
        return run(camera=camera)
    except CameraError as exc:
        _report_camera_error(exc)
        return 4


if __name__ == "__main__":
    sys.exit(main())