"""Webcam capture built on OpenCV.

``ThreadedCamera`` decodes frames on a background thread and hands
``read()`` the most recent one without blocking, so the render loop never waits
on capture latency.

Frames are kept in memory only. Nothing is written to disk or transmitted.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time

import cv2


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect the OS-level stderr descriptor to the null device.

    OpenCV's FFMPEG and Media Foundation backends write to file descriptor 2
    directly from C++, bypassing Python's stream redirection, which would
    interleave with the rendered frame.
    """
    if os.name not in ("posix", "nt"):
        yield
        return
    saved_fd = None
    try:
        sys.stderr.flush()
        saved_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
        finally:
            os.close(devnull)
    except OSError:
        if saved_fd is not None:
            os.close(saved_fd)
            saved_fd = None
    try:
        yield
    finally:
        if saved_fd is not None:
            try:
                sys.stderr.flush()
                os.dup2(saved_fd, 2)
            except OSError:
                pass
            finally:
                os.close(saved_fd)


def _quiet_opencv() -> None:
    """Lower OpenCV's log verbosity below ERROR.

    The Media Foundation backend emits C++-level warnings during warm-up and
    reconnect, which would corrupt the rendered frame.
    """
    with contextlib.suppress(Exception):
        import cv2.utils.logging as cv_logging

        cv_logging.setLogLevel(cv_logging.LOG_LEVEL_ERROR)
        return
    with contextlib.suppress(Exception):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)


class CameraError(Exception):
    """Raised when the camera cannot be opened or used."""


class Camera:
    """Wrapper around ``cv2.VideoCapture`` with explicit error handling."""

    def __init__(
        self, index: int = 0, width: int | None = None, height: int | None = None
    ):
        self.index = index
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None
        self._opened = False

    def open(self) -> None:
        """Open the camera. Raises ``CameraError`` on failure."""
        if self._opened:
            return
        _quiet_opencv()
        try:
            with _suppress_c_stderr():
                self.cap = cv2.VideoCapture(self.index)
        except cv2.error as exc:  # pragma: no cover - defensive
            raise CameraError(f"Failed to open camera {self.index}: {exc}") from exc

        if not self.cap.isOpened():
            raise CameraError(
                f"Camera {self.index} is unavailable. Check that one is connected, "
                "that no other application is using it, and that OS privacy settings "
                "allow camera access."
            )

        # A single-frame buffer keeps read() on the newest frame rather than
        # draining a queue of stale ones.
        with contextlib.suppress(cv2.error):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))

        self._opened = True

    def read(self):
        """Return the newest frame as a ``numpy.ndarray``, or ``None``.

        The frame is handed back in the driver's native order (BGR on Windows)
        with no conversion. Converting at camera resolution costs more than the
        conversion is worth, since the result is immediately downscaled; the
        converter performs it on the much smaller working image instead. Pass
        ``color_order="bgr"`` to the converter to compensate.
        """
        if not self.is_open or self.cap is None:
            raise CameraError("Camera is not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    @property
    def is_open(self) -> bool:
        """True while the capture device is open and usable."""
        return self._opened and self.cap is not None

    @property
    def actual_resolution(self) -> tuple[int, int] | None:
        """Return (width, height) of the active stream, or ``None`` if closed."""
        if not self.is_open:
            return None
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height) if width > 0 and height > 0 else None

    def close(self) -> None:
        """Release the camera. Safe to call more than once."""
        if self.cap is not None:
            with contextlib.suppress(cv2.error):
                self.cap.release()
        self.cap = None
        self._opened = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class ThreadedCamera(Camera):
    """Camera that decodes frames on a background thread.

    ``read()`` returns the most recently decoded frame without blocking, so
    rendering is decoupled from capture latency.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame = None
        self._running = False

    def open(self) -> None:
        super().open()
        self._running = True
        self._thread = threading.Thread(
            target=self._reader, name="ascii-cam-reader", daemon=True
        )
        self._thread.start()

    def _reader(self) -> None:
        while self._running and self.cap is not None:
            try:
                ok, frame = self.cap.read()
            except cv2.error:  # pragma: no cover - defensive
                ok, frame = False, None
            if ok and frame is not None:
                # The frame is stored as decoded. Converting it here would run at
                # camera resolution for output the converter discards.
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        if not self.is_open:
            raise CameraError("Camera is not open")
        with self._lock:
            return self._frame

    def close(self) -> None:
        """Stop the reader thread and release the device."""
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._thread = None
        super().close()


def list_cameras(max_index: int = 4) -> list[int]:
    """Probe a small range of device indices and return those that open."""
    found: list[int] = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                found.append(i)
        finally:
            cap.release()
    return found


# Webcam drivers reject arbitrary sizes and fall back to their default, so only
# standard resolutions are requested.
STANDARD_CAPTURE_SIZES = (
    (320, 240),
    (640, 480),
    (800, 600),
    (1280, 720),
    (1920, 1080),
)


def capture_size_for_grid(
    cols: int, rows: int, enhance_scale: int = 2
) -> tuple[int, int]:
    """Pick a standard capture size for a character grid.

    The camera should supply enough pixels for area averaging over the working
    image and no more. At 61x23 cells, 320x240 retains about 90% of the
    distinguishable structure of a 640x480 source while costing roughly a sixth
    of the capture time; 160x120 falls to about half. The working image is
    therefore doubled and matched against the smallest standard size that covers
    it.

    Returns (width, height) in pixels.
    """
    work_w = max(1, cols) * max(1, enhance_scale)
    work_h = max(1, rows) * max(1, enhance_scale)
    want_w = max(320, work_w * 2)
    want_h = max(240, work_h * 2)
    for width, height in STANDARD_CAPTURE_SIZES:
        if width >= want_w and height >= want_h:
            return width, height
    return STANDARD_CAPTURE_SIZES[-1]