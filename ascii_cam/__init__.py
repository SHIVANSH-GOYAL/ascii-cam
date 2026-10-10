"""Live ASCII camera: real-time webcam-to-ASCII rendering in the terminal."""

from .config import (
    ASCII_RAMP,
    DEFAULT_CONFIG,
    GAMMA_PRESETS,
    MODE_MONOCHROME,
    MODE_PHOSPHOR,
    MODE_TRUE_COLOR,
    STATUS_TEMPLATE,
)
from .camera import Camera, CameraError, ThreadedCamera
from .converter import compute_grid, luminance, ramp_indices, to_ascii
from .renderer import Renderer

__all__ = [
    "ASCII_RAMP",
    "Camera",
    "CameraError",
    "DEFAULT_CONFIG",
    "GAMMA_PRESETS",
    "MODE_MONOCHROME",
    "MODE_PHOSPHOR",
    "MODE_TRUE_COLOR",
    "Renderer",
    "STATUS_TEMPLATE",
    "ThreadedCamera",
    "compute_grid",
    "luminance",
    "ramp_indices",
    "to_ascii",
]

__version__ = "1.0.0"


def main(argv=None):
    """CLI entry point."""
    from .main import main as _main

    return _main(argv)


def run(camera=None, renderer=None, config=None):
    """Run the capture loop with injected dependencies."""
    from .main import run as _run

    return _run(camera, renderer, config)