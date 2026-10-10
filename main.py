#!/usr/bin/env python3
"""Command line entry point.

Run directly::

    python main.py                  # default camera
    python main.py --camera=1       # a second camera
"""

import sys

from ascii_cam.main import main

if __name__ == "__main__":
    sys.exit(main())