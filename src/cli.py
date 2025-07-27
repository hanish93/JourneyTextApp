#!/usr/bin/env python3
import os
import sys

# Ensure this file's directory (the `src` package) is on sys.path
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Relative import inside the src package
from .app import run_pipeline

if __name__ == "__main__":
    import argparse
    import warnings
    import logging

    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Console journey summariser")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to a .mp4 video or a directory of frame JPGs"
    )
    args = parser.parse_args()

    run_pipeline(args.input)
