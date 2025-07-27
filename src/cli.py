#!/usr/bin/env python3
import os, sys

# make sure `src` is on the path
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from .app import run_pipeline

if __name__ == "__main__":
    import argparse, warnings, logging
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Journey CLI")
    parser.add_argument(
        "-i","--input", required=True,
        help="MP4 video or folder of JPG frames"
    )
    args = parser.parse_args()
    run_pipeline(args.input)
