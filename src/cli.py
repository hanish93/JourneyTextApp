#!/usr/bin/env python3
# src/cli.py

import os, sys

# Ensure we can import from src/
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app import run

if __name__ == "__main__":
    import argparse, warnings, logging
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True,
                   help="folder of .jpg frames or a single .mp4")
    p.add_argument("--yolo-model", "-m", default=None,
                   help="optional custom YOLOv8 .pt weights")
    args = p.parse_args()

    run(args.input, args.yolo_model)
