#!/usr/bin/env python3
# src/cli.py

import os
import sys
import argparse
import warnings
import logging

# ─── Ensure `src/` is on the import path ──────────────────────────
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Now import your `run` entrypoint from app.py
from app import run

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Journey summariser CLI")
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to a folder of JPG frames or a single MP4"
    )
    parser.add_argument(
        "-m", "--yolo-model", default=None,
        help="Path to custom YOLOv8 .pt weights (omit to use default yolov8n)"
    )
    args = parser.parse_args()

    run(args.input, args.yolo_model)

if __name__ == "__main__":
    main()
