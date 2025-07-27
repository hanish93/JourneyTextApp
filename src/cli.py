#!/usr/bin/env python3
# src/cli.py  —  stand‑alone entrypoint

import os
import sys
import argparse
import warnings
import logging

# ensure src/ is on PYTHONPATH
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app import run

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Console journey summariser")
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to .mp4, folder of .mp4s, or folder of JPG/PNG frames"
    )
    parser.add_argument(
        "--yolo-model", "-m", default=None,
        help="Optional path to custom YOLOv8 .pt weights"
    )
    args = parser.parse_args()
    run(args.input, args.yolo_model)
