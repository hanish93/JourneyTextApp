#!/usr/bin/env python3
# src/cli.py

import os, sys

# make sure src/ is on PYTHONPATH
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app import run

if __name__ == "__main__":
    import argparse, warnings, logging
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Journey summariser CLI")
    p.add_argument("--input", "-i", required=True,
                   help="Folder of JPG frames or single MP4")
    p.add_argument("--yolo-model", "-m", default=None,
                   help="Path to your custom-trained YOLOv8 .pt")
    args = p.parse_args()
    run(args.input, args.yolo_model)
