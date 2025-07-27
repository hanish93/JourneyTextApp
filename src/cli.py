#!/usr/bin/env python3
import os, sys

# ensure src/ itself is on the PYTHONPATH
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
                   help="folder of JPGs or single .mp4")
    p.add_argument("--yolo-model", "-m", default=None,
                   help="path to custom YOLO weights")
    args = p.parse_args()
    run(args.input, args.yolo_model)
