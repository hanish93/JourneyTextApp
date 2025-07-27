#!/usr/bin/env python3
import os, sys
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app import run

if __name__=="__main__":
    import argparse, warnings, logging
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", required=True,
                        help="Folder of JPG frames or MP4")
    parser.add_argument("-m","--yolo-model", default=None,
                        help="Custom YOLOv8 .pt (omit for yolov8n)")
    args = parser.parse_args()
    run(args.input, args.yolo_model)
