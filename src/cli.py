#!/usr/bin/env python3
import os, sys
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app import run

if __name__ == "__main__":
    import argparse, warnings, logging
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Journey CLI")
    parser.add_argument("-i","--input", required=True,
                        help="Folder of JPG frames or single MP4")
    args = parser.parse_args()
    run(args.input)
