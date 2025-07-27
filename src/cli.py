#!/usr/bin/env python3
import argparse
from .app import run_pipeline

def main():
    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument(
        "-i","--input", required=True,
        help="Path to a .mp4 video or a folder of .jpg frames"
    )
    args = p.parse_args()
    run_pipeline(args.input)

if __name__=="__main__":
    main()
