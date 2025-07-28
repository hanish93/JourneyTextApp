#!/usr/bin/env python3
import argparse
from .app import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Journey summariser")
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to .mp4 video or folder of .jpg frames"
    )
    args = parser.parse_args()
    run_pipeline(args.input)

if __name__ == "__main__":
    main()
