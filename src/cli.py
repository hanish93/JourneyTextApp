#!/usr/bin/env python3
import argparse
from src.app import run_pipeline

def main():
    p = argparse.ArgumentParser(description="Console journey summariser")
    p.add_argument("--input", required=True,
                   help="Path to a .mp4 file, image folder, or directory of .jpgs")
    args = p.parse_args()
    run_pipeline(args.input)

if __name__ == "__main__":
    main()
