# cli.py
import argparse
from app import run_pipeline

def main():
    p = argparse.ArgumentParser(description="Journey Summarisation CLI")
    p.add_argument("--video", required=True, help="Path to video file")
    args = p.parse_args()
    run_pipeline(args.video)   # run_pipeline now auto-detects device

if __name__ == "__main__":
    main()
