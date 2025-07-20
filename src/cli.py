# cli.py – print‑only edition
import argparse
from app import run_pipeline

def main():
    p = argparse.ArgumentParser(description="Console journey summariser")
    p.add_argument("--video", required=True, help="Path to video file")
    args = p.parse_args()
    run_pipeline(args.video)

if __name__ == "__main__":
    main()
