# cli.py
import argparse
from app import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Batch journey summariser")
    parser.add_argument("--video", required=True,
                        help="Path to a video or a directory of videos")
    args = parser.parse_args()
    run_pipeline(args.video)

if __name__ == "__main__":
    main()
