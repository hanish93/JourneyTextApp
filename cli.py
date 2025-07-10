import argparse
from app import run_pipeline

def main():
    import torch
    parser = argparse.ArgumentParser(description="Journey Summarisation CLI")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_pipeline(args.video, device)

if __name__ == "__main__":
    main()
