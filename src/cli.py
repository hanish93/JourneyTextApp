import argparse
from app import run_pipeline

def main():
    p=argparse.ArgumentParser(description="Console journey summariser")
    p.add_argument("--video",required=True,
                   help="Path to .mp4 file or directory")
    run_pipeline(p.parse_args().video)

if __name__=="__main__":
    main()
