import argparse
from .app import run_pipeline

def main():
    p=argparse.ArgumentParser(description="Journey summariser")
    p.add_argument("-i","--input",required=True,
                   help="Path to .mp4 video or folder of .jpg frames")
    args=p.parse_args()
    run_pipeline(args.input)

if __name__=="__main__":
    main()
