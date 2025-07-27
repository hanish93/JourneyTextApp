import argparse
from .app import run_pipeline

def main():
    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument("-i","--input", required=True,
                   help="Video (.mp4) or folder of frames (.jpg)")
    args = p.parse_args()
    run_pipeline(args.input)

if __name__=="__main__":
    main()
