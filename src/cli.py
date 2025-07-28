from .app import run_pipeline

def main():
    import argparse
    p = argparse.ArgumentParser(description="Console journey summariser")
    p.add_argument("--input", required=True,
                   help="Path to .mp4 or folder of frames (jpg)")
    run_pipeline(p.parse_args().input)

if __name__ == "__main__":
    main()
