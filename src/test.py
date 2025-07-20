import os
from app import run_pipeline

def test_pipeline():
    # Use a dummy video path for now
    video_path = 'JourneyText/assets/sample.mp4'
    device = 'cpu'
    # Create a dummy video if not exists
    if not os.path.exists(video_path):
        import numpy as np
        import cv2
        os.makedirs('JourneyText/assets', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 1.0, (64, 64))
        for _ in range(3):
            frame = (np.random.rand(64, 64, 3) * 255).astype('uint8')
            out.write(frame)
        out.release()
    run_pipeline(video_path, device)
    output_path = video_path + '_summary.json'
    assert os.path.exists(output_path), "Output summary not found!"
    print("Test passed: Output summary generated.")

if __name__ == "__main__":
    test_pipeline()
