#!/bin/bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip ffmpeg libsm6 libxext6
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
