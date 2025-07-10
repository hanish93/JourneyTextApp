FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 && \
    pip3 install --upgrade pip

COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY . /app/
CMD ["python3", "cli.py", "--video", "assets/sample.mp4"]
