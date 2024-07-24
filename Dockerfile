FROM ubuntu:20.04

# Set the working directory
WORKDIR /lightspeed

# Install necessary packages and clean up
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    wget \
    libsndfile1 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install specific version of torch
RUN pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Create ckpts directory
RUN mkdir -p ckpts

# Download male models
RUN wget https://huggingface.co/spaces/ntt123/Vietnam-male-voice-TTS/resolve/main/gen_619k.pth -O ckpts/generator_male.pth && \
    wget https://huggingface.co/spaces/ntt123/Vietnam-male-voice-TTS/resolve/main/vbx_phone_set.json -O ckpts/vbx_phone_set.json && \
    wget https://huggingface.co/spaces/ntt123/Vietnam-male-voice-TTS/resolve/main/vbx_duration_model.pth -O ckpts/vbx_duration_model.pth

# Download female models
RUN wget https://huggingface.co/spaces/ntt123/Vietnam-female-voice-TTS/resolve/main/gen_630k.pth -O ckpts/generator_female.pth && \
    wget https://huggingface.co/spaces/ntt123/Vietnam-female-voice-TTS/resolve/main/duration_model.pth -O ckpts/duration_model.pth && \
    wget https://huggingface.co/spaces/ntt123/Vietnam-female-voice-TTS/resolve/main/phone_set.json -O ckpts/phone_set.json

COPY . .
RUN pip install -r requirements.txt