FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing --no-install-recommends && \
    DEBIAN_FRONTEND=noninteractive apt-get install python3.8 python3-pip nano libsm6 \
    libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 python3-tk qt5-default curl git nano htop -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip && \
    pip3 install future -U && \
    pip3 install tensorboard matplotlib numpy pillow==7.2.0 \
    opencv-python scikit-image numba tqdm \
    cython fire easydict pyquaternion pytest

ARG CUDA_VER="110"
ARG TORCH_VER="1.7.1"
ARG VISION_VER="0.8.2"

RUN pip3 install torch==${TORCH_VER} torchvision==${VISION_VER} --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}
