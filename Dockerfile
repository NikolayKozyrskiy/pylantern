ARG CUDA_VERSION=11.8.0-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    cmake \
    curl \
    wget \
    ninja-build \
    ffmpeg \
    git

ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda update --all && \
    conda clean -ay 

ARG TORCH_VERSION=2.1.1
ARG TORCHVISION_VERSION=0.16.1
ARG TORCH_URL=https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url ${TORCH_URL}
ENV TORCH_CUDA_ARCH_LIST="6.1 8.6 8.9"

COPY libs/matches/ /libs/matches
RUN pip install /libs/matches

COPY requirements.txt ./
RUN pip install -r requirements.txt
