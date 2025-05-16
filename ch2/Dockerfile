FROM docker.io/nvidia/cuda:12.0.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    xz-utils \
    build-essential \
    libssl-dev \
    git \
    cmake \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*