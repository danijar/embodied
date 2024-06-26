# Instructions
#
# 1) Test setup:
#
#   docker run -it --rm --gpus all --privileged <base image> \
#     sh -c 'ldconfig; nvidia-smi'
#
# 2) Start training:
#
#   docker build -f agents/ppo/Dockerfile -t img . && \
#   docker run -it --rm --gpus all -v ~/logdir/docker:/logdir img \
#     sh -c 'ldconfig; scripts/xvfb_run.sh python agents/ppo/train.py \
#       --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#       --configs crafter --task crafter_reward'
#
# 3) See results:
#
#   tensorboard --logdir ~/logdir/docker
#

# System
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common \
  libglew-dev x11-xserver-utils xvfb \
  && apt-get clean

# Workdir
RUN mkdir /embodied
WORKDIR /embodied

# Python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
RUN python3.11 -m venv ./venv --upgrade-deps
ENV PATH="/embodied/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools

# Envs
COPY scripts/install-dmlab.sh .
RUN sh install-dmlab.sh
COPY scripts/install-minecraft.sh .
RUN sh install-minecraft.sh
RUN pip install ale_py autorom[accept-rom-license]
RUN pip install procgen_mirror
RUN pip install crafter
RUN pip install dm_control
RUN pip install memory_maze
ENV MUJOCO_GL egl
ENV DMLAB_DATASET_PATH /dmlab_data
ENV NUMBA_CACHE_DIR=/tmp

# Agent
RUN mkdir -p agents/ppo
COPY agents/ppo/requirements.txt agents/ppo/requirements.txt
RUN pip install -r agents/ppo/requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Embodied
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Source
COPY . .

# Cloud
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600
RUN chown 1000:root . && chmod 775 .
