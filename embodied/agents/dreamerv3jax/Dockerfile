# 1. Test setup:
# docker run -it --rm --gpus all tensorflow/tensorflow:2.8.0-gpu nvidia-smi
#
# 2. Start training:
# docker build -f  agents/dreamerv3jax/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh xvfb_run.sh python3 agents/dreamerv3jax/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
# FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04
FROM gcr.io/deeplearning-platform-release/base-cu113

ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco

RUN apt-get update && apt-get install -y \
  apt ffmpeg git python3-pip vim \
  && apt-get clean
# pip3 install --no-cache-dir --upgrade pip

# Envs
RUN pip3 install --no-cache-dir crafter
RUN pip3 install --no-cache-dir robodesk
RUN pip3 install --no-cache-dir dm_control
COPY scripts scripts
# RUN sh scripts/install-atari.sh
# RUN sh scripts/install-minecraft.sh
# RUN sh scripts/install-dmlab.sh
ENV MUJOCO_GL egl
ENV DMLAB_DATASET_PATH /dmlab_data

# Agent
RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir jaxlib==0.3.10+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install --no-cache-dir jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install --no-cache-dir tensorflow_probability
RUN pip3 install --no-cache-dir flax
RUN pip3 install --no-cache-dir dm-haiku
RUN pip3 install --no-cache-dir optax
RUN pip3 install --no-cache-dir tensorflow-cpu
RUN pip3 install --no-cache-dir zmq

# ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Embodied
# RUN pip3 install --no-cache-dir numpy cloudpickle ruamel.yaml rich
RUN pip3 install --no-cache-dir numpy cloudpickle ruamel.yaml
COPY . /embodied
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied
WORKDIR embodied
