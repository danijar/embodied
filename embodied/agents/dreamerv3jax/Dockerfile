# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
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
# FROM gcr.io/deeplearning-platform-release/base-cu113
# FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu18.04
# FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev \
  && apt-get clean
RUN pip3 install --upgrade pip

# Envs
ENV MUJOCO_GL egl
ENV DMLAB_DATASET_PATH /dmlab_data
COPY scripts scripts
RUN sh scripts/install-atari.sh
# RUN sh scripts/install-minecraft.sh
# RUN sh scripts/install-dmlab.sh
ENV NUMBA_CACHE_DIR=/tmp
RUN pip3 install crafter
RUN pip3 install dm_control
# RUN pip3 install robodesk
RUN pip3 install procgen
RUN pip3 install bsuite

# Agent
RUN pip3 install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install jaxlib
RUN pip3 install tensorflow_probability
RUN pip3 install optax
RUN pip3 install tensorflow-cpu
RUN pip3 install zmq
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Embodied
RUN pip3 install numpy cloudpickle ruamel.yaml rich
COPY . /embodied
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied
RUN mkdir /logdir
RUN chown -R 1000:root /logdir && chmod -R 775 /logdir
WORKDIR embodied
