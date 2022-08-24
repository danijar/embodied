# 1. Test setup:
# docker run -it --rm --gpus all tensorflow/tensorflow:2.8.0-gpu nvidia-smi
#
# 2. Start training:
# docker build -f  agents/dreamerv3/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh xvfb_run.sh python3 agents/dreamerv3/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM tensorflow/tensorflow:2.9.0rc1-gpu
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim \
  && apt-get clean
RUN pip3 install --upgrade pip

# Envs
ENV MUJOCO_GL egl
ENV DMLAB_DATASET_PATH /dmlab_data
COPY scripts scripts
# RUN sh scripts/install-atari.sh
# RUN sh scripts/install-minecraft.sh
# RUN sh scripts/install-dmlab.sh
RUN pip3 install crafter
RUN pip3 install dm_control
# RUN pip3 install robodesk
RUN pip3 install procgen
RUN pip3 install bsuite

# Agent
RUN pip3 install dm-sonnet
RUN pip3 install tensorflow_probability
RUN pip3 install zmq
ENV TF_FUNCTION_JIT_COMPILE_DEFAULT 1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Embodied
RUN pip3 install numpy cloudpickle ruamel.yaml rich
COPY . /embodied
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied
WORKDIR embodied
