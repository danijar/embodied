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
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y \
  apt ffmpeg git python3-pip vim wget unrar \
  && apt-get clean

# Envs
RUN pip3 install --no-cache-dir crafter
RUN pip3 install --no-cache-dir robodesk
RUN pip3 install --no-cache-dir dm_control
COPY scripts scripts
RUN sh scripts/install-atari.sh
RUN sh scripts/install-minecraft.sh
# RUN sh scripts/install-dmlab.sh
ENV MUJOCO_GL egl
ENV DMLAB_DATASET_PATH /dmlab_data

# Agent
RUN pip3 install --no-cache-dir dm-sonnet
RUN pip3 install --no-cache-dir tensorflow_probability
ENV TF_FUNCTION_JIT_COMPILE_DEFAULT 1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8
# ENV TF_FORCE_UNIFIED_MEMORY 1

# # HIRO env
# RUN mkdir -p /root/.mujoco && \
#   wget -nv https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O - | \
#   tar -xz -C /root/.mujoco
# ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin"
# RUN apt-get update && apt-get install -y \
#   libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
#   && apt-get clean
# RUN pip3 install --no-cache-dir mujoco_py
# RUN chown -R 1000:root /root/.mujoco && chmod -R 775 /root/.mujoco
# USER 1000
# RUN python3 -c 'import mujoco_py'
# USER root

# # HIRO env
# RUN apt-get update && apt-get install -y \
#   libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
#   && apt-get clean
# RUN groupadd -r myuser
# RUN useradd -rm -d /home/myuser -s /bin/bash -g myuser -G sudo -u 1000 myuser
# USER myuser
# WORKDIR /home/myuser
# RUN mkdir -p /home/myuser/.mujoco && \
#   wget -nv https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O - | \
#   tar -xz -C /home/myuser/.mujoco
# ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/home/myuser/.mujoco/mujoco210/bin"
# # RUN chown -R 1000:root /root/.mujoco && chmod -R 775 /root/.mujoco
# RUN pip3 install --no-cache-dir --user mujoco_py
# RUN python3 -c 'import mujoco_py'
# USER root

# # HIRO env
# RUN apt-get update && apt-get install -y \
#   libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
#   && apt-get clean
# ENV MJROOT /.mujoco
# RUN mkdir -p $MJROOT && \
#   wget -nv https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O - | \
#   tar -xz -C $MJROOT
# ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:${MJROOT}/mujoco210/bin"
# ENV MUJOCO_PY_MUJOCO_PATH "${MJROOT}/mujoco210"
# RUN pip3 install --no-cache-dir mujoco_py
# RUN python3 -c 'import mujoco_py'

# # HIRO env
RUN apt-get update && apt-get install -y \
  libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf \
  && apt-get clean
ENV MJROOT /.mujoco
RUN mkdir -p $MJROOT && \
  wget -nv https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O - | \
  tar -xz -C $MJROOT
ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:${MJROOT}/mujoco210/bin"
ENV MUJOCO_PY_MUJOCO_PATH "${MJROOT}/mujoco210"
RUN pip3 install --no-cache-dir mujoco_py
RUN python3 -c 'import mujoco_py'
RUN python3 -c 'import mujoco_py'
RUN python3 -c 'import mujoco_py'
RUN chown -R 1000:root /usr/local/lib/python3.8/dist-packages/mujoco_py
# # RUN chmod -R 775 /usr/local/lib/python3.8/dist-packages/mujoco_py

RUN mkdir -p /home/myuser
RUN useradd myuser --uid 1000 -b /home
RUN chown -R 1000:root /home/myuser
RUN ls -l /home/myuser

# USER myuser
# ENV MJROOT /.mujoco
# ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:${MJROOT}/mujoco210/bin"
# ENV MUJOCO_PY_MUJOCO_PATH "${MJROOT}/mujoco210"
# RUN pip3 install --no-cache-dir --user mujoco_py
# RUN python3 -c 'import mujoco_py'
# RUN python3 -c 'from gym.envs.mujoco.mujoco_env import MujocoEnv'
# # USER myuser
# # # RUN su - myuser -c "python3 -c 'import mujoco_py'"
# # RUN python3 -c 'import mujoco_py'
# USER root
# # RUN chown -R 1000:root /usr/local/lib/python3.8/dist-packages/mujoco_py

# Embodied
RUN pip3 install --no-cache-dir numpy cloudpickle ruamel.yaml rich
COPY . /embodied
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied
WORKDIR /embodied
