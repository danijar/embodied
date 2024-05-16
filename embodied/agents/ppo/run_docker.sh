#!/bin/sh
set -e
docker build -f agents/ppo/Dockerfile -t img .
# docker run -it --rm --gpus all --privileged -v /dev:/dev -v ~/logdir:/logdir img \
docker run -it --rm --privileged -v /dev:/dev -v ~/logdir:/logdir img \
  sh -c 'ldconfig; sh scripts/xvfb_run.sh python agents/ppo/train.py \
    --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
    --configs crafter'
