#!/bin/sh
set -eu

apt-get update
apt-get install -y qtbase5-dev
apt-get install -y cmake
apt-get clean

git clone https://github.com/openai/procgen.git && cd procgen
pip install -e .
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='coinrun')"
cd .. && rm -rf procgen
