#!/bin/sh
set -eu

pip install ale_py
pip install autorom[accept-rom-license]

# apt-get update
# apt-get install -y cmake
# apt-get install -y unrar
# apt-get install -y wget
# apt-get clean

# pip3 install atari-py==0.2.9

# mkdir roms && cd roms
# wget -L -nv http://www.atarimania.com/roms/Roms.rar
# unrar x -o+ Roms.rar
# python3 -m atari_py.import_roms ROMS
# cd .. && rm -rf roms
