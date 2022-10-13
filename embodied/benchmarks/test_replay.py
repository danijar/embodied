import pathlib
import sys
import time
from functools import partial as bind

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np
import pytest


SLACK = 1.3

UniformPrioritized = bind(
    embodied.replay.Prioritized,
    exponent=0.0, initial=1.0, zero_on_sample=False)

REPLAYS_ALL = [
    # ('UniformDict', embodied.replay.UniformDict),
    # ('UniformChunks', embodied.replay.UniformChunks),
    ('Uniform', embodied.replay.Uniform),
    ('Prioritized1', bind(embodied.replay.Prioritized, zero_on_sample=False)),
    ('Prioritized2', bind(UniformPrioritized, branching=16)),
    ('Prioritized3', bind(UniformPrioritized, branching=100)),
    # ('Prioritized4', bind(UniformPrioritized, branching=1000)),
]

STEP = {
    'image': np.zeros((64, 64, 3), np.uint8),
    'vector': np.zeros(1024, np.float32),
    'action': np.zeros(12, np.float32),
    'is_first': np.array(False),
    'is_last': np.array(False),
    'is_terminal': np.array(False),
}


class TestReplay:

  @pytest.mark.parametrize('name,Replay', REPLAYS_ALL)
  def test_speed(self, name, Replay, inserts=2e5, workers=8, samples=1e5):
    print('')
    initial = time.time()
    replay = Replay(length=32, capacity=1e5, chunks=1024)
    start = time.time()
    for step in range(int(inserts / workers)):
      for worker in range(workers):
        replay.add(self._make_step(), worker)
    duration = time.time() - start
    print(name, 'inserts/sec:', int(inserts / duration))
    start = time.time()
    dataset = iter(replay.dataset())
    for _ in range(int(samples)):
      seq = next(dataset)
      replay.prioritize(seq['id'], [1.0] * len(seq['id']))
    duration = time.time() - start
    print(name, 'samples/sec:', int(samples / duration))
    print(name, 'total duration:', time.time() - initial)

  @pytest.mark.parametrize('chunks', [64, 128, 256, 512, 1024, 2048, 4096])
  def test_chunk_size(self, chunks, inserts=2e5, workers=8, samples=2e5):
    print('')
    initial = time.time()
    replay = embodied.replay.UniformChunks(length=64, chunks=chunks)
    start = time.time()
    for step in range(int(inserts / workers)):
      for worker in range(workers):
        replay.add(self._make_step(), worker)
    duration = time.time() - start
    print('chunks', chunks, 'inserts/sec:', int(inserts / duration))
    start = time.time()
    dataset = iter(replay.dataset())
    for _ in range(int(samples)):
      next(dataset)
    duration = time.time() - start
    print('chunks', chunks, 'samples/sec:', int(samples / duration))
    print('chunks', chunks, 'total duration:', time.time() - initial)

  @pytest.mark.parametrize('name,Replay', REPLAYS_ALL)
  def test_removal(self, name, Replay, inserts=1e6, workers=1):
    print('')
    replay = Replay(length=32, capacity=1e5, chunks=1024)
    start = time.time()
    for step in range(int(inserts)):
        replay.add(self._make_step())
    duration = time.time() - start
    print(name, 'inserts/sec:', int(inserts / duration))

  def _make_step(self):
    return STEP  # TODO
    return {
        'image': np.zeros((64, 64, 3), np.uint8),
        'vector': np.zeros(1024, np.float32),
        'action': np.zeros(12, np.float32),
        'is_first': np.array(False),
        'is_last': np.array(False),
        'is_terminal': np.array(False),
    }


if __name__ == '__main__':
  # import cProfile
  # cProfile.run(
  #     'TestReplay().test_speed("", UniformPrioritized, inserts=2e4, samples=1e4)',
  #     sort='tottime')
  TestReplay().test_speed("", UniformPrioritized, inserts=2e4, samples=1e4)
  # TestReplay().test_speed("", embodied.replay.Uniform, inserts=2e4, samples=1e4)
