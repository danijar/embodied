import collections
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np
import pytest


REPLAYS_ALL = [
    embodied.replay.UniformDict,
    embodied.replay.UniformChunks,
]

REPLAYS_UNIFORM = [
    embodied.replay.UniformDict,
    embodied.replay.UniformChunks,
]


class TestReplay:

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  def test_multiple_keys(self, Replay):
    replay = Replay(length=5, capacity=10)
    for step in range(30):
      replay.add({'image': np.zeros((64, 64, 3)), 'action': np.zeros(12)})
    seq = next(iter(replay.dataset()))
    assert set(seq.keys()) == {'id', 'image', 'action'}
    assert seq['id'].shape == (5, 16)
    assert seq['image'].shape == (5, 64, 64, 3)
    assert seq['action'].shape == (5, 12)

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  @pytest.mark.parametrize(
      'length,workers,capacity',
      [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)])
  def test_capacity_exact(self, Replay, length, workers, capacity):
    replay = Replay(length, capacity)
    for step in range(30):
      for worker in range(workers):
        replay.add({'step': step}, worker)
      target = min(workers * max(0, (step + 1) - length + 1), capacity)
      assert len(replay) == target

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  @pytest.mark.parametrize(
      'length,workers,capacity',
      [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)])
  def test_sample_sequences(self, Replay, length, workers, capacity):
    replay = Replay(length, capacity)
    for step in range(30):
      for worker in range(workers):
        replay.add({'step': step, 'worker': worker}, worker)
    dataset = iter(replay.dataset())
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] - seq['step'][0] == np.arange(length)).all()
      assert (seq['worker'] == seq['worker'][0]).all()

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  @pytest.mark.parametrize(
      'length,capacity', [(1, 1), (2, 2), (5, 10), (1, 2), (5, 15), (2, 20)])
  def test_sample_single(self, Replay, length, capacity):
    replay = Replay(length, capacity)
    for step in range(length):
      replay.add({'step': step})
    dataset = iter(replay.dataset())
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] == np.arange(length)).all()

  @pytest.mark.parametrize('Replay', REPLAYS_UNIFORM)
  def test_sample_uniform(self, Replay):
    replay = Replay(capacity=20, length=5, seed=0)
    for step in range(7):
      replay.add({'step': step})
    histogram = collections.defaultdict(int)
    dataset = iter(replay.dataset())
    for _ in range(100):
      seq = next(dataset)
      histogram[seq['step'][0]] += 1
    histogram = tuple(histogram.values())
    assert len(histogram) == 3
    assert histogram[0] > 20
    assert histogram[1] > 20
    assert histogram[2] > 20

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  def test_workers_simple(self, Replay):
    replay = Replay(length=2, capacity=20)
    replay.add({'step': 0}, worker=0)
    replay.add({'step': 1}, worker=1)
    replay.add({'step': 2}, worker=0)
    replay.add({'step': 3}, worker=1)
    dataset = iter(replay.dataset())
    for _ in range(10):
      seq = next(dataset)
      assert tuple(seq['step']) in ((0, 2), (1, 3))

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  def test_workers_random(self, Replay, length=4, capacity=30):
    rng = np.random.default_rng(seed=0)
    replay = Replay(length, capacity)
    streams = {i: iter(range(10)) for i in range(3)}
    for _ in range(40):
      worker = int(rng.integers(0, 3, ()))
      try:
        step = {'step': next(streams[worker]), 'stream': worker}
        replay.add(step, worker=worker)
      except StopIteration:
        pass
    histogram = collections.defaultdict(int)
    dataset = iter(replay.dataset())
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] - seq['step'][0] == np.arange(length)).all()
      assert (seq['stream'] == seq['stream'][0]).all()
      histogram[int(seq['stream'][0])] += 1
    assert all(count > 0 for count in histogram.values())

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  @pytest.mark.parametrize(
      'length,workers,capacity',
      [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)])
  def test_worker_delay(self, Replay, length, workers, capacity):
    # embodied.uuid.reset(debug=True)
    replay = Replay(length, capacity)
    rng = np.random.default_rng(seed=0)
    streams = [iter(range(10)) for _ in range(workers)]
    while streams:
      try:
        worker = rng.integers(0, len(streams))
        replay.add({'step': next(streams[worker])}, worker)
      except StopIteration:
        del streams[worker]

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  @pytest.mark.parametrize(
      'length,capacity,chunks', [(1, 1, 1), (3, 10, 5), (5, 100, 12)])
  def test_restore_exact(self, tmpdir, Replay, length, capacity, chunks):
    embodied.uuid.reset(debug=True)
    assert len(list(embodied.Path(tmpdir).glob('*.npz'))) == 0
    replay = Replay(length, capacity, directory=tmpdir, chunks=chunks)
    for step in range(30):
      replay.add({'step': step})
    num_items = np.clip(30 - length + 1, 0, capacity)
    assert len(replay) == num_items
    replay.save(wait=True)
    filenames = list(embodied.Path(tmpdir).glob('*.npz'))
    lengths = [int(x.stem.split('-')[3]) for x in filenames]
    assert len(filenames) == (int(np.ceil(30 / chunks)))
    assert sum(lengths) == 30
    assert all(1 <= x <= chunks for x in lengths)
    replay = Replay(length, capacity, tmpdir, chunks)
    assert sorted(embodied.Path(tmpdir).glob('*.npz')) == sorted(filenames)
    assert len(replay) == num_items
    dataset = iter(replay.dataset())
    for _ in range(10):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_ALL)
  @pytest.mark.parametrize('workers', [1, 2, 5])
  @pytest.mark.parametrize(
      'length,capacity,chunks', [(1, 1, 1), (3, 10, 5), (5, 100, 12)])
  def test_restore_workers(
      self, tmpdir, Replay, workers, length, capacity, chunks):
    capacity *= workers
    replay = Replay(length, capacity, directory=tmpdir, chunks=chunks)
    for step in range(50):
      for worker in range(workers):
        replay.add({'step': step}, worker)
    num_items = np.clip((50 - length + 1) * workers, 0, capacity)
    assert len(replay) == num_items
    replay.save(wait=True)
    filenames = list(embodied.Path(tmpdir).glob('*.npz'))
    lengths = [int(x.stem.split('-')[3]) for x in filenames]
    assert sum(lengths) == 50 * workers
    replay = Replay(length, capacity, directory=tmpdir, chunks=chunks)
    assert len(replay) == num_items
    dataset = iter(replay.dataset())
    for _ in range(10):
      assert len(next(dataset)['step']) == length
