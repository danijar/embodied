import io
import time
from collections import defaultdict, deque
from functools import partial as bind

import embodied
import numpy as np

from . import indexdict


class UniformChunks:

  def __init__(
      self, length, capacity=None, directory=None, chunks=1024, seed=0):
    assert not capacity or length <= capacity, (length, capacity)
    assert length <= chunks, (length, chunks)
    self.length = length
    self.capacity = capacity
    self.directory = directory and embodied.Path(directory)
    self.chunks = chunks
    self.items = indexdict.IndexDict()  # {itemid: (Chunk, start, length)}
    self.fifo = deque()
    self.ongoing = {}  # {worker: Chunk}
    self.histories = defaultdict(bind(deque, maxlen=length))
    # TODO: use worker pool
    # self.saver = directory and embodied.Worker(self._save)
    self.saver = directory and embodied.Worker(self._save, 'blocking')
    self.rng = np.random.default_rng(seed)
    self.specs = None
    # TODO
    # self._load()

  def __len__(self):
    return len(self.items)

  @property
  def stats(self):
    return {}

  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step = {k: embodied.convert(v) for k, v in step.items()}
    stepid = embodied.uuid(step.get('id'))
    step['id'] = np.asarray(stepid)
    self._check_specs(step)

    if worker not in self.ongoing:
      chunk = Chunk(self.specs, self.chunks)
      self.ongoing[worker] = chunk
      self._check_capacity()

    chunk = self.ongoing[worker]
    index = len(chunk)
    chunk.append(step)

    history = self.histories[worker]
    history.append((stepid, chunk, index))
    if len(history) >= self.length:
      stepid, chunk2, start = history[0]
      key = embodied.uuid()
      self.items[key] = (chunk2, start, self.length)
      self.fifo.append(key)
      assert chunk2 is chunk or chunk2.nxt is chunk, (chunk2, chunk)

    if len(chunk) >= self.chunks:
      new = Chunk(self.specs, self.chunks)
      chunk.nxt = new
      self.ongoing[worker] = new
      self.directory and self.saver(chunk)

    while self.capacity and len(self) > self.capacity:
      self._remove_oldest()

  def dataset(self):
    while True:
      counter = 0
      while not len(self):
        if counter % 100 == 0:
          print('Replay is waiting for insertions.')
        time.sleep(0.1)
        counter += 1
      yield self._sample()

  def save(self, wait=True):
    if self.directory:
      for chunk in self.ongoing.values():
        if len(chunk):
          self.saver(chunk)
    wait and self.saver.wait()

  def load(self):
    filenames, total = [], 0
    for filename in reversed(sorted(self.directory.glob('*.npz'))):
      filenames.append(filename)
      total += int(filename.stem.split('-')[3])
      # TODO: This is capacity in items not steps so we may need to load more!
      if total > self.capacity:
        break
    chunks, nexts = {}, {}
    # Sorting is important to insert in temporal order, because we rely on
    # insertion order for discarding items when we are over capacity.
    # TODO: Load chunks with a thread pool.
    for filename in sorted(filenames):
      _, id, nxt, length  = filename.stem.split('-')[:4]
      id, nxt, length = embodied.uuid(id), embodied.uuid(nxt), int(length)
      with filename.open('rb') as f:
        data = np.load(f)
        data = {k: data[k][:length] for k in data.keys()}
      self._check_specs({k: v[0] for k, v in data.items()})
      chunks[id] = Chunk(self.specs, self.chunks, id, data, length)
      nexts[id] = nxt
    for chunk in chunks.values():
      chunk.nxt = chunks.get(nexts[chunk.id])
    for chunk in chunks.values():
      amount = chunk.length if chunk.nxt else chunk.length - self.length + 1
      for index in range(amount):
        self.items[embodied.uuid()] = (chunk, index, self.length)
    while self.capacity and len(self) > self.capacity:
      self._remove_oldest()

  def _sample(self):
    chunk, start, length = self.items[
        self.rng.integers(0, len(self.items)).item()]
    seq = chunk.get(start, start + length)
    missing = start + length - len(chunk)
    if missing > 0:
      part = chunk.nxt.get(0, missing)
      seq = {k: np.concatenate([v, part[k]], 0) for k, v in seq.items()}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def _remove_oldest(self):
    del self.items[self.fifo.popleft()]

  def _check_specs(self, step):
    if not self.specs:
      self.specs = {k: (v.dtype, v.shape) for k, v in step.items()}
    assert tuple(step.keys()) == tuple(self.specs.keys())
    for key, value in step.items():
      dtype, shape = self.specs[key]
      assert value.shape == shape
      assert value.dtype == dtype

  def _check_capacity(self):
    needed = self.length * len(self.ongoing)
    if self.capacity and self.capacity < needed:
      raise ValueError(
          f'Need at least capacity of {needed} steps to support '
          f'sequence length {self.length} with {len(self.ongoing)} workers.')

  def _save(self, chunk):
    nxt = chunk.nxt.id if chunk.nxt else embodied.uuid(0)
    filename = f'{chunk.time}-{chunk.id}-{nxt}-{chunk.length}.npz'
    filename = self.directory / filename
    with io.BytesIO() as stream:
      np.savez_compressed(stream, **chunk.data)
      stream.seek(0)
      filename.write(stream.read(), mode='wb')
    print(f'Saved chunk: {filename.name}')


class Chunk:

  def __init__(self, specs, size, id=None, data=None, length=None):
    self.time = time.strftime('%Y%m%dT%H%M%S', time.gmtime(time.time()))
    self.id = embodied.uuid(id)
    self.nxt = None  # Chunk
    self.data = (
        {k: np.empty((size,) + s, d) for k, (d, s) in specs.items()}
        if data is None else data)
    self.length = length or 0

  def __repr__(self):
    return (
        f'Chunk(id={self.id}, '
        f'next={self.nxt and self.nxt.id}, '
        f'len={self.length})')

  def __len__(self):
    return self.length

  def __bool__(self):
    return True

  def append(self, step):
    for key, value in step.items():
      self.data[key][self.length] = value
    self.length += 1

  def get(self, start, stop):
    return {k: v[start: min(stop, self.length)] for k, v in self.data.items()}
