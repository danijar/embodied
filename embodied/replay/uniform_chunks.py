import concurrent.futures
import time
from collections import defaultdict, deque
from functools import partial as bind

import embodied
import numpy as np

from . import chunk as chunklib
from . import indexdict


class UniformChunks:

  def __init__(
      self, length, capacity=None, directory=None, chunks=1024, seed=0):
    assert not capacity or length <= capacity, (length, capacity)
    assert length <= chunks, (length, chunks)
    self.length = length
    self.capacity = capacity
    self.directory = directory
    self.chunks = chunks
    self.items = indexdict.IndexDict()  # {itemid: (Chunk, start, length)}
    self.ongoing = defaultdict(bind(chunklib.Chunk, chunks))  # {worker: Chunk}
    self.histories = defaultdict(bind(deque, maxlen=length))
    self.saver = concurrent.futures.ThreadPoolExecutor(16)
    self.promises = deque()
    self.fifo = deque()
    self.rng = np.random.default_rng(seed)
    self.load(None)

  def __len__(self):
    return len(self.items)

  @property
  def stats(self):
    return {}

  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
    chunk = self.ongoing[worker]
    chunk.append(step)
    self._check_capacity()
    history = self.histories[worker]
    history.append((chunk, chunk.length - 1))
    if len(history) >= self.length:
      itemid = embodied.uuid()
      self.items[itemid] = (*history[0], self.length)
      self.fifo.append(itemid)
    if chunk.length >= self.chunks:
      self.ongoing[worker] = chunk.successor = chunklib.Chunk(self.chunks)
      if self.directory:
        self.promises.append(self.saver.submit(chunk.save, self.directory))
        self._check_errors()
    while self.capacity and len(self) > self.capacity:
      del self.items[self.fifo.popleft()]

  def dataset(self):
    while True:
      counter = 0
      while not self.items:
        if counter % 100 == 0:
          print('Replay is waiting for insertions.')
        time.sleep(0.1)
        counter += 1
      yield self._sample()

  def save(self, wait=False):
    if not self.directory:
      return
    for chunk in self.ongoing.values():
      if chunk.length:
        self.promises.append(self.saver.submit(chunk.save, self.directory))
    if wait:
      [x.result() for x in self.promises]
      self.promises.clear()

  def load(self, data=None):
    self.ongoing.clear()
    if not self.directory:
      return
    filenames = chunklib.Chunk.scan(
        self.directory, self.capacity, self.length - 1)
    if not filenames:
      return
    threads = min(len(filenames), 32)
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
      chunks = list(executor.map(chunklib.Chunk.load, filenames))
    mapping = {x.uuid: x for x in chunks}
    for chunk in sorted(chunks, key=lambda x: x.time):
      chunk.successor = mapping.get(chunk.successor, None)
      total = max(0, chunk.length - self.length + 1)
      if chunk.successor:
        total = min(total + chunk.successor.length, chunk.length)
      for start in range(total):
        itemid = embodied.uuid()
        self.items[itemid] = (chunk, start, self.length)
        self.fifo.append(itemid)
    while self.capacity and len(self) > self.capacity:
      del self.items[self.fifo.popleft()]

  def _sample(self):
    choice = self.rng.integers(0, len(self.items)).item()
    chunk, start, length = self.items[choice]
    seq = {k: v[start: start + length] for k, v in chunk.data.items()}
    missing = start + length - chunk.length
    if missing > 0:
      extension = {k: v[:missing] for k, v in chunk.successor.data.items()}
      seq = {k: np.concatenate([v, extension[k]], 0) for k, v in seq.items()}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def _check_errors(self):
    for promise in [x for x in self.promises if x.done()]:
      promise.result()
      self.promises.remove(promise)

  def _check_capacity(self):
    needed = self.length * len(self.ongoing)
    if self.capacity and self.capacity < needed:
      raise ValueError(
          f'Need at least capacity of {needed} steps to support '
          f'sequence length {self.length} with {len(self.ongoing)} workers.')
