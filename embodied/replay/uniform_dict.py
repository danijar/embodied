import time
from collections import defaultdict, deque
from functools import partial as bind

import embodied
import numpy as np

from . import saver
from . import indexdict


class UniformDict:

  def __init__(
      self, length, capacity=None, directory=None, chunks=1024, seed=0):
    assert not capacity or length <= capacity, (length, capacity)
    self.length = length
    self.capacity = capacity
    self.directory = directory
    self.chunks = chunks
    self.items = indexdict.IndexDict()  # {itemid: [step]}
    self.streams = defaultdict(bind(deque, maxlen=length))
    self.saver = directory and saver.Saver(directory, chunks)
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
    self.directory and self.saver.add(step, worker)
    stream = self.streams[worker]
    stream.append(step)
    self._check_capacity()
    if len(stream) >= self.length:
      itemid = embodied.uuid()
      self.items[itemid] = tuple(stream)
      self.fifo.append(itemid)
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
    self.directory and self.saver.save(wait)

  def load(self, data=None):
    self.streams.clear()
    if not self.directory:
      return
    for step, worker in self.saver.load(self.capacity, self.length):
      self.add(step, worker)
    self.streams.clear()
    while self.capacity and len(self) > self.capacity:
      del self.items[self.fifo.popleft()]

  def _sample(self):
    seq = self.items[self.rng.integers(0, len(self.items)).item()]
    for step in seq:  # Small optimization.
      step['id'] = step['id'].__array__()
    seq = {k: embodied.convert([step[k] for step in seq]) for k in seq[0]}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def _check_capacity(self):
    needed = self.length * len(self.streams)
    if self.capacity and self.capacity < needed:
      raise ValueError(
          f'Need at least capacity of {needed} steps to support '
          f'sequence length {self.length} with {len(self.streams)} workers.')
