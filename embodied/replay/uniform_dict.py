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
    self.table = indexdict.IndexDict()  # {itemid: [step]}
    self.workers = defaultdict(bind(deque, maxlen=length))
    self.order = deque()
    self.rng = np.random.default_rng(seed)
    # self.remover = selectors.Fifo()
    # self.sampler = selectors.Uniform()
    self.saver = directory and saver.Saver(directory, chunks)
    self.load()

  def __len__(self):
    return len(self.table)

  @property
  def stats(self):
    return {}

  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
    self.saver and self.saver.add(step, worker)
    stream = self.workers[worker]
    stream.append(step)
    if len(stream) >= self.length:
      itemid = embodied.uuid()
      self.table[itemid] = tuple(stream)
      self.order.append(itemid)
    self._ensure_capacity()

  def dataset(self):
    while True:
      counter = 0
      while not self.table:
        if counter % 100 == 0:
          print('Replay is waiting for insertions.')
        time.sleep(0.1)
        counter += 1
      yield self._sample()

  def save(self, wait=False):
    self.saver and self.saver.save(wait)

  def load(self, data=None):
    if self.saver:
      for step, worker in self.saver.load(self.capacity, self.length):
        self.add(step, worker)
    self.workers.clear()
    self._ensure_capacity()

  def _sample(self):
    seq = self.table[self.rng.integers(0, len(self.table)).item()]
    seq = {k: embodied.convert([step[k] for step in seq]) for k in seq[0]}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def _ensure_capacity(self):
    needed = self.length * len(self.workers)
    if self.capacity and self.capacity < needed:
      raise ValueError(
          f'Need at least capacity of {needed} steps to support '
          f'sequence length {self.length} with {len(self.workers)} workers.')
    while self.capacity and len(self) > self.capacity:
      del self.table[self.order.popleft()]
