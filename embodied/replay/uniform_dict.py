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
    self.items = indexdict.IndexDict()  # {itemid: [step]}
    self.fifo = deque()
    self.streams = defaultdict(bind(deque, maxlen=length))
    self.saver = directory and saver.Saver(directory, chunks)
    self.rng = np.random.default_rng(seed)
    # TODO
    # for step, stream in self.saver.load(self.capacity):
    #   self.add(step, stream)

  def __len__(self):
    return len(self.items)

  @property
  def stats(self):
    return {}

  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    self.directory and self.saver.add(step, worker)
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
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

  def save(self, wait=True):
    self.directory and self.saver.save(wait)

  def load(self):
    if self.directory:
      for step, stream in self.saver.load(self.capacity):
        self.add(step, stream)
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

  # def draw(self, width=79):
  #   lines = []
  #   allids = tuple(self.steps.keys()) + tuple(self.items.keys())
  #   grid = max(len(str(x)) for x in allids) + 1
  #   stepcols, cols = {}, []
  #   for stepid in self.steps.keys():
  #     stepcols[stepid] = len(cols)
  #     cols.append('|' + str(stepid).rjust(grid - 1, '_'))
  #   lines.append(' ' * (grid + 6) + '_' * (len(cols) * grid - 1))
  #   lines.append('Steps:' + ' ' * (grid - 1) + ''.join(cols) + '|')
  #   for itemid, steps in self.items.items():
  #     cols = ['|' + ' ' * (grid - 1)] * len(stepcols) + ['|']
  #     for step in steps:
  #       if step['id'] not in stepcols:
  #         cols.append(' ?')
  #       else:
  #         cols[stepcols[step['id']]] = '|' + '#' * (grid - 1)
  #     lines.append('Item ' + str(itemid).rjust(grid - 1) + ' ' + ''.join(cols))
  #   lines = [x[:width - 4] + ' ...' if len(x) > width else x for x in lines]
  #   return '\n' + '\n'.join(lines) + '\n'
