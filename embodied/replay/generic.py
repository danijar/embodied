import time
from collections import defaultdict, deque
from functools import partial as bind

import embodied
import numpy as np

from . import saver


class Generic:

  def __init__(
      self, length, overlap, capacity, remover, sampler, limiter, directory,
      chunks):
    assert overlap < length
    self.length = length
    self.overlap = overlap
    self.capacity = capacity
    self.remover = remover
    self.sampler = sampler
    self.limiter = limiter
    self.streams = defaultdict(bind(deque, maxlen=length))
    self.offsets = defaultdict(int)
    self.table = {}
    self.saver = directory and saver.Saver(directory, chunks)
    self.load()

  def __len__(self):
    return len(self.table)

  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
    stream = self.streams[worker]
    stream.append(step)
    self.saver and self.saver.add(step, worker)
    self.offsets[worker] += 1
    if len(stream) < self.length:
      return
    if self.offsets[worker] < self.length - self.overlap:
      return
    self.offsets[worker] = 0
    wait(self.limiter.want_insert, 'Replay insert is waiting')
    key = embodied.uuid()
    seq = tuple(stream)
    self.table[key] = seq
    self.remover[key] = seq
    self.sampler[key] = seq
    while self.capacity and len(self) > self.capacity:
      wait(self.limiter.want_remove, 'Replay remove is waiting')
      key = self.remover()
      del self.table[key]
      del self.remover[key]
      del self.sampler[key]

  def _sample(self):
    wait(self.limiter.want_sample, 'Replay asmple is waiting')
    seq = self.table[self.sampler()]
    seq = {k: [step[k] for step in seq] for k in seq[0]}
    seq = {k: embodied.convert(v) for k, v in seq.items()}
    return seq

  def dataset(self):
    while True:
      yield self._sample()

  def prioritize(self, ids, prios):
    if hasattr(self.sampler, 'prioritize'):
      self.sampler.prioritize(ids, prios)

  def save(self, wait=False):
    if not self.saver:
      return
    self.saver.save(wait)
    # return {
    #     'saver': self.saver.save(wait),
    #     # 'remover': self.remover.save(wait),
    #     # 'sampler': self.sampler.save(wait),
    #     # 'limiter': self.limiter.save(wait),
    # }

  def load(self, data=None):
    if not self.saver:
      return
    for step, worker in self.saver.load(self.capacity, self.length):
      self.add(step, worker)
    # self.remover.load(data['remover'])
    # self.sampler.load(data['sampler'])
    # self.limiter.load(data['limiter'])


def wait(predicate, message):
  counter = 0
  while not predicate():
    if counter % 100 == 0:
      print(message)
    time.sleep(0.1)
    counter += 1
