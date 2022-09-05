import numpy as np

from . import generic
from . import selectors
from . import limiters


class Uniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, chunks=1024, seed=0):
    super().__init__(
        length=length,
        overlap=length - 1,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiters.MinSize(1),
        directory=directory,
        chunks=chunks,
    )


class Queue(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, overlap=0, chunks=1024):
    super().__init__(
        length=length,
        overlap=overlap,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Fifo(),
        limiter=limiters.Queue(capacity),
        directory=directory,
        chunks=chunks,
        max_times_sampled=1,
    )


class UniformWithOnline:

  def __init__(
      self, length,
      capacity=None,
      directory=None,
      batch=None,
      online_fraction=0.1,
      online_overlap=0,
      offline_minsize=1,
      chunks=1024,
      seed=0,
  ):
    assert batch
    self.online_fraction = online_fraction
    self.rng = np.random.default_rng(seed)
    self.offline = generic.Generic(
        length=length,
        overlap=length - 1,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiters.MinSize(offline_minsize),
        directory=directory,
        chunks=chunks,
    )
    queue_size = 3 * np.ceil(batch * online_fraction)
    self.online = generic.Generic(
        length=length,
        overlap=online_overlap,
        capacity=queue_size,
        remover=selectors.Fifo(),
        sampler=selectors.Fifo(),
        limiter=limiters.Queue(queue_size),
        directory=None,
        max_times_sampled=1,
    )

  def __len__(self):
    return len(self.offline)

  def add(self, step, worker=0):
    self.offline.add(step, worker)
    self.online.add(step, worker)

  def _sample(self):
    if self.rng.uniform() < self.online_fraction:
      return self.online._sample()
    else:
      return self.offline._sample()

  def dataset(self):
    while True:
      yield self._sample()

  def prioritize(self, ids, prios):
    pass

  def save(self, wait=False):
    return self.offline.save(wait)

  def load(self, data=None):
    self.offline.load(data)


class Prioritized(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, chunks=1024, **kwargs):
    # TODO: Currently too slow to be really usedful.
    super().__init__(
        length=length,
        overlap=length - 1,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Prioritized(**kwargs),
        limiter=limiters.MinSize(1),
        directory=directory,
        chunks=chunks,
    )
