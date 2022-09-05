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
