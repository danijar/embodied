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


class Prioritized(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, chunks=1024, **kwargs):
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
