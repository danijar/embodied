import collections
from functools import partial as bind

from . import limiters
from . import selectors
from . import tables


class FixedLength:

  def __init__(
      self, capacity, length, chunks=1024, ratio=None, directory=None, seed=0):
    assert capacity >= length + chunks, (capacity, length, chunks)
    batch = length  # Approximation.
    self.capacity = capacity
    self.length = length
    self.chunks = chunks
    self.remover = selectors.Fifo()
    self.sampler = selectors.Uniform(seed)
    self.limiters = [limiters.MinSize(1)]  # Better batch?
    if ratio:
      tolerance = 10 * batch * length
      self.limiters.append(limiters.SamplesPerInsert(ratio, tolerance))
    if chunks:
      self.table = tables.ChunkTable(directory, chunks)
    else:
      self.table = tables.DictTable(directory)
    self.histories = collections.defaultdict(
        bind(collections.deque, maxlen=length))

  def __len__(self):
    return self.table.num_steps

  def add(self, step, worker=0):
    limiters.wait_to_insert(self.limiters)
    stepid = tables.make_id()
    self.table.add_step(stepid, step, worker)
    history = self.histories[worker]
    history.append(stepid)
    self._check_capacity()
    if len(history) >= self.length:
      itemid = tables.make_id()
      self.table.insert(itemid, history)
      self.remover.insert(itemid, history)
      self.sampler.insert(itemid, history)
      [l.insert(itemid, history) for l in self.limiters]
    while self.table.num_steps > self.capacity:
      itemid = self.remover.select()
      self.table.remove(itemid)
      self.remover.remove(itemid)
      self.sampler.remove(itemid)
      [l.remove(itemid) for l in self.limiters]

  def sample(self):
    limiters.wait_to_sample(self.limiters)
    itemid = self.sampler.select()
    seq = self.table.retrieve(itemid)
    [l.retrieve(itemid) for l in self.limiters]
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def dataset(self):
    while True:
      yield self.sample()

  def save(self):
    # TODO: save selectors and limitors if needed
    return {
        'table': self.table.save(),
    }

  def load(self, data):
    self.table.load(data['table'])
    for itemid, stepids in self.table.items.items():
      self.sampler.insert(itemid, stepids)
      self.remover.insert(itemid, stepids)
      # TODO
      # [l.insert(itemid, stepids) for l in self.limiters]

  def _check_capacity(self):
    workers = len(self.histories)
    needed = (self.chunks + self.length) * workers
    if self.capacity < needed:
      raise ValueError(
          f'Capacity of at least {needed} steps needed, or use shorter '
          f'sequences ({self.length}), shorter chunks ({self.chunks}), or '
          f'fewer workers ({workers}).')
