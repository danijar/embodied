import collections
import time
import uuid

import numpy as np
import embodied


class Consecutive(embodied.Replay):

  def __init__(self, store, chunk=64, slots=512):
    self.store = store
    self.chunk = chunk
    self.slots = slots
    self.random = np.random.RandomState(seed=0)
    self.ongoing = collections.defaultdict(
        lambda: collections.defaultdict(list))

  def __len__(self):
    return self.store.steps

  @property
  def stats(self):
    return {f'replay_{k}': v for k, v in self.store.stats().items()}

  def add(self, tran, worker=0):
    if tran['is_first'] and not self.slots:
      self.ongoing[worker].clear()
    ep = self.ongoing[worker]
    [ep[k].append(v) for k, v in tran.items()]
    if (len(ep['is_first']) >= self.slots) if self.slots else tran['is_last']:
      self.add_traj(self.ongoing.pop(worker))

  def add_traj(self, traj):
    traj = {k: v for k, v in traj.items() if not k.startswith('log_')}
    traj = {k: embodied.convert(v) for k, v in traj.items()}
    self.store[uuid.uuid4().hex] = traj

  def dataset(self):
    source, index = None, None
    while True:
      chunk, missing = None, self.chunk
      while missing > 0:
        if not source or index >= len(source['action']):
          source, index = None, 0
          while source is None:
            source = self._sample()
            print('Waiting for episodes.')
            time.sleep(1)
        if not chunk:
          chunk = {k: v[index: index + missing] for k, v in source.items()}
        else:
          chunk = {
              k: np.concatenate([chunk[k], v[index: index + missing]], 0)
              for k, v in source.items()}
        index += missing
        missing = self.chunk - len(chunk['action'])
      assert missing == 0, missing
      yield chunk

  def _sample(self):
    keys = self.store.keys()
    if not keys:
      return None
    traj = self.store[keys[self.random.randint(0, len(keys))]]
    return traj
