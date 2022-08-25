import collections
import time
import uuid

import numpy as np
import embodied

from . import prios


class Prioritized(embodied.Replay):

  def __init__(
      self, store, chunk=64, slots=512, fraction=0.1, softmax=False, temp=1.0,
      constant=0.0, exponent=0.5):
    # TODO: We're currently not removing old episodes from the priority table
    # when the store is reaching its capacity.
    self.store = store
    self.chunk = chunk
    self.slots = slots
    self.random = np.random.RandomState(seed=0)
    self.ongoing = collections.defaultdict(
        lambda: collections.defaultdict(list))
    def aggregate(prios):
      if softmax:
        prios = np.exp(prios / temp)
        prios = np.maximum(prios + constant, 0)
      else:
        prios = np.abs(prios) ** exponent
      return np.convolve(prios, np.ones(chunk), 'valid')
    self.prios = prios.Priorities(aggregate, fraction)
    self.handed_out_keys = set()
    if softmax:
      self.cooldown = np.full(self.chunk, -np.inf, np.float64)
    else:
      self.cooldown = np.full(self.chunk, 0.0, np.float64)

  def __len__(self):
    return self.store.steps

  @property
  def stats(self):
    metrics = {f'replay_{k}': v for k, v in self.store.stats().items()}
    metrics.update(self.prios.stats)
    return metrics

  def add(self, tran, worker=0):
    if tran['is_first'] and not self.slots:
      self.ongoing[worker].clear()
    ep = self.ongoing[worker]
    [ep[k].append(v) for k, v in tran.items()]
    if (len(ep['is_first']) >= self.slots) if self.slots else tran['is_last']:
      self.add_traj(self.ongoing.pop(worker))

  def add_traj(self, traj):
    length = len(next(iter(traj.values())))
    if length < self.chunk:
      print(f'Skipping short trajectory of length {length}.')
      return
    traj = {k: v for k, v in traj.items() if not k.startswith('log_')}
    traj = {k: embodied.convert(v) for k, v in traj.items()}
    key = uuid.uuid4().hex
    self.store[key] = traj
    self.prios.add(key, np.full(length, np.inf, np.float64))

  def prioritize(self, keys, priorities):
    keys = np.asarray(keys, np.int64)[:, 0]  # Key is replicated over time dim.
    priorities = np.asarray(priorities, np.float64)
    assert priorities.shape == (len(keys), self.chunk), priorities.shape
    for key, priority in zip(keys, priorities):
      assert tuple(key.tolist()) in self.handed_out_keys, key
      key, index = self._decode(key)
      try:
        self.prios.update(key, index, priority)
      except KeyError:
        print('Received priorities for an episode that was already removed.')

  def dataset(self):
    while True:
      traj = self.sample()
      if traj is None:
        print('Waiting for episodes.')
        time.sleep(1)
        continue
      yield traj

  def sample(self):
    keys = self.store.keys()
    if not keys:
      return None
    key, index, prob = self.prios.sample()
    self.prios.update(key, index, self.cooldown)
    key = self._encode(key, index)
    self.handed_out_keys.add(tuple(key.tolist()))
    traj = self.store[keys[self.random.randint(0, len(keys))]]
    total = len(next(iter(traj.values())))
    lower = 0
    upper = total - self.chunk + 1
    index = self.random.randint(lower, upper)
    index = np.clip(index, 0, total - self.chunk)
    chunk = {k: traj[k][index: index + self.chunk] for k in traj.keys()}
    chunk['is_first'][0] = True
    chunk['key'] = np.repeat(key[None], self.chunk, axis=0)
    chunk['prob'] = np.repeat(prob[None], self.chunk, axis=0)
    return chunk

  def _encode(self, key, index):
    raw = uuid.UUID(key).bytes + index.to_bytes(8, 'big')
    return np.frombuffer(raw, np.int64)

  def _decode(self, key):
    assert key.dtype == np.int64, key.dtype
    raw = key.tobytes()
    key = uuid.UUID(bytes=raw[:16]).hex
    index = int.from_bytes(raw[16:], 'big')
    return key, index
