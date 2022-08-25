import collections
import time
import uuid

import numpy as np
import embodied


class FixedLength(embodied.Replay):

  def __init__(self, store, length, chunks=1024, seed=0):
    self.store = store
    self.length = length
    self.chunks = chunks
    self.rng = np.random.default_rng(seed)
    self.ongoing = collections.defaultdict(
        lambda: collections.defaultdict(list))

  def __len__(self):
    return self.store.steps + sum(len(x) for x in self.ongoing.values())

  @property
  def stats(self):
    return {f'replay_{k}': v for k, v in self.store.stats().items()}

  def add(self, step, worker=0):
    seq = self.ongoing[worker]
    [seq[k].append(v) for k, v in step.items() if not k.startswith('log_')]
    if self._len(seq) >= self.chunks:
      seq = self.ongoing.pop(worker)
      seq = {k: embodied.convert(v) for k, v in seq.items()}
      self.store[uuid.uuid4().hex] = seq

  def dataset(self):
    while True:
      seq = self.sample()
      if seq is None:
        print('Waiting for episodes:', self.stats)
        time.sleep(1)
        continue
      yield seq

  def sample(self):
    seq = self._chunk(propto=True)
    index = self.rng.integers(0, self._len(seq))
    seq = {k: v[index: index + self.length] for k, v in seq.items()}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    if self._len(seq) == self.length:
      return seq
    parts = [seq]
    missing = self.length - self._len(seq)
    while missing > 0:
      seq = self._chunk(propto=False)
      seq = {k: v[:missing] for k, v in seq.items()}
      missing -= self._len(seq)
      parts.append(seq)
    seq = {k: np.concatenate([p[k] for p in parts], 0) for k in seq.keys()}
    assert self._len(seq) == self.length, (
        self._len(seq), self.length,
        [self._len(part) for part in parts])
    return seq

  def _chunk(self, propto):
    keys = self.store.keys()
    ongoing = tuple(self.ongoing.values())
    if propto:
      lengths = [self.chunks] * len(keys) + [self._len(x) for x in ongoing]
      probs = np.array(lengths, np.float32)
      probs /= probs.sum()
      choice = self.rng.choice(np.arange(len(lengths)), p=probs)
    else:
      choice = self.rng.integers(0, len(keys) + len(ongoing))
    if choice < len(keys):
      seq = self.store[keys[choice]]
    else:
      seq = ongoing[choice - len(keys)]
      seq = {k: embodied.convert(v) for k, v in seq.items()}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def _len(self, seq):
    return len(next(iter(seq.values())))
