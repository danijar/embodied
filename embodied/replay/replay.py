import concurrent.futures
import time
from collections import defaultdict, deque
from functools import partial as bind

import embodied
import numpy as np

from . import chunk as chunklib
from . import limiters
from . import selectors


class Replay:

  def __init__(
      self, length, capacity=None, directory=None, chunksize=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0):
    assert not capacity or min_size <= capacity

    self.length = length
    self.capacity = capacity
    self.directory = directory and embodied.Path(directory)
    self.chunksize = chunksize

    self.sampler = selectors.Uniform(seed)
    if samples_per_insert:
      self.limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      self.limiter = limiters.MinSize(min_size)

    self.chunks = {}
    self.chunkrefs = defaultdict(int)

    self.items = {}
    self.fifo = deque()

    self.current = {}
    self.streams = defaultdict(deque)

    if self.directory:
      self.directory.mkdirs()
      self.workers = concurrent.futures.ThreadPoolExecutor(10)
      self.promises = {}

    self.metrics = {
        'samples': 0,
        'sample_wait_dur': 0,
        'sample_wait_count': 0,
        'inserts': 0,
        'insert_wait_dur': 0,
        'insert_wait_count': 0,
    }

  def __len__(self):
    return len(self.items)

  @property
  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    stats = {
        'size': len(self),
        # 'ram_gb': len(self) * self.itemsize / (1024 ** 3),
        'inserts': m['inserts'],
        'samples': m['samples'],
        'insert_wait_avg': ratio(m['insert_wait_dur'], m['inserts']),
        'insert_wait_frac': ratio(m['insert_wait_count'], m['inserts']),
        'sample_wait_avg': ratio(m['sample_wait_dur'], m['samples']),
        'sample_wait_frac': ratio(m['sample_wait_count'], m['samples']),
    }
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))

    if worker not in self.current:
      chunk = chunklib.Chunk(self.chunksize)
      self.chunks[chunk.uuid] = chunk
      self.current[worker] = (chunk, 0)
      self.chunkrefs[chunk.uuid] += 1

    chunk, index = self.current[worker]
    stream = self.streams[worker]
    chunk.append(step)
    stream.append((chunk.uuid, index))
    self.chunkrefs[chunk.uuid] += 1
    index += 1
    self.current[worker] = (chunk, index)

    if index == chunk.size:
      succ = chunklib.Chunk(self.chunksize)
      chunk.succ = succ.uuid
      self.chunks[succ.uuid] = succ
      self.current[worker] = (succ, 0)
      self.chunkrefs[chunk.uuid] -= 1
      self.chunkrefs[succ.uuid] += 1
      if self.directory:
        (worker in self.promises) and self.promises.pop(worker).result()
        self.promises[worker] = self.workers.submit(chunk.save, self.directory)

    if len(stream) >= self.length:
      dur = wait(self.limiter.want_insert, 'Replay insert is waiting')
      self.metrics['inserts'] += 1
      self.metrics['insert_wait_dur'] += dur
      self.metrics['insert_wait_count'] += int(dur > 0)
      chunkid, index = stream.popleft()
      self._insert(chunkid, index)

  def _sample(self):
    dur = wait(self.limiter.want_sample, 'Replay sample is waiting')
    self.metrics['samples'] += 1
    self.metrics['sample_wait_dur'] += dur
    self.metrics['sample_wait_count'] += int(dur > 0)

    chunkid, index = self.items[self.sampler()]
    chunk = self.chunks[chunkid]

    available = chunk.length - index
    if available >= self.length:
      seq = chunk.slice(index, self.length)
    else:
      parts = [chunk.slice(index, available)]
      remaining = self.length - available
      while remaining > 0:
        chunk = self.chunks[chunk.succ]
        take = min(remaining, chunk.length)
        parts.append(chunk.slice(0, take))
        remaining -= take
      seq = {
          k: np.concatenate([p[k] for p in parts], 0)
          for k in parts[0].keys()}

    if 'is_first' in seq:
      seq['is_first'] = seq['is_first'].copy()
      seq['is_first'][0] = True
    return seq

  def _insert(self, chunkid, index):
    # assert 0 <= index < self.chunks[chunkid].length
    itemid = embodied.uuid()
    self.items[itemid] = (chunkid, index)
    self.sampler[itemid] = (chunkid, index)
    self.fifo.append(itemid)
    while self.capacity and len(self.items) > self.capacity:
      self._remove()

  def _remove(self):
    self.limiter.want_remove()
    itemid = self.fifo.popleft()
    del self.sampler[itemid]
    chunkid, index = self.items.pop(itemid)
    self.chunkrefs[chunkid] -= 1
    if self.chunkrefs[chunkid] < 1:
      del self.chunkrefs[chunkid]
      del self.chunks[chunkid]
      # for chunk in self.chunks.values():
      #   assert chunk.succ != chunkid

  def dataset(self):
    while True:
      yield self._sample()

  def save(self, wait=False):
    if not self.directory:
      return
    promises = self.promises.copy()
    self.promises.clear()
    [promise.result() for promise in promises.values()]
    chunks = [chunk for chunk, index in self.current.values() if index]
    save = bind(chunklib.Chunk.save, directory=self.directory)
    promises = self.workers.map(save, chunks)
    wait and list(promises)
    return None

  def load(self, data=None, directory=None, amount=None):
    assert data is None
    directory = directory or self.directory
    amount = amount or self.capacity or np.inf
    if not directory:
      return
    revsorted = lambda x: list(reversed(sorted(list(x))))
    directory = embodied.Path(directory)
    names_loaded = revsorted(x.filename for x in self.chunks.values())
    names_ondisk = revsorted(x.name for x in directory.glob('*.npz'))

    names_all = revsorted(names_loaded + names_ondisk)
    uuids, succs, lens = zip(*[
        x.rsplit('.', 1)[0].split('-')[1:] for x in names_all])
    uuids = [embodied.uuid(x) for x in uuids]
    succs = [embodied.uuid(x) for x in succs]
    lens = {k: int(v) for k, v in zip(uuids, lens)}
    numitems = {}
    for uuid, succ in zip(uuids, succs):
      numitems[uuid] = lens[uuid] - self.length + 1 + lens.get(succ, 0)
    numitems = {k: np.clip(v, 0, lens[k]) for k, v in numitems.items()}

    uuids = [embodied.uuid(x.split('-')[1]) for x in names_ondisk]
    total = 0
    numchunks = 0
    for uuid in uuids:
      numchunks += 1
      total += numitems[uuid]
      if total >= amount:
        break

    load = bind(chunklib.Chunk.load, error='none')
    filenames = [directory / x for x in names_ondisk[:numchunks]]
    chunks = [x for x in self.workers.map(load, filenames) if x]
    self.chunks.update({x.uuid: x for x in chunks})
    for chunk in reversed(chunks):
      amount = numitems[chunk.uuid]
      self.chunkrefs[chunk.uuid] += amount
      for index in range(amount):
        self.limiter.want_load()
        self._insert(chunk.uuid, index)


def wait(predicate, message, sleep=0.001, notify=1.0):
  first = True
  start = time.time()
  notified = False
  while True:
    allowed, detail = predicate()
    duration = time.time() - start
    if allowed:
      return 0 if first else duration
    if not notified and duration >= notify:
      print(f'{message} ({detail})')
      notified = True
    time.sleep(sleep)
    first = False
