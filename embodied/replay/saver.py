import io
from collections import defaultdict
from datetime import datetime

import embodied
import numpy as np


# TODO: Merge this back into the replay buffers. It will be much faster in the
# case of the chunked buffer.

class Saver:

  def __init__(self, directory, chunks=1024):
    self.directory = embodied.Path(directory)
    self.directory.mkdirs()
    self.chunks = chunks
    self.savebuffers = defaultdict(SaveBuffer)  # {worker: SaveBuffer}
    self.saver = embodied.Worker(self._save)
    self.loading = False

  def add(self, step, worker):
    if self.loading:
      return
    buffer = self.savebuffers[worker]
    buffer.append(step)
    if buffer.length >= self.chunks:
      new = SaveBuffer()
      buffer = self.savebuffers.pop(worker)
      buffer.next = new
      self.saver(buffer)
      self.savebuffers[worker] = new

  def save(self, wait=False):
    for worker, buffer in tuple(self.savebuffers.items()):
      if len(buffer):
        new = SaveBuffer()
        buffer = self.savebuffers.pop(worker)
        buffer.next = new
        self.saver(buffer)

        self.savebuffers[worker] = SaveBuffer(prev=buffer.id)
    wait and self.saver.wait()

  def load(self, capacity=None):
    filenames, prevs = self._find(capacity)
    streams = self._sort(prevs)
    print(streams)
    self.loading = True
    yield from self._read(streams, filenames)
    self.loading = False

  def _save(self, buffer):
    filename = f'{buffer.time}-{buffer.id}-{buffer.prev}-{buffer.length}.npz'
    filename = self.directory / filename
    data = {k: embodied.convert(v) for k, v in buffer.data.items()}
    with io.BytesIO() as stream:
      np.savez_compressed(stream, **data)
      stream.seek(0)
      filename.write(stream.read(), mode='wb')
    print(f'Saved buffer: {filename.name}')

  def _find(self, capacity):
    filenames, prevs, total = {}, {}, 0
    for filename in reversed(sorted(self.directory.glob('*.npz'))):
      _, bid, prev, length = filename.stem.split('-')[:4]
      bid = embodied.uuid(bid)
      prev = embodied.uuid(prev)
      length = int(length)
      filenames[bid] = filename
      prevs[bid] = prev
      total += length
      if capacity and total >= capacity:
        break
    return filenames, prevs

  def _sort(self, prevs):
    streams = {}
    while prevs:
      for bid, prev in list(prevs.items()):
        if prev in streams:
          streams[bid] = streams[prev]
          del prevs[bid]
        elif not prev or prev not in prevs:
          streams[bid] = embodied.uuid()
          del prevs[bid]
    return streams

  def _read(self, streams, filenames):
    for bid, stream in streams.items():
      with filenames[bid].open('rb') as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
      length = len(next(iter(data.values())))
      for index in range(length):
        step = {k: v[index] for k, v in data.items()}
        yield step, stream


class SaveBuffer:

  def __init__(self, prev=None):
    self.time = datetime.now().strftime("%Y%m%dT%H%M%SF%f")
    self.id = embodied.uuid()
    self.prev = prev or embodied.uuid(0)
    self.data = defaultdict(list)
    self.length = 0

  def append(self, step):
    [self.data[k].append(v) for k, v in step.items()]
    self.length += 1

  def __len__(self):
    return self.length
