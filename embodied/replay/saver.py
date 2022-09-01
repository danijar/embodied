import concurrent.futures
from collections import defaultdict, deque
from functools import partial as bind

import embodied

from . import chunk as chunklib


class Saver:

  def __init__(self, directory, chunks=1024):
    self.directory = embodied.Path(directory)
    self.directory.mkdirs()
    self.chunks = chunks
    self.buffers = defaultdict(bind(chunklib.Chunk, chunks))
    self.saver = concurrent.futures.ThreadPoolExecutor(16)
    self.promises = deque()
    self.loading = False

  def add(self, step, worker):
    if self.loading:
      return
    buffer = self.buffers[worker]
    buffer.append(step)
    if buffer.length >= self.chunks:
      self.buffers[worker] = buffer.successor = chunklib.Chunk(self.chunks)
      self.promises.append(self.saver.submit(buffer.save, self.directory))
      for promise in [x for x in self.promises if x.done()]:
        promise.result()
        self.promises.remove(promise)

  def save(self, wait=False):
    for buffer in self.buffers.values():
      if buffer.length:
        self.promises.append(self.saver.submit(buffer.save, self.directory))
    if wait:
      [x.result() for x in self.promises]
      self.promises.clear()

  def load(self, capacity=None, length=1):
    filenames = chunklib.Chunk.scan(self.directory, capacity, length - 1)
    if not filenames:
      return
    threads = min(len(filenames), 32)
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
      chunks = list(executor.map(chunklib.Chunk.load, filenames))
    streams = {}
    for chunk in reversed(sorted(chunks, key=lambda x: x.time)):
      if chunk.successor in streams:
        streams[chunk.uuid] = streams[chunk.successor]
      else:
        streams[chunk.uuid] = int(embodied.uuid())
    self.loading = True
    for chunk in chunks:
      stream = streams[chunk.uuid]
      for index in range(chunk.length):
        step = {k: v[index] for k, v in chunk.data.items()}
        yield step, stream
    self.loading = False
