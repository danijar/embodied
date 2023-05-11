import io
from datetime import datetime

import embodied
import numpy as np


class Chunk:

  __slots__ = ('time', 'uuid', 'succ', 'length', 'size', 'data', 'modified')

  def __init__(self, size=1024):
    now = datetime.now()
    self.time = now.strftime("%Y%m%dT%H%M%S") + f'F{now.microsecond:06d}'
    self.uuid = embodied.uuid()
    self.succ = embodied.uuid(0)
    self.length = 0
    self.size = size
    self.data = None
    self.modified = False

  def __repr__(self):
    return f'Chunk({self.filename})'

  @property
  def filename(self):
    return f'{self.time}-{str(self.uuid)}-{str(self.succ)}-{self.length}.npz'

  def append(self, step):
    assert self.length < self.size
    if not self.data:
      example = {k: embodied.convert(v) for k, v in step.items()}
      self.data = {
          k: np.empty((self.size, *v.shape), v.dtype)
          for k, v in example.items()}
    for key, value in step.items():
      self.data[key][self.length] = value
    self.length += 1
    self.modified = True

  def slice(self, index, length):
    assert 0 <= index and index + length <= self.length
    return {k: v[index: index + length] for k, v in self.data.items()}

  def save(self, directory):
    if not self.modified:
      return
    self.modified = False
    filename = embodied.Path(directory) / self.filename
    data = {k: v[:self.length] for k, v in self.data.items()}
    with io.BytesIO() as stream:
      np.savez_compressed(stream, **data)
      stream.seek(0)
      filename.write(stream.read(), mode='wb')
    print(f'Saved chunk: {filename.name}')

  @classmethod
  def load(cls, filename, error='raise'):
    assert error in ('raise', 'none')
    time, uuid, succ, length = filename.stem.split('-')
    length = int(length)
    try:
      with embodied.Path(filename).open('rb') as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
    except Exception as e:
      print(f'Error loading chunk {filename}: {e}')
      if error == 'raise':
        raise
      else:
        return None
    chunk = cls(length)
    chunk.time = time
    chunk.uuid = embodied.uuid(uuid)
    chunk.succ = embodied.uuid(succ)
    chunk.length = length
    chunk.data = data
    return chunk
