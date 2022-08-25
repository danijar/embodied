import collections
import io
import time
import uuid

import embodied
import numpy as np

# TODO: If we want to speed up the replay buffers, the uuids and
# concatenating sequence parts of subsequent chunks are the main bottlenecks.
# These are the uuid.uuid4() calls and np.concatenate() calls in this file.


MODE = 'int'
COUNT = 0
def reset_ids(mode='int'):
  global MODE, COUNT
  MODE = mode
  COUNT = 0


def make_id():
  global MODE, COUNT
  if MODE == 'int':
    # TODO: benchmark: hex, bytes, int
    return uuid.uuid4().int
  elif MODE == 'debug':
    COUNT += 1
    return COUNT
  else:
    raise NotImplementedError(MODE)


class DictTable:

  def __init__(self, directory=None):
    self.directory = directory and embodied.Path(directory)
    self.steps = {}
    self.items = {}
    self.stepitems = collections.defaultdict(set)

  @property
  def num_steps(self):
    return len(self.steps)

  @property
  def num_items(self):
    return len(self.items)

  def add_step(self, stepid, npdict, worker):
    self.steps[stepid] = npdict

  def insert(self, itemid, stepids):
    self.items[itemid] = tuple(stepids)
    for stepid in stepids:
      self.stepitems[stepid].add(itemid)

  def retrieve(self, itemid):
    steps = [self.steps[stepid] for stepid in self.items[itemid]]
    return {
        k: embodied.convert([steps[i][k] for i in range(len(steps))])
        for k in steps[0].keys()}

  def remove(self, itemid):
    stepids = self.items.pop(itemid)
    for stepid in stepids:
      stepitems = self.stepitems[stepid]
      stepitems.remove(itemid)
      if not stepitems:
        del self.stepitems[stepid]
        del self.steps[stepid]

  def save(self):
    assert self.directory
    import pickle
    (self.directory / 'steps.pkl').write(pickle.dumps(self.steps), 'wb')
    return self.items

  def load(self, data):
    assert self.directory
    import pickle
    self.items = data
    self.steps = pickle.loads((self.directory / 'steps.pkl').read('rb'))
    self.stepitems.clear()
    for itemid, stepids in self.items.items():
      for stepid in stepids:
        self.stepitems[stepid].add(itemid)

  def visualize(self, width=79):
    return ''


class ChunkTable:

  def __init__(self, directory=None, length=8192):
    self.directory = directory and embodied.Path(directory)
    self.length = length
    self.steps = {}  # {stepid: (chunkid, index)}
    self.items = {}  # {itemid: (chunkids, index, length)}
    self.chunks = {}  # {chunkid: Chunk}
    self.ongoing = {}  # {worker: Chunk}
    self.chunkitems = collections.defaultdict(set)  # {chunkid: itemids}
    # TODO
    # self.saver = self.directory and embodied.Worker(Chunk.save, 'thread')
    self.saver = self.directory and embodied.Worker(Chunk.save, 'blocking')

  @property
  def num_steps(self):
    return len(self.steps)

  @property
  def num_items(self):
    return len(self.items)

  def add_step(self, stepid, npdict, worker):
    if worker not in self.ongoing:
      chunk = Chunk()
      self.chunks[chunk.id] = chunk
      self.ongoing[worker] = chunk
    chunk = self.ongoing[worker]
    [chunk.data[k].append(v) for k, v in npdict.items()]
    chunk.stepids.append(stepid)
    chunk.length += 1
    self.steps[stepid] = (chunk.id, chunk.length - 1)
    if chunk.length >= self.length:
      new = Chunk()
      new.prev = chunk.id
      chunk.next = new.id
      chunk.data = {k: embodied.convert(v) for k, v in chunk.data.items()}
      self.chunks[new.id] = new
      self.ongoing[worker] = new
      self.directory and self.saver(chunk, self.directory)

  def insert(self, itemid, stepids):
    stepids = tuple(stepids)
    chunkid, start = self.steps[stepids[0]]
    chunkids = [chunkid]
    if len(stepids) > 1:
      chunk = self.chunks[chunkid]
      index = start
      for step in stepids[1:]:
        index += 1
        if index >= chunk.length:
          chunk = self.chunks[chunk.next]
          chunkids.append(chunk.id)
          index = 0
        # assert chunk.stepids[index] == step
    for chunkid in chunkids:
      self.chunkitems[chunkid].add(itemid)
    self.items[itemid] = (chunkids, start, len(stepids))

  def retrieve(self, itemid):
    chunkids, start, length = self.items[itemid]
    chunk = self.chunks[chunkids[0]]
    if start + length <= chunk.length:
      return chunk.get(start, start + length)
    parts = [chunk.get(start, start + length)]
    missing = length - (chunk.length - start)
    while missing > 0:
      chunk = self.chunks[chunk.next]
      amount = min(missing, chunk.length)
      parts.append(chunk.get(0, amount))
      missing -= amount
    return {
        k: np.concatenate([p[k] for p in parts], 0)
        for k in parts[0].keys()}

  def remove(self, itemid):
    chunkids, index, length = self.items.pop(itemid)
    for chunkid in chunkids:
      chunkitems = self.chunkitems[chunkid]
      chunkitems.remove(itemid)
      if not chunkitems:
        del self.chunkitems[chunkid]
        chunk = self.chunks.pop(chunkid)
        for stepid in chunk.stepids:
          del self.steps[stepid]

  def save(self):
    assert self.directory
    for chunk in self.ongoing.values():
      if chunk.length:
        self.saver(chunk, self.directory)
    self.saver.wait()
    return self.items

  def load(self, items):
    assert self.directory
    self.steps.clear()
    self.chunks.clear()
    self.ongoing.clear()
    self.chunkitems.clear()
    self.items = items
    filenames = {}
    for filename in self.directory.glob('*.npz'):
      chunkid = int(filename.stem.split('-')[1])
      filenames[chunkid] = filename
    chunkids = set()
    for cids, _, _ in items.values():
      chunkids.update(cids)
    for chunkid in chunkids:  # TODO: parallelize this
      if chunkid not in filenames:
        print('Did not find missing chunk:', chunkid)
        continue
      filename = filenames[chunkid]
      try:
        self.chunks[chunkid] = Chunk.load(filename)
      except Exception as e:
        print(f'Failed to load chunk {filename}:', e)
    for chunk in self.chunks.values():
      for index, stepid in enumerate(chunk.stepids):
        self.steps[stepid] = (chunk.id, index)
    for itemid, (chunkids, index, length) in self.items.items():
      for chunkid in chunkids:
        self.chunkitems[chunkid].add(itemid)

  def visualize(self, width=79):
    grid = len(str(max(
        tuple(self.steps.keys()) +
        tuple(self.chunks.keys()) +
        tuple(self.items.keys())))) + 3
    cols = width // grid
    lines = []
    # Stats
    lines.append(f'Chunks: {len(self.chunks):>5}')
    lines.append(f'Steps:  {len(self.steps):>5}')
    lines.append(f'Items:  {len(self.items):>5}')
    lines.append('')
    # Chunks
    line = [' ' * grid]
    chunkpos = {}
    for chunk in self.chunks.values():
      if len(line) >= cols:
        line.append(' ...')
        break
      chunkpos[chunk.id] = len(line) - 1
      line.append('/' + str(chunk.id).ljust(grid - 1, '_'))
      line += ['_' * grid] * max(0, len(chunk.stepids) - 1)
      line[-1] = line[-1][:-1] + '\\'
    lines.append(''.join(line))
    # Steps
    line = [' ' * grid]
    for chunk in self.chunks.values():
      stepids = list(chunk.stepids) or [' ']
      for stepid in stepids:
        line.append('|' + str(stepid).rjust(grid - 3) + '  ')
        if len(line) >= cols:
          line.append(' ...')
          break
      line[-1] = line[-1][:-1] + '|'
    lines.append(''.join(line))
    # Items
    for itemid, (chunkids, start, length) in self.items.items():
      chunkid = chunkids[0]
      if chunkid not in chunkpos:
        lines.append('...')
        break
      start += chunkpos[chunkid]
      line = str(itemid).ljust(grid)
      line += ' ' * (grid * start)
      line += '[' + ('-' * (grid * length))[1:-1] + ']'
      lines.append(line)
    div = ':' + '-' * (width - 1) + '\n'
    return div + ''.join(['| ' + line + '\n' for line in lines]) + div


class Chunk:

  def __init__(self):
    self.id = make_id()
    self.time = time.strftime('%Y%m%dT%H%M%S', time.gmtime(time.time()))
    self.prev = ''
    self.next = ''
    self.data = collections.defaultdict(list)
    self.stepids = []
    self.length = 0

  def __repr__(self):
    return f'Chunk(id={self.id}, stepids={self.stepids})'

  def add(self, stepid, npdict):
    [self.data[k].append(v) for k, v in npdict.items()]
    self.stepids.append(stepid)
    self.length += 1

  def get(self, start, stop):
    return {k: embodied.convert(v[start: stop]) for k, v in self.data.items()}

  def save(self, directory):
    filename = directory / f'{self.time}-{self.id}-{self.length}.npz'
    chunk = {f'chunk.{k}': getattr(self, k) for k in (
        'id', 'prev', 'next', 'stepids', 'length')}
    chunk = {k: np.asarray(v) for k, v in chunk.items()}
    with io.BytesIO() as stream:
      np.savez_compressed(stream, **chunk, **self.data)
      stream.seek(0)
      filename.write(stream.read(), mode='wb')
    print(f'Saved chunk: {filename.name}')

  @classmethod
  def load(cls, filename):
    with filename.open('rb') as f:
      data = np.load(f)
      data = {k: data[k] for k in data.keys()}
    chunk = cls()
    chunk.time = filename.stem.split('-')[0]
    chunk.id = data.pop('chunk.id').item()
    chunk.next = data.pop('chunk.next').item()
    chunk.prev = data.pop('chunk.prev').item()
    chunk.stepids = data.pop('chunk.stepids')
    chunk.length = data.pop('chunk.length').item()
    chunk.data = data
    print(f'Loaded chunk: {filename.name}')
    return chunk
