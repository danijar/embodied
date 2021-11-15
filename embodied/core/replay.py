import collections
import concurrent.futures
import datetime
import io
import pathlib
import time
import uuid

import numpy as np


class SequenceReplay:

  def __init__(
      self, directory, capacity=0, length=50, ongoing=True,
      prioritize_ends=True, parallel=True, seed=None):
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(parents=True, exist_ok=True)
    self._capacity = capacity
    self._length = length
    self._ongoing = ongoing
    self._prioritize_ends = prioritize_ends
    self._parallel = parallel
    if parallel:
      self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      self._future = None
    self._random = np.random.RandomState(seed)
    # filename -> key -> value_sequence
    self._complete_eps = load_episodes(self._directory, capacity, length)
    # worker -> key -> value_sequence
    self._ongoing_eps = collections.defaultdict(
        lambda: collections.defaultdict(list))
    self._total_episodes, self._total_steps = count_episodes(self._directory)
    self._loaded_episodes = len(self._complete_eps)
    self._loaded_steps = sum(
        episode_length(x) for x in self._complete_eps.keys())

  @property
  def stats(self):
    return {
        'total_steps': self._total_steps,
        'total_episodes': self._total_episodes,
        'loaded_steps': self._loaded_steps,
        'loaded_episodes': self._loaded_episodes,
    }

  def add(self, transition, worker=0):
    episode = self._ongoing_eps[worker]
    if transition['is_first']:
      self._ongoing_eps[worker].clear()
    for key, value in transition.items():
      episode[key].append(value)
    if not transition['is_last']:
      return
    length = episode_length(episode)
    if length < self._length:
      self._ongoing_eps[worker].clear()
      print(f'Skipping short episode of length {length}.')
      return
    self._total_steps += length
    self._loaded_steps += length
    self._total_episodes += 1
    self._loaded_episodes += 1
    if self._parallel:
      if self._future:
        self._future.result()
      self._future = self._executor.submit(self._add_episode, episode.copy())
    else:
      self._add_episode(episode.copy())
    self._ongoing_eps[worker].clear()

  def sample(self):
    while True:
      episodes = list(self._complete_eps.values())
      if self._ongoing:
        episodes += [
            x for x in self._ongoing_eps.values()
            if episode_length(x) >= self._length]
      if not episodes:
        time.sleep(1)
        continue
      index = self._random.randint(0, len(episodes))
      episode = episodes[index]
      total = len(episode['action'])
      upper = total - self._length + 1
      if self._prioritize_ends:
        upper += self._length
      index = min(self._random.randint(upper), total - self._length)
      sequence = {
          k: simplify_dtype(v[index: index + self._length])
          for k, v in episode.items() if not k.startswith('log_')}
      sequence['is_first'] = np.zeros(len(sequence['action']), bool)
      sequence['is_first'][0] = True
      yield sequence

  def _add_episode(self, episode):
    episode = {key: simplify_dtype(value) for key, value in episode.items()}
    filename = save_episode(self._directory, episode)
    self._complete_eps[str(filename)] = episode
    self._enforce_limit()

  def _enforce_limit(self):
    if not self._capacity:
      return
    while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
      # Relying on Python preserving the insertion order of dicts.
      oldest = next(iter(self._complete_eps.keys()))
      self._loaded_steps -= episode_length(oldest)
      self._loaded_episodes -= 1
      del self._complete_eps[oldest]


def count_episodes(directory):
  filenames = list(pathlib.Path(directory).glob('*.npz'))
  num_episodes = len(filenames)
  num_steps = sum(int(str(n).split('-')[-1][:-4]) - 1 for n in filenames)
  return num_episodes, num_steps


def save_episode(directory, episode):
  directory = pathlib.Path(directory)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  identifier = str(uuid.uuid4().hex)
  length = episode_length(episode)
  filename = directory / f'{timestamp}-{identifier}-{length}.npz'
  with io.BytesIO() as stream:
    np.savez_compressed(stream, **episode)
    stream.seek(0)
    value = stream.read()
  with filename.open('wb') as f:
    f.write(value)
  return filename


def load_episodes(directory, capacity=None, minlen=1):
  filenames = sorted(pathlib.Path(directory).glob('*.npz'))
  if capacity:
    num_steps = 0
    num_episodes = 0
    for filename in reversed(filenames):
      length = episode_length(filename)
      num_steps += length
      num_episodes += 1
      if num_steps >= capacity:
        break
    filenames = filenames[-num_episodes:]
  episodes = {}
  for filename in filenames:
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode {str(filename)}: {e}')
      continue
    episodes[str(filename)] = episode
  return episodes


def simplify_dtype(value):
  value = np.array(value)
  if np.issubdtype(value.dtype, np.floating):
    return value.astype(np.float32)
  elif np.issubdtype(value.dtype, np.signedinteger):
    return value.astype(np.int32)
  elif np.issubdtype(value.dtype, np.uint8):
    return value.astype(np.uint8)
  return value


def episode_length(episode):
  if isinstance(episode, (str, pathlib.Path)):
      return int(str(episode).split('-')[-1][:-4])
  return len(episode['action']) - 1
