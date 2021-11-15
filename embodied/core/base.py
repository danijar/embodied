import numpy as np


class Space:

  def __init__(self, dtype, shape=(), low=None, high=None):
    # For integer types, high is the excluside upper bound.
    self.dtype = np.dtype(dtype)
    self.low = self._infer_low(dtype, shape, low, high)
    self.high = self._infer_high(dtype, shape, low, high)
    self.shape = self._infer_shape(dtype, shape, low, high)
    self.discrete = (
        np.issubdtype(self.dtype, np.integer) or self.dtype == bool)
    self._random = np.random.RandomState()

  def __repr__(self):
    return (
        f'Space(dtype={self.dtype.name}, '
        f'shape={self.shape}, '
        f'low={self.low.min()}, '
        f'high={self.high.max()})')

  def sample(self):
    low, high = self.low, self.high
    if np.issubdtype(self.dtype, np.floating):
      low = np.maximum(np.ones(self.shape) * np.finfo(self.dtype).min, low)
      high = np.minimum(np.ones(self.shape) * np.finfo(self.dtype).max, high)
    return self._random.uniform(low, high, self.shape).astype(self.dtype)

  def _infer_low(self, dtype, shape, low, high):
    if low is None:
      if np.issubdtype(dtype, np.floating):
        low = -np.inf * np.ones(shape)
      elif np.issubdtype(dtype, np.integer):
        low = np.iinfo(dtype).min * np.ones(shape, dtype)
      elif np.issubdtype(dtype, bool):
        low = np.zeros(shape, bool)
      else:
        raise ValueError('Cannot infer low bound from shape and dtype.')
    return np.array(low)

  def _infer_high(self, dtype, shape, low, high):
    if high is None:
      if np.issubdtype(dtype, np.floating):
        high = np.inf * np.ones(shape)
      elif np.issubdtype(dtype, np.integer):
        high = np.iinfo(dtype).max * np.ones(shape, dtype)
      elif np.issubdtype(dtype, bool):
        high = np.ones(shape, bool)
      else:
        raise ValueError('Cannot infer high bound from shape and dtype.')
    return np.array(high)

  def _infer_shape(self, dtype, shape, low, high):
    if shape is None and low is not None:
      shape = low.shape
    if shape is None and high is not None:
      shape = high.shape
    if not hasattr(shape, '__len__'):
      shape = (shape,)
    assert all(dim and dim > 0 for dim in shape), shape
    return tuple(shape)


class Agent:

  configs = {}  # dict of dicts

  def __init__(self, obs_space, act_space, step, config):
    pass

  def dataset(self, generator):
    raise NotImplementedError(
        'dataset(generator) -> generator')

  def policy(self, obs, state=None, mode='train'):
    raise NotImplementedError(
        "policy(obs, state=None, mode='train') -> act, state")

  def train(self, data, state=None):
    raise NotImplementedError(
        'train(data, state=None) -> state, metrics')

  def report(self, data):
    raise NotImplementedError(
        'report(data) -> metrics')

  def save(self):
    raise NotImplementedError('save() -> data')

  def load(data):
    raise NotImplementedError('load(data) -> None')


class Env:

  def __len__(self):
    # Return positive integer for batched envs.
    return 0

  def __repr__(self):
    return (
        f'{self.__class__.__name__}('
        f'len={len(self)}, '
        f'obs_space={self.obs_space}, '
        f'act_space={self.act_space})')

  @property
  def obs_space(self):
    # The observation space must contain the keys is_first, is_last, and
    # is_terminal. Commonly, it also contains the keys reward and image. By
    # convention, keys starting with log_ are not consumed by the agent.
    raise NotImplementedError('Returns: dict of spaces')

  @property
  def act_space(self):
    # The observation space must contain the keys action and reset. This
    # restriction may be lifted in the future.
    raise NotImplementedError('Returns: dict of spaces')

  def step(self, action):
    raise NotImplementedError('Returns: dict')

  def render(self):
    raise NotImplementedError('Returns: array')

  def close(self):
    pass


class Wrapper:

  def __init__(self, env):
    self.env = env

  def __len__(self):
    return len(self.env)

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      return getattr(self.env, name)
    except AttributeError:
      raise ValueError(name)


class Replay:

  def add(self, transition, worker=0):
    raise NotImplementedError('Returns: None')

  def sample(self):
    raise NotImplementedError('Returns: generator')
