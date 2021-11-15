import embodied
import numpy as np


class Gym(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action'):
    import gym
    self._env = gym.make(env) if isinstance(env, str) else env
    self._obs_dict = getattr(self._env.observation_space, 'spaces', None)
    self._act_dict = getattr(self._env.action_space, 'spaces', None)
    self._scalar_obs = [
        k for k, v in self._obs_dict.items()
        if isinstance(v, gym.spaces.Box) and v.shape == ()
    ] if self._obs_dict else []
    self._scalar_act = [
        k for k, v in self._act_dict.items()
        if isinstance(v, gym.spaces.Box) and v.shape == ()
    ] if self._act_dict else []
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True

  @property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._env.observation_space.spaces.copy()
      spaces = self._flatten(spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    if self._act_dict:
      spaces = self._env.action_space.spaces.copy()
      spaces = self._flatten(spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    for key in self._scalar_act:
      action[key] = np.squeeze(action[key], -1)
    obs, reward, self._done, info = self._env.step(action)
    return self._obs(
        obs, reward,
        is_last=self._done,
        is_terminal=info.get('is_terminal', self._done))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.array(v) for k, v in obs.items()}
    for key in self._scalar_obs:
      obs[key] = obs[key][None]
    obs.update(
        reward=float(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def render(self):
    return self._env.render('rgb_array')

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    import gym
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    shape, low, high = space.shape, space.low, space.high
    if shape == ():
      shape, low, high = (1,), low[None], high[None]
    return embodied.Space(space.dtype, shape, low, high)
