import threading

import embodied
import numpy as np


class Atari(embodied.Env):

  _LOCK = threading.Lock()

  def __init__(
      self, name, repeat=4, size=(84, 84), gray=True, noops=30,
      life_done=False, sticky=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self._LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name,
          obs_type='image',  # TODO: Internal old version.
          # obs_type='grayscale' if gray else 'rgb',
          frameskip=1, repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    self._env = gym.wrappers.AtariPreprocessing(
        env, noops, repeat, size[0], life_done)
    self._size = size
    self._gray = gray
    self._done = True

  @property
  def obs_space(self):
    shape = self._size + (1 if self._gray else 3,)
    return {
        'image': embodied.Space(np.uint8, shape),
        # 'ram': embodied.Space(np.uint8, 128),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      with self._LOCK:
        image = self._env.reset()
      self._done = False
      return self._obs(image, 0.0, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    return self._obs(image, reward, is_last=self._done, is_terminal=self._done)

  def _obs(
      self, image, reward, is_first=False, is_last=False,
      is_terminal=False):
    if len(image.shape) == 2:
      image = image[:, :, None]
    return dict(
        image=image,
        # ram=self._env.env._get_ram(),
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last,
    )

  def close(self):
    return self._env.close()
