import logging
import threading

import elements
import embodied
import numpy as np

from . import from_gym

LOGS = (
    'oak_log', 'birch_log', 'dark_oak_log', 'jungle_log', 'acacia_log',
    'spruce_log')

PLANKS = (
    'oak_planks', 'birch_planks', 'dark_oak_planks', 'jungle_planks',
    'acacia_planks', 'spruce_planks')


class Diamond(embodied.Wrapper):

  REWARDS = {
      LOGS:              (8, 1 / 8),
      PLANKS:            (20, 1 / 20),
      'stick':           (16, 1 / 16),
      'crafting_table':  (1, 1),
      'wooden_pickaxe':  (1, 1),
      'cobblestone':     (11, 1 / 11),
      'stone_pickaxe':   (1, 1),
      'furnace':         (1, 1),
      'coal':            (5, 2 / 5),
      'torch':           (16, 1 / 8),
      'iron_ore':        (3, 4 / 3),
      'iron_ingot':      (3, 4 / 3),
      'iron_pickaxe':    (1, 4),
      'diamond':         (None, 8 / 3),
      'diamond_pickaxe': (None, 8),
  }

  def __init__(self, *args, **kwargs):
    self.rewards = [
        CollectReward(k, r, t) for k, (t, r) in self.REWARDS.items()]
    # VPT RL: 12000 = 10min
    # VPT BC: 72000 = 60min
    # DreamerV3: 36000 = 30min
    length = kwargs.pop('length', 36000)
    env = MinecraftBase(ActionsV1(), *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    obs['reward'] = np.float32(reward)
    return obs


class CollectReward:

  def __init__(self, items, reward, times=None):
    self.items = (items,) if isinstance(items, str) else items
    self.reward = reward
    self.times = times or float('inf')
    self.previous = None
    self.total = None

  def __call__(self, obs, inventory):
    current = sum(inventory[x] for x in self.items)
    if obs['is_first']:
      self.previous = current
      self.total = 0
      return 0
    obtained = max(0, current - self.previous)
    remaining = max(0, self.times - self.total)
    rewarded = min(obtained, remaining)
    reward = self.reward * rewarded
    self.previous = current
    self.total += obtained
    return reward


class MinecraftBase(embodied.Env):

  LOCK = threading.Lock()

  def __init__(
      self,
      actions,
      size=(128, 128),
      invlogs=('logs', 'planks', 'cobblestone', 'iron_ingot', 'diamond'),
      guiscale=1.0,
      logs=False,
  ):
    assert size[0] >= size[1], size
    if logs:
      logging.basicConfig(level=logging.DEBUG)
    self.actions = actions
    self.size = size
    self.invlogs = invlogs
    self.guiscale = guiscale
    self.env = None

  def _once(self):
    if self.env:
      return
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    with self.LOCK:
      guiscale = float(self.guiscale * self.size[1] / 360)
      self.gymenv = HumanSurvival(
          resolution=self.size,           # VPT: [640, 360]
          fov_range=[70, 70],             # Bounds: [30, 140]
          gamma_range=[2, 2],             # VPT: 2, vanilla MC: 0
          guiscale_range=[guiscale] * 2,  # VPT: 1
          cursor_size_range=[16, 16],     # 32 hangs
          frameskip=1,
      ).make()
    assert len(self.gymenv.action_space.spaces) == 24, (
        sorted(self.gymenv.action_space.keys()))
    self.env = from_gym.FromGym(self.gymenv)
    self.invkeys = [
        k.removeprefix('inventory/') for k in self.env.obs_space
        if k.startswith('inventory/')]
    self.inventory = {k: 0 for k in self.invkeys}
    self.invlogs = {}
    for key in self.invlogs:
      if key == 'logs':
        items = LOGS
      elif key == 'planks':
        items = PLANKS
      else:
        items = (key,)
      assert all(it in self.invkeys for it in items), items
      self.invlogs[f'log/inventory_{key}'] = items

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self.size + (3,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        # 'inventory': elements.Space(np.float32, len(self.invkeys), 0),
        **{key: elements.Space(np.int32) for key in self.invlogs.keys()},
    }

  @property
  def act_space(self):
    return {'reset': elements.Space(bool), **self.actions.spaces}

  def step(self, action):
    self._once()
    if action.pop('reset'):
      with self.LOCK:
        obs = self.env.step({'reset': True})
    else:
      command = self.actions.act2mrl(action)
      obs = self.env.step({**command, 'reset': False})
    if self.env.info and 'error' in self.env.info:
      print('Attempting reset after MineRL error:', self.env.info['error'])
      with self.LOCK:
        obs = self.env.step({'reset': True})
    self.inventory = {k: obs[f'inventory/{k}'] for k in self.invkeys}
    return self._obs(obs)

  def _obs(self, obs):
    invlogs = {
        key: np.int32(sum(self.inventory[it] for it in items))
        for key, items in self.invlogs.items()}
    return {
        'image': obs['pov'],
        # 'inventory': np.array([self.inventory.values()], np.float32),
        'reward': np.float32(0.0),
        'is_first': obs['is_first'],
        'is_last': obs['is_last'],
        'is_terminal': obs['is_terminal'],
        **invlogs,
    }


class ActionsV1:

  MINERL_NOOP = {
      'ESC': 0, 'back': 0, 'drop': 0, 'forward': 0, 'hotbar.1': 0,
      'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0,
      'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0,
      'inventory': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0,
      'sprint': 0, 'swapHands': 0, 'camera': (0, 0), 'attack': 0, 'use': 0,
      'pickItem': 0}

  def __init__(self):
    self.bins = 11
    self.keys = (  # Name, MineRL, VPT recordings
        ('attack', 'attack', 'mouse.button.0'),
        ('back', 'back', 'key.keyboard.s'),
        ('drop', 'drop', 'key.keyboard.q'),
        ('escape', 'ESC', 'key.keyboard.escape'),
        ('forward', 'forward', 'key.keyboard.w'),
        ('hotbar1', 'hotbar.1', 'key.keyboard.1'),
        ('hotbar2', 'hotbar.1', 'key.keyboard.2'),
        ('hotbar3', 'hotbar.1', 'key.keyboard.3'),
        ('hotbar4', 'hotbar.1', 'key.keyboard.4'),
        ('hotbar5', 'hotbar.1', 'key.keyboard.5'),
        ('hotbar6', 'hotbar.1', 'key.keyboard.6'),
        ('hotbar7', 'hotbar.1', 'key.keyboard.7'),
        ('hotbar8', 'hotbar.1', 'key.keyboard.8'),
        ('hotbar9', 'hotbar.1', 'key.keyboard.9'),
        ('inventory', 'inventory', 'key.keyboard.e'),
        ('jump', 'jump', 'key.keyboard.space'),
        ('left', 'left', 'key.keyboard.a'),
        ('pick', 'pickItem', 'mouse.button.2'),
        ('right', 'right', 'key.keyboard.d'),
        ('sneak', 'sneak', 'key.keyboard.left.shift'),
        ('sprint', 'sprint', 'key.keyboard.left.control'),
        ('swaphands', 'swapHands', 'key.keyboard.f'),
        ('use', 'use', 'mouse.button.1'),
    )
    assert len(self.keys) == 23, self.keys

  @property
  def spaces(self):
    return {
        'mouse': elements.Space(np.int32, (), 0, self.bins * self.bins),
        'keys': elements.Space(np.int32, (len(self.keys)), 0, 2),
    }

  def act2mrl(self, action):
    assert set(action.keys()) == {'mouse', 'keys'}, action
    assert np.isin(action['keys'], [0, 1]).all(), action
    assert len(action['keys'] == len(self.keys)), action
    result = self.MINERL_NOOP.copy()
    binx = action['mouse'] // self.bins
    biny = action['mouse'] % self.bins
    result['camera'] = undiscretize(
        np.array([binx, biny], np.int32),
        limit=66.6667, bins=self.bins, mu=10)
    for (a, m, r), x in zip(self.keys, action['keys']):
      result[m] = x
    return result

  def mrl2act(self, action):
    binx, biny = discretize(
        np.array(action['camera'], np.float32),
        limit=66.6667, bins=self.bins, mu=10)
    mouse = binx * self.bins + biny
    keys = np.zeros(len(self.keys), np.int32)
    for i, (a, m, r) in enumerate(self.keys):
      keys[i] = action[m]
    result = {'mouse': mouse, 'keys': keys}
    return result

  def rec2mrl(self, action):
    result = {}
    result['camera'] = np.clip((
        float(action['mouse']['dx']) / 2400 * 360,
        float(action['mouse']['dy']) / 2400 * 360,
    ), -180, 180)
    pressed = list(action['keyboard']['keys'])
    for button in action['mouse']['buttons']:
      assert isinstance(button, int), action
      pressed.append(f'mouse.button.{button}')
    for a, m, r in self.keys:
      result[m] = int(r in pressed)
    return result


def discretize(x, limit, bins, mu):
  assert np.issubdtype(x.dtype, np.float32), (x, x.dtype)
  x = np.clip(x / limit, -1, 1)
  x = np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))
  x = (x + 1) / 2
  x = np.round(x * (bins - 1)).astype(np.int32)
  return x


def undiscretize(x, limit, bins, mu):
  assert np.issubdtype(x.dtype, np.int32), (x, x.dtype)
  assert np.all(0 <= bins).all() and np.all(x < bins), x
  x = x / (bins - 1)
  x = x * 2 - 1
  x = np.sign(x) * (1 / mu) * ((1 + mu) ** np.abs(x) - 1)
  x = x * limit
  return x
