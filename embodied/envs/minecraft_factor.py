import logging
import threading

import elements
import embodied
import numpy as np

np.float = float
np.int = int
np.bool = bool

from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc
from minerl.herobraine.hero.mc import INVERSE_KEYMAP


class Diamond1(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = dict(
        main=(
            dict(),
            dict(attack=1),
            dict(camera=(-15, 0)),
            dict(camera=(15, 0)),
            dict(camera=(0, -15)),
            dict(camera=(0, 15)),
            dict(forward=1),
            dict(back=1),
            dict(left=1),
            dict(right=1),
            dict(jump=1, forward=1),
        ),
        other=(
            dict(),
            dict(place='dirt'),
            dict(place='crafting_table'),
            dict(place='furnace'),
            dict(craft='planks'),
            dict(craft='stick'),
            dict(craft='crafting_table'),
            dict(nearbyCraft='wooden_pickaxe'),
            dict(nearbyCraft='stone_pickaxe'),
            dict(nearbyCraft='iron_pickaxe'),
            dict(nearbyCraft='furnace'),
            dict(nearbySmelt='iron_ingot'),
            dict(equip='stone_pickaxe'),
            dict(equip='wooden_pickaxe'),
            dict(equip='iron_pickaxe'),
        )
    )
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('planks', once=1),
        CollectReward('stick', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_pickaxe', once=1),
        CollectReward('cobblestone', once=1),
        CollectReward('stone_pickaxe', once=1),
        CollectReward('iron_ore', once=1),
        CollectReward('furnace', once=1),
        CollectReward('iron_ingot', once=1),
        CollectReward('iron_pickaxe', once=1),
        CollectReward('diamond', once=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    obs['reward'] = np.float32(reward)
    return obs


class Diamond2(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = dict(
        move=(
            dict(),
            dict(forward=1),
            dict(back=1),
            dict(left=1),
            dict(right=1),
            dict(jump=1, forward=1),
        ),
        look=(
            dict(),
            dict(camera=(-15, 0)),
            dict(camera=(15, 0)),
            dict(camera=(0, -15)),
            dict(camera=(0, 15)),
        ),
        attack=(
            dict(),
            dict(attack=1),
        ),
        place=(
            dict(),
            dict(place='dirt'),
            dict(place='crafting_table'),
            dict(place='furnace'),
        ),
        make=(
            dict(),
            dict(craft='planks'),
            dict(craft='stick'),
            dict(craft='crafting_table'),
            dict(nearbyCraft='wooden_pickaxe'),
            dict(nearbyCraft='stone_pickaxe'),
            dict(nearbyCraft='iron_pickaxe'),
            dict(nearbyCraft='furnace'),
            dict(nearbySmelt='iron_ingot'),
        ),
        equip=(
            dict(),
            dict(equip='stone_pickaxe'),
            dict(equip='wooden_pickaxe'),
            dict(equip='iron_pickaxe'),
        )
    )
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('planks', once=1),
        CollectReward('stick', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_pickaxe', once=1),
        CollectReward('cobblestone', once=1),
        CollectReward('stone_pickaxe', once=1),
        CollectReward('iron_ore', once=1),
        CollectReward('furnace', once=1),
        CollectReward('iron_ingot', once=1),
        CollectReward('iron_pickaxe', once=1),
        CollectReward('diamond', once=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    obs['reward'] = np.float32(reward)
    return obs


class CollectReward:

  def __init__(self, item, once=0, repeated=0):
    self.item = item
    self.once = once
    self.repeated = repeated
    self.previous = 0
    self.maximum = 0

  def __call__(self, obs, inventory):
    current = inventory[self.item]
    if obs['is_first']:
      self.previous = current
      self.maximum = current
      return 0
    reward = self.repeated * max(0, current - self.previous)
    if self.maximum == 0 and current > 0:
      reward += self.once
    self.previous = current
    self.maximum = max(self.maximum, current)
    return reward


class HealthReward:

  def __init__(self, scale=0.01):
    self.scale = scale
    self.previous = None

  def __call__(self, obs, inventory=None):
    health = obs['health']
    if obs['is_first']:
      self.previous = health
      return 0
    reward = self.scale * (health - self.previous)
    self.previous = health
    return np.float32(reward)


class MinecraftBase(embodied.Env):

  LOCK = threading.Lock()
  NOOP = dict(
      camera=(0, 0), forward=0, back=0, left=0, right=0, attack=0, sprint=0,
      jump=0, sneak=0, craft='none', nearbyCraft='none', nearbySmelt='none',
      place='none', equip='none')

  def __init__(
      self, actions,
      size=(64, 64),
      break_speed=100.0,
      sticky_jump=10,
      pitch_limit=(-60, 60),
      log_inv_keys=('log', 'cobblestone', 'iron_ingot', 'diamond'),
      logs=False,
  ):
    if logs:
      logging.basicConfig(level=logging.DEBUG)
    self._size = size

    # Make env
    with self.LOCK:
      self._gymenv = MineRLEnv(size, break_speed).make()
    from . import from_gym
    self._env = from_gym.FromGym(self._gymenv)
    self._inventory = {}

    # Observations
    self._inv_keys = [
        k for k in self._env.obs_space if k.startswith('inventory/')
        if k != 'inventory/log2']
    self._inv_log_keys = [f'inventory/{k}' for k in log_inv_keys]
    assert all(k in self._inv_keys for k in self._inv_log_keys), (
        self._inv_keys, self._inv_log_keys)
    self._step = 0
    self._max_inventory = None
    self._equip_enum = self._gymenv.observation_space[
        'equipped_items']['mainhand']['type'].values.tolist()
    self._obs_space = self.obs_space

    # Actions
    self._actions = actions
    print('Minecraft action spaces:')
    for name, subactions in self._actions.items():
      print(f'- {name}: ({len(subactions)}): {subactions}')

    self._sticky_jump_length = sticky_jump
    self._sticky_jump_counter = 0
    self._pitch_limit = pitch_limit
    self._pitch = 0

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self._size + (3,)),
        'inventory': elements.Space(np.float32, len(self._inv_keys), 0),
        'inventory_max': elements.Space(np.float32, len(self._inv_keys), 0),
        'equipped': elements.Space(np.float32, len(self._equip_enum), 0, 1),
        'reward': elements.Space(np.float32),
        'health': elements.Space(np.float32),
        'hunger': elements.Space(np.float32),
        'breath': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        **{f'log/{k}': elements.Space(np.int32) for k in self._inv_log_keys},
        'log/player_pos': elements.Space(np.float32, 3),
    }

  @property
  def act_space(self):
    spaces = {'reset': elements.Space(bool)}
    for name, subactions in self._actions.items():
      spaces[name] = elements.Space(np.int32, (), 0, len(subactions))
    return spaces

  def step(self, action):
    command = self.NOOP.copy()
    for key, subactions in self._actions.items():
      command.update(subactions[action[key]])
    command = self._action(command)
    # print(action, '->', command)
    if action['reset']:
      obs = self._reset()
    else:
      obs = self._env.step({**command, 'reset': False})
    if self._env.info and 'error' in self._env.info:
      return self._reset()
    obs = self._obs(obs)
    self._step += 1
    return obs

  @property
  def inventory(self):
    return self._inventory

  def _reset(self):
    with self.LOCK:
      obs = self._env.step({'reset': True})
    self._step = 0
    self._max_inventory = None
    self._sticky_jump_counter = 0
    self._pitch = 0
    self._inventory = {}
    return obs

  def _obs(self, obs):
    obs['inventory/log'] += obs.pop('inventory/log2')
    self._inventory = {
        k.split('/', 1)[1]: obs[k] for k in self._inv_keys
        if k != 'inventory/air'}
    inventory = np.array([obs[k] for k in self._inv_keys], np.float32)
    if self._max_inventory is None:
      self._max_inventory = inventory
    else:
      self._max_inventory = np.maximum(self._max_inventory, inventory)
    index = self._equip_enum.index(obs['equipped_items/mainhand/type'])
    equipped = np.zeros(len(self._equip_enum), np.float32)
    equipped[index] = 1.0
    player_x = obs['location_stats/xpos']
    player_y = obs['location_stats/ypos']
    player_z = obs['location_stats/zpos']
    obs = {
        'image': obs['pov'],
        'inventory': inventory,
        'inventory_max': self._max_inventory.copy(),
        'equipped': equipped,
        'health': np.float32(obs['life_stats/life'] / 20),
        'hunger': np.float32(obs['life_stats/food'] / 20),
        'breath': np.float32(obs['life_stats/air'] / 300),
        'reward': np.float32(0.0),
        'is_first': obs['is_first'],
        'is_last': obs['is_last'],
        'is_terminal': obs['is_terminal'],
        **{f'log/{k}': np.int32(obs[k]) for k in self._inv_log_keys},
        'log/player_pos': np.array([player_x, player_y, player_z], np.float32),
    }
    for key, value in obs.items():
      space = self._obs_space[key]
      if not isinstance(value, np.ndarray):
        value = np.array(value)
      assert value in space, (key, value, value.dtype, value.shape, space)
    return obs

  def _action(self, action):
    if self._sticky_jump_length:
      if action['jump']:
        self._sticky_jump_counter = self._sticky_jump_length
      if self._sticky_jump_counter > 0:
        action['jump'] = 1
        action['forward'] = 1
        self._sticky_jump_counter -= 1
    if self._pitch_limit and action['camera'][0]:
      lo, hi = self._pitch_limit
      if not (lo <= self._pitch + action['camera'][0] <= hi):
        action['camera'] = (0, action['camera'][1])
      self._pitch += action['camera'][0]
    return action


class MineRLEnv(EnvSpec):

  def __init__(self, resolution=(64, 64), break_speed=50):
    self.resolution = resolution
    self.break_speed = break_speed
    super().__init__(name='MineRLEnv-v1')

  def create_agent_start(self):
    return [BreakSpeedMultiplier(self.break_speed)]

  def create_agent_handlers(self):
    return []

  def create_server_world_generators(self):
    return [handlers.DefaultWorldGenerator(force_reset=True)]

  def create_server_quit_producers(self):
    return [handlers.ServerQuitWhenAnyAgentFinishes()]

  def create_server_initial_conditions(self):
    return [
        handlers.TimeInitialCondition(
            allow_passage_of_time=True, start_time=0),
        handlers.SpawningInitialCondition(allow_spawning=True),
    ]

  def create_observables(self):
    return [
        handlers.POVObservation(self.resolution),
        handlers.FlatInventoryObservation(mc.ALL_ITEMS),
        handlers.EquippedItemObservation(
            mc.ALL_ITEMS, _default='air', _other='other'),
        handlers.ObservationFromCurrentLocation(),
        handlers.ObservationFromLifeStats(),
    ]

  def create_actionables(self):
    kw = dict(_other='none', _default='none')
    return [
        handlers.KeybasedCommandAction('forward', INVERSE_KEYMAP['forward']),
        handlers.KeybasedCommandAction('back', INVERSE_KEYMAP['back']),
        handlers.KeybasedCommandAction('left', INVERSE_KEYMAP['left']),
        handlers.KeybasedCommandAction('right', INVERSE_KEYMAP['right']),
        handlers.KeybasedCommandAction('jump', INVERSE_KEYMAP['jump']),
        handlers.KeybasedCommandAction('sneak', INVERSE_KEYMAP['sneak']),
        handlers.KeybasedCommandAction('attack', INVERSE_KEYMAP['attack']),
        handlers.CameraAction(),
        handlers.PlaceBlock(['none'] + mc.ALL_ITEMS, **kw),
        handlers.EquipAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftNearbyAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.SmeltItemNearby(['none'] + mc.ALL_ITEMS, **kw),
    ]

  def is_from_folder(self, folder):
    return folder == 'none'

  def get_docstring(self):
    return ''

  def determine_success_from_rewards(self, rewards):
    return True

  def create_rewardables(self):
    return []

  def create_server_decorators(self):
    return []

  def create_mission_handlers(self):
    return []

  def create_monitors(self):
    return []


class BreakSpeedMultiplier(handler.Handler):

  def __init__(self, multiplier=1.0):
    self.multiplier = multiplier

  def to_string(self):
    return f'break_speed({self.multiplier})'

  def xml_template(self):
    return '<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>'
