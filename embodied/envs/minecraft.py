import embodied
import numpy as np

from . import minecraft_base


class Minecraft(embodied.Wrapper):

  def __init__(self, task, *args, **kwargs):
    super().__init__({
        'wood': MinecraftWood,
        'climb': MinecraftClimb,
        'torch': MinecraftTorch,
        'bookshelf': MinecraftBookshelf,
        'bed': MinecraftBed,
        'furnace': MinecraftFurnace,
        'wooden_tools': MinecraftWoodenTools,
        'stone_tools': MinecraftStoneTools,
        'hunt': MinecraftHunt,
        'diamond': MinecraftDiamond,
    }[task](*args, **kwargs))


class MinecraftWood(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = BASIC_ACTIONS
    self.rewards = [
        CollectReward('log', repeated=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs


class MinecraftClimb(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = BASIC_ACTIONS
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)
    self._previous = None
    self._health_reward = HealthReward()

  def step(self, action):
    obs = self.env.step(action)
    x, y, z = obs['log_player_pos']
    height = np.float32(y)
    if obs['is_first']:
      self._previous = height
    obs['reward'] = height - self._previous
    obs['reward'] += self._health_reward(obs)
    self._previous = height
    return obs


class MinecraftHunt(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_sword': dict(nearbyCraft='wooden_sword'),
        'equip_wooden_sword': dict(equip='wooden_sword'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_sword', once=1),
        CollectReward('chicken', repeated=1),
        CollectReward('beef', repeated=1),
        CollectReward('porkchop', repeated=1),
        CollectReward('mutton', repeated=1),
        CollectReward('gunpowder', repeated=1),
        CollectReward('bone', repeated=1),
        CollectReward('slime_ball', repeated=1),
        CollectReward('rotten_flesh', repeated=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs


class MinecraftTorch(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
        'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
        'craft_torch': dict(craft='torch'),
        'place_torch': dict(place='torch'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_pickaxe', once=1),
        CollectReward('cobblestone', once=1),
        CollectReward('coal', once=1),
        CollectReward('torch', repeated=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs


class MinecraftBookshelf(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_sword': dict(nearbyCraft='wooden_sword'),
        'equip_wooden_sword': dict(equip='wooden_sword'),
        'craft_paper': dict(nearbyCraft='paper'),
        'craft_book': dict(craft='book'),
        'craft_bookshelf': dict(nearbyCraft='bookshelf'),
        'place_bookshelf': dict(place='bookshelf'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_sword', once=1),
        CollectReward('leather', once=1),
        CollectReward('reeds', once=1),
        CollectReward('paper', once=1),
        CollectReward('book', once=1),
        CollectReward('bookshelf', repeated=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs


class MinecraftBed(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_sword': dict(nearbyCraft='wooden_sword'),
        'equip_wooden_sword': dict(equip='wooden_sword'),
        'craft_bed': dict(nearbyCraft='bed'),
        'place_bed': dict(place='bed'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_sword', once=1),
        CollectReward('wool', once=1),
        CollectReward('bed', once=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs


class MinecraftFurnace(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
        'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
        'craft_furnace': dict(nearbyCraft='furnace'),
        'place_furnace': dict(place='furnace'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_pickaxe', once=1),
        CollectReward('cobblestone', once=1),
        CollectReward('furnace', repeated=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs


class MinecraftWoodenTools(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_axe': dict(nearbyCraft='wooden_axe'),
        'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
        'craft_wooden_sword': dict(nearbyCraft='wooden_sword'),
        'craft_wooden_shovel': dict(nearbyCraft='wooden_shovel'),
        'craft_wooden_hoe': dict(nearbyCraft='wooden_hoe'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_axe', once=1),
        CollectReward('wooden_pickaxe', once=1),
        CollectReward('wooden_sword', once=1),
        CollectReward('wooden_shovel', once=1),
        CollectReward('wooden_hoe', once=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs


class MinecraftStoneTools(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
        'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
        'craft_stone_axe': dict(nearbyCraft='stone_axe'),
        'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
        'craft_stone_sword': dict(nearbyCraft='stone_sword'),
        'craft_stone_shovel': dict(nearbyCraft='stone_shovel'),
        'craft_stone_hoe': dict(nearbyCraft='stone_hoe'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_pickaxe', once=1),
        CollectReward('cobblestone', once=1),
        CollectReward('stone_axe', once=1),
        CollectReward('stone_pickaxe', once=1),
        CollectReward('stone_sword', once=1),
        CollectReward('stone_shovel', once=1),
        CollectReward('stone_hoe', once=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs


class MinecraftDiamond(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
        'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
        'craft_iron_pickaxe': dict(nearbyCraft='iron_pickaxe'),
        'equip_stone_pickaxe': dict(equip='stone_pickaxe'),
        'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
        'equip_iron_pickaxe': dict(equip='iron_pickaxe'),
        'craft_furnace': dict(nearbyCraft='furnace'),
        'place_furnace': dict(place='furnace'),
        'smelt_iron_ingot': dict(nearbySmelt='iron_ingot'),
    }
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
    env = minecraft_base.MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
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


BASIC_ACTIONS = {
    'noop': dict(),
    'attack': dict(attack=1),
    'turn_up': dict(camera=(-15, 0)),
    'turn_down': dict(camera=(15, 0)),
    'turn_left': dict(camera=(0, -15)),
    'turn_right': dict(camera=(0, 15)),
    'forward': dict(forward=1),
    'back': dict(back=1),
    'left': dict(left=1),
    'right': dict(right=1),
    'jump': dict(jump=1, forward=1),
    'place_dirt': dict(place='dirt'),
}


# DIAMOND_ACTIONS = {
#     **BASIC_ACTIONS,
#     'craft_planks': dict(craft='planks'),
#     'craft_stick': dict(craft='stick'),
#     'craft_torch': dict(craft='torch'),
#     'craft_crafting_table': dict(craft='crafting_table'),
#     'craft_furnace': dict(nearbyCraft='furnace'),
#     'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
#     'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
#     'craft_iron_pickaxe': dict(nearbyCraft='iron_pickaxe'),
#     'smelt_coal': dict(nearbySmelt='coal'),
#     'smelt_iron_ingot': dict(nearbySmelt='iron_ingot'),
#     'place_torch': dict(place='torch'),
#     'place_cobblestone': dict(place='cobblestone'),
#     'place_crafting_table': dict(place='crafting_table'),
#     'place_furnace': dict(place='furnace'),
#     'equip_iron_pickaxe': dict(equip='iron_pickaxe'),
#     'equip_stone_pickaxe': dict(equip='stone_pickaxe'),
#     'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
# }

# DISCOVER_ACTIONS = {
#     **BASIC_ACTIONS,
#
#     'craft_planks': dict(craft='planks'),
#     'craft_stick': dict(craft='stick'),
#     'craft_torch': dict(craft='torch'),
#     'craft_wheat': dict(craft='wheat'),
#     'craft_crafting_table': dict(craft='crafting_table'),
#
#     'craft_furnace': dict(nearbyCraft='furnace'),
#     'craft_trapdoor': dict(nearbyCraft='trapdoor'),
#     'craft_boat': dict(nearbyCraft='boat'),
#     'craft_bread': dict(nearbyCraft='bread'),
#     'craft_bucket': dict(nearbyCraft='bucket'),
#     'craft_ladder': dict(nearbyCraft='ladder'),
#     'craft_fence': dict(nearbyCraft='fence'),
#     'craft_chest': dict(nearbyCraft='chest'),
#     'craft_bowl': dict(nearbyCraft='bowl'),
#
#     'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
#     'craft_wooden_sword': dict(nearbyCraft='wooden_sword'),
#     'craft_wooden_shovel': dict(nearbyCraft='wooden_shovel'),
#     'craft_wooden_axe': dict(nearbyCraft='wooden_axe'),
#
#     'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
#     'craft_stone_sword': dict(nearbyCraft='stone_sword'),
#     'craft_stone_shovel': dict(nearbyCraft='stone_shovel'),
#     'craft_stone_axe': dict(nearbyCraft='stone_axe'),
#
#     'craft_iron_pickaxe': dict(nearbyCraft='iron_pickaxe'),
#     'craft_iron_sword': dict(nearbyCraft='iron_sword'),
#     'craft_iron_shovel': dict(nearbyCraft='iron_shovel'),
#     'craft_iron_axe': dict(nearbyCraft='iron_axe'),
#
#     'smelt_coal': dict(nearbySmelt='coal'),
#     'smelt_iron_ingot': dict(nearbySmelt='iron_ingot'),
#
#     'place_torch': dict(place='torch'),
#     'place_cobblestone': dict(place='cobblestone'),
#     'place_crafting_table': dict(place='crafting_table'),
#     'place_dirt': dict(place='dirt'),
#     'place_furnace': dict(place='furnace'),
#
#     'equip_iron_pickaxe': dict(equip='iron_pickaxe'),
#     'equip_stone_pickaxe': dict(equip='stone_pickaxe'),
#     'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
# }


