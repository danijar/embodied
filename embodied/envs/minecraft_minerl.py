# from minerl.herobraine.env_specs import obtain_specs
# from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc
from minerl.herobraine.hero.mc import INVERSE_KEYMAP


def edit_options(**kwargs):
  import os, pathlib, re
  for word in os.popen('pip3 --version').read().split(' '):
    if '-packages/pip' in word:
      break
  else:
    raise RuntimeError('Could not found python package directory.')
  packages = pathlib.Path(word).parent
  filename = packages / 'minerl/Malmo/Minecraft/run/options.txt'
  options = filename.read_text()
  if 'fovEffectScale:' not in options:
    options += 'fovEffectScale:1.0\n'
  if 'simulationDistance:' not in options:
    options += 'simulationDistance:12\n'
  for key, value in kwargs.items():
    assert f'{key}:' in options, key
    assert isinstance(value, str), (value, type(value))
    options = re.sub(f'{key}:.*\n', f'{key}:{value}\n', options)
  # print('Minecraft options.txt:')
  # print(options + '\n')
  filename.write_text(options)


edit_options(
    difficulty='2',
    renderDistance='6',
    simulationDistance='6',
    fovEffectScale='0.0',
    ao='1',
    gamma='5.0',
    # maxFps='260',
)


class MineRLEnv(EnvSpec):

  def __init__(self, resolution=(64, 64), break_speed=50, gamma=10.0):
    self.resolution = resolution
    self.break_speed = break_speed
    self.gamma = gamma
    super().__init__(name='MineRLEnv-v1')

  def create_agent_start(self):
    return [
        BreakSpeedMultiplier(self.break_speed),
        # Gamma(self.gamma),  # TODO: Not supported in MineRL 0.4.4
    ]

  def create_agent_handlers(self):
    return []

  def create_server_world_generators(self):
    return [handlers.DefaultWorldGenerator(force_reset=True)]

  def create_server_quit_producers(self):
    return [handlers.ServerQuitWhenAnyAgentFinishes()]

  def create_server_initial_conditions(self):
    return [
        handlers.TimeInitialCondition(
            allow_passage_of_time=True,
            start_time=0,
        ),
        handlers.SpawningInitialCondition(
            allow_spawning=True,
        )
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
        # handlers.KeybasedCommandAction('sprint', INVERSE_KEYMAP['sprint']),
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
    # return f'agent_start_break_speed_multiplier({self.multiplier})'
    return f'break_speed({self.multiplier})'

  def xml_template(self):
    return '<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>'


class Gamma(handler.Handler):

  def __init__(self, gamma=2.0):
    self.gamma = gamma

  def to_string(self):
    return f'gamma({self.gamma})'

  def xml_template(self):
    return '<GammaSetting>{{gamma}}</GammaSetting>'


NOOP_ACTION = dict(
    camera=(0, 0), forward=0, back=0, left=0, right=0, attack=0, sprint=0,
    jump=0, sneak=0, craft='none', nearbyCraft='none', nearbySmelt='none',
    place='none', equip='none',
)

# SIZE = (64, 64)
# BREAK_SPEED = 1.0
#
#
# class WoodTools(obtain_specs.Obtain):
#
#   def __init__(self):
#     super().__init__(
#         reward_schedule=[
#             dict(type='log', amount=1, reward=1),
#             dict(type='crafting_table', amount=1, reward=1),
#             dict(type='wooden_pickaxe', amount=1, reward=1),
#             dict(type='wooden_axe', amount=1, reward=1),
#             dict(type='wooden_shovel', amount=1, reward=1),
#             dict(type='wooden_sword', amount=1, reward=1),
#             dict(type='wooden_hoe', amount=1, reward=1),
#         ],
#         target_item='furnace',
#         dense=False,
#         max_episode_steps=int(1e6),
#         resolution=SIZE,
#     )
#     self.name = 'MinecraftWoodTools-v1'
#
#   def create_agent_handlers(self):
#     return []
#
#   def create_agent_start(self):
#     handlers = []
#     if BREAK_SPEED != 1.0:
#       handlers.append(BreakSpeedMultiplier(BREAK_SPEED))
#     return handlers
#
#
# class Furnace(obtain_specs.Obtain):
#
#   def __init__(self):
#     super().__init__(
#         reward_schedule=[
#             dict(type='log', amount=1, reward=1),
#             dict(type='crafting_table', amount=1, reward=1),
#             dict(type='wooden_pickaxe', amount=1, reward=1),
#             dict(type='cobblestone', amount=1, reward=1),
#             dict(type='furnace', amount=1, reward=1),
#         ],
#         target_item='furnace',
#         dense=False,
#         max_episode_steps=int(1e6),
#         resolution=SIZE,
#     )
#     self.name = 'MinecraftFurnace-v1'
#
#   def create_agent_handlers(self):
#     return []
#
#   def create_agent_start(self):
#     handlers = []
#     if BREAK_SPEED != 1.0:
#       handlers.append(BreakSpeedMultiplier(BREAK_SPEED))
#     return handlers
#
#
# class Axe(obtain_specs.Obtain):
#
#   def __init__(self):
#     super().__init__(
#         target_item='wooden_axe',
#         dense=False,
#         reward_schedule=[
#             dict(type='log', amount=1, reward=1),
#             dict(type='crafting_table', amount=1, reward=10),
#             dict(type='wooden_axe', amount=1, reward=100),
#         ],
#         max_episode_steps=int(1e6),
#         resolution=SIZE,
#     )
#     self.name = 'MinecraftAxe-v1'
#
#   def create_agent_handlers(self):
#     return []
#
#
# class Table(obtain_specs.Obtain):
#
#   def __init__(self):
#     super().__init__(
#         target_item='crafting_table',
#         dense=True,
#         reward_schedule=[
#             dict(type='log', amount=1, reward=1),
#             dict(type='crafting_table', amount=1, reward=10),
#         ],
#         max_episode_steps=int(1e6),
#         resolution=SIZE,
#     )
#     self.name = 'MinecraftTable-v1'
#
#   def create_agent_handlers(self):
#     return []
#
#
# class Wood(obtain_specs.Obtain):
#
#   def __init__(self):
#     super().__init__(
#         target_item='log',
#         dense=True,
#         reward_schedule=[
#             dict(type='log', amount=1, reward=1),
#         ],
#         max_episode_steps=int(1e6),
#         resolution=SIZE,
#     )
#     self.name = 'MinecraftWood-v1'
#
#   def create_agent_handlers(self):
#     return []
#
#   def create_agent_start(self):
#     handlers = []
#     if BREAK_SPEED != 1.0:
#       handlers.append(BreakSpeedMultiplier(BREAK_SPEED))
#     return handlers
#
#
# class Diamond(obtain_specs.Obtain):
#
#   def __init__(self):
#     super().__init__(
#         target_item='diamond',
#         dense=False,
#         reward_schedule=[
#             dict(type='log', amount=1, reward=1),
#             dict(type='planks', amount=1, reward=2),
#             dict(type='stick', amount=1, reward=4),
#             dict(type='crafting_table', amount=1, reward=4),
#             dict(type='wooden_pickaxe', amount=1, reward=8),
#             dict(type='cobblestone', amount=1, reward=16),
#             dict(type='furnace', amount=1, reward=32),
#             dict(type='stone_pickaxe', amount=1, reward=32),
#             dict(type='iron_ore', amount=1, reward=64),
#             dict(type='iron_ingot', amount=1, reward=128),
#             dict(type='iron_pickaxe', amount=1, reward=256),
#             dict(type='diamond', amount=1, reward=1024)
#         ],
#         # The time limit used to be 18000 steps but MineRL did not enforce it
#         # exactly. We disable the MineRL time limit to apply our own exact
#         # time limit on the outside.
#         max_episode_steps=int(1e6),
#         resolution=SIZE,
#     )
#     self.name = 'MinecraftDiamond-v1'
#
#   def create_agent_handlers(self):
#     # There used to be a handler that terminates the episode after breaking a
#     # diamond block. However, it often did not leave enough time to collect
#     # the diamond item and receive a reward, so we just continue the episode.
#     return []


