import importlib

import embodied


class Minecraft(embodied.Wrapper):

  def __init__(self, task, *args, **kwargs):
    module, cls = {
        'wood': 'minecraft_flat:Wood',
        'climb': 'minecraft_flat:Climb',
        'diamond': 'minecraft_flat:Diamond',
        'diamond_f1': 'minecraft_factor:Diamond1',
        'diamond_f2': 'minecraft_factor:Diamond2',
        'diamond_k': 'minecraft_keyboard:Diamond',
    }[task].split(':')
    if module not in ('minecraft_flat', 'minecraft_factor'):
      kwargs.pop('break_speed', None)
    if module not in ('minecraft_keyboard',):
      kwargs.pop('gui_scale', None)
    module = importlib.import_module(f'.{module}', __package__)
    cls = getattr(module, cls)
    env = cls(*args, **kwargs)
    super().__init__(env)


if __name__ == '__main__':
  from embodied.envs import minecraft_keyboard
  import numpy as np
  env = minecraft_keyboard.Diamond(size=(64, 64), logs=True)
  act = {
      'mouse': np.zeros(11 * 11, np.int32),
      'keys': np.zeros(23, np.int32),
  }
  obs = env.step({'reset': True, **act})
  obs = env.step({'reset': False, **act})
  print('DONE')
