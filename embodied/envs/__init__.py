import functools
import pathlib

from .atari import Atari
from .crafter import Crafter
from .dmc import DMC
from .dummy import Dummy
from .gym import Gym

import embodied


def load_env(task, amount=1, parallel='none', **kwargs):
  ctor = functools.partial(load_single_env, task, **kwargs)
  if parallel != 'none':
    ctor = functools.partial(embodied.Parallel, ctor, parallel)
  envs = [ctor() for _ in range(amount)]
  return embodied.BatchEnv(envs, parallel=(parallel != 'none'))


def load_single_env(
    task, size=(64, 64), repeat=1, mode='train', camera=-1, gray=False,
    length=0, logdir=''):
  suite, task = task.split('_', 1)
  if suite == 'dummy':
    env = Dummy(task, size, length or 100)
  elif suite == 'gym':
    env = Gym(task)
  elif suite == 'dmc':
    env = DMC(task, repeat, size, camera)
  elif suite == 'atari':
    env = Atari(task, repeat, size, gray)
  elif suite == 'crafter':
    assert repeat == 1
    outdir = pathlib.Path(logdir) / 'crafter' if mode == 'train' else None
    env = Crafter(task, size, outdir)
  else:
    raise NotImplementedError(suite)
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    if space.discrete:
      env = embodied.wrappers.OneHotAction(env, name)
    else:
      env = embodied.wrappers.NormalizeAction(env, name)
  if length:
    env = embodied.wrappers.TimeLimit(env, length)
  return env


__all__ = [
    k for k, v in list(locals().items())
    if type(v).__name__ in ('type', 'function') and not k.startswith('_')]
