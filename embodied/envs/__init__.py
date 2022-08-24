import functools

import embodied


def load_env(
    task, amount=1, parallel='none', daemon=False, restart=False, seed=None,
    **kwargs):
  ctors = []
  for index in range(amount):
    ctor = functools.partial(load_single_env, task, **kwargs)
    if seed is not None:
      ctor = functools.partial(ctor, seed=hash((seed, index)) % (2 ** 31 - 1))
    if parallel != 'none':
      ctor = functools.partial(embodied.Parallel, ctor, parallel, daemon)
    if restart:
      ctor = functools.partial(embodied.wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(parallel != 'none'))


def load_single_env(
    task, size=(64, 64), repeat=1, mode='train', camera=-1, gray=False,
    length=0, logdir='/dev/null', discretize=0, sticky=True, lives=False,
    episodic=True, again=False, termination=False, weaker=1.0, checks=False,
    resets=True, seed=None):
  suite, task = task.split('_', 1)
  if suite == 'dummy':
    from . import dummy
    env = dummy.Dummy(task, size, length or 100)
  elif suite == 'gym':
    from . import gym
    env = gym.Gym(task)
  elif suite == 'procgen':
    import procgen
    from . import gym
    env = gym.Gym(f'procgen:procgen-{task}-v0')

  elif suite == 'bsuite':

    try:
      import bsuite.google as bsuite
      from . import dmenv
      env = bsuite.load_from_sweep(task if '/' in task else task + '/0')
      num_episodes = env.bsuite_num_episodes
      env = dmenv.DMEnv(env)
      env = embodied.wrappers.StopAfterEpisodes(env, num_episodes, 600)
      env = embodied.wrappers.FlattenTwoDimObs(env)

    except ImportError:
      import bsuite
      from . import dmenv
      env = bsuite.load_from_id(task if '/' in task else task + '/0')
      env = dmenv.DMEnv(env)
      env = embodied.wrappers.FlattenTwoDimObs(env)

  elif suite == 'bsuiteimg':

    from bsuite import utils
    import bsuite.google as bsuite
    from . import dmenv
    env = bsuite.load_from_sweep(task if '/' in task else task + '/0')
    num_episodes = env.bsuite_num_episodes
    env = utils.wrappers.ImageObservation(env, (64, 64, 3))
    env = dmenv.DMEnv(env, obs_key='image')
    env = embodied.wrappers.StopAfterEpisodes(env, num_episodes, 600)
    env = embodied.wrappers.FlattenTwoDimObs(env)

  elif suite == 'dmc':
    from . import dmc
    env = dmc.DMC(task, repeat, render=True, size=size, camera=camera)
  elif suite == 'atari':
    from . import atari
    env = atari.Atari(task, repeat, size, gray, lives=lives, sticky=sticky)
  elif suite == 'crafter':
    from . import crafter
    assert repeat == 1
    # outdir = embodied.Path(logdir) / 'crafter' if mode == 'train' else None
    outdir = None
    env = crafter.Crafter(task, size, outdir)
  elif suite == 'dmlab':
    from . import dmlab
    env = dmlab.DMLab(task, repeat, size, mode, seed=seed, episodic=episodic)
  elif suite == 'robodesk':
    from . import robodesk
    env = robodesk.RoboDesk(task, mode, repeat, length or 2000)
  elif suite == 'minecraft':
    from . import minecraft
    env = minecraft.Minecraft(task, repeat, size, length or 24000)
  elif suite == 'loconav':
    from . import loconav
    env = loconav.LocoNav(
        task, repeat, size, camera,
        again=again, termination=termination, weaker=weaker)
  elif suite == 'pinpad':
    from . import pinpad
    assert repeat == 1
    assert size == (64, 64)
    env = pinpad.PinPad(task, length or 2000)
  else:
    raise NotImplementedError(suite)
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    if space.discrete:
      env = embodied.wrappers.OneHotAction(env, name)
    elif discretize:
      env = embodied.wrappers.DiscretizeAction(env, name, discretize)
    else:
      env = embodied.wrappers.NormalizeAction(env, name)
  if length:
    env = embodied.wrappers.TimeLimit(env, length, resets)
  env = embodied.wrappers.ExpandScalars(env)
  if checks:
    env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


__all__ = [
    k for k, v in list(locals().items())
    if type(v).__name__ in ('type', 'function') and not k.startswith('_')]