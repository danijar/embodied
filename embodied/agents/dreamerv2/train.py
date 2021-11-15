import pathlib
import sys

directory = pathlib.Path(__file__).parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied


def main():
  # Importing these locally so that multiprocessing workers don't inherit them.
  # This prevents TensorFlow from initializing CUDA and allocating some amount
  # of GPU memory for environment processes that don't use TensorFlow.
  from . import agent as agnt
  from . import tfutils

  parsed, remaining = embodied.Flags({
      'configs': ['defaults'],
  }).parse(known_only=True)

  config = embodied.Config(agnt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agnt.Agent.configs[name])
  config = embodied.Flags(config).parse(remaining)
  config = config.update(logdir=str(pathlib.Path(config.logdir).expanduser()))
  args = embodied.Config(logdir=config.logdir, **config.train)
  print(config)

  logdir = pathlib.Path(config.logdir).expanduser()
  step = embodied.Counter()
  env = embodied.envs.load_env(config.task, mode='train', **config.env)
  replay = embodied.SequenceReplay(logdir / 'episodes', **config.replay)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(config.logdir),
      embodied.logger.TensorBoardOutput(config.logdir),
  ], multiplier=config.env.repeat)

  tfutils.setup(**config.tf)
  agent = agnt.Agent(env.obs_space, env.act_space, step, config)

  if config.run == 'train_eval':
    eval_env = embodied.envs.load_env(config.task, mode='eval', **config.env)
    eval_replay = embodied.SequenceReplay(
        logdir / 'eval_episodes', **config.replay.update(
            capacity=config.replay.capacity // 10))
    embodied.run.train_eval(
        agent, env, eval_env, replay, eval_replay, logger, args)
    eval_env.close()
  elif config.run == 'train':
    embodied.run.train(agent, env, replay, logger, args)
  else:
    raise NotImplementedError(config.run)
  env.close()


if __name__ == '__main__':
  main()
