import pathlib
import sys
import warnings

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')

directory = pathlib.Path(__file__)
try:
  import google3  # noqa
except ImportError:
  directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied


def main(argv=None):
  from . import agent as agnt

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agnt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agnt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)

  config = config.update(logdir=str(embodied.Path(config.logdir)))
  args = embodied.Config(logdir=config.logdir, **config.train)
  args = args.update(expl_until=args.expl_until // config.env.repeat)
  print(config)

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()

  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
  ], multiplier=config.env.repeat)

  store = embodied.replay.CkptRAMStore(
      logdir / 'episodes', config.replay_size, parallel=True)
  if config.replay == 'fixed':
    replay = embodied.replay.FixedLength(store, **config.replay_fixed)
  elif config.replay == 'consec':
    replay = embodied.replay.Consecutive(store, **config.replay_consec)
  elif config.replay == 'prio':
    replay = embodied.replay.Prioritized(store, **config.replay_prio)
  else:
    raise NotImplementedError(config.replay)
  make_small_replay = lambda subdir: embodied.replay.FixedLength(
      embodied.replay.CkptRAMStore(
          logdir / subdir, config.replay_size // 10, parallel=True),
      **config.replay_fixed)

  cleanup = []
  try:
    config = config.update({'env.seed': config.seed})
    env = embodied.envs.load_env(
        config.task, mode='train', logdir=logdir, **config.env)
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    if config.run == 'train':
      embodied.run.train(agent, env, replay, logger, args)
    elif config.run == 'train_eval':
      eval_env = embodied.envs.load_env(
          config.task, mode='eval', logdir=logdir, **config.env)
      eval_replay = make_small_replay('eval_episodes')
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)
      cleanup.append(eval_env)
    elif config.run == 'train_fixed_eval':
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_small_replay(config.eval_dir)
      else:
        assert config.train.eval_fill
        eval_replay = make_small_replay('eval_episodes')
      embodied.run.train_fixed_eval(
          agent, env, replay, eval_replay, logger, args)
    else:
      raise NotImplementedError(config.run)
  finally:
    for obj in cleanup:
      obj.close()


if __name__ == '__main__':
  main()
