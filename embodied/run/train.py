import collections
import pathlib
import re

import embodied
import numpy as np


def train(agent, env, replay, logger, args):

  logdir = pathlib.Path(args.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', logdir)
  should_train = embodied.when.Every(args.train_every)
  should_log = embodied.when.Every(args.log_every)
  should_expl = embodied.when.Until(args.expl_until)
  should_video = embodied.when.Every(args.eval_every)
  step = logger.step

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])

  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'Episode has {length} steps and return {score:.1f}.')
    logger.scalar('return', score)
    logger.scalar('length', length)
    for key, value in ep.items():
      if re.match(args.log_keys_sum, key):
        logger.scalar(f'sum_{key}', ep[key].sum())
      if re.match(args.log_keys_mean, key):
        logger.scalar(f'mean_{key}', ep[key].mean())
      if re.match(args.log_keys_max, key):
        logger.scalar(f'max_{key}', ep[key].max(0).mean())
    if should_video(step):
      for key in args.log_keys_video:
        logger.video(f'policy_{key}', ep[key])
    logger.add(replay.stats)
    logger.write()

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(
      lambda tran, _: None if tran['is_first'] else step.increment())
  driver.on_step(replay.add)

  prefill = max(0, args.prefill - replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = embodied.RandomAgent(env.act_space)
    driver(random_agent.policy, steps=prefill, episodes=1)
    driver.reset()

  dataset = iter(agent.dataset(replay.sample))
  state = [None]  # To be writable from train step function below.
  state[0], _ = agent.train(next(dataset))  # Initialize variables.
  for _ in range(max(0, args.pretrain - 1)):
    state[0], _ = agent.train(next(dataset), state[0])

  metrics = collections.defaultdict(list)
  def train_step(tran, worker):
    if should_train(step):
      for _ in range(args.train_steps):
        state[0], mets = agent.train(next(dataset), state[0])
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agent.report(next(dataset)))
      if args.log_timings:
        logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.load_or_save()

  print('Start training.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  while step < args.steps:
    logger.write()
    driver(policy, steps=args.eval_every)
    checkpoint.save()
