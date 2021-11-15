import collections
import pathlib
import re

import embodied
import numpy as np


def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args):

  logdir = pathlib.Path(args.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', logdir)
  should_train = embodied.when.Every(args.train_every)
  should_expl = embodied.when.Until(args.expl_until)
  should_log = embodied.when.Every(args.log_every)
  should_video_train = embodied.when.Every(args.eval_every)
  should_video_eval = embodied.when.Every(args.eval_every)
  step = logger.step

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(args.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(args.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(args.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in args.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(
      lambda tran, _: None if tran['is_first'] else step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  prefill = max(0, args.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = embodied.RandomAgent(train_env.act_space)
    driver_train(random_agent.policy, steps=prefill, episodes=1)
    driver_eval(random_agent.policy, episodes=1)
    driver_train.reset()
    driver_eval.reset()

  dataset_train = iter(agent.dataset(train_replay.sample))
  dataset_report = iter(agent.dataset(train_replay.sample))
  dataset_eval = iter(agent.dataset(eval_replay.sample))
  state = [None]  # To be writable from train step function below.
  state[0], _ = agent.train(next(dataset_train))  # Initialize variables.
  for _ in range(max(0, args.pretrain - 1)):
    state[0], _ = agent.train(next(dataset_train), state[0])

  metrics = collections.defaultdict(list)
  def train_step(tran, worker):
    if should_train(step):
      for _ in range(args.train_steps):
        state[0], mets = agent.train(next(dataset_train), state[0])
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agent.report(next(dataset_report)), prefix='train')
      if args.log_timings:
        logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.load_or_save()

  print('Start training.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    logger.write()
    with timer.scope('eval'):
      logger.add(agent.report(next(dataset_eval)), prefix='eval')
      driver_eval(policy_eval, episodes=args.eval_eps)
    with timer.scope('train'):
      driver_train(policy_train, steps=args.eval_every)
    checkpoint.save()
