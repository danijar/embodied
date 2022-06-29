import contextlib
import os
import pathlib
import sys
import time

directory = pathlib.Path(__file__).parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name
sys.argv[0] = str(pathlib.Path(__file__))

import embodied
import numpy as np
import tensorflow as tf
from parameterized import parameterized

from . import agent as agnt


class XLATest(tf.test.TestCase):

  TIMES_V100 = {
      'train_compile_nostate': 38.1,
      'train_compile_state': 25.7,
      'train_run': 0.3,
      'policy_compile_nostate': 3.0,
      'policy_compile_state': 0.3,
      'policy_run': 0.2,
      'report_compile': 23.5,
      'report_run': 0.12,
  }

  @parameterized.expand([
      'dummy_continuous',
      'dummy_discrete',
  ])
  def test_train(self, task):
    assert tf.test.is_gpu_available()
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    config = embodied.Config(agnt.Agent.configs['defaults'])
    config = config.update(task=task)
    env = embodied.envs.load_env(config.task, **config.env)
    step = embodied.Counter()
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    data = self._make_data(
        env, batch_dims=[config.batch_size, config.replay.length])
    timer = Timer(self.TIMES_V100)
    with timer('train_compile_nostate'):
      state, _ = agent.train(data)
    with timer('train_compile_state'):
      state, _ = agent.train(data, state)
    with timer('train_run'):
      state, _ = agent.train(data, state)

  @parameterized.expand([
      'dummy_continuous',
      'dummy_discrete',
  ])
  def test_policy(self, task):
    assert tf.test.is_gpu_available()
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    config = embodied.Config(agnt.Agent.configs['defaults'])
    config = config.update(task=task)
    env = embodied.envs.load_env(config.task, **config.env)
    step = embodied.Counter()
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    data = self._make_data(env, batch_dims=[1])
    timer = Timer(self.TIMES_V100)
    with timer('policy_compile_nostate'):
      _, state = agent.policy(data)
    with timer('policy_compile_state'):
      _, state = agent.policy(data, state)
    with timer('policy_run'):
      _, state = agent.policy(data, state)

  @parameterized.expand([
      'dummy_continuous',
      'dummy_discrete',
  ])
  def test_report(self, task):
    assert tf.test.is_gpu_available()
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    config = embodied.Config(agnt.Agent.configs['defaults'])
    config = config.update(task=task)
    env = embodied.envs.load_env(config.task, **config.env)
    step = embodied.Counter()
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    data = self._make_data(
        env, batch_dims=[config.batch_size, config.replay.length])
    timer = Timer(self.TIMES_V100)
    with timer('report_compile'):
      agent.report(data)
    with timer('report_run'):
      agent.report(data)

  def _make_data(self, env, batch_dims):
    spaces = list(env.obs_space.items()) + list(env.act_space.items())
    data = {k: v.sample() for k, v in spaces}
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data


class Timer:

  def __init__(self, times, slack=0.7):
    self.times = times
    self.slack = slack

  @contextlib.contextmanager
  def __call__(self, name):
    self.name = name
    start = time.time()
    yield self
    duration = time.time() - start
    target = self.times[self.name]
    print(f'{self.name} duration: {duration:.3f}s, target: {target:.3f}s')
    assert self.slack * duration <= target, (
        f'{name} with target {target} took {duration}.')


if __name__ == '__main__':
  tf.test.main()
