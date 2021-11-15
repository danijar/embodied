import collections
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
from parameterized import parameterized


class TestParallel:

  @parameterized.expand(['none', 'thread', 'process'])
  def test_parallel_strategy(self, parallel):
    env = embodied.envs.load_env(
        'dummy_discrete', parallel=parallel, amount=4, length=10)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    episodes = collections.defaultdict(list)
    driver.on_episode(lambda ep, worker: episodes[worker].append(ep))
    driver(agent.policy, episodes=8)
    env.close()
    assert len(episodes) == 4
    assert set(episodes.keys()) == {0, 1, 2, 3}
    for worker, eps in episodes.items():
      assert len(eps) == 2
      assert len(eps[0]['reward']) == 11
      assert len(eps[1]['reward']) == 11

  @parameterized.expand(['thread', 'process'])
  def test_parallel_fast(self, parallel):
    def ctor():
      env = embodied.envs.Dummy('discrete')
      orig = env.step
      env.step = lambda a: [time.sleep(0.1), orig(a)][-1]
      return env
    envs = [embodied.Parallel(ctor, parallel) for _ in range(4)]
    env = embodied.BatchEnv(envs, parallel=True)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    start = time.time()
    driver(agent.policy, steps=4)
    duration = time.time() - start
    env.close()
    assert duration <= 0.2

  def test_sequential_slow(self):
    def ctor():
      env = embodied.envs.Dummy('discrete')
      orig = env.step
      env.step = lambda a: [time.sleep(0.1), orig(a)][-1]
      return env
    envs = [ctor() for _ in range(4)]
    env = embodied.BatchEnv(envs, parallel=False)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    start = time.time()
    driver(agent.policy, steps=4)
    duration = time.time() - start
    env.close()
    assert 0.4 <= duration
