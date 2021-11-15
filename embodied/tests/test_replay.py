import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np


class TestReplay:

  # Example trajectory:
  # idx: -1    0        1      ...  9      10      11
  # obs: zeros is_first mid         mid    is_last is_first
  # act: reset policy   policy      policy reset   policy

  def test_internal_content(self, tmpdir):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    replay = embodied.SequenceReplay(tmpdir, length=5, parallel=False)
    driver = embodied.Driver(env)
    driver.on_step(replay.add)
    driver(agent.policy, episodes=2)
    assert replay.stats['total_steps'] == 20
    assert replay.stats['total_episodes'] == 2
    assert len(replay._complete_eps) == 2
    for completed in replay._complete_eps.values():
      assert len(completed['action']) == 11
      assert (completed['step'] == np.arange(11)).all()
    for ongoing in replay._ongoing_eps.values():
      assert len(ongoing) == 0

  def test_sample_uniform(self, tmpdir):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    replay = embodied.SequenceReplay(
        tmpdir, length=10, prioritize_ends=False, parallel=False, seed=0)
    driver = embodied.Driver(env)
    driver.on_step(replay.add)
    driver(agent.policy, episodes=1)
    count1, count2 = 0, 0
    iterator = replay.sample()
    for _ in range(100):
      sample = next(iterator)['step']
      count1 += (sample == np.arange(0, 10)).all()
      count2 += (sample == np.arange(1, 11)).all()
    assert count1 + count2 == 100
    assert count1 > 30
    assert count2 > 30

  def test_unexpected_reset(self, tmpdir):

    class UnexpectedReset(embodied.Wrapper):
      """Send is_first without preceeding is_last."""
      def __init__(self, env, when):
        super().__init__(env)
        self._when = when
        self._step = 0
      def step(self, action):
        if self._step == self._when:
          action = action.copy()
          action['reset'] = np.ones_like(action['reset'])
        self._step += 1
        return self.env.step(action)

    env = embodied.envs.load_env('dummy_discrete', length=10)
    env = UnexpectedReset(env, when=8)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    replay = embodied.SequenceReplay(tmpdir, length=5, parallel=False)
    driver.on_step(replay.add)
    driver.on_step(lambda t, _: print(t['is_first']))
    driver(agent.policy, episodes=2)
    assert replay.stats['total_steps'] == 20
    assert replay.stats['total_episodes'] == 2
    assert len(replay._complete_eps) == 2
    for completed in replay._complete_eps.values():
      assert len(completed['action']) == 11
      assert (completed['step'] == np.arange(11)).all()
