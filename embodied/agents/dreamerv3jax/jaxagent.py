from functools import partial as bind
import os

import embodied
import jax
import jax.numpy as jnp
import numpy as np

from . import ninjax as nj


def Wrapper(agent_cls):
  class Agent(JAXAgent):
    configs = agent_cls.configs
    def __init__(self, obs_space, act_space, step, config):
      super().__init__(agent_cls, obs_space, act_space, step, config)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, step, config):
    self.config = config.jax
    self.setup()
    self.agent = agent_cls(obs_space, act_space, step, config)
    self.rng = jax.random.PRNGKey(config.seed)
    self.varibs = {}
    self._init_policy = bind(nj.run, self.agent.initial_policy_state)
    self._init_train = bind(nj.run, self.agent.initial_train_state)
    self._policy = bind(nj.run, self.agent.policy)
    self._train = bind(nj.run, self.agent.train)
    self._report = bind(nj.run, self.agent.report)
    if self.config.parallel:
      self._init_train = jax.pmap(self._init_train, 'devices')
      self._init_policy = jax.pmap(self._init_policy, 'devices')
      self._train = jax.pmap(self._train, 'devices')
      self._policy = jax.pmap(
          self._policy, 'devices', static_broadcasted_argnums=[4])
      self._report = jax.pmap(self._report, 'devices')
    else:
      self._train = jax.jit(self._train)
      self._policy = jax.jit(self._policy, static_argnums=[4])
      self._report = jax.jit(self._report)

  def setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
    except Exception as e:
      print('Could not disable TensorFlow GPU:', e)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    xla_flags = []
    xla_flags.append('--xla_gpu_strict_conv_algorithm_picker=false')  # TODO
    if self.config.logical_cpus:
      count = self.config.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.config.platform)
    jax.config.update('jax_disable_jit', not self.config.jit)
    jax.config.update('jax_debug_nans', self.config.debug_nans)
    # jax.config.update('jax_log_compiles', True)
    # jax.config.update('jax_enable_x64', self.config.enable_x64)
    print(f'JAX DEVICES ({jax.local_device_count()}):', jax.devices())

  def train(self, data, state=None):
    data = self._convert_inps(data)
    rng = self._shard_rng(self._next_rng(), mirror=not bool(self.varibs))
    if state is None:
      state, self.varibs = self._init_train(self.varibs, rng, data)
    (outs, state, mets), self.varibs = self._train(
        self.varibs, rng, data, state)
    outs = self._convert_outs(outs)
    mets = self._convert_mets(mets)
    return outs, state, mets

  def policy(self, obs, state=None, mode='train'):
    assert self.varibs, 'call train() first to initialize variables'

    # varibs = jax.tree_map(lambda x: x[0], self.varibs)
    # if state is None:
    #   state, _ = self._init_policy(varibs, self._next_rng(), obs)
    # (outs, state), _ = self._policy(
    #     varibs, self._next_rng(), obs, state, mode)

    obs = self._convert_inps(obs)
    rng = self._shard_rng(self._next_rng())
    if state is None:
      state, _ = self._init_policy(self.varibs, rng, obs)
    (outs, state), _ = self._policy(self.varibs, rng, obs, state, mode)
    outs = self._convert_outs(outs)

    return outs, state

  def report(self, data):
    assert self.varibs, 'call train() first to initialize variables'
    data = self._convert_inps(data)
    rng = self._shard_rng(self._next_rng())
    mets, _ = self._report(self.varibs, rng, data)
    mets = self._convert_mets(mets)
    return mets

  def dataset(self, generator):
    return self.agent.dataset(generator)

  def save(self):
    data = jax.tree_util.tree_flatten(
        jax.tree_map(jnp.asarray, self.varibs))[0]
    data = [np.array(x) for x in data]
    return data

  def load(self, state):
    self.varibs = jax.tree_util.tree_flatten(self.varibs)[1].unflatten(state)

  def _convert_inps(self, value):
    if isinstance(value, (tuple, list, dict)):
      return jax.tree_map(self._convert_inps, value)
    if self.config.parallel:
      replicas = jax.local_device_count()
      assert len(value) % replicas == 0, (value.shape, replicas)
      return value.reshape((replicas, -1) + value.shape[1:])
    return value

  def _convert_outs(self, value):
    if isinstance(value, (tuple, list, dict)):
      return jax.tree_map(self._convert_outs, value)
    if self.config.parallel:
      return jnp.asarray(value.reshape((-1,) + value.shape[2:]))
    return jnp.asarray(value)  # TODO: Force copy here with np.array()?

  def _convert_mets(self, value):
    if isinstance(value, (tuple, list, dict)):
      return jax.tree_map(self._convert_mets, value)
    if self.config.parallel:
      return jnp.asarray(value[0])
    return jnp.asarray(value)

  # def _convert_inps(self, value):
  #   if self.config.parallel:
  #     replicas = jax.local_device_count()
  #     flat = jax.tree_util.tree_flatten(value)[0]
  #     print([x.shape])
  #     for x in flat:
  #       assert len(x) == len(flat[0]), (x.shape, flat[0].shape)
  #     size = len(flat[0])
  #     split = jax.tree_map(lambda x: np.split(x, replicas), value)
  #     shards = [jax.tree_map(lambda x: x[i], split) for i in range(size)]
  #     value = jax.device_put_sharded(shards, jax.devices())
  #   else:
  #     value = jax.device_put(value)
  #   return value

  # def _convert_outs(self, value):
  #   value = jax.device_get(value)
  #   value = jax.tree_map(jnp.asarray, value)
  #   if self.config.parallel:
  #     value = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
  #   return value

  # def _convert_mets(self, value):
  #   value = jax.device_get(value)
  #   value = jax.tree_map(jnp.asarray, value)
  #   if self.config.parallel:
  #     value = jax.tree_map(lambda x: x[0], value)
  #   return value

  def _next_rng(self):
    self.rng, rng = jax.random.split(self.rng)
    return rng

  def _shard_rng(self, rng, mirror=False):
    if not self.config.parallel:
      return rng
    elif mirror:
      replicas = jax.local_device_count()
      return jnp.repeat(rng[None], replicas, axis=0)
    else:
      replicas = jax.local_device_count()
      return jax.random.split(rng, replicas)
