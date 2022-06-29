import logging
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd

try:
  from tensorflow.python.distribute import values
except Exception:
  from google3.third_party.tensorflow.python.distribute import values


tf.tensor = tf.convert_to_tensor
for base in (tf.Tensor, tf.Variable, values.PerReplica):
  base.mean = tf.math.reduce_mean
  base.std = tf.math.reduce_std
  base.var = tf.math.reduce_variance
  base.sum = tf.math.reduce_sum
  base.any = tf.math.reduce_any
  base.all = tf.math.reduce_all
  base.min = tf.math.reduce_min
  base.max = tf.math.reduce_max
  base.abs = tf.math.abs
  base.logsumexp = tf.math.reduce_logsumexp
  base.transpose = tf.transpose
  base.reshape = tf.reshape
  base.astype = tf.cast


# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_categorical = tf.random.categorical
def random_categorical(*args, **kwargs):
  kwargs['seed'] = None
  return _orig_random_categorical(*args, **kwargs)
tf.random.categorical = random_categorical

# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_normal = tf.random.normal
def random_normal(*args, **kwargs):
  kwargs['seed'] = None
  return _orig_random_normal(*args, **kwargs)
tf.random.normal = random_normal


def setup(jit=True, platform='gpu', precision=16):
  assert precision in (16, 32)
  tf.config.run_functions_eagerly(not jit)
  tf.config.set_soft_device_placement(False)
  if platform == 'cpu':
    pass
  elif platform == 'gpu':
    assert tf.config.experimental.list_physical_devices('GPU')
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
    if precision == 16:
      prec.set_global_policy(prec.Policy('mixed_float16'))
  else:
    raise NotImplementedError(platform)


def scan(fn, inputs, start, static=True, reverse=False):
  if not static:
    return tf.scan(fn, inputs, start, reverse=reverse)
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(tf.nest.flatten(inputs)[0].shape[0])
  if reverse:
    indices = reversed(indices)
  for index in indices:
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, 0) for x in outputs]
  return tf.nest.pack_sequence_as(start, outputs)


def action_noise(action, amount, act_space):
  if amount == 0:
    return action
  amount = tf.cast(amount, action.dtype)
  if act_space.discrete:
    probs = amount / action.shape[-1] + (1 - amount) * action
    return OneHotDist(probs=probs).sample()
  else:
    return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)


def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * tf.ones_like(reward)
  dims = list(range(reward.shape.ndims))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if axis != 0:
    reward = tf.transpose(reward, dims)
    value = tf.transpose(value, dims)
    pcont = tf.transpose(pcont, dims)
  if bootstrap is None:
    bootstrap = tf.zeros_like(value[-1])
  next_values = tf.concat([value[1:], bootstrap[None]], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = scan(
      lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
      (inputs, pcont), bootstrap, static=True, reverse=True)
  if axis != 0:
    returns = tf.transpose(returns, dims)
  return returns


class Module(tf.Module):

  def save(self):
    values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Save module with {amount} tensors and {count} parameters.')
    return values

  def load(self, values):
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Load module with {amount} tensors and {count} parameters.')
    tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

  def get(self, name, ctor, *args, **kwargs):
    if not hasattr(self, '_modules'):
      self._modules = {}
    if name not in self._modules:
      self._modules[name] = ctor(*args, **kwargs)
    return self._modules[name]


class Optimizer(Module):

  def __init__(
      self, name, lr, opt='adam', eps=1e-4, clip=None,
      wd=None, wd_pattern=r'.*'):
    assert 0 <= wd < 1
    assert not clip or 1 <= clip
    self._name = name
    self._clip = clip
    self._wd = wd
    self._wd_pattern = wd_pattern
    self._opt = {
        'adam': lambda: tf.optimizers.Adam(lr, epsilon=eps),
        'sgd': lambda: tf.optimizers.SGD(lr),
        'momentum': lambda: tf.optimizers.SGD(lr, 0.9),
    }[opt]()
    self._mixed = (prec.global_policy().compute_dtype == tf.float16)
    if self._mixed:
      self._opt = prec.LossScaleOptimizer(self._opt, dynamic=True)
    self._once = True

  @property
  def variables(self):
    return self._opt.variables()

  def __call__(self, tape, loss, modules):
    assert loss.dtype is tf.float32, (self._name, loss.dtype)
    assert len(loss.shape) == 0, (self._name, loss.shape)
    metrics = {}

    # Find variables.
    modules = modules if hasattr(modules, '__len__') else (modules,)
    varibs = tf.nest.flatten([module.variables for module in modules])
    count = sum(np.prod(x.shape) for x in varibs)
    if self._once:
      print(f'Found {count} {self._name} parameters.')
      self._once = False

    # Check loss.
    tf.debugging.check_numerics(loss, self._name + '_loss')
    metrics[f'{self._name}_loss'] = loss

    # Compute scaled gradient.
    if self._mixed:
      with tape:
        loss = self._opt.get_scaled_loss(loss)
    grads = tape.gradient(loss, varibs)
    if self._mixed:
      grads = self._opt.get_unscaled_gradients(grads)
    if self._mixed:
      metrics[f'{self._name}_loss_scale'] = self._opt.loss_scale

    # Distributed sync.
    if tf.distribute.has_strategy():
      context = tf.distribute.get_replica_context()
      grads = context.all_reduce('mean', grads)

    # Gradient clipping.
    norm = tf.linalg.global_norm(grads)
    if not self._mixed:
      tf.debugging.check_numerics(norm, self._name + '_norm')
    if self._clip:
      grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
    metrics[f'{self._name}_grad_norm'] = norm

    # Weight decay.
    if self._wd:
      self._apply_weight_decay(varibs)

    # Apply gradients.
    self._opt.apply_gradients(
        zip(grads, varibs),
        experimental_aggregate_gradients=False)

    return metrics

  def _apply_weight_decay(self, varibs):
    nontrivial = (self._wd_pattern != r'.*')
    if nontrivial:
      print('Applied weight decay to variables:')
    for var in varibs:
      if re.search(self._wd_pattern, self._name + '/' + var.name):
        if nontrivial:
          print('- ' + self._name + '/' + var.name)
        var.assign((1 - self._wd) * var)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=tf.float32):
    super().__init__(logits=logits, probs=probs, dtype=dtype)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
     return super()._parameter_properties(dtype)

  def sample(self, sample_shape=(), seed=None):
    # Straight through biased gradient estimator.
    sample = super().sample(sample_shape, seed)
    probs = self._pad(self.probs_parameter(), sample.shape)
    sample += tf.cast(probs - tf.stop_gradient(probs), sample.dtype)
    return sample

  def _pad(self, tensor, shape):
    while len(tensor.shape) < len(shape):
      tensor = tensor[None]
    return tensor
