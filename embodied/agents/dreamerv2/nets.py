import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras import mixed_precision as prec

from . import tfutils


class EnsembleRSSM(tfutils.Module):

  def __init__(
      self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
      act='elu', norm='none', std_act='softplus', min_std=0.1,
      unroll=True):
    super().__init__()
    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = get_act(act)
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std
    self._unroll = unroll
    self._cell = GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    start = state, state
    post, prior = tfutils.scan(step, inputs, start, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = tfutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, ensemble=False):
    if ensemble:
      state = self._suff_stats_ensemble(state['deter'])
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(tfutils.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
    # if is_first.any():
    prev_state, prev_action = tf.nest.map_structure(
        lambda x: tf.einsum(
            'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
        (prev_state, prev_action))
    prior = self.img_step(prev_state, prev_action, sample)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
    x = self.get('obs_out_norm', NormLayer, self._norm)(x)
    x = self._act(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    post = {k: v.astype(prev_state[k].dtype) for k, v in post.items()}
    return post, prior

  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', NormLayer, self._norm)(x)
    x = self._act(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    stats = self._suff_stats_ensemble(x)
    index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    prior = {k: v.astype(prev_state[k].dtype) for k, v in prior.items()}
    return prior

  def _suff_stats_ensemble(self, inp):
    bs = list(inp.shape[:-1])
    inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
      x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
      x = self._act(x)
      stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
    stats = {
        k: tf.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    stats = {
        k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
        for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class MultiEncoder(tfutils.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      mlp_layers=4, mlp_units=512, cnn_depth=48, cnn_kernels=(4, 4, 4, 4)):
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) == 1}
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)
    self._cnn = ImageEncoder(cnn_depth, cnn_kernels, act, norm)
    self._mlps = {
        k: MLP(None, mlp_layers, mlp_units, act, norm, dist='none')
        for k in self.mlp_shapes.keys()}

  def __call__(self, data):
    some_key, some_shape = (
        list(self.mlp_shapes.items()) +
        list(self.cnn_shapes.items()))[0]
    batch_dims = tuple(data[some_key].shape[:-len(some_shape)])
    data = {
        k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_shapes:
      inputs = tf.concat([data[k] for k in self.cnn_shapes], -1)
      output = self._cnn(inputs)
      output = output.reshape((output.shape[0], -1))
      outputs.append(output)
    for key, net in self._mlps.items():
      outputs.append(net(data[key]))
    outputs = tf.concat(outputs, -1)
    outputs = outputs.reshape(batch_dims + outputs.shape[1:])
    return outputs


class MultiDecoder(tfutils.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      mlp_layers=4, mlp_units=512, cnn_depth=48, cnn_kernels=(5, 5, 6, 6)):
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) == 1}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)
    if self.cnn_shapes:
      shapes = list(self.cnn_shapes.values())
      assert all(x[:-1] == shapes[0][:-1] for x in shapes)
      merged = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      self._cnn = ImageDecoder(merged, cnn_depth, cnn_kernels, act, norm)
    if self.mlp_shapes:
      self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, act, norm)

  def __call__(self, features):
    dists = {}
    if self.cnn_shapes:
      flat = features.reshape([-1, features.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(features.shape[:-1] + output.shape[1:])
      means = tf.split(output, [v[-1] for v in self.cnn_shapes.values()], -1)
      dists.update({
          key: tfd.Independent(tfd.Normal(mean, 1), 3)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
    if self.mlp_shapes:
      dists.update(self._mlp(features))
    return dists


class ImageEncoder(tfutils.Module):

  def __init__(self, depth, kernels, act, norm):
    self._depth = depth
    self._kernels = kernels
    self._act = get_act(act)
    self._norm = norm

  def __call__(self, features):
    x = features.astype(prec.global_policy().compute_dtype)
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** i * self._depth
      x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
      x = self.get(f'norm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x


class ImageDecoder(tfutils.Module):

  def __init__(self, shape, depth, kernels, act, norm):
    self._shape = shape
    self._depth = depth
    self._kernels = kernels
    self._act = get_act(act)
    self._norm = norm

  def __call__(self, features):
    features = tf.cast(features, prec.global_policy().compute_dtype)
    ConvT = tfkl.Conv2DTranspose
    x = self.get('in', tfkl.Dense, 32 * self._depth)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act, norm = self._act, self._norm
      if i == len(self._kernels) - 1:
        depth, act, norm = self._shape[-1], tf.identity, 'none'
      x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
      x = self.get(f'norm{i}', NormLayer, norm)(x)
      x = act(x)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    return x


class MLP(tfutils.Module):

  def __init__(self, shape, layers, units, act='elu', norm='none', **out):
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    self._shape = shape
    self._layers = layers
    self._units = units
    self._norm = norm
    self._act = get_act(act)
    self._out = out

  def __call__(self, features):
    x = tf.cast(features, prec.global_policy().compute_dtype)
    x = x.reshape([-1, x.shape[-1]])
    for index in range(self._layers):
      x = self.get(f'dense{index}', tfkl.Dense, self._units)(x)
      x = self.get(f'norm{index}', NormLayer, self._norm)(x)
      x = self._act(x)
    x = x.reshape(features.shape[:-1] + [x.shape[-1]])
    if self._shape is None:
      return x
    elif isinstance(self._shape, tuple):
      return self.get('out', DistLayer, self._shape, **self._out)(x)
    elif isinstance(self._shape, dict):
      return {
          k: self.get(f'out{k}', DistLayer, v, **self._out)(x)
          for k, v in self._shape.items()}
    else:
      raise ValueError(self._shape)


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = get_act(act)
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    inputs = inputs.astype(state.dtype)
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, inputs.dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class DistLayer(tfutils.Module):

  def __init__(
      self, shape, dist='mse', min_std=0.1, max_std=2.0):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._dist = dist
    self._min_std = min_std
    self._max_std = max_std

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
    out = tf.reshape(out, tuple(inputs.shape[:-1]) + self._shape)
    out = out.astype(tf.float32)
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tuple(inputs.shape[:-1]) + self._shape)
      std = std.astype(tf.float32)
    if self._dist == 'mse':
      dist = tfd.Normal(out, 1.0)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'trunc_normal':
      std = self._max_std * tf.nn.sigmoid(std) + self._min_std
      dist = tfd.TruncatedNormal(tf.tanh(out), std, -1, 1)
      return tfd.Independent(dist, 1)
    if self._dist == 'onehot':
      return tfutils.OneHotDist(out)
    raise NotImplementedError(self._dist)


class NormLayer(tfutils.Module):

  def __init__(self, name):
    if name == 'none':
      self._layer = None
    elif name == 'layer':
      self._layer = tfkl.LayerNormalization()
    else:
      raise NotImplementedError(name)

  def __call__(self, features):
    if not self._layer:
      return features
    return self._layer(features)


def get_act(name):
  if name == 'none':
    return tf.identity
  if name == 'mish':
    return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
  elif hasattr(tf.nn, name):
    return getattr(tf.nn, name)
  elif hasattr(tf, name):
    return getattr(tf, name)
  else:
    raise NotImplementedError(name)
