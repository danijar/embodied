import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import ninjax as nj


class RSSM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=True, initial='zeros',
      unimix=0.0, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._kw = kw
    self._cast = jaxutils.cast_to_compute

  def initial(self, bs):
    if self._classes:
      state = dict(
          deter=jnp.zeros([bs, self._deter], jnp.float32),
          logit=jnp.zeros([bs, self._stoch, self._classes], jnp.float32),
          stoch=jnp.zeros([bs, self._stoch, self._classes], jnp.float32))
    else:
      state = dict(
          deter=jnp.zeros([bs, self._deter], jnp.float32),
          mean=jnp.zeros([bs, self._stoch], jnp.float32),
          std=jnp.ones([bs, self._stoch], jnp.float32),
          stoch=jnp.zeros([bs, self._stoch], jnp.float32))
    if self._initial == 'zeros':
      return self._cast(state)
    elif self._initial == 'learned':
      # This will cut gradients when the state is created outside of the
      # training graph, but this only happens once at the beginning of the
      # training loop. Afterwards, the state is reset inside the obs_step().
      deter = self.get(
          'initial_deter', jnp.zeros, state['deter'][0].shape, jnp.float32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      state['stoch'] = self.get_stoch(self._cast(state['deter']))
      return self._cast(state)
    else:
      raise NotImplementedError(self._initial)

  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    start = state, state
    post, prior = jaxutils.scan(step, inputs, start, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = state['logit'].astype(jnp.float32)
      dist = tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(jnp.float32)
      std = state['std'].astype(jnp.float32)
      dist = tfp.MultivariateNormalDiag(mean, std)
    return dist

  def obs_step(self, prev_state, prev_action, embed, is_first):
    prev_action = self._cast(prev_action)
    is_first = self._cast(is_first)
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    prior = self.img_step(prev_state, prev_action)
    x = jnp.concatenate([prior['deter'], embed], -1)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats_layer('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return self._cast(post), self._cast(prior)

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = self._cast(prev_action)
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    x, deter = self._gru(x, prev_state['deter'])
    x = self.get('img_out', Linear, **self._kw)(x)
    stats = self._stats_layer('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return self._cast(prior)

  def get_stoch(self, deter):
    x = self.get('img_out', Linear, **self._kw)(deter)
    stats = self._stats_layer('img_stats', x)
    dist = self.get_dist(stats)
    return self._cast(dist.mode())

  def _gru(self, x, deter):
    x = jnp.concatenate([deter, x], -1)
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = self.get('gru', Linear, **kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter, deter

  def _stats_layer(self, name, x):
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      return {'logit': logit}
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    mask = mask.astype(value.dtype)
    # if jaxutils.parallel():
    #   return jnp.einsum('db...,db->db...', value, mask)
    # else:
    return jnp.einsum('b...,b->b...', value, mask)

  def kl_loss(self, post, prior, balance=0.8):
    lhs = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
    rhs = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    return balance * lhs + (1 - balance) * rhs


class MultiEncoder(nj.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
      mlp_units=512, cnn='stride', cnn_depth=48, cnn_kernels=(4, 4, 4, 4),
      cnn_blocks=2, **kw):
    excluded = ('is_first', 'is_last')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) in (0, 1)}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)
    if cnn == 'stride':
      self._cnn = ImageEncoderStride(cnn_depth, cnn_kernels, **kw)
    elif cnn == 'resize':
      self._cnn = ImageEncoderResize(cnn_depth, cnn_kernels, **kw)
    elif cnn == 'resnet':
      self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, **kw)
    else:
      raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(None, mlp_layers, mlp_units, dist='none', **kw)

  def __call__(self, data):
    some_key, some_shape = list(self.shapes.items())[0]
    batch_dims = data[some_key].shape[:-len(some_shape)]
    data = {
        k: v.reshape((-1,) + v.shape[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_shapes:
      inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)
      output = self._cnn(inputs)
      output = output.reshape((output.shape[0], -1))
      outputs.append(output)
    if self.mlp_shapes:
      inputs = [
          data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
          for k in self.mlp_shapes]
      inputs = jnp.concatenate([x.astype(jnp.float32) for x in inputs], -1)
      outputs.append(self._mlp(jaxutils.cast_to_compute(inputs)))
    outputs = jnp.concatenate(outputs, -1)
    outputs = outputs.reshape(batch_dims + outputs.shape[1:])
    return outputs


class MultiDecoder(nj.Module):

  def __init__(
      self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
      mlp_layers=4, mlp_units=512, cnn='stride', cnn_depth=48,
      cnn_kernels=(5, 5, 6, 6), cnn_blocks=2, image_dist='mse', **kw):
    excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) == 1}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)
    if self.cnn_shapes:
      shapes = list(self.cnn_shapes.values())
      assert all(x[:-1] == shapes[0][:-1] for x in shapes)
      merged = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      if cnn == 'stride':
        self._cnn = ImageDecoderStride(merged, cnn_depth, cnn_kernels, **kw)
      elif cnn == 'resize':
        self._cnn = ImageDecoderResize(merged, cnn_depth, cnn_kernels, **kw)
      elif cnn == 'resnet':
        self._cnn = ImageDecoderResnet(merged, cnn_depth, cnn_blocks, **kw)
      else:
        raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, **kw)
    self._inputs = Input(inputs)
    self._image_dist = image_dist

  def __call__(self, inputs):
    features = self._inputs(inputs)
    dists = {}
    if self.cnn_shapes:
      flat = features.reshape([-1, features.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(features.shape[:-1] + output.shape[1:])
      means = jnp.split(output, [v[-1] for v in self.cnn_shapes.values()], -1)
      dists.update({
          key: self._make_image_dist(key, mean)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
    if self.mlp_shapes:
      dists.update(self._mlp(features))
    return dists

  def _make_image_dist(self, name, mean):
    mean = mean.astype(jnp.float32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3)
    if self._image_dist == 'mse':
      return jaxutils.MSEDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)


class ImageEncoderStride(nj.Module):

  def __init__(self, depth, kernels, **kw):
    self._depth = depth
    self._kernels = kernels
    self._kw = kw

  def __call__(self, x):
    Conv = functools.partial(Conv2D, stride=2, pad='valid')
    x = jaxutils.cast_to_compute(x)
    depth = self._depth
    for i, kernel in enumerate(self._kernels):
      x = self.get(f'conv{i}', Conv, depth, kernel, **self._kw)(x)
      depth *= 2
    return x


class ImageDecoderStride(nj.Module):

  def __init__(self, shape, depth, kernels, **kw):
    self._shape = shape
    self._depth = depth
    self._kernels = kernels
    self._kw = kw

  def __call__(self, x):
    ConvT = functools.partial(Conv2D, transp=True, stride=2, pad='valid')
    x = jaxutils.cast_to_compute(x)
    x = x.reshape([-1, 1, 1, x.shape[-1]])
    depth = self._depth * 2 ** (len(self._kernels) - 2)
    for i, kernel in enumerate(self._kernels[:-1]):
      x = self.get(f'conv{i}', ConvT, depth, kernel, **self._kw)(x)
      depth //= 2
    x = self.get('out', ConvT, self._shape[-1], self._kernels[-1])(x)
    x = jax.nn.sigmoid(x)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    return x


class ImageEncoderResize(nj.Module):

  def __init__(self, depth, kernels, **kw):
    self._depth = depth
    self._kernels = kernels
    self._kw = kw

  def __call__(self, x):
    x = jaxutils.cast_to_compute(x)
    depth = self._depth
    for i, kernel in enumerate(self._kernels):
      x = self.get(f'conv{i}', Conv2D, depth, kernel, **self._kw)(x)
      # print(x.shape)
      x = jax.lax.reduce_window(
          x, 0.0, jax.lax.add, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME') / 4.0
      # print(x.shape)
      depth *= 2
    return x


class ImageDecoderResize(nj.Module):

  def __init__(self, shape, depth, kernels, **kw):
    self._shape = shape
    self._depth = depth
    self._kernels = kernels
    self._kw = kw

  def __call__(self, features):
    depth = self._depth * 2 ** (len(self._kernels) - 1)
    x = jaxutils.cast_to_compute(features)
    x = self.get('in', Linear, (4, 4, depth), **self._kw)(x)
    # print(x.shape)
    for i, kernel in enumerate(self._kernels):
      last = (i == len(self._kernels) - 1)
      depth = self._shape[-1] if last else depth // 2
      kw = {'act': jax.nn.sigmoid} if last else self._kw
      x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)  # Upsample
      # print(x.shape)
      x = self.get(f'conv{i}', Conv2D, depth, kernel, **kw)(x)
      # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    return x


class ImageEncoderResnet(nj.Module):

  def __init__(self, depth, blocks, **kw):
    self._depth = depth
    self._blocks = blocks
    self._kw = {**kw, 'preact': True}

  def __call__(self, x):
    x = jaxutils.cast_to_compute(x)
    stages = int(np.log2(x.shape[-2])) - 2
    depth = self._depth
    x = self.get('in', Conv2D, depth, 3)(x)
    for i in range(stages):
      x = jax.lax.reduce_window(
          x, 0.0, jax.lax.add, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME') / 4.0
      for j in range(self._blocks):
        x = self._block(f's{i}b{j}', depth, x)
        # print(i, j, x.shape)
      depth *= 2
    x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
    x = self.get('out', Linear, 1024)(x)
    return x

  def _block(self, name, depth, x):
    skip = x
    if skip.shape[-1] != depth:
      skip = self.get(f'{name}s', Conv2D, depth, 1, bias=False)(skip)
    x = self.get(f'{name}a', Conv2D, depth, 3, **self._kw)(x)
    x = self.get(f'{name}b', Conv2D, depth, 3, **self._kw)(x)
    return skip + 0.1 * x


class ImageDecoderResnet(nj.Module):

  def __init__(self, shape, depth, blocks, **kw):
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._kw = {**kw, 'preact': True}

  def __call__(self, x):
    x = jaxutils.cast_to_compute(x)
    stages = int(np.log2(self._shape[0])) - 2
    depth = 2 ** stages * self._depth
    x = self.get('in', Linear, 16 * depth)(x)
    x = x.reshape([-1, 4, 4, depth])
    for i in range(stages):
      for j in range(self._blocks):
        x = self._block(f's{i}b{j}', depth, x)
        # print(i, j, x.shape)
      x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)  # Upsample
      depth //= 2
    x = self.get('out', Conv2D, self._shape[-1], 3)(x)
    x = jax.nn.sigmoid(x)
    return x

  def _block(self, name, depth, x):
    skip = x
    if skip.shape[-1] != depth:
      skip = self.get(f'{name}s', Conv2D, depth, 1, bias=False)(skip)
    x = self.get(f'{name}a', Conv2D, depth, 3, **self._kw)(x)
    x = self.get(f'{name}b', Conv2D, depth, 3, **self._kw)(x)
    return skip + 0.1 * x


class MLP(nj.Module):

  def __init__(self, shape, layers, units, inputs=['tensor'], dims=None, **kw):
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    self._shape = shape
    self._layers = layers
    self._units = units
    self._inputs = Input(inputs, dims=dims)
    distkeys = ('dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs):
    feat = self._inputs(inputs)
    x = jaxutils.cast_to_compute(feat)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self._layers):
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
    if self._shape is None:
      return x
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x)
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()}
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', Dist, shape, **self._dist)(x)


class Dist(nj.Module):

  def __init__(
      self, shape, dist='mse', outscale=0.1, minstd=0.1, maxstd=1.0,
      unimix=0.0):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._dist = dist
    self._minstd = minstd
    self._maxstd = maxstd
    self._unimix = unimix
    self._outscale = outscale

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    kw = {}
    kw['outscale'] = self._outscale
    out = self.get('out', Linear, int(np.prod(self._shape)), **kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + self._shape).astype(jnp.float32)
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', Linear, np.prod(self._shape), **kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self._shape).astype(jnp.float32)
    if self._dist == 'symlog':
      return jaxutils.SymlogDist(out, len(self._shape), 'sum')
    if self._dist == 'mse':
      return jaxutils.MSEDist(out, len(self._shape), 'sum')
    if self._dist == 'normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * jax.nn.sigmoid(std) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self._shape))
      dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'trunc_normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * jax.nn.sigmoid(std) + lo
      dist = tfd.TruncatedNormal(jnp.tanh(out), std, -1, 1)
      dist = tfd.Independent(dist, 1)
      dist.minent = np.prod(self._shape) * tfd.Normal(0.99, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'onehot':
      if self._unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self._shape) > 1:
        dist = tfd.Independent(dist, len(self._shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
      return dist
    raise NotImplementedError(self._dist)


class Conv2D(nj.Module):

  def __init__(
      self, depth, kernel, stride=1, transp=False, act='none', norm='none',
      pad='same', bias=True, preact=False):
    self._depth = depth
    self._kernel = kernel
    self._stride = stride
    self._pad = pad.upper()
    self._transp = transp
    self._act = get_act(act)
    self._norm = Norm(norm)
    self._preact = preact
    self._bias = bias

  def __call__(self, hidden):
    if self._preact:
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
      hidden = self._layer(hidden)
    else:
      hidden = self._layer(hidden)
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
    return hidden

  def _layer(self, x):
    if self._transp:
      shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
      limit = np.sqrt(3.0 / (np.prod(shape[:-2]) * np.mean(shape[-2:])))
      kernel = self.get(
          'kernel', jax.random.uniform, nj.rng(), shape, jnp.float32,
          -limit, limit)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_transpose(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
    else:
      shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
      limit = np.sqrt(3.0 / (np.prod(shape[:-2]) * np.mean(shape[-2:])))
      kernel = self.get(
          'kernel', jax.random.uniform, nj.rng(), shape, jnp.float32,
          -limit, limit)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self._bias:
      bias = self.get('bias', jnp.zeros, self._depth, np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    return x


class Linear(nj.Module):

  def __init__(self, units, act='none', norm='none', bias=True, outscale=1.0):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,)
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._outscale = outscale

  def __call__(self, x):
    fanin, fanout = x.shape[-1], np.prod(self._units)
    total = np.prod(self._units)
    limit = np.sqrt(3.0 * self._outscale / np.mean((fanin, fanout)))
    kernel = self.get(
        'kernel', jax.random.uniform, nj.rng(), (fanin, fanout), jnp.float32,
        -limit, limit)
    kernel = jaxutils.cast_to_compute(kernel)
    x = x @ kernel
    if self._bias:
      bias = self.get('bias', jnp.zeros, fanout, np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    if len(self._units) > 1:
      x = x.reshape(x.shape[:-1] + self._units)
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x


class Norm(nj.Module):

  def __init__(self, impl):
    self._impl = impl

  def __call__(self, x):
    dtype = x.dtype
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(jnp.float32)
      x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
      x *= self.get('scale', jnp.ones, x.shape[-1], jnp.float32)
      x += self.get('bias', jnp.zeros, x.shape[-1], jnp.float32)
      return x.astype(dtype)
    else:
      raise NotImplementedError(self._impl)


class Input:

  def __init__(self, keys=['tensor'], dims=None):
    assert isinstance(keys, (list, tuple)), keys
    self._keys = tuple(keys)
    self._dims = dims or self._keys[0]

  def __call__(self, inputs):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    if not all(k in inputs for k in self._keys):
      needs = f'{{{", ".join(self._keys)}}}'
      found = f'{{{", ".join(inputs.keys())}}}'
      raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
    values = [inputs[k] for k in self._keys]
    dims = len(inputs[self._dims].shape)
    for i, value in enumerate(values):
      if len(value.shape) > dims:
        values[i] = value.reshape(
            value.shape[:dims - 1] + (np.prod(value.shape[dims - 1:]),))
    values = [x.astype(inputs[self._dims].dtype) for x in values]
    return jnp.concatenate(values, -1)


def get_act(name):
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)
