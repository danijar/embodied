from functools import partial as bind

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree.map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import ninjax as nj
cast = jaxutils.cast_to_compute


class GRU(nj.Module):

  def __init__(self, units=1024, bottleneck=-1, **kw):
    self._units = units
    self._bottleneck = bottleneck
    self._kw = kw

  def initial(self, batch_size):
    return cast(jnp.zeros([batch_size, self._units], f32))

  def __call__(self, state, action, embed, reset, single=False):
    state = cast(state)
    if single:
      state = self._step(state, action, embed, reset)
      return state, state
    state = jaxutils.scan(
        lambda state, inputs: self._step(state, *inputs),
        (action, embed, reset), state, axis=1)
    return state, state[:, -1]

  def _step(self, state, action, embed, reset):
    action = cast(jaxutils.concat_dict(action))
    state = jaxutils.reset(state, reset)
    action = jaxutils.reset(action, reset)
    state = self._gru(state, action, embed)
    return state

  def _gru(self, state, action, embed):
    batch_shape = state.shape[:-1]
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    x = jnp.concatenate([
        cast(action).reshape((*batch_shape, -1)),
        cast(embed).reshape((*batch_shape, -1)),
    ], -1)
    x = jnp.concatenate([state, x], -1)
    kw = {**self._kw, 'units': 3 * self._units, 'act': 'none'}
    x = self.get('gru', Linear, **kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    state = update * cand + (1 - update) * state
    return state


class ImpalaEncoder(nj.Module):

  depth: int = 32
  mults: tuple = (1, 2, 2)
  outmult: int = 16
  blocks: int = 2
  act: str = 'relu'
  norm: str = 'none'
  symlog: bool = True
  layers: int = 5
  units: int = 512

  def __init__(self, spaces, **kw):
    assert all(len(s.shape) <= 3 for s in spaces.values()), spaces
    self.veckeys = [k for k, s in spaces.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in spaces.items() if len(s.shape) == 3]
    self.vecinp = Input(self.veckeys, featdims=1)
    self.imginp = Input(self.imgkeys, featdims=3)
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  def __call__(self, data, bdims=2):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    outs = []

    if self.veckeys:
      x = self.vecinp(data, bdims, f32)
      x = x.reshape((-1, *x.shape[bdims:]))
      x = jaxutils.symlog(x) if self.symlog else x
      x = jaxutils.cast_to_compute(x)
      for i in range(self.layers):
        x = self.get(f'mlp{i}', Linear, self.units, **kw)(x)
      outs.append(x)

    if self.imgkeys:
      print('ENC')
      x = self.imginp(data, bdims, jaxutils.COMPUTE_DTYPE) - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for s, depth in enumerate(self.depths):
        x = self.get(f's{s}in', Conv2D, depth, 3)(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same')
        for b in range(self.blocks):
          skip = x
          x = self.get(f's{s}b{b}n1', Norm, self.norm, act=self.act)(x)
          x = self.get(f's{s}b{b}c1', Conv2D, depth, 3, **self.kw)(x)
          x = self.get(f's{s}b{b}n2', Norm, self.norm, act=self.act)(x)
          x = self.get(f's{s}b{b}c2', Conv2D, depth, 3, **self.kw)(x)
          x += skip
      x = x.reshape((x.shape[0], -1))
      x = self.get('outn1', Norm, self.norm, act=self.act)(x)
      x = self.get('outl', Linear, self.outmult * self.depth, **self.kw)(x)
      x = self.get('outn2', Norm, self.norm, act=self.act)(x)
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    x = x.reshape((*data['is_first'].shape, *x.shape[1:]))
    return x


class MLP(nj.Module):

  layers: int = None
  units: int = None

  def __init__(self, shape, dist='mse', inputs=['tensor'], **kw):
    shape = (shape,) if isinstance(shape, (int, np.integer)) else shape
    assert isinstance(shape, (tuple, dict, type(None))), shape
    assert isinstance(dist, (str, dict)), dist
    assert isinstance(dist, dict) == isinstance(shape, dict), (dist, shape)
    self.shape = shape
    self.dist = dist
    self.inputs = Input(inputs, featdims=1)
    distonly = ('outscale', 'minstd', 'maxstd', 'unimix', 'bins')
    self.lkw = {k: v for k, v in kw.items() if k not in distonly}
    forbidden = ('binit', 'norm', 'act')
    self.dkw = {k: v for k, v in kw.items() if k not in forbidden}

  def __call__(self, inputs, bdims=2, training=False):
    feat = self.inputs(inputs, bdims, jaxutils.COMPUTE_DTYPE)
    x = feat.reshape([-1, feat.shape[-1]])
    for i in range(self.layers):
      x = self.get(f'h{i}', Linear, self.units, **self.lkw)(x)
    x = x.reshape((*feat.shape[:bdims], -1))
    if self.shape is None:
      return x
    elif isinstance(self.shape, dict):
      return {
          k: self._out(k, v, self.dist[k], x) for k, v in self.shape.items()}
    else:
      return self._out('dist', self.shape, self.dist, x)

  def _out(self, name, shape, dist, x):
    name = name.replace('/', '_').replace('.', '_')
    return self.get(name, Dist, shape, dist, **self.dkw)(x)


class Dist(nj.Module):

  outscale: float = 0.1
  minstd: float = 1.0
  maxstd: float = 1.0
  unimix: float = 0.0
  bins: int = 255

  def __init__(self, shape, dist='mse', **kw):
    assert all(isinstance(dim, (int, np.integer)) for dim in shape), shape
    forbidden = ('binit', 'norm', 'act')
    assert all(k not in kw for k in forbidden), (forbidden, kw)
    self.shape = shape
    self.dist = dist
    self.kw = dict(**kw, outscale=self.outscale)

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    shape = self.shape
    padding = 0
    if 'twohot' in self.dist or self.dist == 'softmax':
      padding = int(self.bins % 2)
      shape = (*self.shape, self.bins + padding)

    out = self.get('out', Linear, int(np.prod(shape)), **self.kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
    out = out[..., :-padding] if padding else out

    if 'normal' in self.dist:
      units = int(np.prod(self.shape))
      std = self.get('std', Linear, units, **self.kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self.shape).astype(f32)

    if self.dist == 'symlog_mse':
      return jaxutils.SymlogDist(out, len(self.shape), 'mse', 'sum')

    if self.dist == 'symlog_and_twohot':
      bins = np.linspace(-20, 20, out.shape[-1])
      return jaxutils.TwoHotDist(
          out, bins, len(self.shape), jaxutils.symlog, jaxutils.symexp)

    if self.dist == 'symexp_twohot':
      if out.shape[-1] % 2 == 1:
        half = jnp.linspace(-20, 0, (out.shape[-1] - 1) // 2 + 1, dtype=f32)
        half = jaxutils.symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
      else:
        half = jnp.linspace(-20, 0, out.shape[-1] // 2, dtype=f32)
        half = jaxutils.symexp(half)
        bins = jnp.concatenate([half, -half[::-1]], 0)
      return jaxutils.TwoHotDist(out, bins, len(self.shape))

    if self.dist == 'parab_twohot':
      eps = 0.001
      f = lambda x: np.sign(x) * (np.square(np.sqrt(
          1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps) - 1)
      bins = f(np.linspace(-300, 300, out.shape[-1]))
      return jaxutils.TwoHotDist(out, bins, len(self.shape))

    if self.dist == 'mse':
      return jaxutils.MSEDist(out, len(self.shape), 'sum')

    if self.dist == 'huber':
      return jaxutils.HuberDist(out, len(self.shape), 'sum')

    if self.dist == 'normal_logstd':
      dist = tfd.Normal(out, jnp.exp(std) + 1e-8)
      return tfd.Independent(dist, len(self.shape))

    if self.dist == 'normal_sigmoidstd':
      lo, hi = self.minstd, self.maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self.shape))
      dist.minent = np.prod(self.shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self.shape) * tfd.Normal(0.0, hi).entropy()
      return dist

    if self.dist == 'trunc_normal':
      lo, hi = self.minstd, self.maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.TruncatedNormal(jnp.tanh(out), std, -1, 1)
      dist = tfd.Independent(dist, len(self.shape))
      dist.minent = np.prod(self.shape) * (
          tfd.TruncatedNormal(1.0, lo, -1, 1).entropy())
      dist.maxent = np.prod(self.shape) * (
          tfd.TruncatedNormal(0.0, hi, -1, 1).entropy())
      return dist

    if self.dist == 'binary':
      dist = tfd.Bernoulli(out)
      if self.shape:
        dist = tfd.Independent(dist, len(self.shape))
      return dist

    if self.dist == 'softmax':
      dist = tfd.Categorical(out)
      if len(self.shape) > 1:
        dist = tfd.Independent(dist, len(self.shape) - 1)
      return dist

    if self.dist == 'onehot':
      if self.unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self.unimix) * probs + self.unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self.shape) > 1:
        dist = tfd.Independent(dist, len(self.shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self.shape[:-1]) * np.log(self.shape[-1])
      return dist

    raise NotImplementedError(self.dist)


class ConvBlock(nj.Module):

  norm: str = 'rms'
  act: str = 'gelu'

  def __init__(self, impl, depth, kernel=4, **kw):
    self.impl = impl
    self.depth = depth
    self.kernel = kernel
    self.kw = kw

  def __call__(self, x):
    skip = x
    if self.impl == 'naive':
      kw = dict(**self.kw, norm=self.norm, act=self.act)
      x = self.get('mix', Conv2D, self.depth, self.kernel, **kw)(x)
      x = self.get('mix', Conv2D, self.depth, self.kernel, **kw)(x)
    elif self.impl == 'convnext':
      dkw = dict(**self.kw, groups=self.depth, norm=self.norm)
      x = self.get('mix', Conv2D, self.depth, 7, **dkw)(x)
      x = self.get('up', Linear, 4 * self.depth, act=self.act, **self.kw)(x)
      # x = self.get('grn', Norm, 'grn')(x)
      x = self.get('down', Linear, self.depth, **self.kw)(x)
      x += skip
    elif self.impl == 'convnextv2':
      dkw = dict(**self.kw, groups=self.depth, norm=self.norm)
      x = self.get('mix', Conv2D, self.depth, 7, **dkw)(x)
      x = self.get('up', Linear, 4 * self.depth, act=self.act, **self.kw)(x)
      x = self.get('grn', Norm, 'grn')(x)
      x = self.get('down', Linear, self.depth, **self.kw)(x)
      x += skip
    elif self.impl == 'resnet':
      x = self.get('norm1', Norm, self.norm, act=self.act)(x)
      x = self.get('conv1', Conv2D, self.depth, self.kernel, **self.kw)(x)
      x = self.get('norm2', Norm, self.norm, act=self.act)(x)
      x = self.get('conv2', Conv2D, self.depth, self.kernel, **self.kw)(x)
      x += skip
    elif self.impl == 'resnet_scaled':
      x = self.get('norm1', Norm, self.norm, act=self.act)(x)
      x = self.get('conv1', Conv2D, self.depth, self.kernel, **self.kw)(x)
      x = self.get('norm2', Norm, self.norm, act=self.act)(x)
      x = self.get('conv2', Conv2D, self.depth, self.kernel, **self.kw)(x)
      x = (x + skip) / float(np.sqrt(2))
    elif self.impl == 'resnet_zeroinit':
      x = self.get('norm1', Norm, self.norm, act=self.act)(x)
      x = self.get('conv1', Conv2D, self.depth, self.kernel, **self.kw)(x)
      x = self.get('norm2', Norm, self.norm, act=self.act)(x)
      kw = dict(**self.kw, outscale=1e-10)
      x = self.get('conv2', Conv2D, self.depth, self.kernel, **kw)(x)
      x = (x + skip) / float(np.sqrt(2))
    elif self.impl == 'taming':
      x = self.get('norm1', Norm, self.norm, act=self.act)(x)
      x = self.get('conv1', Conv2D, self.depth, 3, **self.kw)(x)
      x = self.get('norm2', Norm, self.norm, act=self.act)(x)
      x = self.get('conv2', Conv2D, self.depth, 3, **self.kw)(x)
      if skip.shape[-1] != self.depth:
        skip = self.get('skip', Conv2D, self.depth, 1, **self.kw)(skip)
      x += skip
    else:
      raise NotImplementedError(self.impl)
    print(x.shape, self.path)
    return x


class ConvDown(nj.Module):

  norm: str = 'rms'
  act: str = 'gelu'

  def __init__(self, impl, depth, kernel=4, **kw):
    self.impl = impl
    self.depth = depth
    self.kernel = kernel
    self.kw = kw

  def __call__(self, x):
    b, h, w, c = x.shape
    d, k = self.depth, self.kernel
    if self.impl == 'stride_act':
      kw = dict(**self.kw, norm=self.norm, act=self.act)
      x = self.get('conv', Conv2D, d, k, 2, **kw)(x)
    elif self.impl == 'stride':
      x = self.get('conv', Conv2D, d, k, 2, **self.kw)(x)
    elif self.impl == 'norm_stride':
      x = self.get('norm', Norm, self.norm)(x)
      x = self.get('conv', Conv2D, d, k, 2, **self.kw)(x)
    elif self.impl == 'stride_norm':
      x = self.get('conv', Conv2D, d, k, 2, **self.kw)(x)
      x = self.get('norm', Norm, self.norm)(x)
    elif self.impl == 'average_conv':
      x = x.reshape((b, h // 2, w // 2, 4, d)).mean(-2)
      x = self.get('conv', Conv2D, d, k, **self.kw)(x)
    elif self.impl == 'bilinear_conv':
      x = jax.image.resize(x, (b, h // 2, w // 2, d), 'bilinear')
      x = self.get('conv', Conv2D, d, k, **self.kw)(x)
    else:
      raise NotImplementedError(self.impl)
    print(x.shape, self.path)
    return x


class ConvUp(nj.Module):

  norm: str = 'rms'
  act: str = 'gelu'

  def __init__(self, impl, depth, kernel=4, **kw):
    self.impl = impl
    self.depth = depth
    self.kernel = kernel
    self.kw = kw

  def __call__(self, x):
    b, h, w, c = x.shape
    d, k = self.depth, self.kernel
    if self.impl == 'stride_act':
      kw = dict(**self.kw, norm=self.norm, act=self.act)
      x = self.get('conv', Conv2D, d, k, 2, **kw, transp=True)(x)
    elif self.impl == 'stride':
      x = self.get('conv', Conv2D, d, k, 2, **self.kw, transp=True)(x)
    elif self.impl == 'norm_stride':
      x = self.get('norm', Norm, self.norm)(x)
      x = self.get('conv', Conv2D, d, k, 2, **self.kw, transp=True)(x)
    elif self.impl == 'stride_norm':
      x = self.get('conv', Conv2D, d, k, 2, **self.kw, transp=True)(x)
      x = self.get('norm', Norm, self.norm)(x)
    elif self.impl == 'nearest_conv':
      x = x.reshape((b, h, 1, w, 1, c))
      x = jnp.tile(x, [1, 1, 2, 1, 2, 1])
      x = x.reshape((b, h * 2, w * 2, c))
      # x = x.repeat(2, -3).repeat(2, -2)
      x = self.get('conv', Conv2D, d, k, **self.kw)(x)
    elif self.impl == 'bilinear_conv':
      x = jax.image.resize(x, (b, h * 2, w * 2, d), 'bilinear')
      x = self.get('conv', Conv2D, d, k, **self.kw)(x)
    else:
      raise NotImplementedError(self.impl)
    print(x.shape, self.path)
    return x


class Conv2D(nj.Module):

  groups: int = 1
  transp: bool = False
  act: str = 'none'
  norm: str = 'none'
  pad: str = 'same'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = kernel
    self.stride = stride
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    if self.transp:
      assert self.groups == 1, self.groups
      shape = (self.kernel, self.kernel, self.depth, x.shape[-1])
      kernel = self.get('kernel', self._winit, shape)
      kernel = jaxutils.cast_to_compute(kernel)
      flops = int(np.prod(shape)) * x.shape[-3] * x.shape[-2]
      x = jax.lax.conv_transpose(
          x, kernel, (self.stride, self.stride), self.pad.upper(),
          dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
    else:
      G = self.groups
      shape = (self.kernel, self.kernel, x.shape[-1] // G, self.depth)
      kernel = self.get('kernel', self._winit, shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self.stride, self.stride), self.pad.upper(),
          feature_group_count=self.groups,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
      flops = int(np.prod(shape)) * x.shape[-3] * x.shape[-2]
    if self.bias:
      if self.binit:
        args = (self._winit, self.depth, shape)
      else:
        args = (self._binit, self.depth)
      x += self.get('bias', *args).astype(x.dtype)
      flops += int(np.prod(x.shape[-3:]))
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    jaxutils.TRACKSHAPES.append((self.path, x.shape[-3:]))
    jaxutils.TRACKACTS.append((self.path, x.shape))
    jaxutils.TRACKFLOPS.append((self.path, flops))
    return x


class Linear(nj.Module):

  act: str = 'none'
  norm: str = 'none'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, units):
    self.units = (units,) if isinstance(units, int) else tuple(units)
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    shape = (x.shape[-1], int(np.prod(self.units)))
    x = x @ self.get('kernel', self._winit, shape).astype(x.dtype)
    flops = int(np.prod(shape))
    if self.bias:
      if self.binit:
        args = (self._winit, np.prod(self.units), shape)
      else:
        args = (self._binit, np.prod(self.units))
      x += self.get('bias', *args).astype(x.dtype)
      flops += int(np.prod(self.units))
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    jaxutils.TRACKSHAPES.append((self.path, x.shape[-1:]))
    jaxutils.TRACKACTS.append((self.path, x.shape))
    jaxutils.TRACKFLOPS.append((self.path, flops))
    if len(self.units) > 1:
      x = x.reshape(x.shape[:-1] + self.units)
    return x


class BlockLinear(nj.Module):

  act: str = 'none'
  norm: str = 'none'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, units, groups):
    self.units = (units,) if isinstance(units, int) else tuple(units)
    assert groups <= np.prod(units), (groups, units)
    self.groups = groups
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    bdims, indim, outdim = x.shape[:-1], x.shape[-1], np.prod(self.units)
    if indim % self.groups != 0:
      pad = int(np.ceil(indim / self.groups)) * self.groups - indim
      x = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], pad), x.dtype)], -1)
      indim = x.shape[-1]
    assert indim % self.groups == outdim % self.groups == 0, (
        indim, outdim, self.groups, self.units)
    shape = (self.groups, indim // self.groups, outdim // self.groups)
    kernel = self.get('kernel', self._winit, shape).astype(x.dtype)
    flops = int(np.prod(shape))
    x = x.reshape((*bdims, self.groups, indim // self.groups))
    x = jnp.einsum('...ki,kio->...ko', x, kernel)
    x = x.reshape((*bdims, outdim))
    if self.bias:
      if self.binit:
        args = (self._winit, np.prod(self.units), shape)
      else:
        args = (self._binit, np.prod(self.units))
      bias = self.get('bias', *args)
      x += bias.astype(x.dtype)
      flops += int(np.prod(self.units))
    if len(self.units) > 1:
      x = x.reshape(x.shape[:-1] + self.units)
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    jaxutils.TRACKSHAPES.append((self.path, x.shape[-1:]))
    jaxutils.TRACKACTS.append((self.path, x.shape))
    jaxutils.TRACKFLOPS.append((self.path, flops))
    return x


class Norm(nj.Module):

  act: str = 'none'

  def __init__(self, impl, eps=1e-4):
    if '1em' in impl:
      impl, exponent = impl.split('1em')
      eps = 10 ** -int(exponent)
    self._impl = impl
    self._eps = eps

  def __call__(self, x):
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _norm(self, x):
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(f32)
      mean = x.mean(-1)[..., None]
      mean2 = jnp.square(x).mean(-1)[..., None]
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      mult = scale * jax.lax.rsqrt(var + self._eps)
      x = (x - mean) * mult + offset
      return cast(x)
    elif self._impl == 'rms':
      x = x.astype(f32)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      mult = jax.lax.rsqrt((x * x).mean(-1)[..., None] + self._eps) * scale
      return cast(x * mult)
    elif self._impl == 'rms_lowprec':
      scale = self.get('scale', jnp.ones, x.shape[-1], f32).astype(x.dtype)
      mult = jax.lax.rsqrt((x * x).mean(-1)[..., None] + self._eps) * scale
      return x * mult
    elif self._impl == 'rms_instance':
      x = x.astype(f32)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      mult = jax.lax.rsqrt((x * x).mean((-3, -2), keepdims=True) + self._eps)
      mult = mult * scale
      return cast(x * mult)
    elif self._impl == 'grn':
      assert len(x.shape) >= 4, x.shape
      x = x.astype(f32)
      norm = jnp.linalg.norm(x, 2, (-3, -2), keepdims=True)
      norm /= (norm.mean(-1, keepdims=True) + self._eps)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      x = (norm * scale + 1) * x + offset
      return cast(x)
    elif self._impl == 'instance':
      x = x.astype(f32)
      mean = x.mean(axis=(-3, -2), keepdims=True)
      var = x.var(axis=(-3, -2), keepdims=True)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      x = (scale * jax.lax.rsqrt(var + self._eps)) * (x - mean) + offset
      return cast(x)
    else:
      raise NotImplementedError(self._impl)


class Input:

  def __init__(self, keys=['tensor'], featdims=1):
    self.keys = tuple(keys)
    self.featdims = featdims

  def __call__(self, inputs, bdims=2, dtype=None):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    try:
      xs = []
      for key in self.keys:
        x = inputs[key]
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
          x = jnp.concatenate([x.real, x.imag], -1)
        x = x.astype(dtype or inputs[self.keys[0]].dtype)
        x = x.reshape((*x.shape[:bdims + self.featdims - 1], -1))
        msg = f'Invalid input ({nj.SCOPE}, {key}, {x.shape}, {x.dtype}): {{x}}'
        jaxutils.check(jnp.isfinite(x).all(), msg, x=x)
        xs.append(x)
      xs = jnp.concatenate(xs, -1)
    except (KeyError, ValueError, TypeError) as e:
      shapes = {k: v.shape for k, v in inputs.items()}
      raise ValueError(
          f'Error: {e}\n'
          f'Input shapes: {shapes}\n' +
          f'Requested keys: {self.keys}')
    return xs


class Initializer:

  def __init__(self, dist='normed', scale=1.0, fan='in', dtype='default'):
    self.dist = dist
    self.scale = scale
    self.fan = fan
    self.dtype = dtype

  def __call__(self, shape, fan_shape=None):
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    assert all(x > 0 for x in shape), shape
    dtype = jaxutils.PARAM_DTYPE if self.dtype == 'default' else self.dtype
    dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype
    fanin, fanout = self._fans(fan_shape or shape)
    fan = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}[self.fan]
    if self.dist == 'zeros':
      value = jnp.zeros(shape, dtype)
    elif self.dist == 'normed':
      value = jax.random.uniform(nj.seed(), shape, dtype, -1, 1)
      value /= jnp.linalg.norm(value.reshape((-1, shape[-1])), 2, 0)
    elif self.dist == 'uniform':
      limit = np.sqrt(1 / fan)
      value = jax.random.uniform(nj.seed(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      value = jax.random.truncated_normal(nj.seed(), -2, 2, shape)
      value *= 1.137 * np.sqrt(1 / fan)
      value = value.astype(dtype)
    elif self.dist == 'complex':
      assert jnp.issubdtype(dtype, jnp.complexfloating), dtype
      realdt = jnp.finfo(dtype).dtype
      value = jax.random.truncated_normal(
          nj.seed(), -2, 2, (2, *shape), realdt)
      value = value[0] + 1j * value[1]
      value *= jax.lax.convert_element_type(1.137 * np.sqrt(1 / fan), realdt)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.seed(), matshape, dtype)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    value *= self.scale
    return value

  def _fans(self, shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    else:
      space = int(np.prod(shape[:-2]))
      return (shape[-2] * space, shape[-1] * space)


def get_act(name):
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'cswiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      y1, y2 = jnp.split(y, 2, -1)
      pad = jnp.ones_like(y1)
      x = jax.nn.swish(jnp.concatenate([x, -x], -1))
      y = jnp.concatenate([y1, pad, y2, pad], -1)
      return x * y
    return fn
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)
