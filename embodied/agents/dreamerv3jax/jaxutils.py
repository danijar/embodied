import re

import jax
import jax.numpy as jnp
import optax
from tensorflow_probability.substrates import jax as tfp

from . import ninjax as nj

tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
COMPUTE_DTYPE = jnp.float32


def cast_to_compute(values):
  return tree_map(lambda x: x.astype(COMPUTE_DTYPE), values)


def parallel():
  try:
    jax.lax.axis_index('devices')
    return True
  except NameError:
    return False


def scan(fn, inputs, start, unroll=True, modify=False):
  fn2 = lambda carry, inp: (fn(carry, inp),) * 2
  if not unroll:
    return nj.scan(fn2, start, inputs, modify=modify)[1]
  length = len(jax.tree_util.tree_leaves(inputs)[0])
  carrydef = jax.tree_util.tree_structure(start)
  carry = start
  outs = []
  for index in range(length):
    carry, out = fn2(carry, tree_map(lambda x: x[index], inputs))
    flat, treedef = jax.tree_util.tree_flatten(out)
    assert treedef == carrydef, (treedef, carrydef)
    outs.append(flat)
  outs = [
      jnp.stack([carry[i] for carry in outs], 0)
      for i in range(len(outs[0]))]
  return carrydef.unflatten(outs)


def symlog(x):
  return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def action_noise(action, amount, act_space):
  if amount == 0:
    return action
  amount = jnp.cast(amount, action.dtype)
  if act_space.discrete:
    probs = amount / action.shape[-1] + (1 - amount) * action
    return OneHotDist(probs=probs).sample(nj.rng())
  else:
    return jnp.clip(tfd.Normal(action, amount).sample(), -1, 1)


class MSEDist:

  def __init__(self, mode, dims, agg='sum'):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._agg = agg
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return self._mode

  def mean(self):
    return self._mode

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = ((self._mode - value) ** 2)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class SymlogDist:

  def __init__(self, mode, dims, agg='sum'):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._agg = agg
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return symexp(self._mode)

  def mean(self):
    return symexp(self._mode)

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = (self._mode - symlog(value)) ** 2
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=jnp.float32):
    super().__init__(logits, probs, dtype)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
     return super()._parameter_properties(dtype)

  def sample(self, sample_shape=(), seed=None):
    sample = sg(super().sample(sample_shape, seed))
    probs = self._pad(super().probs_parameter(), sample.shape)
    return sg(sample) + (probs - sg(probs)).astype(sample.dtype)

  def _pad(self, tensor, shape):
    while len(tensor.shape) < len(shape):
      tensor = tensor[None]
    return tensor


def video_grid(video):
  B, T, H, W, C = video.shape
  return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def balance_stats(dist, target, thres):
  # Values are NaN when there are no positives or negatives in the current
  # batch, which means they will be ignored when aggregating metrics via
  # np.nanmean() later, as they should.
  pos = (target.astype(jnp.float32) > thres).astype(jnp.float32)
  neg = (target.astype(jnp.float32) <= thres).astype(jnp.float32)
  pred = (dist.mean().astype(jnp.float32) > thres).astype(jnp.float32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      rate=pos.mean(),
      avg=target.astype(jnp.float32).mean(),
      pred=dist.mean().astype(jnp.float32).mean(),
  )


class AutoAdapt(nj.Module):

  def __init__(
      self, shape, impl, scale, target, min, max,
      vel=0.1, thres=0.1, inverse=False):
    self._shape = tuple(shape)
    self._impl = impl
    self._fixed_scale = scale
    self._target = target
    self._min = min
    self._max = max
    self._vel = vel
    self._inverse = inverse
    self._thres = thres

  @property
  def shape(self):
    return self._shape

  def __call__(self, reg, update=True):
    update and self.update(reg)
    scale = self.scale()
    loss = scale * (-reg if self._inverse else reg)
    metrics = {
        'mean': reg.mean(),
        'std': reg.std(),
        'scale_mean': scale.mean(),
        'scale_std': scale.std(),
    }
    return loss, metrics

  def scale(self):
    if self._impl == 'fixed':
      return jnp.float32(self._fixed_scale)
    elif self._impl == 'mult':
      return sg(self.get('scale', jnp.ones, self._shape, jnp.float32))
    elif self._impl == 'prop':
      return sg(self.get('scale', jnp.ones, self._shape, jnp.float32))
    else:
      raise NotImplementedError(self._impl)

  def update(self, reg):
    avg = reg.mean(list(range(len(reg.shape) - len(self._shape))))
    if self._impl == 'fixed':
      pass
    elif self._impl == 'mult':
      scale = self.scale()
      below = avg < (1 / (1 + self._thres)) * self._target
      above = avg > (1 + self._thres) * self._target
      if self._inverse:
        below, above = above, below
      inside = ~below & ~above
      adjusted = (
          above.astype(jnp.float32) * scale * (1 + self._vel) +
          below.astype(jnp.float32) * scale / (1 + self._vel) +
          inside.astype(jnp.float32) * scale)
      self.put('scale', jnp.clip(adjusted, self._min, self._max))
    elif self._impl == 'prop':
      direction = avg - self._target
      if self._inverse:
        direction = -direction
      self.put('scale', jnp.clip(
          self._scale + self._vel * direction, self._min, self._max))
    else:
      raise NotImplementedError(self._impl)


class Normalize(nj.Module):

  def __init__(
      self, impl='mean_std', decay=0.99, max=1e8, vareps=0.0, stdeps=0.0):
    self._impl = impl
    self._decay = decay
    self._max = max
    self._stdeps = stdeps
    self._vareps = vareps

  def __call__(self, values, update=True):
    update and self.update(values)
    return self.transform(values)

  def update(self, values):
    x = values.astype(jnp.float32)
    m = self._decay
    mean = self.get('mean', jnp.zeros, (), jnp.float32)
    sqrs = self.get('sqrs', jnp.zeros, (), jnp.float32)
    step = self.get('step', jnp.zeros, (), jnp.int32)
    self.put('step', step + 1)
    self.put('mean', m * mean + (1 - m) * x.mean())
    self.put('sqrs', m * sqrs + (1 - m) * (x ** 2).mean())

  def transform(self, values):
    mean = self.get('mean')
    sqrs = self.get('sqrs')
    step = self.get('step')
    correction = 1 - self._decay ** step.astype(jnp.float32)
    mean = mean / correction
    var = (sqrs / correction) - mean ** 2
    if self._max > 0.0:
      scale = 1 / jnp.sqrt(
          jnp.maximum(var, 1 / self._max ** 2 + self._vareps) + self._stdeps)
    else:
      scale = 1 / jnp.sqrt(var + self._vareps) + self._stdeps
    if self._impl == 'off':
      pass
    elif self._impl == 'mean_std':
      values -= sg(mean.astype(values.dtype))
      values *= sg(scale.astype(values.dtype))
    elif self._impl == 'std':
      values *= sg(scale.astype(values.dtype))
    else:
      raise NotImplementedError(self._impl)
    return values


class Optimizer(nj.Module):

  def __init__(
      self, name, lr, opt='adam', eps=1e-5, clip=100.0, warmup=0, wd=0.0,
      wd_pattern=r'/(w|kernel)$'):
    assert opt in ('adam',)
    assert wd_pattern[0] not in ('0', '1')
    self.name = name
    wd_pattern = re.compile(wd_pattern)
    chain = []
    chain.append(optax.clip_by_global_norm(clip))
    chain.append(optax.scale_by_adam(eps=eps))
    if wd:
      chain.append(optax.additive_weight_decay(wd, lambda params: (
          tree_map(lambda k: bool(wd_pattern.search(k)), tree_keys(params)))))
    chain.append(optax.scale(-lr))
    self.opt = optax.chain(*chain)
    self.step = nj.Variable(jnp.array, 0, jnp.int32)
    self.mixed = (COMPUTE_DTYPE != jnp.float32)
    if self.mixed:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, jnp.float32)
      self.good_steps = nj.Variable(jnp.array, 0, jnp.int32)

  def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
    # TODO: Exclude non-trainable variables.
    def lossfn2(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == jnp.float32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.mixed:
        loss *= sg(self.grad_scale.read())
      return loss, aux
    metrics = {}
    loss, params, grads, aux = nj.grad(
        lossfn2, modules, has_aux=True)(*args, **kwargs)
    if parallel():
      grads = tree_map(lambda x: jax.lax.pmean(x, 'devices'), grads)
    if self.mixed:
      grads = tree_map(lambda x: x / self.grad_scale.read(), grads)
      finite = self._update_scale(grads)
      metrics[f'{self.name}_grad_scale'] = self.grad_scale.read()
      metrics[f'{self.name}_grad_overflow'] = (~finite).astype(jnp.float32)
    optstate = self.get('state', self.opt.init, params)
    updates, optstate = self.opt.update(grads, optstate, params)
    self.put('state', optstate)
    nj.context().update(optax.apply_updates(params, updates))
    norm = optax.global_norm(grads)
    if self.mixed:
      norm = jnp.where(jnp.isfinite(norm), norm, jnp.nan)
    self.step.write(self.step.read() + jnp.isfinite(norm).astype(jnp.int32))
    metrics['loss'] = loss.mean()
    metrics['grad_norm'] = norm
    metrics['grad_steps'] = self.step.read()
    metrics = {f'{self.name}_{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads):
    finite = jnp.array([
        jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(grads)]).all()
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(
        keep.astype(jnp.int32) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        keep.astype(jnp.float32) * self.grad_scale.read() +
        incr.astype(jnp.float32) * self.grad_scale.read() * 2 +
        decr.astype(jnp.float32) * self.grad_scale.read() / 2,
        1e-4, 1e4))
    return finite


def tree_keys(params, prefix=''):
  if hasattr(params, 'items'):
    return type(params)({
        k: tree_keys(v, prefix + '/' + k.lstrip('/'))
        for k, v in params.items()})
  elif isinstance(params, (tuple, list)):
    return [tree_keys(x, prefix) for x in params]
  elif isinstance(params, jnp.ndarray):
    return prefix
  else:
    raise TypeError(type(params))
