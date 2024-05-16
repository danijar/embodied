import re
from functools import partial as bind

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import checkify
from tensorflow_probability.substrates import jax as tfp

from . import ninjax as nj

tfd = tfp.distributions
sg = lambda x: jax.tree.map(jax.lax.stop_gradient, x)
f32 = jnp.float32
i32 = jnp.int32
COMPUTE_DTYPE = f32
PARAM_DTYPE = f32
ENABLE_CHECKS = False
TRACKSHAPES = []
TRACKACTS = []
TRACKFLOPS = []


def cast_to_compute(values):
  return jax.tree.map(lambda x: x.astype(COMPUTE_DTYPE), values)


def get_param_dtype():
  return PARAM_DTYPE


def check(predicate, message, **kwargs):
  if ENABLE_CHECKS:
    checkify.check(predicate, message, **kwargs)


def parallel():
  try:
    jax.lax.axis_index('i')
    return True
  except NameError:
    return False


def tensorstats(tensor, prefix=None):
  assert tensor.size > 0, tensor.shape
  assert jnp.issubdtype(tensor.dtype, jnp.floating), tensor.dtype
  tensor = tensor.astype(f32)  # To avoid overflows.
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).mean(),
      'min': tensor.min(),
      'max': tensor.max(),
      'dist': subsample(tensor),
  }
  if prefix:
    metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
  return metrics


def subsample(values, amount=1024):
  values = values.flatten()
  if len(values) > amount:
    values = jax.random.permutation(nj.seed(), values)[:amount]
  return values


def masked_mean(tensor, mask, axis=None):
  assert tensor.shape == mask.shape
  mask = mask.astype(tensor.dtype)
  return (mask * tensor).sum(axis) / mask.sum(axis)


def scan(fn, inputs, start, unroll=False, axis=0):
  if axis:
    inputs = jax.tree.map(lambda x: x.swapaxes(0, axis), inputs)
  fn2 = lambda carry, inp: (fn(carry, inp),) * 2
  if unroll:
    length = len(jax.tree_util.tree_leaves(inputs)[0])
    carrydef = jax.tree_util.tree_structure(start)
    carry = start
    outs = []
    for index in range(length):
      carry, out = fn2(carry, jax.tree.map(lambda x: x[index], inputs))
      flat, treedef = jax.tree_util.tree_flatten(out)
      assert treedef == carrydef, (treedef, carrydef)
      outs.append(flat)
    outs = [
        jnp.stack([carry[i] for carry in outs], 0)
        for i in range(len(outs[0]))]
    outs = carrydef.unflatten(outs)
  else:
    carry, outs = nj.scan(fn2, start, inputs)
  if axis:
    outs = jax.tree.map(lambda x: x.swapaxes(0, axis), outs)
  return outs


def symlog(x):
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def switch(pred, lhs, rhs):
  def fn(lhs, rhs):
    assert lhs.shape == rhs.shape, (pred.shape, lhs.shape, rhs.shape)
    mask = pred
    while len(mask.shape) < len(lhs.shape):
      mask = mask[..., None]
    return jnp.where(mask, lhs, rhs)
  return jax.tree.map(fn, lhs, rhs)


def reset(xs, reset):
  def fn(x):
    mask = reset
    while len(mask.shape) < len(x.shape):
      mask = mask[..., None]
    return x * (1 - mask.astype(x.dtype))
  return jax.tree.map(fn, xs)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=f32):
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

  def __init__(self, mode, dims, dist='mse', agg='sum', tol=1e-8):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._dist = dist
    self._agg = agg
    self._tol = tol
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return symexp(self._mode)

  def mean(self):
    return symexp(self._mode)

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    if self._dist == 'mse':
      distance = (self._mode - symlog(value)) ** 2
      distance = jnp.where(distance < self._tol, 0, distance)
    elif self._dist == 'abs':
      distance = jnp.abs(self._mode - symlog(value))
      distance = jnp.where(distance < self._tol, 0, distance)
    else:
      raise NotImplementedError(self._dist)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class TwoHotDist:

  def __init__(self, logits, bins, dims=0, transfwd=None, transbwd=None):
    assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
    assert logits.dtype == f32, logits.dtype
    assert bins.dtype == f32, bins.dtype
    self.logits = logits
    self.probs = jax.nn.softmax(logits)
    self.dims = tuple([-x for x in range(1, dims + 1)])
    self.bins = jnp.array(bins)
    self.transfwd = transfwd or (lambda x: x)
    self.transbwd = transbwd or (lambda x: x)
    self.batch_shape = logits.shape[:len(logits.shape) - dims - 1]
    self.event_shape = logits.shape[len(logits.shape) - dims: -1]

  def mean(self):
    # The naive implementation results in a non-zero result even if the bins
    # are symmetric and the probabilities uniform, because the sum operation
    # goes left to right, accumulating numerical errors. Instead, we use a
    # symmetric sum to ensure that the predicted rewards and values are
    # actually zero at initialization.
    # return self.transbwd((self.probs * self.bins).sum(-1))
    n = self.logits.shape[-1]
    if n % 2 == 1:
      m = (n - 1) // 2
      p1 = self.probs[..., :m]
      p2 = self.probs[..., m: m + 1]
      p3 = self.probs[..., m + 1:]
      b1 = self.bins[..., :m]
      b2 = self.bins[..., m: m + 1]
      b3 = self.bins[..., m + 1:]
      wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
      return self.transbwd(wavg)
    else:
      p1 = self.probs[..., :n // 2]
      p2 = self.probs[..., n // 2:]
      b1 = self.bins[..., :n // 2]
      b2 = self.bins[..., n // 2:]
      wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
      return self.transbwd(wavg)

  def mode(self):
    return self.transbwd((self.probs * self.bins).sum(-1))

  def log_prob(self, x):
    assert x.dtype == f32, x.dtype
    x = self.transfwd(x)
    below = (self.bins <= x[..., None]).astype(i32).sum(-1) - 1
    above = len(self.bins) - (
        self.bins > x[..., None]).astype(i32).sum(-1)
    below = jnp.clip(below, 0, len(self.bins) - 1)
    above = jnp.clip(above, 0, len(self.bins) - 1)
    equal = (below == above)
    dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - x))
    dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - x))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = (
        jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None] +
        jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None])
    log_pred = self.logits - jax.scipy.special.logsumexp(
        self.logits, -1, keepdims=True)
    return (target * log_pred).sum(-1).sum(self.dims)


def video_grid(video):
  B, T, H, W, C = video.shape
  return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def balance_stats(dist, target, thres):
  # Values are NaN when there are no positives or negatives in the current
  # batch, which means they will be ignored when aggregating metrics via
  # np.nanmean() later, as they should.
  pos = (target.astype(f32) > thres).astype(f32)
  neg = (target.astype(f32) <= thres).astype(f32)
  pred = (dist.mean().astype(f32) > thres).astype(f32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      rate=pos.mean(),
      avg=target.astype(f32).mean(),
      pred=dist.mean().astype(f32).mean(),
  )


class Moments(nj.Module):

  def __init__(
      self, impl='mean_std', rate=0.01, limit=1e-8, perclo=5, perchi=95):
    self.impl = impl
    self.rate = rate
    self.limit = limit
    self.perclo = perclo
    self.perchi = perchi
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.mean = nj.Variable(jnp.zeros, (), f32, name='mean')
      self.sqrs = nj.Variable(jnp.zeros, (), f32, name='sqrs')
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    elif self.impl == 'min_max':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
    elif self.impl == 'perc':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
    elif self.impl == 'perc_corr':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    else:
      raise NotImplementedError(self.impl)

  def __call__(self, x, update=True):
    update and self.update(x)
    return self.stats()

  def update(self, x):
    if parallel():
      mean = lambda x: jax.lax.pmean(x.mean(), 'i')
      min_ = lambda x: jax.lax.pmin(x.min(), 'i')
      max_ = lambda x: jax.lax.pmax(x.max(), 'i')
      per = lambda x, q: jnp.percentile(jax.lax.all_gather(x, 'i'), q)
    else:
      mean = jnp.mean
      min_ = jnp.min
      max_ = jnp.max
      per = jnp.percentile
    x = sg(x.astype(f32))
    m = self.rate
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.mean.write((1 - m) * self.mean.read() + m * mean(x))
      self.sqrs.write((1 - m) * self.sqrs.read() + m * mean(x * x))
      self.corr.write((1 - m) * self.corr.read() + m * 1.0)
    elif self.impl == 'min_max':
      low, high = min_(x), max_(x)
      self.low.write((1 - m) * jnp.minimum(self.low.read(), low) + m * low)
      self.high.write((1 - m) * jnp.maximum(self.high.read(), high) + m * high)
    elif self.impl == 'perc':
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write((1 - m) * self.low.read() + m * low)
      self.high.write((1 - m) * self.high.read() + m * high)
    elif self.impl == 'perc_corr':
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write((1 - m) * self.low.read() + m * low)
      self.high.write((1 - m) * self.high.read() + m * high)
      self.corr.write((1 - m) * self.corr.read() + m * 1.0)
    else:
      raise NotImplementedError(self.impl)

  def stats(self):
    if self.impl == 'off':
      return 0.0, 1.0
    elif self.impl == 'mean_std':
      corr = jnp.maximum(self.rate, self.corr.read())
      mean = self.mean.read() / corr
      std = jnp.sqrt(jax.nn.relu(self.sqrs.read() / corr - mean ** 2))
      std = jnp.maximum(self.limit, std)
      return sg(mean), sg(std)
    elif self.impl == 'min_max':
      offset = self.low.read()
      span = self.high.read() - self.low.read()
      span = jnp.maximum(self.limit, span)
      return sg(offset), sg(span)
    elif self.impl == 'perc':
      offset = self.low.read()
      span = self.high.read() - self.low.read()
      span = jnp.maximum(self.limit, span)
      return sg(offset), sg(span)
    elif self.impl == 'perc_corr':
      corr = jnp.maximum(self.rate, self.corr.read())
      lo = self.low.read() / corr
      hi = self.high.read() / corr
      span = hi - lo
      span = jnp.maximum(self.limit, span)
      return sg(lo), sg(span)
    else:
      raise NotImplementedError(self.impl)


class Optimizer(nj.Module):

  def __init__(
      self, lr, opt='adam', eps=1e-7, clip=0.0, warmup=1000, wd=0.0,
      wd_pattern=r'/kernel$', agc=0.03, postagc=0.0, anneal=0,
      beta1=0.9, beta2=0.999, pmin=1e-3, details=False):
    chain = []

    if clip:
      chain.append(optax.clip_by_global_norm(clip))
    if agc:
      chain.append(scale_by_agc(agc, pmin))

    if opt == 'adam':
      chain.append(optax.scale_by_adam(beta1, beta2, eps=eps))
    else:
      raise NotImplementedError(opt)

    if wd:
      assert not wd_pattern[0].isnumeric(), wd_pattern
      wd_pattern = re.compile(wd_pattern)
      wdmaskfn = lambda params: {k: bool(wd_pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmaskfn))

    self.lr = lr
    if isinstance(lr, dict):
      chain.append(scale_by_groups({k + '/': -v for k, v in lr.items()}))
    else:
      chain.append(optax.scale(-lr))

    if postagc:
      chain.append(scale_by_agc(postagc, pmin))

    self.opt = optax.chain(*chain)
    self.warmup = warmup
    self.anneal = anneal
    self.details = details
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')
    self.once = True

  def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
    def wrapped(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == f32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux

    metrics = {}
    loss, params, grads, aux = nj.grad(
        wrapped, modules, has_aux=True)(*args, **kwargs)
    if self.scaling:
      loss /= self.grad_scale.read()
    if not isinstance(modules, (list, tuple)):
      modules = [modules]
    counts = {k: int(np.prod(v.shape)) for k, v in params.items()}
    if self.once:
      self.once = False
      prefs = []
      for key in counts:
        parts = key.split('/')
        prefs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), 2))]
      subcounts = {
          prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
          for prefix in set(prefs)}
      print(f'Optimizer {self.name} has {sum(counts.values()):,} variables:')
      for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
        print(f'{count:>14,} {prefix}')

    if parallel():
      grads = jax.tree.map(lambda x: jax.lax.pmean(x, 'i'), grads)
    if self.scaling:
      invscale = 1.0 / self.grad_scale.read()
      grads = jax.tree.map(lambda x: x * invscale, grads)
    optstate = self.get('state', self.opt.init, params)
    updates, optstate = self.opt.update(grads, optstate, params)
    self.put('state', optstate)

    if self.details:
      metrics.update(self._detailed_stats(optstate, params, updates))

    if self.warmup > 0:
      scale = jnp.clip(self.step.read().astype(f32) / self.warmup, 0, 1)
      updates = jax.tree.map(lambda x: x * scale, updates)
    if self.anneal > 0:
      scale = jnp.clip(1 - self.step.read().astype(f32) / self.anneal, 0, 1)
      updates = jax.tree.map(lambda x: x * scale, updates)

    nj.context().update(optax.apply_updates(params, updates))
    grad_norm = optax.global_norm(grads)
    update_norm = optax.global_norm(updates)
    param_norm = optax.global_norm([x.find() for x in modules])
    isfin = jnp.isfinite
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm))
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = (~jnp.isfinite(grad_norm)).astype(f32)
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
      self.step.write(self.step.read() + isfin(grad_norm).astype(i32))
    else:
      check(isfin(grad_norm), f'{self.path} grad norm: {{x}}', x=grad_norm)
      self.step.write(self.step.read() + 1)
    check(isfin(update_norm), f'{self.path} updates: {{x}}', x=update_norm)
    check(isfin(param_norm), f'{self.path} params: {{x}}', x=param_norm)

    metrics['loss'] = loss.mean()
    metrics['grad_norm'] = grad_norm
    metrics['update_norm'] = update_norm
    metrics['param_norm'] = param_norm
    metrics['grad_steps'] = self.step.read()
    metrics['param_count'] = jnp.array(sum(counts.values()))
    metrics = {f'{self.name}_{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(
        keep.astype(i32) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        keep.astype(f32) * self.grad_scale.read() +
        incr.astype(f32) * self.grad_scale.read() * 2 +
        decr.astype(f32) * self.grad_scale.read() / 2,
        1e-4, 1e5))
    return finite

  def _detailed_stats(self, optstate, params, updates):
    groups = {
        'all': r'.*',
        'enc': r'/enc/.*/kernel$',
        'dec': r'/dec/.*/kernel$',
        'rssm': r'/rssm/.*/kernel$',
        'cont': r'/cont/.*/kernel$',
        'rew': r'/rew/.*/kernel$',
        'actor': r'/actor/.*/kernel$',
        'critic': r'/critic/.*/kernel$',
        'gru': r'/gru/kernel$',
        'bias': r'/bias$',
        'out': r'/out/kernel$',
        'repr': r'/repr_logit/kernel$',
        'prior': r'/prior_logit/kernel$',
        'offset': r'/offset$',
        'scale': r'/scale$',
    }
    metrics = {}
    stddev = None
    for state in getattr(optstate, 'inner_state', optstate):
      if isinstance(state, optax.ScaleByAdamState):
        corr = 1 / (1 - 0.999 ** state.count)
        stddev = jax.tree.map(lambda x: jnp.sqrt(x * corr), state.nu)
    for name, pattern in groups.items():
      keys = [k for k in params if re.search(pattern, k)]
      ps = [params[k] for k in keys]
      us = [updates[k] for k in keys]
      if not ps:
        continue
      metrics.update({f'{k}/{name}': v for k, v in dict(
          param_abs_max=jnp.stack([jnp.abs(x).max() for x in ps]).max(),
          param_abs_mean=jnp.stack([jnp.abs(x).mean() for x in ps]).mean(),
          param_norm=optax.global_norm(ps),
          update_abs_max=jnp.stack([jnp.abs(x).max() for x in us]).max(),
          update_abs_mean=jnp.stack([jnp.abs(x).mean() for x in us]).mean(),
          update_norm=optax.global_norm(us),
      ).items()})
      if stddev is not None:
        sc = [stddev[k] for k in keys]
        pr = [
            jnp.abs(x) / jnp.maximum(1e-3, jnp.abs(y)) for x, y in zip(us, ps)]
        metrics.update({f'{k}/{name}': v for k, v in dict(
            scale_abs_max=jnp.stack([jnp.abs(x).max() for x in sc]).max(),
            scale_abs_min=jnp.stack([jnp.abs(x).min() for x in sc]).min(),
            scale_abs_mean=jnp.stack([jnp.abs(x).mean() for x in sc]).mean(),
            prop_max=jnp.stack([x.max() for x in pr]).max(),
            prop_min=jnp.stack([x.min() for x in pr]).min(),
            prop_mean=jnp.stack([x.mean() for x in pr]).mean(),
      ).items()})
    return metrics


def scale_by_groups(mapping):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    scaled = {}
    for key, update in updates.items():
      matches = [prefix for prefix in mapping if key.startswith(prefix)]
      assert len(matches) == 1, (key, matches)
      scale = mapping[matches[0]]
      assert 0 < abs(scale) < 1, (key, matches[0], scale)
      scaled[key] = scale * update
    return scaled, state

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_agc(clip=0.03, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates)
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def onehot_dict(mapping, spaces):
  mapping = mapping.copy()
  for key, space in spaces.items():
    if space.discrete and key in mapping:
      mapping[key] = jax.nn.one_hot(mapping[key], space.high.max().item())
  return mapping


def concat_dict(mapping, batch_shape=None):
  tensors = [v for _, v in sorted(mapping.items(), key=lambda x: x[0])]
  if batch_shape is not None:
    tensors = [x.reshape((*batch_shape, -1)) for x in tensors]
  return jnp.concatenate(tensors, -1)


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


class SlowUpdater(nj.Module):

  def __init__(self, src, dst, fraction=1.0, period=1):
    self.src = src
    self.dst = dst
    self.fraction = fraction
    self.period = period
    self.updates = nj.Variable(jnp.zeros, (), i32, name='updates')

  def __call__(self):
    assert self.src.find()
    updates = self.updates.read()
    need_init = (updates == 0).astype(f32)
    need_update = (updates % self.period == 0).astype(f32)
    mix = jnp.clip(1.0 * need_init + self.fraction * need_update, 0, 1)
    params = {
        k.replace(f'/{self.src.name}/', f'/{self.dst.name}/'): v
        for k, v in self.src.find().items()}
    ema = jax.tree.map(
        lambda s, d: mix * s + (1 - mix) * d,
        params, self.dst.find())
    for name, param in ema.items():
      assert param.dtype == jnp.float32, (
          f'EMA of {name} should be float32 not {param.dtype}')
    self.dst.put(sg(ema))
    self.updates.write(updates + 1)


def draw_time_hist(data, bins=32, zoom=4, range=None, **kwargs):
  range = (data.min(), data.max()) if range is None else range
  B, T = data.shape
  fn = bind(jnp.histogram, bins=bins, range=range, **kwargs)
  hist, _ = jax.vmap(fn, 1, 1)(data)
  hist = hist.astype(f32)
  hist /= hist.max(0, keepdims=True)
  hist = jnp.nan_to_num(hist, 1)
  hist = jnp.repeat(hist, zoom, 0)
  hist = jnp.repeat(hist, zoom, 1)
  return (255 * hist[::-1]).astype(np.uint8)
