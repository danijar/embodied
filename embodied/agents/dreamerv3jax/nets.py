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
cast = jaxutils.cast_to_compute


class RSSM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=True, initial='zeros',
      unimix=0.0, dynpost=True, sepdyn=False, argmax=False,
      action_clip=-1.0, smooth=0.0, gru_depth=0, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._dynpost = dynpost
    self._sepdyn = sepdyn
    self._argmax = argmax
    self._action_clip = action_clip
    self._smooth = smooth
    self._gru_depth = gru_depth
    self._kw = kw

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
      return cast(state)
    elif self._initial == 'learned':
      # This will cut gradients when the state is created outside of the
      # training graph, but this only happens once at the beginning of the
      # training loop. Afterwards, the state is reset inside the obs_step().
      deter = self.get(
          'initial_deter', jnp.zeros, state['deter'][0].shape, jnp.float32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      state['stoch'] = self.get_stoch(cast(state['deter']))
      return cast(state)
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

  def get_dist(self, state, argmax=False, smooth=False):
    if self._classes:
      logit = state['logit'].astype(jnp.float32)
      if smooth and self._smooth > 0.0:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._smooth) * probs + self._smooth * uniform
        logit = jnp.log(probs)
      dist = tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(jnp.float32)
      std = state['std'].astype(jnp.float32)
      dist = tfp.MultivariateNormalDiag(mean, std)
    return dist

  def obs_step(self, prev_state, prev_action, embed, is_first):
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    is_first = cast(is_first)
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    if self._sepdyn:
      prev_state['stoch'] = sg(prev_state['stoch'])
    prior = self.img_step(prev_state, prev_action)
    if self._dynpost:
      x = jnp.concatenate([prior['deter'], embed], -1)
    else:
      x = embed
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats_layer('obs_stats', x)
    if self._argmax:
      stoch = jax.nn.one_hot(
          jnp.argmax(stats['logit'], -1), self._classes)
    else:
      dist = self.get_dist(stats)
      stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return cast(post), cast(prior)

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
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
    return cast(prior)

  def get_stoch(self, deter):
    x = self.get('img_out', Linear, **self._kw)(deter)
    stats = self._stats_layer('img_stats', x)
    dist = self.get_dist(stats)
    return cast(dist.mode())

  def _gru(self, x, deter):
    x = jnp.concatenate([deter, x], -1)
    for i in range(self._gru_depth):
      kw = {**self._kw, 'units': self._deter}
      x = self.get(f'depth{i}', Linear, **kw)(x)
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
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def dyn_loss(self, post, prior, impl='kl'):
    if impl == 'kl':
      return self.get_dist(sg(post), smooth=True).kl_divergence(
          self.get_dist(prior))
    elif impl == 'logprob':
      return -self.get_dist(prior).log_prob(sg(post['stoch']))
    else:
      raise NotImplementedError(impl)

  def rep_loss(self, post, prior, impl='kl', free=0.0, commit=0.25):
    if impl == 'kl':
      loss = self.get_dist(post).kl_divergence(
          self.get_dist(sg(prior), smooth=True))
    elif impl == 'uniform':
      uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
      loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
    elif impl == 'entropy':
      loss = -self.get_dist(post).entropy()
    elif impl == 'none':
      loss = jnp.zeros(post['deter'].shape[:-1])
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss


class GroupRSSM(nj.Module):

  def __init__(
      self, deter=256, stoch=32, classes=32, unroll=True,
      initial='zeros', unimix=0.0, dynpost=True, sepdyn=False, argmax=False,
      action_clip=-1.0, smooth=0.0, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._dynpost = dynpost
    self._sepdyn = sepdyn
    self._argmax = argmax
    self._action_clip = action_clip
    self._smooth = smooth
    self._kw = kw

  def initial(self, bs):
    state = dict(
        deter=jnp.zeros([bs, self._deter], jnp.float32),
        logit=jnp.zeros([bs, self._stoch, self._classes], jnp.float32),
        stoch=jnp.zeros([bs, self._stoch, self._classes], jnp.float32))
    if self._initial == 'zeros':
      return cast(state)
    elif self._initial == 'learned':
      # This will cut gradients when the state is created outside of the
      # training graph, but this only happens once at the beginning of the
      # training loop. Afterwards, the state is reset inside the obs_step().
      deter = self.get(
          'initial_deter', jnp.zeros, state['deter'][0].shape, jnp.float32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      state['stoch'] = self.get_stoch(cast(state['deter']))
      return cast(state)
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

  def get_dist(self, state, argmax=False, smooth=False):
    logit = state['logit'].astype(jnp.float32)
    if smooth and self._smooth > 0.0:
      probs = jax.nn.softmax(logit, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - self._smooth) * probs + self._smooth * uniform
      logit = jnp.log(probs)
    return tfd.Independent(jaxutils.OneHotDist(logit), 1)

  def obs_step(self, prev_state, prev_action, embed, is_first):
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    is_first = cast(is_first)
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    if self._sepdyn:
      prev_state['stoch'] = sg(prev_state['stoch'])
    prior = self.img_step(prev_state, prev_action)
    if self._dynpost:
      x = jnp.concatenate([prior['deter'], embed], -1)
    else:
      x = embed
    x = self.get('obs_out', Block, **self._kw)(x)
    stats = self._stats_layer('obs_stats', x)
    if self._argmax:
      stoch = jax.nn.one_hot(
          jnp.argmax(stats['logit'], -1), self._classes)
    else:
      dist = self.get_dist(stats)
      stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return cast(post), cast(prior)

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
    prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Block, **self._kw)(x)
    x, deter = self._gru(x, prev_state['deter'])
    x = self.get('img_out', Block, **self._kw)(x)
    stats = self._stats_layer('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return cast(prior)

  def get_stoch(self, deter):
    x = self.get('img_out', Block, **self._kw)(deter)
    stats = self._stats_layer('img_stats', x)
    dist = self.get_dist(stats)
    return cast(dist.mode())

  def _gru(self, x, deter):
    x = jnp.concatenate([deter, x], -1)
    kw = {**self._kw, 'act': 'none', 'size': 3 * self._deter}
    x = self.get('gru', Block, **kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter, deter

  def _stats_layer(self, name, x):
    x = self.get(name, Block, self._stoch * self._classes)(x)
    logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
    if self._unimix:
      probs = jax.nn.softmax(logit, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - self._unimix) * probs + self._unimix * uniform
      logit = jnp.log(probs)
    return {'logit': logit}

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def dyn_loss(self, post, prior, impl='kl'):
    if impl == 'kl':
      return self.get_dist(sg(post), smooth=True).kl_divergence(
          self.get_dist(prior))
    elif impl == 'logprob':
      return -self.get_dist(prior).log_prob(sg(post['stoch']))
    else:
      raise NotImplementedError(impl)

  def rep_loss(self, post, prior, impl='kl', free=0.0, commit=0.25):
    if impl == 'kl':
      loss = self.get_dist(post).kl_divergence(
          self.get_dist(sg(prior), smooth=True))
    elif impl == 'uniform':
      uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
      loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
    elif impl == 'entropy':
      loss = -self.get_dist(post).entropy()
    elif impl == 'none':
      loss = jnp.zeros(post['deter'].shape[:-1])
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss


class STGRU(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=True, dynpost=True,
      sepdyn=False, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._dynpost = dynpost
    self._sepdyn = sepdyn
    self._kw = kw
    cast = jaxutils.cast_to_compute

  def initial(self, bs):
    return cast(dict(
        deter=jnp.zeros([bs, self._deter], jnp.float32),
        logit=jnp.zeros([bs, self._stoch, self._classes], jnp.float32),
        stoch=jnp.zeros([bs, self._stoch, self._classes], jnp.float32)))

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

  def get_dist(self, state):
    return jaxutils.OneHotDist(state['logit'].astype(jnp.float32))

  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def obs_step(self, prev_state, prev_action, embed, is_first):
    prev_action = cast(prev_action)
    is_first = cast(is_first)
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    if self._sepdyn:
      prev_state['stoch'] = sg(prev_state['stoch'])
    prior = self.img_step(prev_state, prev_action)
    if self._dynpost:
      x = jnp.concatenate([prior['deter'], embed], -1)
    else:
      x = embed
    x = self.get('obs_out', Linear, **self._kw)(x)
    logits = self.get('obs_stats', Linear, (self._stoch, self._classes))(x)
    indices = jnp.argmax(logits, -1)
    stoch = jax.nn.one_hot(indices, self._classes)
    probs = jax.nn.softmax(logits)
    stoch = sg(stoch) + probs - sg(probs)
    post = {'deter': prior['deter'], 'logit': logits, 'stoch': stoch}
    return cast(post), cast(prior)

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_stoch = prev_stoch.reshape(prev_stoch.shape[:-2] + (-1,))
    prev_action = cast(prev_action).reshape(
        prev_stoch.shape[:-1] + (-1,))
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    x, deter = self._gru(x, prev_state['deter'])
    x = self.get('img_out', Linear, **self._kw)(x)
    logits = self.get(
        'img_logits', Linear, (self._stoch, self._classes))(x)
    indices = jax.random.categorical(nj.rng(), logits)
    stoch = jax.nn.one_hot(indices, self._classes)
    probs = jax.nn.softmax(logits)
    stoch = sg(stoch) + probs - sg(stoch)  # For planning by backprop.
    prior = {'deter': deter, 'logit': logits, 'stoch': stoch}
    return cast(prior)

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

  def _mask(self, value, mask):
    mask = mask.astype(value.dtype)
    # if jaxutils.parallel():
    #   return jnp.einsum('db...,db->db...', value, mask)
    # else:
    return jnp.einsum('b...,b->b...', value, mask)

  def dyn_loss(self, post, prior, impl='kl'):
    return -self.get_dist(prior).log_prob(sg(post['stoch'])).sum(-1)

  def rep_loss(self, post, prior, impl='kl', free=0.0, commit=0.25):
    return jnp.zeros(post['deter'].shape[:-1])


class VQGRU(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, embed=64, unroll=True,
      dynpost=True, sepdyn=False, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._embed = embed
    self._unroll = unroll
    self._dynpost = dynpost
    self._sepdyn = sepdyn
    self._vq = VectorQuantizer(classes, embed)
    self._kw = kw
    cast = jaxutils.cast_to_compute

  def initial(self, bs):
    return cast(dict(
        deter=jnp.zeros([bs, self._deter], jnp.float32),
        inputs=jnp.zeros([bs, self._stoch, self._embed], jnp.float32),
        logits=jnp.zeros([bs, self._stoch, self._classes], jnp.float32),
        stoch=jnp.zeros([bs, self._stoch, self._embed], jnp.float32)))

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

  def get_dist(self, state):
    return jaxutils.OneHotDist(state['logits'].astype(jnp.float32))

  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def obs_step(self, prev_state, prev_action, embed, is_first):
    prev_action = cast(prev_action)
    is_first = cast(is_first)
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    if self._sepdyn:
      prev_state['stoch'] = sg(prev_state['stoch'])
    prior = self.img_step(prev_state, prev_action)
    if self._dynpost:
      x = jnp.concatenate([prior['deter'], embed], -1)
    else:
      x = embed
    x = self.get('obs_out', Linear, **self._kw)(x)
    inputs = self.get('obs_stats', Linear, (self._stoch, self._embed))(x)
    outputs, indices = self._vq(inputs)
    logits = jnp.log(jax.nn.one_hot(indices, self._classes))
    post = {
        'deter': prior['deter'], 'logits': logits,
        'stoch': outputs, 'inputs': inputs}
    return cast(post), cast(prior)

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_stoch = prev_stoch.reshape(prev_stoch.shape[:-2] + (-1,))
    prev_action = cast(prev_action).reshape(
        prev_stoch.shape[:-1] + (-1,))
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    x, deter = self._gru(x, prev_state['deter'])
    x = self.get('img_out', Linear, **self._kw)(x)
    logits = self.get('img_logits', Linear, (self._stoch, self._classes))(x)
    indices = jax.random.categorical(nj.rng(), logits)
    outputs = self._vq.embed(indices)
    prior = {
        'deter': deter, 'logits': logits,
        'stoch': outputs, 'inputs': outputs}
    return cast(prior)

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

  def _mask(self, value, mask):
    mask = mask.astype(value.dtype)
    # if jaxutils.parallel():
    #   return jnp.einsum('db...,db->db...', value, mask)
    # else:
    return jnp.einsum('b...,b->b...', value, mask)

  def dyn_loss(self, post, prior, impl='kl'):
    target = jax.nn.one_hot(jnp.argmax(post['logits'], -1), self._classes)
    return -self.get_dist(prior).log_prob(sg(target)).sum(-1)

  def rep_loss(self, post, prior, impl='kl', free=0.0, commit=0.25):
    indices = jnp.argmax(post['logits'], -1)
    return self._vq.loss(post['inputs'], indices).sum(-1)  # TODO: mean?


class MultiEncoder(nj.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
      mlp_units=512, cnn='resize', cnn_depth=48, cnn_kernels=(4, 4, 4, 4),
      cnn_blocks=2, resize='stride', **kw):
    excluded = ('is_first', 'is_last')
    shapes = {k: v for k, v in shapes.items() if (
        k not in excluded and not k.startswith('log_'))}
    self.cnn_shapes = {k: v for k, v in shapes.items() if (
        len(v) == 3 and re.match(cnn_keys, k))}
    self.mlp_shapes = {k: v for k, v in shapes.items() if (
        len(v) in (1, 2) and re.match(mlp_keys, k))}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)
    if cnn == 'resize':
      self._cnn = ImageEncoderResize(cnn_depth, cnn_kernels, **kw)
    elif cnn == 'valid':
      self._cnn = ImageEncoderValid(cnn_depth, cnn_kernels, **kw)
    elif cnn == 'resnet':
      self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **kw)
    elif cnn == 'stem':
      self._cnn = ImageEncoderStem(cnn_depth, **kw)
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
      mlp_layers=4, mlp_units=512, cnn='resize', cnn_depth=48,
      cnn_kernels=(5, 5, 6, 6), cnn_blocks=2, image_dist='mse',
      vector_dist='mse', resize='resize', bins=256, outscale=1.0, **kw):
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
      shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      if cnn == 'resize':
        self._cnn = ImageDecoderResize(shape, cnn_depth, cnn_kernels, **kw)
      elif cnn == 'valid':
        self._cnn = ImageDecoderValid(shape, cnn_depth, cnn_kernels, **kw)
      elif cnn == 'resnet':
        self._cnn = ImageDecoderResnet(
            shape, cnn_depth, cnn_blocks, resize, **kw)
      elif cnn == 'stem':
        self._cnn = ImageDecoderStem(shape, cnn_depth, **kw)
      else:
        raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(
          self.mlp_shapes, mlp_layers, mlp_units, **kw, dist=vector_dist,
          outscale=outscale, bins=bins)
    self._inputs = Input(inputs, dims='deter')
    self._image_dist = image_dist

  def __call__(self, inputs, drop_loss_indices=None):
    features = self._inputs(inputs)
    dists = {}
    if self.cnn_shapes:
      feat = features
      if drop_loss_indices is not None:
        feat = feat[:, drop_loss_indices]
      flat = feat.reshape([-1, feat.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(feat.shape[:-1] + output.shape[1:])
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
    if self._image_dist == 'mse_max':
      return jaxutils.MSEMaxDist(mean, 3, 'sum')
    if self._image_dist == 'abs':
      return jaxutils.AbsDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)


class ImageEncoderValid(nj.Module):

  def __init__(self, depth, kernels, **kw):
    self._depth = depth
    self._kernels = kernels
    self._kw = kw

  def __call__(self, x):
    Conv = functools.partial(Conv2D, stride=2, pad='valid')
    x = jaxutils.cast_to_compute(x) - 0.5
    depth = self._depth
    for i, kernel in enumerate(self._kernels):
      x = self.get(f'conv{i}', Conv, depth, kernel, **self._kw)(x)
      depth *= 2
    return x


class ImageDecoderValid(nj.Module):

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
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    return x + 0.5


class ImageEncoderResize(nj.Module):

  def __init__(self, depth, kernels, **kw):
    self._depth = depth
    self._kernels = kernels
    self._kw = kw

  def __call__(self, x):
    x = jaxutils.cast_to_compute(x) - 0.5
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
      kw = {'act': 'none'} if last else self._kw
      x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)  # Upsample
      # print(x.shape)
      x = self.get(f'conv{i}', Conv2D, depth, kernel, **kw)(x)
      # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    return x + 0.5


class ImageEncoderResnet(nj.Module):

  def __init__(self, depth, blocks, resize, **kw):
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._kw = kw

  def __call__(self, x):
    depth = self._depth
    x = jaxutils.cast_to_compute(x) - 0.5
    # print(x.shape)
    for i in range(int(np.log2(x.shape[-2])) - 2):
      kw = {**self._kw, 'preact': False}
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, **kw)(x)
      elif self._resize == 'mean':
        N, H, W, D = x.shape
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
      elif self._resize == 'max':
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same')
      else:
        raise NotImplementedError(self._resize)
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth *= 2
    if self._blocks:
      x = get_act(self._kw['act'])(x)
    x = x.reshape((x.shape[0], -1))
    # print(x.shape)
    return x


class ImageDecoderResnet(nj.Module):

  def __init__(self, shape, depth, blocks, resize, **kw):
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(self._shape[-2])) - 2
    depth = self._depth * 2 ** (stages - 1)
    start = 6 if self._shape[0] in (84, 96) else 4
    x = jaxutils.cast_to_compute(x)
    x = self.get('in', Linear, (start, start, depth))(x)
    for i in range(stages):
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth //= 2
      kw = {**self._kw, 'preact': False}
      if i == stages - 1:
        kw = {}
        depth = self._shape[-1]
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, transp=True, **kw)(x)
      elif self._resize == 'resize':
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
      else:
        raise NotImplementedError(self._resize)
    if max(x.shape[1:-1]) > max(self._shape[:-1]):
      padh = (x.shape[1] - self._shape[0]) / 2
      padw = (x.shape[2] - self._shape[1]) / 2
      x = x[:, int(np.ceil(padh)): -int(padh), :]
      x = x[:, :, int(np.ceil(padw)): -int(padw)]
    # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    return x + 0.5


class ImageEncoderStem(nj.Module):

  def __init__(self, depth, **kw):
    self._depth = depth
    self._kw = kw

  def __call__(self, x):
    x = jaxutils.cast_to_compute(x) - 0.5
    # print(x.shape)
    x = self.get(f'str1', Conv2D, self._depth, 3, 2)(x)
    x = self._block('s1b1', x, self._depth)
    x = self._block('s1b2', x, self._depth)
    # print(x.shape)
    x = self.get(f'str2', Conv2D, 2 * self._depth, 3, 2)(x)
    x = self._block('s2b1', x, 2 * self._depth)
    x = self._block('s2b2', x, 2 * self._depth)
    x = self._block('s2b3', x, 2 * self._depth)
    # print(x.shape)
    x = self._pool(x)
    x = self._block('s3b1', x, 2 * self._depth)
    x = self._block('s3b2', x, 2 * self._depth)
    x = self._block('s3b3', x, 2 * self._depth)
    # print(x.shape)
    x = self._pool(x)
    # print(x.shape)
    x = get_act(self._kw['act'])(x)
    x = x.reshape((x.shape[0], -1))
    # print(x.shape)
    return x

  def _block(self, name, x, depth):
    kw = {**self._kw, 'preact': True}
    skip = x
    x = self.get(f'{name}c1', Conv2D, depth, 3, **kw)(x)
    x = self.get(f'{name}c2', Conv2D, depth, 3, **kw)(x)
    x += skip
    return x

  def _pool(self, x):
    N, H, W, D = x.shape
    x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
    return x


class ImageDecoderStem(nj.Module):

  def __init__(self, shape, depth, **kw):
    self._shape = shape
    self._depth = depth
    self._kw = kw

  def __call__(self, x):
    x = jaxutils.cast_to_compute(x)
    start = {64: 4, 84: 6}[self._shape[0]]
    x = self.get('in', Linear, (start, start, 2 * self._depth))(x)
    # print(x.shape)
    x = self._block('s1b1', x, 2 * self._depth)
    x = self._block('s1b2', x, 2 * self._depth)
    x = self._block('s1b3', x, 2 * self._depth)
    x = self._upscale(x)
    # print(x.shape)
    x = self._block('s2b1', x, 2 * self._depth)
    x = self._block('s2b2', x, 2 * self._depth)
    x = self._block('s2b3', x, 2 * self._depth)
    x = self._upscale(x)
    # print(x.shape)
    x = self.get(f'str3', Conv2D, self._depth, 3, 2, transp=True)(x)
    x = self._block('s3b1', x, self._depth)
    x = self._block('s3b2', x, self._depth)
    # print(x.shape)
    x = self.get(f'str4', Conv2D, self._shape[-1], 3, 2, transp=True)(x)
    # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    return x + 0.5

  def _block(self, name, x, depth):
    kw = {**self._kw, 'preact': True}
    skip = x
    x = self.get(f'{name}c1', Conv2D, depth, 3, **kw)(x)
    x = self.get(f'{name}c2', Conv2D, depth, 3, **kw)(x)
    x += skip
    return x

  def _upscale(self, x):
    return jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)


class MLP(nj.Module):

  def __init__(self, shape, layers, units, inputs=['tensor'], dims=None, **kw):
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    self._shape = shape
    self._layers = layers
    self._units = units
    self._inputs = Input(inputs, dims=dims)
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix',
        'rawstd_offset', 'bins')
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
      self, shape, dist='mse', outscale=0.1, outnorm=False, minstd=1.0,
      maxstd=1.0, unimix=0.0, rawstd_offset=0.0, bins=256):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._dist = dist
    self._minstd = minstd
    self._maxstd = maxstd
    self._unimix = unimix
    self._outscale = outscale
    self._outnorm = outnorm
    self._rawstd_offset = rawstd_offset
    self._bins = bins

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    kw = {}
    kw['outscale'] = self._outscale
    kw['outnorm'] = self._outnorm
    shape = self._shape
    if self._dist.endswith('_disc'):
      shape = (*self._shape, self._bins)
    out = self.get('out', Linear, int(np.prod(shape)), **kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(jnp.float32)
    # out = out.astype(jnp.float32)
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', Linear, int(np.prod(self._shape)), **kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self._shape).astype(jnp.float32)
      # std = std.astype(jnp.float32)
    if self._dist == 'symlog_mse':
      return jaxutils.SymlogDist(out, len(self._shape), 'mse', 'sum')
    if self._dist == 'symlog_abs':
      return jaxutils.SymlogDist(out, len(self._shape), 'abs', 'sum')
    if self._dist == 'symlog_disc':
      return jaxutils.DiscDist(
          out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp)
    if self._dist == 'asqrt_disc':
      return jaxutils.DiscDist(
          out, len(self._shape), -300, 300, jaxutils.asqrt, jaxutils.apower)
    if self._dist == 'mse':
      return jaxutils.MSEDist(out, len(self._shape), 'sum')
    if self._dist == 'abs':
      return jaxutils.AbsDist(out, len(self._shape), 'sum')
    if self._dist == 'normal':
      lo, hi = self._minstd, self._maxstd
      std += self._rawstd_offset
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
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
      dist.minent = np.prod(self._shape) * tfd.TruncatedNormal(1.0, lo, -1, 1).entropy()
      dist.maxent = np.prod(self._shape) * tfd.TruncatedNormal(0.0, hi, -1, 1).entropy()
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


class VectorQuantizer(nj.Module):

  def __init__(self, codes=512, embed=32):
    self.codes = codes
    self.book = nj.Variable(lambda: jax.random.normal(
        nj.rng(), (self.codes, embed), jnp.float32))

  def __call__(self, inputs):
    book = self.book.read()
    book /= jnp.linalg.norm(book, 2, -1, True)
    flat = inputs.reshape((-1, inputs.shape[-1]))
    flat /= jnp.linalg.norm(flat, 2, -1, True)
    flat2 = (flat ** 2).sum(-1, keepdims=True)
    book2 = (book ** 2).sum(-1, keepdims=True).T
    dist = flat2 - 2 * (flat @ book.T) + book2
    indices = jnp.argmin(dist, -1).reshape(inputs.shape[:-1])
    outputs = book[indices]
    outputs = inputs + sg(outputs - inputs)
    return outputs, indices

  def embed(self, indices):
    book = self.book.read()
    book /= jnp.linalg.norm(book, 2, -1, True)
    return book[indices]

  def loss(self, inputs, indices, beta=0.25):
    inputs = inputs.astype(jnp.float32)
    embed = self.embed(indices).astype(jnp.float32)
    loss_enc  = ((sg(embed) - inputs) ** 2).mean(-1)
    loss_book = ((embed - sg(inputs)) ** 2).mean(-1)
    return loss_enc + beta * loss_book


class Block(nj.Module):

  def __init__(
      self, size, groups=8, heads=8, act='gelu', norm='layer',
      winit='normal', fan='avg'):
    assert norm == 'layer', norm
    assert size % groups == 0, (size, groups)
    assert (size // groups) % heads == 0, (size, groups, heads)
    self.size = size
    self.act = get_act(act)
    self.groups = groups
    self.heads = heads
    self.kw = dict(winit=winit, fan=fan)

  def __call__(self, x):
    if x.shape[-1] % self.groups != 0:
      want = int(np.ceil(x.shape[-1] / self.groups) * self.groups)
      missing = want - x.shape[-1]
      x = jnp.concatenate([x, x[..., :missing]], -1)
      assert x.shape[-1] % self.groups == 0, (should, x.shape, self.groups)
    embed = self.size // self.groups
    x = x.reshape((*x.shape[:-1], self.groups, x.shape[-1] // self.groups))
    if x.shape[-1] != embed:
      x = self.get('proj', Linear, embed, **self.kw)(x)
    skip = x
    x = self.get('norm1', Norm, 'layer')(x)
    dim = embed // self.heads
    x = self.get('attn1', Attention, self.heads, dim, **self.kw)(x, x, x)
    x += skip
    skip = x
    x = self.get('norm2', Norm, 'layer')(x)
    x = self.get('linear1', Linear, embed, **self.kw)(x)
    x = self.act(x)
    x = self.get('linear2', Linear, embed, **self.kw)(x)
    x += skip
    x = x.reshape((*x.shape[:-2], self.size))
    return x


class Attention(nj.Module):

  def __init__(self, heads, size, winit='normal', fan='avg'):
    self.heads = heads
    self.size = size
    self.kw = dict(winit=winit, fan=fan)

  def __call__(self, query, key, value, mask=None):
    shape = (self.heads, self.size)
    query = self.get('query', Linear, shape, **self.kw)(query)
    key = self.get('key', Linear, shape, **self.kw)(key)
    value = self.get('value', Linear, shape, **self.kw)(value)
    logits = jnp.einsum('...thd,...Thd->...htT', query, key)
    logits /= np.sqrt(self.size).astype(key.dtype)
    if mask is not None:
      assert mask.ndim == logits.ndim
      logits = jnp.where(mask, logits, -np.inf)
    weights = jax.nn.softmax(logits)
    x = jnp.einsum('...htT,...Thd->...thd', weights, value)
    x = x.reshape((*x.shape[:-2], -1))
    x = self.get('out', Linear, self.heads * self.size)(x)
    return x


class Conv2D(nj.Module):

  def __init__(
      self, depth, kernel, stride=1, transp=False, act='none', norm='none',
      pad='same', bias=True, preact=False, winit='uniform', fan='avg'):
    self._depth = depth
    self._kernel = kernel
    self._stride = stride
    self._transp = transp
    self._act = get_act(act)
    self._norm = Norm(norm)
    self._pad = pad.upper()
    self._bias = bias and (preact or norm == 'none')
    self._preact = preact
    self._winit = winit
    self._fan = fan

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
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_transpose(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
    else:
      shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
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

  def __init__(
      self, units, act='none', norm='none', bias=True, outscale=1.0,
      outnorm=False, winit='uniform', fan='avg'):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,)
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._outscale = outscale
    self._outnorm = outnorm
    self._winit = winit
    self._fan = fan

  def __call__(self, x):
    shape = (x.shape[-1], np.prod(self._units))
    kernel = self.get('kernel', Initializer(
        self._winit, self._outscale, fan=self._fan), shape)
    kernel = jaxutils.cast_to_compute(kernel)
    x = x @ kernel
    if self._bias:
      bias = self.get('bias', jnp.zeros, np.prod(self._units), np.float32)
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
    elif self._impl == 'layer1em4':
      x = x.astype(jnp.float32)
      x = jax.nn.standardize(x, axis=-1, epsilon=1e-4)
      x *= self.get('scale', jnp.ones, x.shape[-1], jnp.float32)
      x += self.get('bias', jnp.zeros, x.shape[-1], jnp.float32)
      return x.astype(dtype)
    elif self._impl == 'layer1em2':
      x = x.astype(jnp.float32)
      x = jax.nn.standardize(x, axis=-1, epsilon=1e-2)
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


class Initializer:

  def __init__(self, dist='uniform', scale=1.0, fan='avg', norm=False):
    self.scale = scale
    self.dist = dist
    self.fan = fan
    self.norm = norm

  def __call__(self, shape):
    if self.scale == 0.0:
      value = jnp.zeros(shape, jnp.float32)
    elif self.dist == 'uniform':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      limit = np.sqrt(3 * scale)
      value = jax.random.uniform(
          nj.rng(), shape, jnp.float32, -limit, limit)
    elif self.dist == 'normal':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': np.mean((fanin, fanout)), 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      std = np.sqrt(scale) / 0.87962566103423978
      value = std * jax.random.truncated_normal(
          nj.rng(), -2, 2, shape, jnp.float32)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.rng(), matshape, jnp.float32)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = self.scale * jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    if self.norm:
      value /= jnp.linalg.norm(value, 2, 0, keepdims=True)
    return value

  def _fans(self, shape):
    if len(shape) == 0:
      return 1, 1
    elif len(shape) == 1:
      return shape[0], shape[0]
    elif len(shape) == 2:
      return shape
    else:
      space = int(np.prod(shape[:-2]))
      return shape[-2] * space, shape[-1] * space


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
