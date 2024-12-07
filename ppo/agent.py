import re

import elements
import embodied
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import nets

sg = jax.lax.stop_gradient
f32 = jnp.float32


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___ ___  ___   ---",
      r"--- | _ \ _ \/ _ \  ---",
      r"--- |  _/  _/ (_) | ---",
      r"--- |_| |_|  \___/  ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.model = Model(obs_space, act_space, config, name='model')
    self.opt = embodied.jax.Optimizer(
        self.model, self._make_opt(**config.opt), name='opt')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')

  @property
  def policy_keys(self):
    return '/(enc|actemb|rnn|policy)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    for key in self.act_space.keys():
      spaces[f'logp/{key}'] = elements.Space(np.float32)
    if self.config.recurrent and self.config.replay_context:
      spaces['memory'] = elements.Space(np.float32, self.config.rnn.units)
    return spaces

  def init_policy(self, batch_size):
    memory = self.model.initial(batch_size)
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return memory, prevact

  def init_train(self, batch_size):
    memory = self.model.initial(batch_size)
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return memory, prevact

  def init_report(self, batch_size):
    return ()

  def policy(self, carry, obs, mode='train'):
    (memory, prevact) = carry
    memory, feat, policy, _ = self.model(
        memory, obs, prevact, value=False, single=True)
    act = {k: v.sample(nj.seed()) for k, v in policy.items()}
    out = {f'logp/{k}': policy[k].logp(v) for k, v in act.items()}
    if self.config.recurrent:
      out['memory'] = memory
    carry = (memory, act)
    return carry, act, out

  def train(self, carry, data):
    memory, prevact = carry
    if self.config.replay_context:
      K = self.config.replay_context
      prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
      data = {k: v[:, K:] for k, v in data.items()}
      if self.config.recurrent:
        memory = data.pop('memory').astype(nn.COMPUTE_DTYPE)[:, K - 1]
    else:
      prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
      prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    metrics, (memory, mets) = self.opt(
        self.loss, memory, data, prevact, has_aux=True)
    metrics.update(mets)
    prevact = {k: data[k][:, -1] for k in self.act_space}
    carry = (memory, prevact)
    return carry, {}, metrics

  def report(self, carry, data):
    return carry, {}

  def loss(self, memory, data, prevact):
    memory, feat, policy, value = self.model(memory, data, prevact)
    losses, metrics = ppo_loss(
        data, policy, value, self.advnorm, self.valnorm, self.act_space,
        update=True, **self.config.ppo_loss)
    for k, v in losses.items():
      metrics[f'{k}_loss'] = v.mean()
      metrics[f'{k}_loss_std'] = v.std()
    scales = self.config.loss_scales
    loss = sum([v.mean() * scales[k] for k, v in losses.items()])
    return loss, (memory, metrics)

  def _make_opt(self, lr, warmup, clip, eps, wd, wdregex=r'/kernel$'):
    assert not wdregex[0].isnumeric(), wdregex
    pattern = re.compile(wdregex)
    wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
    lr = optax.linear_schedule(0.0, lr, warmup)
    return optax.chain(
        optax.clip_by_global_norm(clip),
        optax.scale_by_adam(eps=eps),
        optax.add_decayed_weights(wd, wdmask),
        optax.scale_by_learning_rate(lr),
    )


class Model(nj.Module):

  def __init__(self, obs_space, act_space, config):
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    self.obs_space = obs_space
    self.act_space = act_space
    self.enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.config = config
    self.enc = {
        'impala': nets.ImpalaEncoder,
    }[config.enc.typ](self.enc_space, **config.enc[config.enc.typ], name='enc')
    if self.config.recurrent:
      if self.config.rnnact:
        clip = lambda x: x / sg(jnp.maximum(1, jnp.abs(x)))
        kw = dict(**config.actemb, squish=clip)
        self.actemb = nn.DictEmbed(act_space, **kw, name='actemb')
      self.rnn = nn.GRU(**config.rnn, name='rnn')
    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.policy = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='policy')
    self.value = embodied.jax.MLPHead((), **config.value, name='value')

  def initial(self, batch_size):
    if self.config.recurrent:
      return self.rnn.initial(batch_size)
    else:
      return ()

  def __call__(self, carry, obs, prevact, value=True, single=False):
    bdims = 1 if single else 2
    bshape = jax.tree.leaves(obs)[0].shape[:bdims]
    embed = self.enc(obs, bdims=bdims)
    assert embed.shape[:-1] == bshape, (bshape, embed.shape)
    if self.config.recurrent:
      if self.config.rnnact:
        prevact = nn.mask(prevact, ~obs['is_first'])
        inputs = jnp.concatenate([embed, self.actemb(prevact, bshape)], -1)
      else:
        inputs = embed
      resets = obs['is_first']
      carry, feat = self.rnn(carry, inputs, resets, single)
    else:
      feat = embed
    assert feat.shape[:-1] == bshape, (bshape, feat.shape)
    policy = self.policy(feat, bdims=bdims)
    value = self.value(feat, bdims=bdims) if value else None
    return carry, feat, policy, value


def ppo_loss(
    data, policy, value, advnorm, valnorm, act_space, update,
    actent=1e-2, hor=200, lam=0.8, trclip=0.2, tarclip=10.0):

  metrics = {}
  losses = {}

  act = {k: data[k] for k in act_space}
  logpi = sum([policy[k].logp(act[k]) for k in act_space])
  logdata = sum([data['logp/' + k] for k in act_space])

  rew, last, term = data['reward'], data['is_last'], data['is_terminal']
  mask = f32(~last & ~term)
  ratio = jnp.exp(logpi - sg(logdata))
  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset

  live = f32(~term)[:, 1:] * (1 - 1 / hor)
  cont = f32(~last & ~term)[:, 1:] * lam
  delta = rew[:, 1:] + live * val[:, 1:] - val[:, :-1]
  advs = [0]
  for t in reversed(range(delta.shape[1])):
    advs.append(delta[:, t] + live[:, t] * cont[:, t] * advs[-1])
  adv = jnp.stack(list(reversed(advs))[:-1], 1)
  tar = adv + val[:, :-1]

  voffset, vscale = valnorm(tar, update)
  tarnormed = (tar - voffset) / vscale
  tarnormed = jnp.clip(tarnormed, -tarclip, tarclip) if tarclip else tarnormed
  tarnormed_padded = jnp.concatenate([tarnormed, 0 * tarnormed[:, :1]], 1)
  losses['value'] = value.loss(sg(tarnormed_padded)) * mask

  aoffset, ascale = advnorm(adv, update)
  advnormed = (adv - aoffset) / ascale
  reinforce = ratio[:, :-1] * sg(advnormed)
  ents = {k: policy[k].entropy() for k in act_space}
  maxent = actent * sum(ents.values())[:, :-1]

  upper = (ratio[:, :-1] < 1 + trclip) | (advnormed < 0)
  lower = (ratio[:, :-1] > 1 - trclip) | (advnormed > 0)
  tr = f32(upper & lower)
  losses['policy'] = -(reinforce + maxent) * mask[:, :-1] * tr

  for k in act_space:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  metrics['rew'] = data['reward'].mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar.mean()
  metrics['adv'] = adv.mean()
  metrics['advmag'] = jnp.abs(adv).mean()
  metrics['ratio'] = ratio.mean()
  metrics['clipfrac'] = (1 - tr).mean()
  metrics['td'] = jnp.abs(value.pred()[:, :-1] - tarnormed).mean()

  return losses, metrics
