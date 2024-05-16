import re
from functools import partial as bind

import embodied
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml

sg = lambda x: jax.tree.map(jax.lax.stop_gradient, x)
f32 = jnp.float32

from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


class Model(nj.Module):

  def __init__(self, enc_space, act_space, config):
    self.config = config
    self.enc = {
        'impala': bind(nets.ImpalaEncoder, **config.enc.impala),
    }[config.enc.typ](enc_space, name='enc')
    if self.config.recurrent:
      self.core = nets.GRU(**config.gru, name='core')
    fn = lambda s: (*s.shape, s.high.max().item()) if s.discrete else s.shape
    shapes = {k: fn(s) for k, s in act_space.items()}
    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    dists = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.policy = nets.MLP(shapes, **config.policy, dist=dists, name='policy')
    self.value = nets.MLP((), **config.value, name='value')

  def initial(self, batch_size):
    if self.config.recurrent:
      return self.core.initial(batch_size)
    else:
      return ()

  def __call__(self, data, prevact, carry, value=True, bdims=2):
    assert bdims in (1, 2)
    embed = self.enc(data, bdims=bdims)
    if not self.config.gru_actions:
      prevact = jax.tree.map(jnp.zeros_like, prevact)
    if self.config.recurrent:
      feat, carry = self.core(
          carry, prevact, embed, data['is_first'], single=(bdims == 1))
    else:
      feat = embed
    inputs = {**data, 'feat': feat}
    policy = self.policy(inputs, bdims=bdims)
    value = self.value(inputs, bdims=bdims) if value else None
    return feat, policy, value, carry


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, config):
    self.config = config
    self.obs_space = {
        k: v for k, v in obs_space.items() if not k.startswith('log_')}
    self.act_space = {
        k: v for k, v in act_space.items() if k != 'reset'}
    enc_space = {
        k: v for k, v in obs_space.items() if
        k not in ('is_first', 'is_last', 'is_terminal', 'reward', 'bwreturn')
        and not k.startswith('log_') and re.match(config.enc.spaces, k)}
    self.model = Model(enc_space, self.act_space, config, name='model')
    self.opt = jaxutils.Optimizer(**config.opt, name='opt')
    self.advnorm = jaxutils.Moments(**config.advnorm, name='advnorm')
    self.valnorm = jaxutils.Moments(**config.valnorm, name='valnorm')

  @property
  def policy_keys(self):
    return '/(enc|core|policy)/'

  @property
  def aux_spaces(self):
    spaces = {}
    for key in self.act_space.keys():
      spaces[f'logprob_{key}'] = embodied.Space(np.float32)
    if self.config.recurrent:
      spaces['gru_state'] = embodied.Space(np.float32, self.config.gru.units)
    return spaces

  def init_policy(self, batch_size):
    lat = self.model.initial(batch_size)
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return (lat, prevact)

  def init_train(self, batch_size):
    lat = self.model.initial(batch_size)
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return (lat, prevact)

  def init_report(self, batch_size):
    return ()

  def policy(self, obs, carry, mode='train'):
    self.config.jax.jit and embodied.print(
        'Tracing policy function', color='yellow')
    lat, prevact = carry
    outs = {}
    obs = self.preprocess(obs)
    prevact = jaxutils.onehot_dict(prevact, self.act_space)
    _, policy, _, lat = self.model(obs, prevact, lat, value=False, bdims=1)
    acts = {k: v.sample(seed=nj.seed()) for k, v in policy.items()}
    if self.config.recurrent:
      outs['gru_state'] = lat
    outs.update({
        f'logprob_{k}': policy[k].log_prob(v) for k, v in acts.items()})
    acts = {
        k: jnp.nanargmax(acts[k], -1).astype(jnp.int32)
        if s.discrete else acts[k] for k, s in self.act_space.items()}
    return acts, outs, (lat, acts)

  def train(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing train function', color='yellow')
    data = self.preprocess(data)
    if self.config.recurrent and self.config.replay_context:
      K = self.config.replay_context
      lat = data.pop('gru_state')[:, K - 1]
      data = {k: v[:, K:] for k, v in data.items()}
      prevact = {k: data[k][:, K - 1] for k in self.act_space}
      carry = lat, prevact
    metrics, (carry, mets) = self.opt(
        self.model, self.loss, data, carry, has_aux=True)
    metrics.update(mets)
    return {}, carry, metrics

  def report(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing report function', color='yellow')
    return {}, carry

  def loss(self, data, carry):
    lat, prevact = carry
    prevacts = jaxutils.onehot_dict({
        k: jnp.concatenate([prevact[k][:, None], data[k][:, :-1]], 1)
        for k in self.act_space}, self.act_space)
    feat, policy, value, lat = self.model(data, prevacts, lat)
    losses, metrics = ppo_losses(
        data, policy, value, self.advnorm, self.valnorm, self.act_space,
        update=True, **self.config.ppo_losses)
    scales = self.config.loss_scales
    loss = sum([v.mean() * scales[k] for k, v in losses.items()])
    nextact = {k: data[k][:, -1] for k in self.act_space}
    return loss, ((lat, nextact), metrics)

  def preprocess(self, obs):
    spaces = {**self.obs_space, **self.act_space, **self.aux_spaces}
    result = {}
    for key, value in obs.items():
      if key.startswith('log_') or key in ('reset', 'key', 'id'):
        continue
      space = spaces[key]
      if len(space.shape) >= 3 and space.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      result[key] = value
    return result


def ppo_losses(
    data, policy, value, advnorm, valnorm, act_space, update,
    actent=1e-2, hor=200, lam=0.8, trclip=0.2, tarclip=10.0):

  metrics = {}
  losses = {}

  act = {
      k: jax.nn.one_hot(data[k], int(s.high.max())) if s.discrete else data[k]
      for k, s in act_space.items()}
  logpi = sum([policy[k].log_prob(act[k]) for k in act_space])
  logdata = sum([data['logprob_' + k] for k in act_space])

  rew, last, term = data['reward'], data['is_last'], data['is_terminal']
  mask = f32(~last & ~term)
  ratio = jnp.exp(logpi - sg(logdata))
  voffset, vscale = valnorm.stats()
  val = value.mean() * vscale + voffset

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
  losses['value'] = -(value.log_prob(sg(tarnormed_padded)) * mask)[:, :-1]

  aoffset, ascale = advnorm(adv, update)
  advnormed = (adv - aoffset) / ascale
  reinforce = ratio[:, :-1] * sg(advnormed)
  ents = {k: policy[k].entropy() for k in act_space}
  maxent = actent * sum(ents.values())[:, :-1]

  upper = (ratio[:, :-1] < 1 + trclip) | (advnormed < 0)
  lower = (ratio[:, :-1] > 1 - trclip) | (advnormed > 0)
  tr = f32(upper & lower)
  losses['policy'] = -(reinforce + maxent) * mask[:, :-1] * tr

  for k, v in losses.items():
    metrics[f'{k}_loss'] = v.mean()
    metrics[f'{k}_loss_std'] = v.std()

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
  metrics['clipfrac'] = (1 - f32(tr)).mean()
  metrics['td'] = jnp.abs(value.mean()[:, :-1] - tarnormed).mean()

  return losses, metrics
