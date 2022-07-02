import sys

import embodied
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
sg = lambda x: jax.tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load((
      embodied.Path(sys.argv[0]).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, config)
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config)
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config)

  def initial_policy_state(self, obs):
    return (
        self.wm.rssm.initial(len(obs['is_first'])),
        self.task_behavior.initial(len(obs['is_first'])),
        self.expl_behavior.initial(len(obs['is_first'])),
        jnp.zeros((len(obs['is_first']),) + self.act_space.shape),
    )

  def initial_train_state(self, obs):
    return self.wm.rssm.initial(len(obs['is_first']))

  def policy(self, obs, state=None, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    if state is None:
      state = self._initial_policy_state(obs)
    obs = self.preprocess(obs)
    latent, task_state, expl_state, action = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'])
    noise = self.config.expl_noise
    if mode == 'eval':
      noise = self.config.eval_noise
      outs, task_state = self.task_behavior.policy(latent, task_state)
      outs = {**outs, 'action': outs['action'].mode()}
    elif mode == 'explore':
      outs, expl_state = self.expl_behavior.policy(latent, expl_state)
      outs = {**outs, 'action': outs['action'].sample(seed=nj.rng())}
    elif mode == 'train':
      outs, task_state = self.task_behavior.policy(latent, task_state)
      outs = {**outs, 'action': outs['action'].sample(seed=nj.rng())}
    outs = {**outs, 'action': jaxutils.action_noise(
        outs['action'], noise, self.act_space)}
    state = (latent, task_state, expl_state, outs['action'])
    return outs, state

  def train(self, data, state=None):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    if state is None:
      state = self._initial_train_state(data)
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = jax.tree_map(
        lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    if 'key' in data:
      criteria = {**data, **wm_outs}
      outs.update(key=data['key'], priority=criteria[self.config.priority])
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def dataset(self, generator):
    return embodied.Prefetch(
        sources=[generator] * self.config.batch_size,
        workers=8, prefetch=4)

  def save(self):
    data = jax.tree_util.tree_flatten(jax.tree_map(jnp.asarray, self.state))[0]
    data = [np.array(x) for x in data]
    return data

  def load(self, state):
    self.state = jax.tree_util.tree_flatten(self.state)[1].unflatten(state)

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = value.astype(jnp.float32) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['reward'] = {
        'off': lambda x: x, 'sign': jnp.sign,
        'tanh': jnp.tanh, 'symlog': jaxutils.symlog,
    }[self.config.transform_rewards](obs['reward'])
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, config):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.config = config
    self.rssm = nets.RSSM(**config.rssm)
    self.encoder = nets.MultiEncoder(shapes, **config.encoder)
    self.heads = {}
    self.heads['decoder'] = nets.MultiDecoder(shapes, **config.decoder)
    self.heads['reward'] = nets.MLP((), **config.reward_head)
    self.heads['cont'] = nets.MLP((), **config.cont_head)
    self.opt = jaxutils.Optimizer('model', **config.model_opt)
    self.wmkl = jaxutils.AutoAdapt((), **self.config.wmkl, inverse=False)

  def train(self, data, state=None):
    self.loss(data)  # Create variables.
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, training=True, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state=None, training=False):
    metrics = {}
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    dists = {}
    post_const = sg(post)
    for name, head in self.heads.items():
      out = head(post if name in self.config.grad_heads else post_const)
      if not isinstance(out, dict):
        out = {name: out}
      dists.update(out)
    losses = {}
    kl = self.rssm.kl_loss(post, prior, self.config.wmkl_balance)
    kl, mets = self.wmkl(kl, update=training)
    losses['kl'] = kl
    metrics.update({f'wmkl_{k}': v for k, v in mets.items()})
    for key, dist in dists.items():
      losses[key] = -dist.log_prob(data[key].astype(jnp.float32))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    scaled = {}
    for key, loss in losses.items():
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      scaled[key] = loss * self.config.loss_scales.get(key, 1.0)
    model_loss = sum(scaled.values())
    if 'prob' in data and self.config.priority_correct:
      weights = (1.0 / data['prob']) ** self.config.priority_correct
      weights /= weights.max()
      assert weights.shape == model_loss.shape
      model_loss *= weights
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    metrics['prior_ent_mean'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent_mean'] = self.rssm.get_dist(post).entropy().mean()
    metrics['prior_ent_min'] = self.rssm.get_dist(prior).entropy().min()
    metrics['post_ent_min'] = self.rssm.get_dist(post).entropy().min()
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    if not self.config.jax.debug_nans:
      if 'reward' in dists:
        stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
        metrics.update({f'reward_{k}': v for k, v in stats.items()})
      if 'cont' in dists:
        stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
        metrics.update({f'cont_{k}': v for k, v in stats.items()})
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss.mean(), (last_state, out, metrics)

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      action = prev.pop('action')
      state = self.rssm.img_step(prev, action)
      action = policy(state)
      return {**state, 'action': action}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    traj['cont'] = jnp.concatenate([
        first_cont[None], self.heads['cont'](traj).mean()[1:]], 0)
    traj['weight'] = jnp.cumprod(
        self.config.discount * traj['cont'], 0) / self.config.discount
    return traj

  def report(self, data):
    report = {}
    report.update(self.loss(data)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    self.actor = nets.MLP(
        act_space.shape, **self.config.actor, dist=(
            config.actor_dist_disc if act_space.discrete
            else config.actor_dist_cont))
    self.grad = (
        config.actor_grad_disc if act_space.discrete
        else config.actor_grad_cont)
    self.advnorm = jaxutils.Normalize(**self.config.advnorm)
    self.retnorms = {
        k: jaxutils.Normalize(**self.config.retnorm) for k in self.critics}
    self.scorenorms = {
        k: jaxutils.Normalize(**self.config.scorenorm) for k in self.critics}
    self.actent = jaxutils.AutoAdapt(
        act_space.shape[:-1] if act_space.discrete else act_space.shape,
        **self.config.actent, inverse=True)
    self.opt = jaxutils.Optimizer('actor', **self.config.actor_opt)

  def initial(self, batch_size):
    return None

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    score, metrics = self.score(traj)
    policy = self.actor(sg(traj))
    if self.grad == 'backprop':
      loss = -score
    elif self.grad == 'reinforce':
      loss = -policy.log_prob(sg(traj['action']))[:-1] * sg(score)
    else:
      raise NotImplementedError(self.grad)
    if len(self.actent.shape) > 0:
      assert isinstance(policy, tfd.Independent)
      ent = policy.distribution.entropy()[:-1]
      if self.config.actent_norm:
        lo = policy.minent / np.prod(self.actent.shape)
        hi = policy.maxent / np.prod(self.actent.shape)
        ent = (ent - lo) / (hi - lo)
      ent_loss, mets = self.actent(ent)
      assert len(ent_loss.shape) == 2 + len(self.actent.shape), (
          len(ent_loss.shape) == 2 + len(self.actent.shape))
      ent_loss = ent_loss.sum(range(2, len(ent_loss.shape)))
    else:
      ent = policy.entropy()[:-1]
      if self.config.actent_norm:
        lo, hi = policy.minent, policy.maxent
        ent = (ent - lo) / (hi - lo)
      ent_loss, mets = self.actent(ent)
    metrics.update({f'actent_{k}': v for k, v in mets.items()})
    loss += ent_loss
    loss *= sg(traj['weight'])[:-1]
    return loss.mean(), metrics

  def score(self, traj):
    metrics = {}
    scores = []
    for key, critic in self.critics.items():
      ret, baseline = critic.score(traj, self.actor)
      ret = self.retnorms[key](ret)
      baseline = self.retnorms[key](baseline, update=False)
      score = self.scorenorms[key](ret - baseline)
      metrics[f'{key}_score_mean'] = score.mean()
      metrics[f'{key}_score_std'] = score.std()
      metrics[f'{key}_score_mag'] = jnp.abs(score).mean()
      metrics[f'{key}_score_max'] = jnp.abs(score).max()
      scores.append(score * self.scales[key])
    score = self.advnorm(sum(scores, 0))
    return score, metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', **self.config.critic)
    if self.config.slow_target:
      self.target_net = nets.MLP((), name='target_net', **self.config.critic)
    else:
      self.target_net = self.net
    self.opt = jaxutils.Optimizer('critic', **self.config.critic_opt)

  def train(self, traj, actor):
    metrics = {}
    reward = self.rewfn(traj)
    target = sg(self.target(traj, reward, self.config.critic_return)[0])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    metrics['imag_reward_mean'] = reward.mean()
    metrics['imag_reward_std'] = reward.std()
    self.update_slow()
    return metrics

  def loss(self, traj, target):
    dist = self.net({k: v[:-1] for k, v in traj.items()})
    loss = -(dist.log_prob(target) * traj['weight'][:-1]).mean()
    metrics = {}
    metrics['imag_critic_mean'] = dist.mean().mean()
    metrics['imag_critic_std'] = dist.mean().std()
    metrics['imag_return_mean'] = target.mean()
    metrics['imag_return_std'] = target.std()
    return loss, metrics

  def score(self, traj, actor):
    return self.target(traj, self.rewfn(traj), self.config.actor_return)

  def target(self, traj, reward, impl):
    if len(reward) != len(traj['action']) - 1:
      raise AssertionError('Should provide rewards for all but last action.')
    disc = traj['cont'][1:] * self.config.discount
    value = self.target_net(traj).mean()
    if impl == 'gae':
      advs = [jnp.zeros_like(value[0])]
      deltas = reward + disc * value[1:] - value[:-1]
      for t in reversed(range(len(disc))):
        advs.append(deltas[t] + disc[t] * self.config.return_lambda * advs[-1])
      adv = jnp.stack(list(reversed(advs))[:-1])
      return adv + value[:-1], value[:-1]
    elif impl == 'gve':
      vals = [value[-1]]
      interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
      for t in reversed(range(len(disc))):
        vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
      ret = jnp.stack(list(reversed(vals))[:-1])
      return ret, value[:-1]
    else:
      raise NotImplementedError(impl)

  def update_slow(self):
    if not self.config.slow_target:
      return
    assert self.net.get_state()
    updates = self.get('updates', lambda: jnp.zeros((), jnp.int32))
    period = self.config.slow_target_update
    fraction = self.config.slow_target_fraction
    need_init = (updates == 0).astype(jnp.float32)
    need_update = (updates % period == 0).astype(jnp.float32)
    mix = jnp.clip(1.0 * need_init + fraction * need_update, 0, 1)
    source = {
        k.replace('/net/', '/target_net/'): v
        for k, v in self.net.get_state().items()}
    self.target_net.set_state(jax.tree_map(
        lambda s, d: mix * s + (1 - mix) * d,
        source, self.target_net.get_state()))
    self.put('updates', updates + 1)
