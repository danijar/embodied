import sys

import embodied
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)

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

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    if config.scale_images > 1:
      n = config.scale_images
      for key, value in obs_space.items():
        if value.dtype == np.uint8 and len(value.shape) == 3:
          shape = (n * value.shape[0], n * value.shape[1], value.shape[2])
          obs_space[key] = embodied.Space(value.dtype, shape, 0, 255)
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config)
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config)
    self.wm.qhead_actor = self.task_behavior.ac.actor  # TODO
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config)

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
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
    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = jax.tree_util.tree_map(
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
        workers=self.config.data_loaders, prefetch=4)

  def save(self):
    data = jax.tree_util.tree_flatten(jax.tree_util.tree_map(
        jnp.asarray, self.state))[0]
    data = [np.asarray(x) for x in data]
    return data

  def load(self, state):
    self.state = jax.tree_util.tree_flatten(self.state)[1].unflatten(state)

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
        if self.config.scale_images > 1:
          amount = self.config.scale_images
          value = jnp.repeat(jnp.repeat(value, amount, -3), amount, -2)
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

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    self.qhead_actor = None  # TODO
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    if self.config.rssm_type == 'rssm':
      self.rssm = nets.RSSM(**config.rssm)
    elif self.config.rssm_type == 'group':
      self.rssm = nets.GroupRSSM(**config.group_rssm)
    elif self.config.rssm_type == 'vqgru':
      self.rssm = nets.VQGRU(**config.vqgru)
    elif self.config.rssm_type == 'stgru':
      self.rssm = nets.STGRU(**config.stgru)
    else:
      raise NotImplementedError(self.config.rssm_type)
    self.encoder = nets.MultiEncoder(shapes, **config.encoder)
    self.heads = {}
    self.heads['decoder'] = nets.MultiDecoder(shapes, **config.decoder)
    self.heads['reward'] = nets.MLP((), **config.reward_head)
    self.heads['cont'] = nets.MLP((), **config.cont_head)
    if self.config.use_qhead:
      self.qhead = nets.MLP((), **config.qhead, name='qhead')
      self.qslow = nets.MLP((), **config.qhead, name='qslow')
      self.updater = jaxutils.SlowUpdater(
          self.qhead, self.qslow,
          self.config.slow_critic_fraction,
          self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='model', **config.model_opt)
    self.wmkl = jaxutils.AutoAdapt((), **self.config.wmkl, inverse=False)
    self.repauto = jaxutils.AutoAdapt((), **self.config.repauto, inverse=False)

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    if self.config.use_qhead:
      modules += [self.qhead]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, training=True, has_aux=True)
    metrics.update(mets)
    if self.config.use_qhead:
      self.updater()
    return state, outs, metrics

  def loss(self, data, state, training=False):
    metrics = {}
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    post_const = sg(post)
    for name, head in self.heads.items():
      inp = post if name in self.config.grad_heads else post_const
      if name == 'decoder' and self.config.drop_loss:
        indices = jnp.arange(embed.shape[1])
        amount = int(embed.shape[1] * (1 - self.config.drop_loss))
        drop_loss_indices = jax.random.permutation(nj.rng(), indices)[:amount]
        out = head(inp, drop_loss_indices)
      else:
        out = head(inp)
      if not isinstance(out, dict):
        out = {name: out}
      dists.update(out)
    losses = {}
    dyn_loss = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    rep_loss = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    metrics['raw_dyn_loss'] = dyn_loss.mean()
    metrics['raw_rep_loss'] = rep_loss.mean()
    if self.config.kl_combine:
      mix = self.config.kl_balance
      kl = mix * dyn_loss + (1 - mix) * rep_loss
      kl, mets = self.wmkl(kl, update=training)
      losses['kl'] = kl
      metrics.update({f'wmkl_{k}': v for k, v in mets.items()})
    elif self.config.repauto.impl != 'fixed':
      losses['dyn'] = dyn_loss
      rep_loss, mets = self.repauto(rep_loss, update=training)
      losses['rep'] = rep_loss
      metrics.update({f'repauto_{k}': v for k, v in mets.items()})
    else:
      losses['dyn'] = dyn_loss
      losses['rep'] = rep_loss
    if self.config.use_qhead:
      action = self.qhead_actor(post).sample(seed=nj.rng())
      qslow = self.qslow({**post, 'action': action}).mean()
      vals = [qslow[-1]]
      reward = data['reward'][1:]
      disc = 1 - data['is_terminal'][1:].astype(jnp.float32)
      disc *= 1 - (1 / self.config.horizon)
      interm = reward + disc * qslow[1:] * (1 - self.config.qhead_lambda)
      for t in reversed(range(len(disc))):
        vals.append(interm[t] + disc[t] * self.config.qhead_lambda * vals[-1])
      tar = jnp.stack(list(reversed(vals)))
      losses['qhead'] = -self.qhead({**data, **post}).log_prob(sg(tar))
    for key, dist in dists.items():
      if key in self.heads['decoder'].cnn_shapes and self.config.drop_loss:
        target = data[key][:, drop_loss_indices]
        loss = -dist.log_prob(target.astype(jnp.float32))
        padding = jnp.zeros((embed.shape[0], embed.shape[1] - loss.shape[1]))
        loss = jnp.concatenate([loss, sg(padding + loss.mean())], 1)
        losses[key] = loss
        continue
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
    prev_latent = {k: v[:, -1] for k, v in post.items()}
    prev_action = data['action'][:, -1]
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    return model_loss.mean(), ((prev_latent, prev_action), out, metrics)

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
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
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
    kwargs = dict(shape=act_space.shape, **self.config.actor)
    if act_space.discrete:
      kwargs['dist'] = config.actor_dist_disc
      self.grad = config.actor_grad_disc
    else:
      kwargs['dist'] = config.actor_dist_cont
      self.grad = config.actor_grad_cont
    self.actor = nets.MLP(name='actor', **kwargs)
    if self.config.slow_actor:
      self.slow_actor = nets.MLP(name='slow_actor', **kwargs)
      self.updater = jaxutils.SlowUpdater(
          self.actor, self.slow_actor,
          self.config.slow_actor_fraction,
          self.config.slow_actor_update)
    self.advmom = jaxutils.Moments(**self.config.advmom)
    self.retmoms = {k: jaxutils.Moments(**self.config.retmom) for k in critics}
    self.rewmoms = {k: jaxutils.Moments(**self.config.rewmom) for k in critics}
    self.actent = jaxutils.AutoAdapt(
        act_space.shape[:-1] if act_space.discrete else act_space.shape,
        **self.config.actent, inverse=True)
    self.opt = jaxutils.Optimizer(name='actor', **self.config.actor_opt)

  def initial(self, batch_size):
    return {}

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
    if self.config.slow_actor:
      self.updater()
    return traj, metrics

  def loss(self, traj):
    metrics = {}

    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      metrics[f'{key}_reward_mean'] = rew.mean()
      metrics[f'{key}_reward_std'] = rew.std()
      metrics[f'{key}_reward_max'] = jnp.abs(rew).max()
      metrics[f'{key}_reward_dist'] = jaxutils.subsample(rew)
      metrics[f'{key}_return_mean'] = ret.mean()
      metrics[f'{key}_return_std'] = ret.std()
      metrics[f'{key}_return_max'] = jnp.abs(ret).max()
      metrics[f'{key}_return_dist'] = jaxutils.subsample(ret)
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()

      metrics[f'{key}_retstd_rewstd'] = ret.std() / rew.std()

      if self.config.retnorm == 'off':
        pass
      elif self.config.retnorm == 'reward':
        mean, std = self.rewmoms[key](rew)
        ret = (ret / self.config.horizon - mean) / std
        base = (base / self.config.horizon - mean) / std
        metrics[f'{key}_rewmom_mean'] = mean
        metrics[f'{key}_rewmom_std'] = std
      elif self.config.retnorm == 'return':
        mean, std = self.retmoms[key](ret)
        ret = (ret - mean) / std
        base = (base - mean) / std
      else:
        raise NotImplementedError(self.config.retnorm)
      if self.config.retnorm != 'off':
        metrics[f'return_normed_{key}_dist'] = jaxutils.subsample(ret)
        metrics[f'{key}_return_normed_mean'] = ret.mean()
        metrics[f'{key}_return_normed_std'] = ret.std()
        metrics[f'{key}_return_normed_abs_max'] = jnp.abs(ret).max()
        metrics[f'{key}_return_normed_min'] = ret.min()
        metrics[f'{key}_return_normed_max'] = ret.max()
      adv = ret - base
      mets = self._adv_metrics(adv)
      metrics.update({f'{key}_adv_{k}': v for k, v in mets.items()})
      advs.append(adv * self.scales[key] / total)
    adv = jnp.stack(advs).sum(0)
    metrics[f'adv_raw_dist'] = jaxutils.subsample(adv)
    metrics[f'adv_raw_mean'] = adv.mean()
    metrics[f'adv_raw_std'] = adv.std()
    metrics[f'adv_raw_max'] = jnp.abs(adv).max()

    if self.config.advnorm:
      mean, std = self.advmom(adv)
      adv = (adv - mean) / std
    if self.config.adv_symlog:
      adv = jaxutils.symlog(adv)
    if self.config.adv_clip > 0.0:
      adv *= sg(self.config.adv_clip / jnp.maximum(
          self.config.adv_clip, jnp.abs(adv)))
    if self.config.adv_temp > 0.0:
      adv = jnp.exp(adv / self.config.adv_temp)
    metrics[f'adv_dist'] = jaxutils.subsample(adv)
    metrics[f'adv_mean'] = adv.mean()
    metrics[f'adv_std'] = adv.std()
    metrics[f'adv_max'] = jnp.abs(adv).max()

    policy = self.actor(sg(traj))
    if self.config.actor_smooth > 0.0 and self.act_space.discrete:
      mix = self.config.actor_smooth
      action = traj['action']
      uniform = jnp.ones_like(action) / action.shape[-1]
      action = (1 - mix) * action + mix * uniform
      logpi = policy.log_prob(sg(action))[:-1]
    else:
      logpi = policy.log_prob(sg(traj['action']))[:-1]
    ent = policy.entropy()[:-1]
    mets = self._policy_metrics(policy, logpi, ent)
    metrics.update({f'policy_{k}': v for k, v in mets.items()})
    metrics['action_max'] = jnp.abs(traj['action']).max()
    if self.act_space.discrete:
      metrics['action_dist'] = jaxutils.subsample(
          jnp.argmax(traj['action'], -1))
    else:
      metrics['action_dist'] = jaxutils.subsample(traj['action'], -1)
    if self.grad == 'backprop':
      loss = -adv
    elif self.grad == 'reinforce':
      loss = -logpi * sg(adv)
    else:
      raise NotImplementedError(self.grad)

    if self.config.actent.impl == 'fixed':
      scale = self.config.actent.scale
      if self.config.advnorm and self.config.entnorm:
        scale /= self.advmom.stats()[1]
        scale = jnp.clip(scale, 1e-4, 1e-1)
      entloss = -scale * ent
      metrics['entloss_scale'] = scale
      metrics['entloss_dist'] = jaxutils.subsample(ent)
      metrics['entloss_mean'] = entloss.mean()
    else:
      ent = self._policy_randomness(policy, perdim=True)
      entloss, mets = self.actent(ent)
      metrics.update({f'actent_{k}': v for k, v in mets.items()})
      if len(self.actent.shape) > 0:
        entloss = entloss.sum(range(2, len(entloss.shape)))
    loss += entloss

    if self.config.slow_kl:
      # kl = policy.kl_divergence(self.slow_actor(sg(traj)))[:-1].mean()
      kl = self.slow_actor(sg(traj)).kl_divergence(policy)[:-1].mean()
      loss += self.config.slow_kl * kl

    if self.config.muesli:
      assert self.config.slow_actor
      assert self.config.advnorm
      assert self.config.dueling_critic
      zeros = jnp.zeros(traj['action'].shape)
      mix = jnp.array([0.99, 0.01])
      mix = mix[None, None, :] + jnp.zeros((*zeros.shape[:2], 1))
      mix = tfd.Categorical(probs=mix)
      if self.act_space.discrete:
        prior = tfd.Mixture(mix, [
            self.slow_actor(traj),
            tfd.OneHotCategorical(zeros, dtype=jnp.float32),
        ])
      else:
        prior = tfd.Mixture(mix, [
            self.slow_actor(traj),
            tfd.Independent(tfd.Uniform(zeros - 1, zeros + 1), 1),
        ])
      proposals = prior.sample(self.config.muesli, seed=nj.rng())
      inputs = {**{
          k: jnp.repeat(v[None], self.config.muesli, 0)
          for k, v in traj.items()}, 'action': proposals}
      advs = self.critics['extr'].adv_net(inputs).mean()
      mean, std = self.advmom.stats()
      advs = jnp.exp(jnp.clip((advs - mean) / std, -1, 1))
      constant = advs.sum(0, keepdims=True)
      constant = constant + 1 - advs
      advs /= constant
      logpis = policy.log_prob(sg(proposals))
      muesli_loss = -1.0 * (sg(advs) * logpis).mean(0)[:-1]
      metrics['muesli_loss_mean'] = muesli_loss.mean()
      metrics['muesli_loss_std'] = muesli_loss.std()
      loss += muesli_loss

    loss *= sg(traj['weight'])[:-1]
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return loss.mean(), metrics

  def _adv_metrics(self, adv):
    standard = (adv - adv.mean()) / jnp.maximum(adv.std(), 1e-8)
    metrics = {}
    metrics['mag_mean'] = jnp.abs(adv).mean()
    metrics['mag_max'] = jnp.abs(adv).max()
    metrics['mag_min'] = jnp.abs(adv).min()
    metrics['mean'] = adv.mean()
    metrics['std'] = adv.std()
    metrics['skew'] = (standard ** 3).mean()
    metrics['kurt'] = (standard ** 4).mean()
    return metrics

  def _policy_metrics(self, policy, logpi, ent):
    metrics = {}
    randomness = self._policy_randomness(policy)
    metrics['randomness_dist'] = jaxutils.subsample(randomness)
    metrics['randomness'] = randomness.mean()
    metrics['entropy_mean'] = ent.mean()
    metrics['entropy_std'] = ent.std()
    metrics['logprob_mean'] = logpi.mean()
    metrics['logprob_min'] = logpi.min()
    metrics['logprob_max'] = logpi.max()
    if self.act_space.discrete:
      dist = policy
      if hasattr(dist, 'distribution'):
        dist = dist.distribution
      metrics['logit_min'] = dist.probs_parameter().min()
      metrics['logit_max'] = dist.probs_parameter().max()
    else:
      metrics['stddev'] = policy.stddev().mean()
    return metrics

  def _policy_randomness(self, policy, perdim=False):
    if len(self.actent.shape) > 0:
      assert isinstance(policy, tfd.Independent)
      ent = policy.distribution.entropy()[:-1]
      lo = policy.minent / np.prod(self.actent.shape)
      hi = policy.maxent / np.prod(self.actent.shape)
      rand = ((ent - lo) / (hi - lo))
      if not perdim > 0:
        rand = rand.mean(range(2, len(rand.shape)))
      return rand
    else:
      ent = policy.entropy()[:-1]
      lo, hi = policy.minent, policy.maxent
      return ((ent - lo) / (hi - lo))


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', **self.config.critic)
    if self.config.slow_critic:
      kwargs = self.config.critic
      if self.config.slow_critic_zero_init:
        kwargs = kwargs.update(outscale=0.0)
      self.target_net = nets.MLP((), name='target_net', **kwargs)
      self.updater = jaxutils.SlowUpdater(
          self.net, self.target_net,
          self.config.slow_critic_fraction,
          self.config.slow_critic_update)
    else:
      self.target_net = self.net
    if self.config.dueling_critic:
      kwargs = self.config.critic.update(
          inputs=(*self.config.critic.inputs, 'action'))
      self.adv_net = nets.MLP((), **kwargs)
    self.rewmom = jaxutils.Moments(**self.config.rewmom)
    self.opt = jaxutils.Optimizer(name='critic', **self.config.critic_opt)

  def train(self, traj, actor):
    metrics = {}
    reward = self.rewfn(traj)
    if self.config.rewnorm:
      self.rewmom.update(reward)
      if self.config.rewnorm_horizon:
        reward /= self.config.horizon
      reward /= self.rewmom.stats()[1]
    impl = self.config.critic_return_impl
    slow = self.config.critic_return_slow
    target = sg(self.target(traj, reward, impl, slow)[0])
    if self.config.dueling_critic:
      mets, metrics = self.opt(
          [self.net, self.adv_net], self.loss, traj, target, has_aux=True)
    else:
      mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    # metrics['imag_reward_dist'] = jaxutils.subsample(reward)
    # metrics['imag_reward_mean'] = reward.mean()
    # metrics['imag_reward_std'] = reward.std()
    if self.config.slow_critic:
      self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_trust:
      old = self.target_net(traj).mean()
      new = dist.mean()
      inside = (
          (new >= (1 - self.config.critic_trust) * old) &
          (new <= (1 + self.config.critic_trust) * old))
      towards = (jnp.sign(new - old) == jnp.sign(new - target))
      mask = inside | towards
      assert mask.shape == loss.shape
      loss *= sg(mask.astype(loss.dtype))
    loss = (loss * sg(traj['weight'])).mean()
    if self.config.dueling_critic:
      adv_dist = self.adv_net(traj)
      adv_target = sg(target - dist.mean())
      loss += -(adv_dist.log_prob(adv_target) * sg(trae['weight'])).mean()
      metrics['dueling_dist'] = jaxutils.subsample(adv_dist.mean())
      metrics['dueling_mean'] = adv_dist.mean().mean()
      metrics['dueling_std'] = adv_dist.mean().mean()
    metrics['critic_dist'] = jaxutils.subsample(dist.mean())
    metrics['critic_mean'] = dist.mean().mean()
    metrics['critic_max'] = jnp.abs(dist.mean()).max()
    metrics['critic_std'] = dist.mean().std()
    # metrics['imag_return_dist'] = jaxutils.subsample(target)
    # metrics['imag_return_mean'] = target.mean()
    # metrics['imag_return_std'] = target.std()
    return loss, metrics

  def score(self, traj, actor):
    reward = self.rewfn(traj)
    if self.config.rewnorm:
      if self.config.rewnorm_horizon:
        reward /= self.config.horizon
      reward /= self.rewmom.stats()[1]
    impl = self.config.actor_return_impl
    slow = self.config.actor_return_slow
    return reward, *self.target(traj, reward, impl, slow)

  def target(self, traj, reward, impl, slow):
    if len(reward) != len(traj['action']) - 1:
      raise AssertionError('Should provide rewards for all but last action.')
    if self.config.imag_reward_min:
      reward = jnp.where(
          jnp.abs(reward) < self.config.imag_reward_min, 0, reward)
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    if slow:
      value = self.target_net(traj).mean()
    else:
      value = self.net(traj).mean()
    if self.config.slow_target_min:
      value = jnp.minimum(value, self.net(traj).mean())
    if self.config.slow_target_clip:
      lo = 1 - self.config.slow_target_clip
      hi = 1 + self.config.slow_target_clip
      fast = self.net(traj).mean()
      value = jnp.clip(fast, lo * value, hi * value)
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
