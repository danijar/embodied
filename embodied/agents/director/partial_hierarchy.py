import functools

import embodied
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import agent
from . import nets
from . import tfutils


class PartialHierarchy(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.wm = wm
    self.config = config
    self.extr_reward = lambda traj: self.wm.heads['reward'](traj).mean()[1:]

    wconfig = config.update({
        'actor.inputs': self.config.worker_inputs,
        'critic.inputs': self.config.worker_inputs,
    })
    self.worker = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], wconfig),
        'expl': agent.VFunction(lambda s: s['reward_expl'], wconfig),
        'goal': agent.VFunction(lambda s: s['reward_goal'], wconfig),
    }, config.worker_rews, act_space, wconfig)

    mconfig = config.update({
        'actor_grad_cont': 'reinforce',
        'actent.target': config.manager_actent,
    })
    self.skill_space = embodied.Space(
        np.int32 if config.goal_encoder.dist == 'onehot' else np.float32,
        config.skill_shape[:-1] + (config.skill_shape[-1] + 1,))
    self.manager = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], mconfig),
        'expl': agent.VFunction(lambda s: s['reward_expl'], mconfig),
        'goal': agent.VFunction(lambda s: s['reward_goal'], mconfig),
    }, config.manager_rews, self.skill_space, mconfig)

    if self.skill_space.discrete:
      self.prior = tfutils.OneHotDist(tf.zeros(config.skill_shape))
      self.prior = tfd.Independent(self.prior, len(config.skill_shape) - 1)
    else:
      self.prior = tfd.Normal(
          tf.zeros(config.skill_shape), tf.ones(config.skill_shape))
      self.prior = tfd.Independent(self.prior, len(config.skill_shape))

    self.feat = nets.Input(['deter'])
    self.enc = nets.MLP(
        config.skill_shape, dims='context', **config.goal_encoder)
    self.dec = nets.MLP(
        (self.config.rssm.deter,), dims='context', **self.config.goal_decoder)
    self.kl = tfutils.AutoAdapt((), **self.config.encdec_kl)
    self.opt = tfutils.Optimizer('goal', **config.encdec_opt)

  def initial(self, batch_size):
    return {
        'step': tf.zeros((batch_size,), tf.int64),
        'skill': tf.zeros((batch_size,) + self.skill_space.shape, tf.float32),
        'goal': tf.zeros((batch_size, self.config.rssm.deter, 2), tf.float32),
    }

  def policy(self, latent, carry, imag=False):
    duration = self.config.train_skill_duration if imag else (
        self.config.env_skill_duration)
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    update = (carry['step'] % duration) == 0
    switch = lambda x, y: (
        tf.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
        tf.einsum('i,i...->i...', update.astype(x.dtype), y))
    skill = sg(switch(carry['skill'], self.manager.actor(sg(latent)).sample()))
    new_goal = self.skill_to_goal(skill, self.feat(latent))
    goal = sg(switch(carry['goal'], new_goal))
    dist = self.worker.actor(sg({**latent, 'goal': goal}))
    outs = {'action': dist}
    outs['log_goal'] = self.wm.heads['decoder']({
        'deter': goal[..., 0],
        'stoch': self.wm.rssm.get_stoch(goal[..., 0])})['image'].mode()
    carry = {'step': carry['step'] + 1, 'skill': skill, 'goal': goal}
    return outs, carry

  def skill_to_goal(self, skill, context):
    dont_care = (skill[..., -1] > 0.5).astype(tf.float32)[..., None]
    uniform = tf.ones_like(skill[..., :-1]) / (skill.shape[-1] - 1)
    probs = (1 - dont_care) * skill[..., :-1] + dont_care * uniform
    codes = tfutils.OneHotDist(probs=probs).sample(self.config.partial_samples)
    context = tf.repeat(context[None], self.config.partial_samples, 0)
    goals = self.dec({'skill': codes, 'context': context}).mode()
    mean = goals.mean(0)
    if self.config.partial_mask == 'var':
      mask = 1 / (goals.var(0) + self.config.partial_eps)
    elif self.config.partial_mask == 'std':
      mask = 1 / (goals.std(0) + self.config.partial_eps)
    elif self.config.partial_mask == 'ones':
      mask = tf.ones_like(mean)
    else:
      raise NotImplementedError(self.config.partial_mask)
    if self.config.partial_normalize:
      mask /= mask.sum(-1, keepdims=True)
    return tf.stack([mean, mask], -1)

  def train(self, imagine, start, data):
    success = lambda rew: (rew[-1] > 0.7).astype(tf.float32).mean()
    metrics = {}
    if self.config.vae_replay:
      metrics.update(self.train_vae_replay(data))
    traj, mets = self.train_jointly(imagine, start)
    metrics.update(mets)
    metrics['goal_success'] = success(traj['reward_goal'])
    if self.config.vae_imag:
      metrics.update(self.train_vae_imag(traj))
    return None, metrics

  def train_jointly(self, imagine, start):
    start = start.copy()
    metrics = {}
    with tf.GradientTape(persistent=True) as tape:
      policy = functools.partial(self.policy, imag=True)
      traj = self.wm.imagine_carry(
          policy, start, self.config.imag_horizon,
          self.initial(len(start['is_first'])))
      traj['reward_extr'] = self.extr_reward(traj)
      traj['reward_expl'] = self.expl_reward(traj)
      traj['reward_goal'] = self.goal_reward(traj)
      wtraj = self.split_traj(traj)
      mtraj = self.abstract_traj(traj)
    mets = self.worker.update(wtraj, tape)
    metrics.update({f'worker_{k}': v for k, v in mets.items()})
    mets = self.manager.update(mtraj, tape)
    metrics.update({f'manager_{k}': v for k, v in mets.items()})
    return traj, metrics

  def train_vae_replay(self, data):
    metrics = {}
    feat = self.feat(data).astype(tf.float32)
    if 'context' in self.config.goal_decoder.inputs:
      if self.config.vae_span:
        context = feat[:, 0]
        goal = feat[:, -1]
      else:
        assert feat.shape[1] > self.config.train_skill_duration
        context = feat[:, :-self.config.train_skill_duration]
        goal = feat[:, self.config.train_skill_duration:]
    else:
      goal = context = feat
    with tf.GradientTape() as tape:
      enc = self.enc({'goal': goal, 'context': context})
      dec = self.dec({'skill': enc.sample(), 'context': context})
      rec = -dec.log_prob(tf.stop_gradient(goal))
      if self.config.goal_kl:
        kl = tfd.kl_divergence(enc, self.prior)
        kl, mets = self.kl(kl)
        metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
        assert rec.shape == kl.shape, (rec.shape, kl.shape)
      else:
        kl = 0.0
      loss = (rec + kl).mean()
    metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
    metrics['goalrec_mean'] = rec.mean()
    metrics['goalrec_std'] = rec.std()
    return metrics

  def train_vae_imag(self, traj):
    metrics = {}
    feat = self.feat(traj).astype(tf.float32)
    if 'context' in self.config.goal_decoder.inputs:
      if self.config.vae_span:
        context = feat[0]
        goal = feat[-1]
      else:
        assert feat.shape[0] > self.config.train_skill_duration
        context = feat[:-self.config.train_skill_duration]
        goal = feat[self.config.train_skill_duration:]
    else:
      goal = context = feat
    with tf.GradientTape() as tape:
      enc = self.enc({'goal': goal, 'context': context})
      dec = self.dec({'skill': enc.sample(), 'context': context})
      rec = -dec.log_prob(tf.stop_gradient(goal.astype(tf.float32)))
      if self.config.goal_kl:
        kl = tfd.kl_divergence(enc, self.prior)
        kl, mets = self.kl(kl)
        metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
      else:
        kl = 0.0
      loss = (rec + kl).mean()
    metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
    metrics['goalrec_mean'] = rec.mean()
    metrics['goalrec_std'] = rec.std()
    return metrics

  def propose_goal(self, start, impl):
    feat = self.feat(start).astype(tf.float32)
    if impl == 'replay':
      target = tf.random.shuffle(feat).astype(tf.float32)
      skill = self.enc({'goal': target, 'context': feat}).sample()
      skill = tf.concat([skill, 0 * skill[..., :1]], -1)
      return self.skill_to_goal(skill, feat)
    if impl == 'manager':
      skill = self.manager.actor(start).sample()
      return self.skill_to_goal(skill, feat)
    if impl == 'prior':
      skill = self.prior.sample(len(start['is_terminal']))
      skill = tf.concat([skill, 0 * skill[..., :1]], -1)
      return self.skill_to_goal(skill, feat)
    raise NotImplementedError(impl)

  def goal_reward(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    goal = tf.stop_gradient(traj['goal'].astype(tf.float32))
    goal, mask = goal[..., 0], goal[..., 1]
    if self.config.goal_reward == 'normed_inner':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return tf.einsum('...i,...i->...', mask * goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'squared':
      return -(mask * ((goal - feat) ** 2)).sum(-1)[1:]
    elif self.config.goal_reward == 'normed_squared':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return -(mask * ((goal / norm - feat / norm) ** 2)).sum(-1)[1:]
    else:
      raise NotImplementedError(self.config.goal_reward)

  def expl_reward(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    context = tf.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
    enc = self.enc({'goal': feat, 'context': context})
    dec = self.dec({'skill': enc.sample(), 'context': context})
    ll = dec.log_prob(feat)
    if self.config.goal_kl:
      kl = tfd.kl_divergence(enc, self.prior)
    else:
      kl = 0.0
    if self.config.adver_impl == 'abs':
      return tf.abs(dec.mode() - feat).mean(-1)[1:]
    elif self.config.adver_impl == 'squared':
      return ((dec.mode() - feat) ** 2).mean(-1)[1:]
    elif self.config.adver_impl == 'elbo_scaled':
      return (kl - ll / self.kl.scale())[1:]
    elif self.config.adver_impl == 'elbo_unscaled':
      return (kl - ll)[1:]
    raise NotImplementedError(self.config.adver_impl)

  def split_traj(self, traj):
    traj = traj.copy()
    k = self.config.train_skill_duration
    assert len(traj['action']) % k == 1
    reshape = lambda x: x.reshape([x.shape[0] // k, k] + x.shape[1:])
    for key, val in list(traj.items()):
      val = tf.concat([0 * val[:1], val], 0) if 'reward' in key else val
      # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
      val = tf.concat([reshape(val[:-1]), val[k::k][:, None]], 1)
      # N val K val B val F... -> K val (N B) val F...
      val = val.transpose([1, 0] + list(range(2, len(val.shape))))
      val = val.reshape(
          [val.shape[0], np.prod(val.shape[1:3])] + val.shape[3:])
      val = val[1:] if 'reward' in key else val
      traj[key] = val
    # Bootstrap sub trajectory against current not next goal.
    traj['goal'] = tf.concat([traj['goal'][:-1], traj['goal'][:1]], 0)
    traj['weight'] = tf.math.cumprod(
        self.config.discount * traj['cont']) / self.config.discount
    return traj

  def abstract_traj(self, traj):
    traj = traj.copy()
    traj['action'] = traj.pop('skill')
    k = self.config.train_skill_duration
    reshape = lambda x: x.reshape([x.shape[0] // k, k] + x.shape[1:])
    weights = tf.math.cumprod(reshape(traj['cont'][:-1]), 1)
    for key, value in list(traj.items()):
      if 'reward' in key:
        traj[key] = (reshape(value) * weights).mean(1)
      elif key == 'cont':
        traj[key] = tf.concat([value[:1], reshape(value[1:]).prod(1)], 0)
      else:
        traj[key] = tf.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)
    traj['weight'] = tf.math.cumprod(
        self.config.discount * traj['cont']) / self.config.discount
    return traj

  def abstract_traj_old(self, traj):
    traj = traj.copy()
    traj['action'] = traj.pop('skill')
    mult = tf.math.cumprod(traj['cont'][1:], 0)
    for key, value in list(traj.items()):
      if 'reward' in key:
        traj[key] = (mult * value).mean(0)[None]
      elif key == 'cont':
        traj[key] = tf.stack([value[0], value[1:].prod(0)], 0)
      else:
        traj[key] = tf.stack([value[0], value[-1]], 0)
    return traj

  def report(self, data):
    metrics = {}
    for impl in ('manager', 'prior', 'replay'):
      for key, video in self.report_worker(data, impl).items():
        metrics[f'impl_{impl}_{key}'] = video
    return metrics

  def report_worker(self, data, impl):
    # Prepare initial state.
    decoder = self.wm.heads['decoder']
    states, _ = self.wm.rssm.observe(
        self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
    start = {k: v[:, 4] for k, v in states.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    goal = self.propose_goal(start, impl)
    # Worker rollout.
    worker = lambda s: self.worker.actor({**s, 'goal': goal}).sample()
    traj = self.wm.imagine(
        worker, start, self.config.worker_report_horizon)
    # Decoder into images.
    initial = decoder(start)
    target = decoder({
        'deter': goal[..., 0],
        'stoch': self.wm.rssm.get_stoch(goal[..., 0])})
    rollout = decoder(traj)
    # Stich together into videos.
    videos = {}
    for k in rollout.keys():
      if k not in decoder.cnn_shapes:
        continue
      length = 1 + self.config.worker_report_horizon
      rows = []
      rows.append(tf.repeat(initial[k].mode()[:, None], length, 1))
      if target is not None:
        rows.append(tf.repeat(target[k].mode()[:, None], length, 1))
      rows.append(rollout[k].mode().transpose((1, 0, 2, 3, 4)))
      videos[k] = tfutils.video_grid(tf.concat(rows, 2))
    return videos
