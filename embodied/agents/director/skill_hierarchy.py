import embodied
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import nets
from . import tfutils


class SkillHierarchy(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.wm = wm
    self.config = config
    self.cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)
    self.skill_space = embodied.Space(
        np.int32 if config.goal_encoder.dist == 'onehot' else np.float32,
        config.skill_shape)
    if config.expl_rew == 'disag':
      self.disag = expl.Disag(wm, act_space, config)
      self.expl_rew = self.disag
    elif config.expl_rew == 'adver':
      self.expl_rew = self.elbo_reward
    else:
      raise NotImplementedError(config.expl_rew)

    wconfig = config.update({
        'actor.inputs': config.worker_actor_inputs,
        'critic.inputs': config.worker_critic_inputs,
    })
    self.worker = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], wconfig),
        'expl': agent.VFunction(lambda s: s['reward_expl'], wconfig),
        'goal': agent.VFunction(lambda s: s['reward_skill'], wconfig),
    }, config.worker_rews, act_space, wconfig)

    mconfig = config.update({
        'actor_dist_cont': config.manager_dist,
        'actor_grad_cont': config.manager_grad,
        'actent.target': config.manager_actent,
    })
    self.manager = agent.ImagActorCritic({
        'extr': PresumptiveVFunction(wm, lambda s: s['reward_extr'], mconfig),
        'expl': PresumptiveVFunction(wm, lambda s: s['reward_expl'], mconfig),
        'goal': PresumptiveVFunction(
            wm, lambda s: s['reward_skill'], mconfig),
    }, config.manager_rews, self.skill_space, mconfig)

    if config.explorer:
      self.explorer = agent.ImagActorCritic({
          'expl': agent.VFunction(self.expl_rew, config),
      }, {'expl': 1.0}, act_space, config)

    shape = self.skill_space.shape
    if self.skill_space.discrete:
      self.prior = tfutils.OneHotDist(tf.zeros(shape))
      self.prior = tfd.Independent(self.prior, len(shape) - 1)
    else:
      self.prior = tfd.Normal(tf.zeros(shape), tf.ones(shape))
      self.prior = tfd.Independent(self.prior, len(shape))
    self.fixed_skills = None

    self.feat = nets.Input(self.config.goal_feat, dims='deter')
    if self.config.goal_feat == ('deter',):
      self.goal_shape = (self.config.rssm.deter,)
    elif self.config.goal_feat == ('deter', 'stoch'):
      stoch_size = self.config.rssm.stoch * (self.config.rssm.classes or 1)
      self.goal_shape = (self.config.rssm.deter + stoch_size,)
    else:
      raise NotImplementedError(self.config.goal_feat)
    self.enc = nets.MLP(
        config.skill_shape, dims='context', **config.goal_encoder)
    self.dec = nets.MLP(
        self.goal_shape, dims='context', **self.config.goal_decoder)
    self.kl = tfutils.AutoAdapt((), **self.config.encdec_kl)
    self.opt = tfutils.Optimizer('goal', **config.encdec_opt)

  def initial(self, batch_size):
    return (
        -1 * tf.ones([batch_size], tf.int64),
        self.cast(tf.zeros((batch_size,) + self.config.skill_shape)),
        self.cast(tf.zeros((batch_size,) + self.goal_shape)),
    )

  def policy(self, state, carry):
    step, skill, goal = carry
    step = (step + 1) % self.config.skill_duration
    keep = (step > 0)
    switch = lambda x, y: (
        tf.einsum('i,i...->i...', keep.astype(x.dtype), x) +
        tf.einsum('i,i...->i...', 1 - keep.astype(x.dtype), y))
    step = switch(step, tf.zeros_like(step))
    new_skill = self.cast(self.manager.actor(state).sample())
    context = self.feat(state)
    skill = switch(skill, new_skill)
    new_goal = self.cast(self.dec({'skill': skill, 'context': context}).mode())
    goal = switch(goal, new_goal)
    dist = self.worker.actor({**state, 'goal': goal})
    return dist, (step + 1, skill, goal)

  def train(self, imagine, start, data):
    metrics = {}
    if self.config.expl_rew == 'disag':
      metrics.update(self.disag.train(data))
    if self.config.vae_replay:
      metrics.update(self.train_encdec_vae_replay(data))
    if self.config.explorer:
      traj, mets = self.explorer.train(imagine, start, data)
      metrics.update({f'explorer_{k}': v for k, v in mets.items()})
      metrics.update(self.train_encdec_vae_imag(traj))
      if self.config.explorer_repeat:
        metrics.update(self.train_repeat(imagine, start, data, traj))
    if self.config.skill_proposal == 'manager':
      metrics.update(self.train_jointly(imagine, start, data))
    else:
      metrics.update(self.train_worker(imagine, start, data))
      metrics.update(self.train_manager(imagine, start, data))
    return None, metrics

  def train_jointly(self, imagine, start, data):
    start = start.copy()
    metrics = {}
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    with tf.GradientTape(persistent=True) as tape:
      context = self.feat(start)
      skill = self.skill_proposal(start)
      goal = self.dec({'skill': skill, 'context': context}).mode()
      worker = lambda s: self.worker.actor(
          sg({**s, 'skill': skill, 'context': context, 'goal': goal})).sample()
      traj = imagine(worker, start, self.config.imag_horizon)
      traj = self.enrich_traj(traj, skill, goal, context)
      traj['reward_extr'] = self.wm.heads['reward'](traj).mean()[1:]
      traj['reward_expl'] = self.expl_rew(traj)
      traj['reward_skill'] = self.goal_reward(traj)[1:]
      wtraj = traj.copy()
      mtraj = self.abstract_traj(traj)
    metrics['worker_success_mean'] = (
        traj['reward_skill'][-1] > 0.8).astype(tf.float32).mean()
    metrics['worker_success_std'] = (
        traj['reward_skill'][-1] > 0.8).astype(tf.float32).std()
    mets = self.worker.update(wtraj, tape)
    metrics.update({f'worker_{k}': v for k, v in mets.items()})
    mets = self.manager.update(mtraj, tape)
    metrics.update({f'manager_{k}': v for k, v in mets.items()})
    if self.config.vae_imag:
      metrics.update(self.train_encdec_vae_imag(traj))
    return metrics

  def train_manager(self, imagine, start, data):
    start = start.copy()
    metrics = {}
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    context = self.feat(start)
    with tf.GradientTape(persistent=True) as tape:
      skill = self.manager.actor(start).sample()
      goal = self.dec({'skill': skill, 'context': context}).mode()
      worker = lambda s: self.worker.actor({
          **sg(s), 'skill': skill, 'context': context, 'goal': goal}).sample()
      traj = imagine(worker, start, self.config.imag_horizon)
      traj = self.enrich_traj(traj, skill, goal, context)
      traj['reward_extr'] = self.wm.heads['reward'](traj).mean()[1:]
      traj['reward_expl'] = self.expl_rew(traj)
      traj['reward_skill'] = self.goal_reward(traj)[1:]
      traj = self.abstract_traj(traj)
    mets = self.manager.update(traj, tape)
    metrics.update({f'manager_{k}': v for k, v in mets.items()})
    return metrics

  def train_worker(self, imagine, start, data):
    start = start.copy()
    metrics = {}
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    context = self.feat(start)
    with tf.GradientTape(persistent=True) as tape:
      skill = self.skill_proposal(start)
      goal = self.dec({'skill': skill, 'context': context}).mode()
      worker = lambda s: self.worker.actor(
          sg({**s, 'skill': skill, 'context': context, 'goal': goal})).sample()
      traj = imagine(worker, start, self.config.imag_horizon)
      traj = self.enrich_traj(traj, skill, goal, context)
      traj['reward_extr'] = self.wm.heads['reward'](traj).mean()[1:]
      traj['reward_expl'] = self.expl_rew(traj)
      traj['reward_skill'] = self.goal_reward(traj).astype(tf.float32)[1:]
    metrics['worker_success_mean'] = (
        traj['reward_skill'][-1] > 0.8).astype(tf.float32).mean()
    metrics['worker_success_std'] = (
        traj['reward_skill'][-1] > 0.8).astype(tf.float32).std()
    mets = self.worker.update(traj, tape)
    metrics.update({f'worker_{k}': v for k, v in mets.items()})
    if self.config.vae_imag:
      metrics.update(self.train_encdec_vae_imag(traj))
    return metrics

  def train_repeat(self, imagine, start, data, repeat_traj):
    start = start.copy()
    metrics = {}
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    context = self.feat(start).astype(tf.float32)
    goal = self.feat(repeat_traj)[-1].astype(tf.float32)
    with tf.GradientTape(persistent=True) as tape:
      skill = self.enc({'goal': goal, 'context': context}).mode()
      worker = lambda s: self.worker.actor(
          sg({**s, 'skill': skill, 'context': context, 'goal': goal})).sample()
      traj = imagine(worker, start, self.config.imag_horizon)
      traj = self.enrich_traj(traj, skill, goal, context)
      traj['reward_extr'] = self.wm.heads['reward'](traj).mean()[1:]
      traj['reward_expl'] = self.expl_rew(traj)
      traj['reward_skill'] = self.goal_reward(traj).astype(tf.float32)[1:]
    mets = self.worker.update(traj, tape)
    metrics.update({f'worker_{k}': v for k, v in mets.items()})
    if self.config.vae_imag:
      metrics.update(self.train_encdec_vae_imag(traj))
    return metrics

  def train_encdec_vae_replay(self, data):
    metrics = {}
    context = self.feat(data).astype(tf.float32)
    goal = tfutils.shuffle(self.feat(data), axis=1).astype(tf.float32)
    with tf.GradientTape() as tape:
      enc = self.enc({'goal': goal, 'context': context})
      dec = self.dec({'skill': enc.sample(), 'context': context})
      rec = -dec.log_prob(tf.stop_gradient(goal.astype(tf.float32)))
      kl = tfd.kl_divergence(enc, self.prior)
      kl, mets = self.kl(kl)
      metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
      assert rec.shape == kl.shape, (rec.shape, kl.shape)
      loss = (rec + kl).mean()
    metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
    metrics['goalrec_mean'] = rec.mean()
    metrics['goalrec_std'] = rec.std()
    return metrics

  def train_encdec_vae_imag(self, traj):
    metrics = {}
    feat = self.feat(traj).astype(tf.float32)
    if 'context' in self.config.goal_decoder.inputs:
      context = feat[0]
      goal = feat[-1]
    else:
      context = tfutils.shuffle(feat, axis=0)
      goal = feat
    with tf.GradientTape() as tape:
      enc = self.enc({'goal': goal, 'context': context})
      dec = self.dec({'skill': enc.sample(), 'context': context})
      rec = -dec.log_prob(tf.stop_gradient(goal.astype(tf.float32)))
      kl = tfd.kl_divergence(enc, self.prior)
      kl, mets = self.kl(kl)
      metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
      loss = (rec + kl).mean()
    metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
    metrics['goalrec_mean'] = rec.mean()
    metrics['goalrec_std'] = rec.std()
    return metrics

  def skill_proposal(self, state):
    context = self.feat(state).astype(tf.float32)
    if self.config.skill_proposal == 'manager':
      return self.manager.actor(state).sample()
    elif self.config.skill_proposal == 'random':
      return self.prior.sample(len(state['deter']))
    elif self.config.skill_proposal == 'replay':
      target = tf.random.shuffle(self.feat(state)).astype(tf.float32)
      return self.enc({'goal': target, 'context': context}).sample()
    else:
      raise NotImplementedError(self.config.skill_proposal)

  def goal_reward(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    goal = traj['goal']
    if self.config.goal_reward == 'dot':
      return tf.einsum('...i,...i->...', goal, feat)
    elif self.config.goal_reward == 'dir':
      return tf.einsum('...i,...i->...', tf.nn.l2_normalize(goal, -1), feat)
    elif self.config.goal_reward == 'normed_inner':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return tf.einsum('...i,...i->...', goal / norm, feat / norm)
    elif self.config.goal_reward == 'normed_inner_clip':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      dist = tf.einsum('...i,...i->...', goal / norm, feat / norm)
      return tf.clip_by_value(dist, -1.0, 1.0)
    elif self.config.goal_reward == 'normed_inner_clip_pos':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      dist = tf.einsum('...i,...i->...', goal / norm, feat / norm)
      return tf.clip_by_value(dist, 0.0, 1.0)
    elif self.config.goal_reward == 'cosine_lower':
      gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      fnorm = tf.maximum(gnorm, fnorm)
      return tf.einsum('...i,...i->...', goal / gnorm, feat / fnorm)
    elif self.config.goal_reward == 'diff':
      return tf.einsum(
          '...i,...i->...',
          tf.nn.l2_normalize(goal, -1),
          tf.concat([0 * feat[:1], feat[1:] - feat[:-1]], 0))
    elif self.config.goal_reward == 'norm':
      return -tf.norm(goal - feat, 2, -1)
    elif self.config.goal_reward == 'squared':
      return -((goal - feat) ** 2).sum(-1)
    elif self.config.goal_reward == 'epsilon':
      return ((goal - feat).mean(-1) < 1e-3).astype(tf.float32)
    else:
      raise NotImplementedError(self.config.goal_reward)

  def elbo_reward(self, traj, impl='logprob'):
    feat = self.feat(traj).astype(tf.float32)
    context = tf.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
    dist = self.enc({'goal': feat, 'context': context})
    ll = self.dec({'skill': dist.sample(), 'context': context}).log_prob(feat)
    kl = tfd.kl_divergence(dist, self.prior)
    return (kl - ll / self.kl.scale())[1:]

  def enrich_traj(self, traj, skill, goal, context):
    traj = traj.copy()
    traj['skill'] = tf.repeat(skill[None], 1 + self.config.imag_horizon, 0)
    traj['goal'] = tf.repeat(goal[None], 1 + self.config.imag_horizon, 0)
    traj['context'] = tf.repeat(context[None], 1 + self.config.imag_horizon, 0)
    return traj

  def abstract_traj(self, traj):
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
    for skills in ('training', 'manager', 'fixed'):
      for key, video in self.report_skills(data, skills).items():
        metrics[f'skills_{skills}_{key}'] = video
    return metrics

  def report_skills(self, data, skills):

    # Prepare initial state.
    decoder = self.wm.heads['decoder']
    states, _ = self.wm.rssm.observe(
        self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
    start = {k: v[:, 4] for k, v in states.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]

    # Prepare skills.
    if skills == 'training':
      skill = self.skill_proposal(start)
    elif skills == 'manager':
      skill = self.manager.actor(start).sample()
    elif skills == 'fixed':
      if self.fixed_skills is None:
        self.fixed_skills = self.prior.sample(len(start['is_terminal']))
      skill = self.fixed_skills
    context = self.feat(start)
    goal = self.dec({'skill': skill, 'context': context}).mode()

    # Worker rollout.
    traj = self.wm.imagine(
        lambda s: self.worker.actor(
            {**s, 'skill': skill, 'goal': goal, 'context': context}).sample(),
        start, self.config.worker_report_horizon)

    # Decoder into images.
    initial = decoder(start)
    if self.config.goal_feat == ('deter', 'stoch'):
      n = self.config.rssm.deter
      target = decoder({'deter': goal[..., :n], 'stoch': goal[..., n:]})
    elif self.config.goal_feat == ('deter',):
      target = decoder({'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal)})

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


class OneStepReward(tfutils.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), **self.config.critic)
    self.rewnorm = tfutils.Normalize(**self.config.rewnorm)
    self.advnorm = tfutils.Normalize(**self.config.advnorm)
    self.opt = tfutils.Optimizer('critic', **self.config.critic_opt)

  def train(self, traj, actor):
    metrics = {}
    reward = self.rewfn(traj)
    target = tf.stop_gradient(self.rewnorm(reward))
    with tf.GradientTape() as tape:
      dist = self.net({k: v[:-1] for k, v in traj.items()})
      loss = -(dist.log_prob(target) * traj['weight'][:-1]).mean()
    metrics.update(self.opt(tape, loss, self.net))
    metrics.update({
        'imag_reward_mean': reward.mean(),
        'imag_reward_std': reward.std(),
        'imag_baseline_mean': dist.mean().mean(),
        'imag_baseline_std': dist.mean().std(),
    })
    return metrics

  def score(self, traj, actor):
    reward = self.rewnorm.transform(self.rewfn(traj))
    baseline = self.net(traj).mean()[:-1]
    return self.advnorm(reward - baseline)


class PresumptiveVFunction(tfutils.Module):

  def __init__(self, wm, rewfn, config):
    assert 'action' not in config.critic.inputs, config.critic.inputs
    self.wm = wm
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), **self.config.critic)
    if self.config.slow_target:
      self.target_net = nets.MLP((), **self.config.critic)
      self.updates = tf.Variable(-1, dtype=tf.int64)
    else:
      self.target_net = self.net
    self.opt = tfutils.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = tfutils.Normalize(**self.config.rewnorm)
    self.advnorm = tfutils.Normalize(**self.config.advnorm)

  def train(self, traj, actor):
    metrics = {}
    reward = self.rewfn(traj)
    target = tf.stop_gradient(self.target(
        traj, reward, self.config.critic_return, baseline=False, update=True))
    with tf.GradientTape() as tape:
      dist = self.net({k: v[:-1] for k, v in traj.items()})
      loss = -(dist.log_prob(target) * traj['weight'][:-1]).mean()
    metrics.update(self.opt(tape, loss, self.net))
    metrics.update({
        'imag_reward_mean': reward.mean(),
        'imag_reward_std': reward.std(),
        'imag_reward_normed_mean': self.rewnorm.transform(reward).mean(),
        'imag_reward_normed_std': self.rewnorm.transform(reward).std(),
        'imag_critic_mean': dist.mean().mean(),
        'imag_critic_std': dist.mean().std(),
        'imag_return_mean': target.mean(),
        'imag_return_std': target.std(),
    })
    self.update_slow()
    return metrics

  def score(self, traj, actor):
    return self.advnorm(self.target(
        traj, self.rewfn(traj), self.config.actor_return,
        baseline=True, update=False))

  def target(self, traj, reward, impl, baseline, update):
    if len(reward) != len(traj['action']) - 1:
      raise AssertionError('Should provide rewards for all but last action.')
    reward = self.rewnorm(reward, update)
    cont = traj['cont'][1:]
    cont = cont if self.config.episodic else tf.ones_like(cont)
    disc = cont * self.config.discount
    value = self.target_net(traj).mean()
    if self.config.presumptive:
      goal = {
          'deter': traj['goal'], 'stoch': self.wm.rssm.get_stoch(traj['goal'])}
      value = (
          (1 - self.config.presumptive) * value +
          self.config.presumptive * self.target_net(goal).mean())
    if impl == 'gae':
      advs = [tf.zeros_like(value[0])]
      deltas = reward + disc * value[1:] - value[:-1]
      for t in reversed(range(len(disc))):
        advs.append(deltas[t] + disc[t] * self.config.return_lambda * advs[-1])
      adv = tf.stack(list(reversed(advs))[:-1])
      return adv if baseline else adv + value[:-1]
    elif impl == 'gve':
      vals = [value[-1]]
      interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
      for t in reversed(range(len(disc))):
        vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
      tar = tf.stack(list(reversed(vals))[:-1])
      return tar - value[:-1] if baseline else tar
    else:
      raise NotImplementedError(impl)

  def update_slow(self):
    if not self.config.slow_target:
      return
    assert self.net.variables
    initialize = (self.updates == -1)
    if initialize or self.updates >= self.config.slow_target_update:
      self.updates.assign(0)
      mix = 1.0 if initialize else self.config.slow_target_fraction
      for s, d in zip(self.net.variables, self.target_net.variables):
        d.assign(mix * s + (1 - mix) * d)
    self.updates.assign_add(1)
