import embodied
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import nets
from . import tfutils


class PairHierarchy(tfutils.Module):

  def __init__(self, wm, act_space, config):
    assert 'prev_deter' in config.goal_decoder.inputs
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
      self.expl_rew = self.encdec_expl_reward
    else:
      raise NotImplementedError(config.expl_rew)

    wconfig = config.update({
        'actor.inputs': config.worker_actor_inputs,
        'critic.inputs': config.worker_critic_inputs,
    })
    self.worker = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], wconfig),
        'expl': agent.VFunction(lambda s: s['reward_expl'], wconfig),
        'skill': agent.VFunction(lambda s: s['reward_skill'], wconfig),
    }, config.worker_rews, act_space, wconfig)

    mconfig = config.update({
        'actor_dist_cont': config.manager_dist,
        'actor_grad_cont': config.manager_grad,
        'actent.target': config.manager_actent,
    })
    self.manager = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], mconfig),
        'expl': agent.VFunction(lambda s: s['reward_expl'], mconfig),
        'skill': agent.VFunction(lambda s: s['reward_skill'], mconfig),
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
        config.skill_shape, dims='prev_deter', **config.goal_encoder)
    self.dec = nets.MLP(
        self.goal_shape, dims='prev_deter', **self.config.goal_decoder)
    self.kl = tfutils.AutoAdapt(**self.config.encdec_kl)
    self.opt = tfutils.Optimizer('goal', **config.encdec_opt)
    self.prev = lambda x: {f'prev_{k}': v for k, v in x.items()}

  def initial(self, batch_size):
    return (
        tf.zeros([batch_size], tf.int64),
        self.cast(tf.zeros((batch_size,) + self.config.skill_shape)),
    )

  def policy(self, state, carry):
    step, skill = carry
    keep = (step < self.config.skill_duration)
    switch = lambda x, y: (
        tf.einsum('i,i...->i...', keep.astype(x.dtype), x) +
        tf.einsum('i,i...->i...', 1 - keep.astype(x.dtype), y))
    step = switch(step, tf.zeros_like(step))
    new_skill = self.cast(self.manager.actor(state).sample())
    skill = switch(skill, new_skill)
    goal = self.cast(self.dec({'skill': skill, **self.prev(state)}).mode())
    dist = self.worker.actor({**state, 'skill': skill, 'goal': goal})
    return dist, (step + 1, skill)

  def train(self, imagine, start, data):
    metrics = {}
    if self.config.expl_rew == 'disag':
      metrics.update(self.disag.train(data))

    if self.config.encdec_loss == 'vae_replay':
      metrics.update(self.train_encdec_vae_replay(data))

    if self.config.explorer:
      traj, mets = self.explorer.train(imagine, start, data)
      metrics.update({f'explorer_{k}': v for k, v in mets.items()})
      metrics.update(self.train_encdec_vae_imag(traj))
      # metrics.update(self.train_repeat(imagine, start, data, traj))

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
      skill = self.skill_proposal(start)
      worker = lambda s: self.worker.actor(sg({
          **s, 'skill': skill,
          'goal': self.dec({'skill': skill, **self.prev(s)}).mode()})).sample()
      traj = imagine(worker, start, self.config.imag_horizon)
      traj = self.enrich_traj(traj, skill)
      traj['reward_extr'] = self.wm.heads['reward'](traj).mean()[1:]
      traj['reward_expl'] = self.expl_rew(traj)
      traj['reward_skill'] = self.skill_reward(traj)[1:]
      mtraj = self.abstract_traj(traj)
    mets = self.worker.update(traj, tape)
    metrics.update({f'worker_{k}': v for k, v in mets.items()})
    mets = self.manager.update(mtraj, tape)
    metrics.update({f'manager_{k}': v for k, v in mets.items()})
    if self.config.encdec_loss == 'vae_imag':
      metrics.update(self.train_encdec_vae_imag(traj))
    return metrics

  def train_manager(self, imagine, start, data):
    start = start.copy()
    metrics = {}
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    with tf.GradientTape(persistent=True) as tape:
      skill = self.manager.actor(start).sample()
      worker = lambda s: self.worker.actor(sg({
          **s, 'skill': skill,
          'goal': self.dec({'skill': skill, **self.prev(s)}).mode()})).sample()
      traj = imagine(worker, start, self.config.imag_horizon)
      traj = self.enrich_traj(traj, skill)
      traj['reward_extr'] = self.wm.heads['reward'](traj).mean()[1:]
      traj['reward_expl'] = self.expl_rew(traj)
      traj['reward_skill'] = self.skill_reward(traj)[1:]
      traj = self.abstract_traj(traj)
    mets = self.manager.update(traj, tape)
    metrics.update({f'manager_{k}': v for k, v in mets.items()})
    return metrics

  def train_worker(self, imagine, start, data):
    start = start.copy()
    metrics = {}
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    with tf.GradientTape(persistent=True) as tape:
      skill, goal = self.skill_proposal(start)
      worker = lambda s: self.worker.actor(sg({
          **s, 'skill': skill,
          'goal': self.dec({'skill': skill, **self.prev(s)}).mode()})).sample()
      traj = imagine(worker, start, self.config.imag_horizon)
      traj = self.enrich_traj(traj, skill)
      traj['reward_extr'] = self.wm.heads['reward'](traj).mean()[1:]
      traj['reward_expl'] = self.expl_rew(traj)
      traj['reward_skill'] = self.skill_reward(traj).astype(tf.float32)[1:]
    mets = self.worker.update(traj, tape)
    metrics.update({f'worker_{k}': v for k, v in mets.items()})
    if self.config.encdec_loss == 'vae_imag':
      metrics.update(self.train_encdec_vae_imag(traj))
    return metrics

  def train_encdec_vae_replay(self, data):
    metrics = {}
    feat = self.feat(data)
    prev = {k: v[:, :-1] for k, v in data.items()}
    goal = feat[:, 1:]
    with tf.GradientTape() as tape:
      enc = self.enc({'goal': goal, **self.prev(prev)})
      dec = self.dec({'skill': enc.sample(), **self.prev(prev)})
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
    feat = self.feat(traj)
    prev = {k: v[:-1] for k, v in traj.items()}
    goal = feat[1:]
    with tf.GradientTape() as tape:
      enc = self.enc({'goal': goal, **self.prev(prev)})
      dec = self.dec({'skill': enc.sample(), **self.prev(prev)})
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
    if self.config.skill_proposal == 'manager':
      skill = self.manager.actor(state).sample()
    elif self.config.skill_proposal == 'random':
      skill = self.prior.sample(len(state['deter']))
    elif self.config.skill_proposal == 'replay':
      target = tf.random.shuffle(self.feat(state)).astype(tf.float32)
      skill = self.enc({'goal': target, **self.prev(state)}).sample()
    else:
      raise NotImplementedError(self.config.skill_proposal)
    return skill

  def skill_reward(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    goal = traj['goal']
    if self.config.skill_reward == 'dot':
      return tf.einsum('...i,...i->...', goal, feat)
    elif self.config.skill_reward == 'dir':
      return tf.einsum('...i,...i->...', tf.nn.l2_normalize(goal, -1), feat)
    elif self.config.skill_reward == 'normed_inner':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return tf.einsum('...i,...i->...', goal / norm, feat / norm)
    elif self.config.skill_reward == 'normed_square':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return -((goal / norm - feat / norm) ** 2).sum(-1)
    elif self.config.skill_reward == 'dir_norm_clip':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return tf.clip_by_value(
          tf.einsum('...i,...i->...', goal / norm, feat / norm), -1.0, 1.0)
    elif self.config.skill_reward == 'dir_norm_clip_pos':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return tf.clip_by_value(
          tf.einsum('...i,...i->...', goal / norm, feat / norm), 0.0, 1.0)
    elif self.config.skill_reward == 'diff':
      return tf.einsum(
          '...i,...i->...',
          tf.nn.l2_normalize(goal, -1),
          tf.concat([feat[1:] - feat[:-1], 0 * feat[:1]], 0))
    elif self.config.skill_reward == 'norm':
      return -tf.norm(goal - feat, 2, -1)
    elif self.config.skill_reward == 'squared':
      return -((goal - feat) ** 2).sum(-1)
    elif self.config.skill_reward == 'exp_squared':
      return tf.exp(-((goal - feat) ** 2)).sum(-1)
    elif self.config.skill_reward == 'enc_logprob':
      return self.enc(traj).log_prob(tf.stop_gradient(traj['skill']))
    elif self.config.skill_reward == 'enc_prob':
      return self.enc(traj).prob(tf.stop_gradient(traj['skill']))
    elif self.config.skill_reward == 'epsilon':
      return ((goal - feat).mean(-1) < 1e-3).astype(tf.float32)
    else:
      raise NotImplementedError(self.config.skill_reward)

  def encdec_expl_reward(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    prev = {k: v[:-1] for k, v in traj.items()}
    goal = feat[1:]
    skill = self.enc({'goal': goal, **self.prev(prev)}).mode()
    dist = self.dec({'skill': skill, **self.prev(prev)})
    return -dist.log_prob(goal.astype(tf.float32))

  def enrich_traj(self, traj, skill):
    traj = traj.copy()
    prev = {k: tf.concat([v[:-1], v[-1:]], 0) for k, v in traj.items()}
    traj['skill'] = tf.repeat(skill[None], 1 + self.config.imag_horizon, 0)
    traj['goal'] = self.dec({'skill': traj['skill'], **self.prev(prev)}).mode()
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
    goal = self.dec({'skill': skill, **self.prev(start)}).mode()

    # Worker rollout.
    traj = self.wm.imagine(
        lambda s: self.worker.actor({
            **s, 'skill': skill, 'goal': self.dec({
                'skill': skill, **self.prev(s)}).mode()}).sample(),
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
