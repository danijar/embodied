import embodied
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import nets
from . import tfutils


class AbstractHierarchy(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.wm = wm
    self.config = config
    self.extr_reward = lambda traj: self.wm.heads['reward'](traj).mean()[1:]
    self.skill_space = embodied.Space(
        np.int32 if config.goal_encoder.dist == 'onehot' else np.float32,
        config.skill_shape)

    wconfig = config.update({
        'actor.inputs': ['deter', 'stoch', 'goal'],
        'critic.inputs': ['deter', 'stoch', 'goal'],
    })
    self.worker = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], wconfig),
        'expl': agent.VFunction(lambda s: s['reward_expl'], wconfig),
        'goal': agent.VFunction(lambda s: s['reward_goal'], wconfig),
    }, config.worker_rews, act_space, wconfig)

    mconfig = config.update({
        'actor_grad_cont': 'reinforce',
        'actent.target': config.manager_actent,
        'actor.inputs': ['deter'],
        'critic.inputs': ['deter'],
    })
    self.manager = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], mconfig),
        'expl': agent.VFunction(lambda s: s['reward_expl'], mconfig),
        'goal': agent.VFunction(lambda s: s['reward_goal'], mconfig),
    }, config.manager_rews, self.skill_space, mconfig)

    if self.config.expl_rew == 'disag':
      self.expl_reward = expl.Disag(wm, act_space, config)
    elif self.config.expl_rew == 'adver':
      self.expl_reward = self.elbo_reward
    else:
      raise NotImplementedError(self.config.expl_rew)
    if config.explorer:
      self.explorer = agent.ImagActorCritic({
          'expl': agent.VFunction(self.expl_reward, config),
      }, {'expl': 1.0}, act_space, config)

    shape = self.skill_space.shape
    if self.skill_space.discrete:
      self.prior = tfutils.OneHotDist(tf.zeros(shape))
      self.prior = tfd.Independent(self.prior, len(shape) - 1)
    else:
      self.prior = tfd.Normal(tf.zeros(shape), tf.ones(shape))
      self.prior = tfd.Independent(self.prior, len(shape))

    self.feat = nets.Input(['deter'])
    self.goal_shape = (self.config.rssm.deter,)
    self.enc = nets.MLP(
        config.skill_shape, dims='context', **config.goal_encoder)
    self.dec = nets.MLP(
        self.goal_shape, dims='context', **self.config.goal_decoder)
    self.kl = tfutils.AutoAdapt((), **self.config.encdec_kl)
    self.opt = tfutils.Optimizer('goal', **config.encdec_opt)

    self.rsgs = {
        k: nets.MLP((), **self.config.rsg)
        for k, v in self.config.manager_rews.items() if v}
    self.rsg_opt = tfutils.Optimizer('rsg', **self.config.rsg_opt)
    self.dsg = nets.MLP((), **self.config.dsg)
    self.dsg_opt = tfutils.Optimizer('dsg', **self.config.rsg_opt)
    self.jpy = nets.MLP((self.config.rssm.deter,), **self.config.jpy)
    self.jpy_opt = tfutils.Optimizer('jpy', **self.config.jpy_opt)
    self.cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    return {
        'step': tf.zeros((batch_size,), tf.int64),
        'skill': tf.zeros((batch_size,) + self.config.skill_shape, tf.float32),
        'goal': tf.zeros((batch_size,) + self.goal_shape, tf.float32),
    }

  def policy(self, latent, carry):
    duration = self.config.env_skill_duration
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    update = (carry['step'] % duration) == 0
    switch = lambda x, y: (
        tf.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
        tf.einsum('i,i...->i...', update.astype(x.dtype), y))
    skill = sg(switch(carry['skill'], self.manager.actor(sg(latent)).sample()))
    goal = sg(switch(carry['goal'], self.dec(
        {'skill': skill, 'context': self.feat(latent)}).mode()))
    dist = self.worker.actor(sg({**latent, 'goal': goal}))
    outs = {'action': dist}
    outs['log_goal'] = self.wm.heads['decoder'](
        {'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal)})['image'].mode()
    carry = {'step': carry['step'] + 1, 'skill': skill, 'goal': goal}
    return outs, carry

  def train(self, imagine, start, data):
    success = lambda rew: (rew[-1] > 0.7).astype(tf.float32).mean()
    metrics = {}
    if self.config.expl_rew == 'disag':
      metrics.update(self.expl_reward.train(data))
    if self.config.vae_replay:
      metrics.update(self.train_vae_replay(data))
    if self.config.explorer:
      traj, mets = self.explorer.train(imagine, start, data)
      metrics.update({f'explorer_{k}': v for k, v in mets.items()})
      metrics.update(self.train_vae_imag(traj))
      if self.config.explorer_repeat:
        goal = self.feat(traj)[-1]
        traj, mets = self.train_worker(imagine, start, goal)
        metrics.update(mets)
        metrics.update(self.train_rsgs(traj))
        metrics.update(self.train_dsg(traj))
        metrics.update(self.train_jpy(traj))
    for impl in self.config.worker_goals:
      goal = self.propose_goal(start, impl)
      traj, mets = self.train_worker(imagine, start, goal)
      metrics.update(mets)
      metrics[f'success_{impl}'] = success(traj['reward_goal'])
      metrics.update(self.train_rsgs(traj))
      metrics.update(self.train_dsg(traj))
      metrics.update(self.train_jpy(traj))
      if self.config.vae_imag:
        metrics.update(self.train_vae_imag(traj))
    traj, mets = self.train_manager(imagine, start)
    metrics.update(mets)
    return None, metrics

  def train_manager(self, imagine, start):
    start = start.copy()
    metrics = {}
    with tf.GradientTape(persistent=True) as tape:
      traj = self.abstr_imagine(start, self.config.manager_imag_horizon)
      for key, rsg in self.rsgs.items():
        traj[f'reward_{key}'] = rsg(traj).mean()[:-1]
    mets = self.manager.update(traj, tape)
    metrics.update({f'manager_{k}': v for k, v in mets.items()})
    return traj, metrics

  def abstr_imagine(self, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(tf.float32)
    start = {'deter': start['deter']}
    start['action'] = self.manager.actor(start).sample()
    start['goal'] = self.dec({
        'skill': start['action'], 'context': start['deter']}).mode()
    def step(prev, _):
      prev = prev.copy()
      action = prev.pop('action')
      goal = self.dec({'skill': action, 'context': prev['deter']}).mode()
      deter = self.jpy({'deter': prev['deter'], 'goal': goal}).mode()
      state = {'deter': self.cast(deter)}
      action = self.manager.actor(state).sample()
      return {**state, 'action': action, 'goal': goal}
    traj = tfutils.scan(
        step, tf.range(horizon), start, self.config.imag_unroll)
    traj = {k: tf.concat([start[k][None], v], 0) for k, v in traj.items()}
    traj['cont'] = tf.concat([
        first_cont[None], self.dsg(traj).mean()[1:]], 0)
    traj['weight'] = tf.math.cumprod(
        self.config.discount * traj['cont']) / self.config.discount
    return traj

  def train_worker(self, imagine, start, goal):
    start = start.copy()
    metrics = {}
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    with tf.GradientTape(persistent=True) as tape:
      worker = lambda s: self.worker.actor(sg({**s, 'goal': goal})).sample()
      traj = imagine(worker, start, self.config.imag_horizon)
      traj['goal'] = tf.repeat(goal[None], 1 + self.config.imag_horizon, 0)
      traj['reward_extr'] = self.extr_reward(traj)
      traj['reward_expl'] = self.expl_reward(traj)
      traj['reward_goal'] = self.goal_reward(traj)
    mets = self.worker.update(traj, tape)
    metrics.update({f'worker_{k}': v for k, v in mets.items()})
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
      kl = tfd.kl_divergence(enc, self.prior)
      kl, mets = self.kl(kl)
      metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
      assert rec.shape == kl.shape, (rec.shape, kl.shape)
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
      kl = tfd.kl_divergence(enc, self.prior)
      kl, mets = self.kl(kl)
      metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
      loss = (rec + kl).mean()
    metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
    metrics['goalrec_mean'] = rec.mean()
    metrics['goalrec_std'] = rec.std()
    return metrics

  def train_rsgs(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    start, goal = feat[0], traj['goal'][0]
    mask = tf.math.cumsum(traj['cont'][:-1], 0)
    with tf.GradientTape() as tape:
      losses = []
      for key, rsg in self.rsgs.items():
        total = tf.stop_gradient((traj[f'reward_{key}'] * mask).mean(0))
        loss = -rsg({'deter': start, 'goal': goal}).log_prob(total).mean()
        losses.append(loss)
      loss = tf.math.reduce_sum(losses, 0)
    return self.rsg_opt(tape, loss, list(self.rsgs.values()))

  def train_dsg(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    start, goal = feat[0], traj['goal'][0]
    with tf.GradientTape() as tape:
      total = tf.stop_gradient(traj['cont'][1:].prod(0))
      loss = -self.dsg({'deter': start, 'goal': goal}).log_prob(total).mean()
    return self.dsg_opt(tape, loss, self.dsg)

  def train_jpy(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    start, goal, final = feat[0], traj['goal'][0], feat[-1]
    with tf.GradientTape() as tape:
      loss = -self.jpy({'deter': start, 'goal': goal}).log_prob(final).mean()
    return self.jpy_opt(tape, loss, self.jpy)

  def propose_goal(self, start, impl):
    feat = self.feat(start)
    if impl == 'replay':
      target = tf.random.shuffle(feat).astype(tf.float32)
      skill = self.enc({'goal': target, 'context': feat}).sample()
      return self.dec({'skill': skill, 'context': feat}).mode()
    if impl == 'replay_direct':
      return tf.random.shuffle(feat).astype(tf.float32)
    if impl == 'manager':
      skill = self.manager.actor(start).sample()
      return self.dec({'skill': skill, 'context': feat}).mode()
    if impl == 'prior':
      skill = self.prior.sample(len(start['is_terminal']))
      return self.dec({'skill': skill, 'context': feat}).mode()
    raise NotImplementedError(impl)

  def goal_reward(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    goal = tf.stop_gradient(traj['goal'].astype(tf.float32))
    if self.config.goal_reward == 'dot':
      return tf.einsum('...i,...i->...', goal, feat)[1:]
    elif self.config.goal_reward == 'dir':
      return tf.einsum(
          '...i,...i->...', tf.nn.l2_normalize(goal, -1), feat)[1:]
    elif self.config.goal_reward == 'normed_inner':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'cosine_lower':
      gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      fnorm = tf.maximum(gnorm, fnorm)
      return tf.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
    elif self.config.goal_reward == 'cosine_lower_pos':
      gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      fnorm = tf.maximum(gnorm, fnorm)
      cos = tf.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
      return tf.nn.relu(cos)
    elif self.config.goal_reward == 'cosine_frac':
      gnorm = tf.linalg.norm(goal, axis=-1) + 1e-12
      fnorm = tf.linalg.norm(feat, axis=-1) + 1e-12
      goal /= gnorm[..., None]
      feat /= fnorm[..., None]
      cos = tf.einsum('...i,...i->...', goal, feat)
      mag = tf.minimum(gnorm, fnorm) / tf.maximum(gnorm, fnorm)
      return (cos * mag)[1:]
    elif self.config.goal_reward == 'cosine_frac_pos':
      gnorm = tf.linalg.norm(goal, axis=-1) + 1e-12
      fnorm = tf.linalg.norm(feat, axis=-1) + 1e-12
      goal /= gnorm[..., None]
      feat /= fnorm[..., None]
      cos = tf.einsum('...i,...i->...', goal, feat)
      mag = tf.minimum(gnorm, fnorm) / tf.maximum(gnorm, fnorm)
      return tf.nn.relu(cos * mag)[1:]
    elif self.config.goal_reward == 'cosine_max':
      gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      norm = tf.maximum(gnorm, fnorm)
      return tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'normed_inner_clip':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      cosine = tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
      return tf.clip_by_value(cosine, -1.0, 1.0)
    elif self.config.goal_reward == 'normed_inner_clip_pos':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      cosine = tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
      return tf.clip_by_value(cosine, 0.0, 1.0)
    elif self.config.goal_reward == 'diff':
      goal = tf.nn.l2_normalize(goal[:-1], -1)
      diff = tf.concat([feat[1:] - feat[:-1]], 0)
      return tf.einsum('...i,...i->...', goal, diff)
    elif self.config.goal_reward == 'norm':
      return -tf.norm(goal - feat, 2, -1)[1:]
    elif self.config.goal_reward == 'squared':
      return -((goal - feat) ** 2).sum(-1)[1:]
    elif self.config.goal_reward == 'epsilon':
      return ((goal - feat).mean(-1) < 1e-3).astype(tf.float32)[1:]
    else:
      raise NotImplementedError(self.config.goal_reward)

  def elbo_reward(self, traj):
    feat = self.feat(traj).astype(tf.float32)
    context = tf.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
    enc = self.enc({'goal': feat, 'context': context})
    dec = self.dec({'skill': enc.sample(), 'context': context})
    ll = dec.log_prob(feat)
    kl = tfd.kl_divergence(enc, self.prior)
    if self.config.adver_impl == 'abs':
      return tf.abs(dec.mode() - feat).mean(-1)[1:]
    elif self.config.adver_impl == 'squared':
      return ((dec.mode() - feat) ** 2).mean(-1)[1:]
    elif self.config.adver_impl == 'elbo_scaled':
      return (kl - ll / self.kl.scale())[1:]
    elif self.config.adver_impl == 'elbo_unscaled':
      return (kl - ll)[1:]
    raise NotImplementedError(self.config.adver_impl)

  def report(self, data):
    metrics = {}
    for impl in ('manager', 'prior', 'replay'):
      for key, video in self.report_worker(data, impl).items():
        metrics[f'goals_{impl}_{key}'] = video
    for key, video in self.report_manager(data).items():
        metrics[f'abstract_{key}'] = video
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
    traj = self.wm.imagine(
        lambda s: self.worker.actor({**s, 'goal': goal}).sample(),
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

  def report_manager(self, data):
    videos = {}
    decoder = lambda deter: self.wm.heads['decoder']({
        'deter': deter, 'stoch': self.wm.rssm.get_stoch(traj['deter'])})
    states, _ = self.wm.rssm.observe(
        self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
    start = {k: v[:, 4] for k, v in states.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    traj = self.abstr_imagine(start, self.config.manager_imag_horizon)
    start = self.wm.heads['decoder'](start)
    goal = decoder(traj['goal'])
    deter = decoder(traj['deter'])
    for k in deter.keys():
      if k not in self.wm.heads['decoder'].cnn_shapes:
        continue
      video_deter = tf.repeat(deter[k].mode(), self.config.imag_horizon, 0)
      video_goal = tf.repeat(goal[k].mode(), self.config.imag_horizon, 0)
      video_start = tf.repeat(start[k].mode()[None], len(video_deter), 0)
      rows = [video_start, video_deter, video_goal]
      rows = [video.transpose((1, 0, 2, 3, 4)) for video in rows]
      videos[k] = tfutils.video_grid(tf.concat(rows, 2))
    return videos
