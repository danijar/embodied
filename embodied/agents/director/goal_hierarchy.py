import functools

import embodied
import numpy as np
import tensorflow as tf

from . import agent
from . import expl
from . import tfutils


class GoalHierarchy(tfutils.Module):

  # TODO:
  # - learn linear autoencoder (with or without L2 norm activation)
  # - let manager choose latent and latent mask to specify partial goals

  def __init__(self, wm, act_space, config):
    self.wm = wm
    self.config = config
    self.extr_reward = lambda traj: self.wm.heads['reward'](traj).mean()[1:]
    self.goal_space = embodied.Space(np.float32, (self.config.rssm.deter,))

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
    })
    self.manager = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], mconfig),
        'expl': agent.VFunction(lambda s: s['reward_expl'], mconfig),
        'goal': agent.VFunction(lambda s: s['reward_goal'], mconfig),
    }, config.manager_rews, self.goal_space, mconfig)

    self.expl_reward = expl.Disag(wm, act_space, config)
    if config.explorer:
      self.explorer = agent.ImagActorCritic({
          'expl': agent.VFunction(self.expl_reward, config),
      }, {'expl': 1.0}, act_space, config)

  def initial(self, batch_size):
    return {
        'step': tf.zeros((batch_size,), tf.int64),
        'goal': tf.zeros((batch_size,) + self.goal_space.shape, tf.float32),
    }

  def policy(self, latent, carry, duration=None):
    duration = duration or self.config.env_skill_duration
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    update = (carry['step'] % duration) == 0
    switch = lambda x, y: (
        tf.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
        tf.einsum('i,i...->i...', update.astype(x.dtype), y))
    if self.config.manager_delta:
      feat = latent['deter'].astype(tf.float32)
      delta = self.manager.actor(sg(latent)).sample()
      if self.config.manager_delta_max:
        delta = tf.clip_by_norm(delta, self.config.manager_delta_max)
      goal = sg(switch(carry['goal'], feat + delta))
    else:
      goal = sg(switch(carry['goal'], self.manager.actor(sg(latent)).sample()))
    dist = self.worker.actor(sg({**latent, 'goal': goal}))
    outs = {'action': dist}
    outs['log_goal'] = self.wm.heads['decoder'](
        {'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal)})['image'].mode()
    carry = {'step': carry['step'] + 1, 'goal': goal}
    return outs, carry

  def train(self, imagine, start, data):
    success = lambda rew: (rew[-1] > 0.7).astype(tf.float32).mean()
    metrics = {}
    if self.config.expl_rew == 'disag':
      metrics.update(self.expl_reward.train(data))
    if self.config.explorer:
      traj, mets = self.explorer.train(imagine, start, data)
      metrics.update({f'explorer_{k}': v for k, v in mets.items()})
      goal = traj['deter'][-1]
      metrics.update(self.train_worker(imagine, start, goal)[1])
    if self.config.jointly == 'new':
      traj, mets = self.train_jointly(imagine, start)
      metrics.update(mets)
      metrics['success_manager'] = success(traj['reward_goal'])
    elif self.config.jointly == 'off':
      for impl in self.config.worker_goals:
        goal = self.propose_goal(start, impl)
        traj, mets = self.train_worker(imagine, start, goal)
        metrics.update(mets)
        metrics[f'success_{impl}'] = success(traj['reward_goal'])
      traj, mets = self.train_manager(imagine, start)
      metrics.update(mets)
      metrics['success_manager'] = success(traj['reward_goal'])
    else:
      raise NotImplementedError(self.config.jointly)
    return None, metrics

  def train_jointly(self, imagine, start):
    start = start.copy()
    metrics = {}
    with tf.GradientTape(persistent=True) as tape:
      policy = functools.partial(
          self.policy, duration=self.config.train_skill_duration)
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

  def train_manager(self, imagine, start):
    start = start.copy()
    with tf.GradientTape(persistent=True) as tape:
      policy = functools.partial(
          self.policy, duration=self.config.train_skill_duration)
      traj = self.wm.imagine_carry(
          policy, start, self.config.imag_horizon,
          self.initial(len(start['is_first'])))
      traj['reward_extr'] = self.extr_reward(traj)
      traj['reward_expl'] = self.expl_reward(traj)
      traj['reward_goal'] = self.goal_reward(traj)
      mtraj = self.abstract_traj(traj)
    metrics = self.manager.update(mtraj, tape)
    metrics = {f'manager_{k}': v for k, v in metrics.items()}
    return traj, metrics

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

  def propose_goal(self, start, impl):
    feat = start['deter'].astype(tf.float32)
    if impl == 'replay':
      return tf.random.shuffle(feat)
    if impl == 'manager':
      if self.config.manager_delta:
        delta = self.manager.actor(start).sample()
        if self.config.manager_delta_max:
          delta = tf.clip_by_norm(delta, self.config.manager_delta_max)
        return feat + delta
      else:
        return self.manager.actor(start).sample()
    raise NotImplementedError(impl)

  def goal_reward(self, traj):
    feat = traj['deter'].astype(tf.float32)
    goal = tf.stop_gradient(traj['goal'].astype(tf.float32))
    if self.config.goal_reward == 'dot':
      return tf.einsum('...i,...i->...', goal, feat)[1:]
    elif self.config.goal_reward == 'dir':
      return tf.einsum(
          '...i,...i->...', tf.nn.l2_normalize(goal, -1), feat)[1:]
    elif self.config.goal_reward == 'normed_inner':
      norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
      return tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'hiro':
      return -tf.linalg.norm(feat[:-1] + goal[:-1] - feat[:-1], axis=-1)
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
    traj['action'] = traj.pop('goal')
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

  def report(self, data):
    metrics = {}
    for impl in ('manager', 'replay'):
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
    traj = self.wm.imagine(
        lambda s: self.worker.actor({**s, 'goal': goal}).sample(),
        start, self.config.worker_report_horizon)
    # Decoder into images.
    initial = decoder(start)
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
