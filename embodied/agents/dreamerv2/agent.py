import pathlib
import sys

import embodied
import ruamel.yaml as yaml
import tensorflow as tf

from . import expl
from . import nets
from . import tfutils


class Agent(tfutils.Module, embodied.Agent):

  configs = yaml.YAML(typ='safe').load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())

  def __init__(self, obs_space, act_space, step, config):
    tfutils.setup(**config.tf)
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(config, obs_space)
    self._task_behavior = ActorCritic(config, self.act_space)
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm,
          lambda seq: self.wm.heads['reward'](seq['feat']).mode())

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    self.config.tf.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action
    latent, action = state
    embed = self.wm.encoder(obs)
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], sample)
    feat = self.wm.rssm.get_feat(latent)
    if mode == 'eval':
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    action = tfutils.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None):
    self.config.tf.jit and print('Tracing train function.')
    data = self.preprocess(data)
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    start = outputs['post']
    reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
    metrics.update(self._task_behavior.train(
        self.wm, start, data['is_terminal'], reward))
    if self.config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  @tf.function
  def report(self, data):
    self.config.tf.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.loss(data)[-1])
    report.update({
        'openl_' + k.replace('/', '_'): v
        for k, v in self.wm.video_pred(data).items()})
    return report

  def dataset(self, generator):
    example = next(generator())
    dtypes = {k: v.dtype for k, v in example.items()}
    shapes = {k: v.shape for k, v in example.items()}
    dataset = tf.data.Dataset.range(self.config.data_loaders).interleave(
        lambda _: tf.data.Dataset.from_generator(generator, dtypes, shapes),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(self.config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

  def preprocess(self, obs):
    from tensorflow.keras import mixed_precision as prec
    dtype = prec.global_policy().compute_dtype
    obs = {k: tf.tensor(v) for k, v in obs.items()}
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if len(value.shape) > 3 and value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0 - 0.5
      else:
        value = value.astype(tf.float32)
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(tf.float32)
    obs['discount'] *= self.config.discount
    return obs


class WorldModel(tfutils.Module):

  def __init__(self, config, obs_space):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.config = config
    self.rssm = nets.EnsembleRSSM(**config.rssm)
    self.encoder = nets.MultiEncoder(shapes, **config.encoder)
    self.heads = {}
    self.heads['decoder'] = nets.MultiDecoder(shapes, **config.decoder)
    self.heads['reward'] = nets.MLP((), **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = nets.MLP((), **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = tfutils.Optimizer('model', **config.model_opt)

  def train(self, data, state=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def loss(self, data, state=None):
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        like = dist.log_prob(data[key]).astype(tf.float32)
        likes[key] = like
        losses[key] = -like.mean()
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    start['action'] = tf.zeros_like(policy(start['feat']).mode())
    def step(prev, _):
      action = policy(tf.stop_gradient(prev['feat'])).sample()
      state = self.rssm.img_step(prev, action)
      feat = self.rssm.get_feat(state)
      return {**state, 'action': action, 'feat': feat}
    seq = tfutils.scan(step, tf.range(horizon), start, self.config.imag_unroll)
    seq = {k: tf.concat([start[k][None], v], 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= self.config.discount
        disc = tf.concat([true_first[None], disc[1:]], 0)
    else:
      disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

  def video_pred(self, data):
    decoder = self.heads['decoder']
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    recon = decoder(self.rssm.get_feat(states))
    openl = decoder(self.rssm.get_feat(prior))
    videos = {}
    for k in recon.keys():
      if k not in decoder.cnn_shapes:
        continue
      truth = data[k][:6] + 0.5
      model = tf.concat([recon[k].mode()[:6][:, :5], openl[k].mode()], 1) + 0.5
      error = (model - truth + 1) / 2
      video = tf.concat([truth, model, error], 2)
      B, T, H, W, C = video.shape
      videos[k] = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    return videos


class ActorCritic(tfutils.Module):

  def __init__(self, config, act_space):
    self.config = config
    self.act_space = act_space
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if act_space.discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if act_space.discrete else 'dynamics'})
    self.actor = nets.MLP(act_space.shape[0], **self.config.actor)
    self.critic = nets.MLP((), **self.config.critic)
    if self.config.slow_target:
      self._target_critic = nets.MLP((), **self.config.critic)
      self._updates = tf.Variable(-1, dtype=tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = tfutils.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = tfutils.Optimizer('critic', **self.config.critic_opt)

  def train(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.config.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    with tf.GradientTape() as actor_tape:
      seq = world_model.imagine(self.actor, start, is_terminal, hor)
      seq['reward'] = reward_fn(seq)
      target, mets1 = self.target(seq)
      actor_loss, mets2 = self.actor_loss(seq, target)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets3 = self.critic_loss(seq, target)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = self.config.actor_grad_mix
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    objective += self.config.actor_ent * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    dist = self.critic(seq['feat'][:-1])
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    value = self._target_critic(seq['feat']).mode()
    # Skipping last time step because it is used for bootstrapping.
    target = tfutils.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if not self.config.slow_target:
      return
    assert self.critic.variables
    initialize = (self._updates == -1)
    if initialize or self._updates >= self.config.slow_target_update:
      self._updates.assign(0)
      mix = 1.0 if initialize else self.config.slow_target_fraction
      for s, d in zip(self.critic.variables, self._target_critic.variables):
        d.assign(mix * s + (1 - mix) * d)
    self._updates.assign_add(1)
