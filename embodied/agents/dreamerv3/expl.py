import jax
import jax.numpy as jnp
# import numpy as np
# from tensorflow_probability import distributions as tfd
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import nets
from . import jaxutils
from . import ninjax as nj


class Disag(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config.update({'disag_head.inputs': ['tensor']})
    self.opt = jaxutils.Optimizer(name='disag_opt', **config.expl_opt)
    self.inputs = nets.Input(config.disag_head.inputs, dims='deter')
    self.target = nets.Input(self.config.disag_target, dims='deter')
    self.nets = [
        nets.MLP(shape=None, **self.config.disag_head, name=f'disag{i}')
        for i in range(self.config.disag_models)]

  def __call__(self, traj):
    inp = self.inputs(traj)
    preds = jnp.array([net(inp).mode() for net in self.nets])
    return preds.std(0).mean(-1)[1:]

  def train(self, data):
    return self.opt(self.nets, self.loss, data)

  def loss(self, data):
    inp = sg(self.inputs(data)[:, :-1])
    tar = sg(self.target(data)[:, 1:])
    losses = []
    for net in self.nets:
      net._shape = tar.shape[2:]
      losses.append(-net(inp).log_prob(tar).mean())
    return jnp.array(losses).sum()


# class LatentVAE(nj.Module):
#
#   def __init__(self, wm, act_space, config):
#     self.config = config
#     self.enc = nets.MLP(**self.config.expl_enc)
#     self.dec = nets.MLP(self.config.rssm.deter, **self.config.expl_dec)
#     shape = self.config.expl_enc.shape
#     if self.config.expl_enc.dist == 'onehot':
#       self.prior = jaxutils.OneHotDist(tf.zeros(shape))
#       self.prior = tfd.Independent(self.prior, len(shape) - 1)
#     else:
#       self.prior = tfd.Normal(tf.zeros(shape), tf.ones(shape))
#       self.prior = tfd.Independent(self.prior, len(shape))
#     self.kl = jaxutils.AutoAdapt(**self.config.expl_kl)
#     self.opt = jaxutils.Optimizer('disag', **self.config.expl_opt)
#     self.flatten = lambda x: x.reshape(
#         x.shape[:-len(shape)] + [np.prod(x.shape[len(shape):])])
#
#   def __call__(self, traj):
#     dist = self.enc(traj)
#     target = tf.stop_gradient(traj['deter'].astype(tf.float32))
#     ll = self.dec(self.flatten(dist.sample())).log_prob(target)
#     if self.config.expl_vae_elbo:
#       kl = tfd.kl_divergence(dist, self.prior)
#       reward = kl - ll / self.kl.scale()
#     else:
#       reward = -ll
#     return reward[1:]
#
#   def train(self, data):
#     metrics = {}
#     target = tf.stop_gradient(data['deter'].astype(tf.float32))
#     with tf.GradientTape() as tape:
#       dist = self.enc(data)
#       kl = tfd.kl_divergence(dist, self.prior)
#       kl, mets = self.kl(kl)
#       metrics.update({f'kl_{k}': v for k, v in mets.items()})
#       ll = self.dec(self.flatten(dist.sample())).log_prob(target)
#       assert kl.shape == ll.shape
#       loss = (kl - ll).mean()
#     metrics['vae_kl'] = kl.mean()
#     metrics['vae_ll'] = ll.mean()
#     metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
#     return metrics
#
#
# class CtrlDisag(nj.Module):
#
#   def __init__(self, wm, act_space, config):
#     self.disag = Disag(
#         wm, act_space, config.update({'disag_target': ['ctrl']}))
#     self.embed = nets.MLP((config.ctrl_size,), **config.ctrl_embed)
#     self.head = nets.MLP(act_space.shape, **config.ctrl_head)
#     self.opt = jaxutils.Optimizer('ctrl', **config.ctrl_opt)
#
#   def __call__(self, traj):
#     return self.disag(traj)
#
#   def train(self, data):
#     metrics = {}
#     with tf.GradientTape() as tape:
#       ctrl = self.embed(data).mode()
#       dist = self.head({'current': ctrl[:, :-1], 'next': ctrl[:, 1:]})
#       loss = -dist.log_prob(data['action'][:, 1:]).mean()
#     self.opt(tape, loss, [self.embed, self.head])
#     metrics.update(self.disag.train({**data, 'ctrl': ctrl}))
#     return metrics
#
#
# class PBE(nj.Module):
#
#   def __init__(self, wm, act_space, config):
#     self.config = config
#     self.inputs = nets.Input(config.pbe_inputs, dims='deter')
#
#   def __call__(self, traj):
#     feat = self.inputs(traj)
#     flat = feat.reshape([-1, feat.shape[-1]])
#     dists = tf.norm(
#         flat[:, None, :].reshape((len(flat), 1, -1)) -
#         flat[None, :, :].reshape((1, len(flat), -1)), axis=-1)
#     rew = -tf.math.top_k(-dists, self.config.pbe_knn, sorted=True)[0].mean(-1)
#     return rew.reshape(feat.shape[:-1]).astype(tf.float32)
#
#   def train(self, data):
#     return {}
