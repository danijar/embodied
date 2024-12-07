import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn

f32 = jnp.float32
sg = jax.lax.stop_gradient


class ImpalaEncoder(nj.Module):

  depth: int = 32
  mults: tuple = (1, 2, 2)
  outmult: int = 16
  blocks: int = 2
  norm: str = 'none'
  act: str = 'relu'
  symlog: bool = True
  layers: int = 5
  units: int = 512

  def __init__(self, spaces, **kw):
    assert all(len(s.shape) <= 3 for s in spaces.values()), spaces
    self.vecspaces = {k: v for k, v in spaces.items() if len(v.shape) <= 2}
    self.imgspaces = {k: v for k, v in spaces.items() if len(v.shape) == 3}
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  def __call__(self, data, bdims=2):
    bshape = next(iter(data.values())).shape[:bdims]
    outs = []

    if self.vecspaces:
      mkw = dict(norm=self.norm, act=self.act, **self.kw)
      ekw = dict(units=self.units, squish=self.symlog and nn.symlog, **self.kw)
      x = {k: data[k] for k in self.vecspaces}
      x = self.sub('emb', nn.DictEmbed, self.vecspaces, **ekw)(x, bshape)
      x = x.reshape((-1, x.shape[-1]))
      x = self.sub('mlp', nn.MLP, self.layers - 1, self.units, **mkw)(x)
      outs.append(x)

    if self.imgspaces:
      keys = sorted(self.imgspaces.keys())
      x = jnp.concatenate([data[k] for k in keys], -1)
      assert x.dtype == jnp.uint8
      x = nn.COMPUTE_DTYPE(x) * 255 - 0.5
      x = x.reshape((-1, *x.shape[-3:]))
      for s, depth in enumerate(self.depths):
        x = self.sub(f's{s}in', nn.Conv2D, depth, 3, **self.kw)(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same')
        for b in range(self.blocks):
          skip = x
          x = self.sub(f's{s}b{b}n1', nn.Norm, self.norm)(x)
          x = nn.act(self.act)(x)
          x = self.sub(f's{s}b{b}c1', nn.Conv2D, depth, 3, **self.kw)(x)
          x = self.sub(f's{s}b{b}n2', nn.Norm, self.norm)(x)
          x = nn.act(self.act)(x)
          x = self.sub(f's{s}b{b}c2', nn.Conv2D, depth, 3, **self.kw)(x)
          x += skip
      x = x.reshape((x.shape[0], -1))
      x = self.sub('outn1', nn.Norm, self.norm)(x)
      x = nn.act(self.act)(x)
      x = self.sub('outl', nn.Linear, self.outmult * self.depth, **self.kw)(x)
      x = self.sub('outn2', nn.Norm, self.norm)(x)
      x = nn.act(self.act)(x)
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    x = x.reshape((*bshape, -1))
    return x
