import collections
import warnings
from functools import partial as bind

import numpy as np


class Metrics:

  def __init__(self):
    self.scalars = collections.defaultdict(list)
    self.aggs = {}
    self.lasts = {}
    self.fns = {
        'mean': bind(np.nanmean, dtype=np.float64),
        'sum': bind(np.nansum, dtype=np.float64),
        'min': np.nanmin,
        'max': np.nanmax,
    }

  def scalar(self, key, value, agg='mean'):
    if isinstance(agg, str):
      assert agg in self.fns.keys()
    else:
      assert all(x in self.fns.keys() for x in agg)
    self.scalars[key].append(value)
    self.aggs[key] = agg

  def image(self, key, value):
    self.lasts[key] = value

  def video(self, key, value):
    self.lasts[key] = value

  def add(self, mapping, prefix=None):
    for key, value in mapping.items():
      key = prefix + '/' + key if prefix else key
      if hasattr(value, 'shape') and len(value.shape) > 0:
        self.lasts[key] = value
      else:
        self.scalar(key, value)

  def result(self, reset=True):
    result = {}
    result.update(self.lasts)
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      for key, values in self.scalars.items():
        agg = self.aggs[key]
        if isinstance(agg, str):
          result[key] = self.fns[agg](values)
        else:
          for x in agg:
            result[f'{key}_{x}'] = self.fns[x](values)
    reset and self.reset()
    return result

  def reset(self):
    self.scalars.clear()
    self.lasts.clear()
