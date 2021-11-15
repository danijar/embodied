import collections
import contextlib
import time

import numpy as np


class Timer:

  def __init__(self):
    self._durations = collections.defaultdict(list)
    self._start = time.time()

  def reset(self):
    self._durations.clear()
    self._start = time.time()

  @contextlib.contextmanager
  def scope(self, name):
    start = time.time()
    yield
    stop = time.time()
    self._durations[name].append(stop - start)

  def wrap(self, name, obj, methods):
    for method in methods:
      decorator = self.scope(f'{name}.{method}')
      setattr(obj, method, decorator(getattr(obj, method)))

  def stats(self, log=True):
    metrics = {}
    metrics['duration'] = time.time() - self._start
    for name, durs in self._durations.items():
      metrics[f'{name}_frac'] = np.sum(durs) / metrics['duration']
      metrics[f'{name}_sum'] = np.sum(durs)
      metrics[f'{name}_avg'] = np.mean(durs)
      metrics[f'{name}_min'] = np.min(durs)
      metrics[f'{name}_max'] = np.max(durs)
    if log:
      self._log(metrics)
    return metrics

  def _log(self, metrics):
    names = self._durations.keys()
    names = sorted(names, key=lambda k: -metrics[f'{k}_frac'])
    cols = ('frac', 'sum', 'avg', 'min', 'max')
    print('Timer:'.ljust(20), ' '.join(x.rjust(8) for x in cols))
    for name in names:
      values = [metrics[f'{name}_{col}'] for col in cols]
      print(f'{name.ljust(20)}', ' '.join((f'{x:8.4f}' for x in values)))
