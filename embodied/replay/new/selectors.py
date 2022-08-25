import collections

import numpy as np


class Fifo:

  def __init__(self):
    self.items = {}

  def insert(self, itemid, stepids):
    self.items[itemid] = None

  def remove(self, itemid):
    del self.items[itemid]

  def select(self):
    return next(iter(self.items))


class Uniform:

  def __init__(self, seed=0):
    self.items_dict = {}
    self.items_list = []
    self.rng = np.random.default_rng(seed)

  def insert(self, itemid, stepids):
    if itemid in self.items_dict:
      return
    self.items_dict[itemid] = len(self.items_list)
    self.items_list.append(itemid)

  def remove(self, itemid):
    index = self.items_dict.pop(itemid)
    last = self.items_list.pop()
    if index != len(self.items_list):
      self.items_list[index] = last
      self.items_dict[last] = index

  def select(self):
    return self.items_list[self.rng.integers(0, len(self.items_list))]


class Prioritized:

  def __init__(self, exponent=1.0, seed=0):
    # TODO: Implement priority tree and and hierarchical sampling to speed up
    # computation.
    self.exponent = exponent
    self.rng = np.random.default_rng(seed)
    self.items = {}
    self.stepitems = collections.defaultdict(set)
    self.stepprios = {}
    self.itemprios = {}

  def insert(self, itemid, stepids):
    if itemid in self.items:
      # This class checkpoints its items, so on resume we skip reinsertions
      # done by the replay buffer to avoid recomputing the priority tree too
      # often.
      return
    self.items[itemid] = stepids
    for stepid in stepids:
      self.stepitems[stepid].add(itemid)
      self.stepprios[stepid] = np.inf
    self._cache_itemprio(itemid)

  def remove(self, itemid):
    stepids = self.items.pop(itemid)
    del self.itemprios[itemid]
    for stepid in stepids:
      stepitems = self.stepitems[stepid]
      del stepitems[itemid]
      if not stepitems:
        del self.stepprios[stepid]
        del self.stepitems[stepid]

  def select(self):
    itemids, prios = zip(*self.itemprios.items())
    prios **= self.exponent
    probs = prios / prios.sum()
    return self.rng.choice(itemids, p=probs)

  def priorize(self, stepid, priority):
    self.stepprios[stepid] = priority
    for itemid in self.stepitems[stepid]:
      self._cache_itemprio(itemid)

  def save(self):
    return (self.items, self.stepprios)

  def load(self, data):
    self.items, self.stepprios = data
    self.stepitems.clear()
    self.itemprios.clear()
    for itemid, stepids in self.items.items():
      for stepid in stepids:
        self.stepitems[stepid].add(itemid)
    for itemid in self.items.keys():
      self._cache_itemprio(itemid)

  def _cache_itemprio(self, itemid):
    itemprio = sum([self.stepprios[stepid] for stepid in self.items[itemid]])
    self.itemprios[itemid] = itemprio
