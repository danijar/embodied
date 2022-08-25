import time


def wait_to_insert(limiters):
  counter = 0
  while not all(l.can_insert() for l in limiters):
    if counter % 100 == 0:
      print('Replay insertion is waiting for more sampling.')
    time.sleep(0.1)
    counter += 1


def wait_to_sample(limiters):
  counter = 0
  while not all(l.can_sample() for l in limiters):
    if counter % 100 == 0:
      print('Replay sampling is waiting for more insertion.')
    time.sleep(0.1)
    counter += 1


class MinSize:

  def __init__(self, minimum):
    assert 1 <= minimum, minimum
    self.minimum = minimum
    self.size = 0

  def insert(self, itemid, stepids):
    self.size += 1

  def remove(self, itemid):
    self.size -= 1

  def retrieve(self, itemid):
    pass

  def can_insert(self):
    return True

  def can_sample(self):
    return self.size >= self.minimum


class SamplesPerInsert:

  def __init__(self, samples_per_insert, tolerance):
    self.samples_per_insert = samples_per_insert
    self.tolerance = tolerance
    self.available_samples = 0

  def insert(self, itemid, stepids):
    self.available_samples += self.samples_per_insert

  def remove(self, itemid):
    pass

  def retrieve(self, itemid):
    self.available_samples -= 1

  def can_insert(self):
    return self.available_samples <= self.tolerance

  def can_sample(self):
    return self.available_samples > -self.tolerance
