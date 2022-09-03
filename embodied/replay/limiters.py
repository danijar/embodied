class Unlimited:

  def want_insert(self):
    return True

  def want_remove(self):
    return True

  def want_sample(self):
    return True


class MinSize:

  def __init__(self, minimum):
    assert 1 <= minimum, minimum
    self.minimum = minimum
    self.size = 0

  def want_insert(self):
    self.size += 1
    return True

  def want_remove(self):
    if self.size <= self.minimum:
      return False
    self.size -= 1
    return True

  def want_sample(self):
    return self.size >= self.minimum


class SamplesPerInsert:

  def __init__(self, samples_per_insert, tolerance):
    self.samples_per_insert = samples_per_insert
    self.tolerance = tolerance
    self.available_samples = 0

  def want_insert(self):
    if self.available_samples >= self.tolerance:
      return False
    self.available_samples += self.samples_per_insert
    return True

  def want_remove(self):
    return True

  def want_sample(self):
    if self.available_samples <= -self.tolerance:
      return False
    self.available_samples -= 1
    return True
