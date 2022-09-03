import embodied


class Dict:

  def __init__(self, directory=None):
    self.data = {}

  def keys(self):
    return self.data.keys()

  def __len__(self):
    return len(self.data)

  def __setitem__(self, key, steps):
    self.data[key] = steps

  def __getitem__(self, key):
    seq = self.data[key]
    seq = {k: [step[k] for step in seq] for k in seq[0]}
    seq = {k: embodied.convert(v) for k, v in seq.items()}
    return seq

  def __delitem__(self, key):
    del self.data[key]
