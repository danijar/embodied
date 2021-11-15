import pathlib
import pickle


class Checkpoint:

  def __init__(self, filename, log=True):
    self._filename = pathlib.Path(filename).expanduser()
    self._log = log
    self._values = {}

  def __setattr__(self, name, value):
    if name in ('exists', 'save', 'load'):
      return super().__setattr__(name, value)
    if name.startswith('_'):
      return super().__setattr__(name, value)
    has_load = hasattr(value, 'load') and callable(value.load)
    has_save = hasattr(value, 'save') and callable(value.save)
    if not (has_load and has_save):
      message = f"Checkpoint entry '{name}' must implement save() and load()."
      raise ValueError(message)
    self._values[name] = value

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      return getattr(self._values, name)
    except AttributeError:
      raise ValueError(name)

  def exists(self):
    exists = self._filename.exists()
    self._log and exists and print('Existing checkpoint found.')
    self._log and not exists and print('Existing checkpoint not found.')
    return exists

  def save(self):
    self._log and print('Save checkpoint.')
    data = {k: v.save() for k, v in self._values.items()}
    with self._filename.open('wb') as f:
      pickle.dump(data, f)

  def load(self):
    self._log and print('Load checkpoint.')
    with self._filename.open('rb') as f:
      data = pickle.load(f)
    for key, value in data.items():
      self._values[key].load(value)

  def load_or_save(self):
    if self.exists():
      self.load()
    else:
      self.save()
