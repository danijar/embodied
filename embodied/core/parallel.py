import atexit
import enum
import functools
import sys
import traceback


class Message(enum.Enum):

  CALLABLE = 1
  ACCESS = 2
  CALL = 3
  RESULT = 4
  CLOSE = 5
  ERROR = 6


class Parallel:

  def __init__(self, constructor, strategy='thread'):
    import cloudpickle
    if strategy == 'process':
      import multiprocessing
      mp = multiprocessing.get_context('spawn')
    elif strategy == 'thread':
      import multiprocessing.dummy as mp
    else:
      raise NotImplementedError(strategy)
    atexit.register(self.close)
    self._callables = {}
    self._outside, inside = mp.Pipe()
    pickled_ctor = cloudpickle.dumps(constructor)
    self._process = mp.Process(
        target=self._worker, args=(inside, pickled_ctor))
    self._process.start()
    assert self._receive() == 'ready'

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      if name not in self._callables:
        self._outside.send((Message.CALLABLE, name))
        self._callables[name] = self._receive()
      if self._callables[name]:
        return functools.partial(self._call, name)
      else:
        return self._access(name)()
    except AttributeError:
      raise ValueError(name)

  def __len__(self):
    return self._call('__len__')()

  def close(self):
    try:
      self._outside.send((Message.CLOSE, None))
      self._outside.close()
    except (AttributeError, IOError):
      pass  # The connection was already closed.
    try:
      self._process.join(5)
    except (AttributeError, AssertionError):
      pass

  def _access(self, name):
    self._outside.send((Message.ACCESS, name))
    return self._receive

  def _call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._outside.send((Message.CALL, payload))
    return self._receive

  def _receive(self):
    try:
      message, payload = self._outside.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to worker.')
    if message == Message.ERROR:
      raise Exception(payload)
    elif message == Message.RESULT:
      return payload
    else:
      raise KeyError(f'Unknown message type {message}.')

  @staticmethod
  def _worker(inside, pickled_ctor):
    try:
      import cloudpickle
      ctor = cloudpickle.loads(pickled_ctor)
      obj = ctor()
      inside.send((Message.RESULT, 'ready'))
      while True:
        try:
          if not inside.poll(0.1):
            continue
          message, payload = inside.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == Message.CALLABLE:
          name = payload
          result = callable(getattr(obj, name))
        elif message == Message.ACCESS:
          name = payload
          result = getattr(obj, name)
        elif message == Message.CALL:
          name, args, kwargs = payload
          result = getattr(obj, name)(*args, **kwargs)
        elif message == Message.CLOSE:
          break
        else:
          raise ValueError(f'Unknown message type {message}.')
        inside.send((Message.RESULT, result))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error inside worker: {stacktrace}.', flush=True)
      inside.send((Message.ERROR, stacktrace))
    finally:
      try:
        inside.close()
      except IOError:
        pass  # The connection was already closed.
