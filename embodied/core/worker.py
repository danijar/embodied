import atexit
import concurrent.futures
import enum
import functools
import os
import sys
import time
import traceback


class Worker:

  def __init__(self, fn, strategy='thread'):
    self.impl = {
        'blocking': BlockingWorker,
        'thread': ThreadWorker,
        'process': ProcessWorker,
        'daemon': functools.partial(ProcessWorker, daemon=True),
    }[strategy](fn)

  def __call__(self, *args, **kwargs):
    return self.impl(*args, **kwargs)

  def close(self):
    self.impl.close()


class BlockingWorker:

  def __init__(self, fn):
    self.fn = fn
    self.state = {}

  def __call__(self, *args, **kwargs):
    return lambda: self.fn(self.state, *args, **kwargs)

  def close(self):
    pass


class ThreadWorker:

  def __init__(self, fn):
    self.fn = fn
    self.state = {}
    self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

  def __call__(self, *args, **kwargs):
    return self.executor.submit(self.fn, self.state, *args, **kwargs).result

  def close(self):
    self.executor.shutdown(wait=False, cancel_futures=True)


class ProcessWorker:

  initializers = []

  def __init__(self, fn, daemon=False):
    import multiprocessing
    import cloudpickle
    self._context = multiprocessing.get_context('spawn')
    self._pipe, pipe = self._context.Pipe()
    fn = cloudpickle.dumps(fn)
    initializers = cloudpickle.dumps(self.initializers)
    self._process = self._context.Process(
        target=self._loop,
        args=(pipe, fn, initializers),
        daemon=daemon)
    self._process.start()
    assert self._receive() == 'ready'
    atexit.register(self.close)

  def __call__(self, *args, **kwargs):
    self._pipe.send((Message.RUN, (args, kwargs)))
    return self._receive

  def close(self):
    try:
      self._pipe.send((Message.STOP, None))
      self._pipe.close()
    except (AttributeError, IOError):
      pass  # The connection was already closed.
    try:
      self._process.join(0.1)
      if self._process.exitcode is None:
        try:
          os.kill(self._process.pid, 9)
          time.sleep(0.1)
        except Exception as e:
          pass
    except (AttributeError, AssertionError):
      pass

  def _receive(self):
    try:
      message, payload = self._pipe.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to worker.')
    if message == Message.ERROR:
      raise Exception(payload)
    assert message == Message.RESULT, message
    return payload

  @staticmethod
  def _loop(pipe, function, initializers):
    try:
      import cloudpickle
      initializers = cloudpickle.loads(initializers)
      function = cloudpickle.loads(function)
      state = {}
      [fn() for fn in initializers]
      pipe.send((Message.RESULT, 'ready'))
      while True:
        if not pipe.poll(0.1):
          continue  # Wake up for keyboard interrupts.
        message, payload = pipe.recv()
        if message == Message.STOP:
          return
        elif message == Message.RUN:
          args, kwargs = payload
          result = function(state, *args, **kwargs)
          pipe.send((Message.RESULT, result))
        else:
          raise NameError(f'Invalid message: {message}')
    except (EOFError, KeyboardInterrupt):
      return
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error inside process worker: {stacktrace}.', flush=True)
      pipe.send((Message.ERROR, stacktrace))
      return
    finally:
      try:
        pipe.close()
      except Exception:
        pass


class Message(enum.Enum):

  RUN = 1
  RESULT = 2
  STOP = 3
  ERROR = 4
