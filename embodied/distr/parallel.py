import ctypes
import multiprocessing
import sys
import threading
import time
import traceback

import embodied

mp = multiprocessing.get_context('spawn')
PRINT_LOCK = mp.Lock()


class Thread:

  def __init__(self, fn, *args, name=None, start=False):
    self._fn = fn
    self._exitcode = None
    self._exception = None
    name = name or fn.__name__
    self._thread = threading.Thread(
        target=self._wrapper, args=args, name=name, daemon=True)
    start and self.start()

  def start(self):
    self._thread.start()

  @property
  def name(self):
    return self._thread.name

  @property
  def running(self):
    return self._thread.is_alive()

  @property
  def ident(self):
    return self._thread.ident

  @property
  def exitcode(self):
    return self._exitcode

  def check(self):
    if self._exception:
      e = self._exception
      self._exception = None
      raise e
    # TODO
    # assert self._exitcode in (None, 0), self._exitcode

  def join(self):
    self.check()
    self._thread.join()

  def terminate(self):
    thread = self._thread
    if not thread.is_alive():
      return
    if hasattr(thread, '_thread_id'):
      thread_id = thread._thread_id
    else:
      thread_id = [k for k, v in threading._active.items() if v is thread][0]
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
    if result > 1:
      ctypes.pythonapi.PyThreadState_SetAsyncExc(
          ctypes.c_long(thread_id), None)

  def _wrapper(self, *args):
    try:
      self._fn(*args)
      self._exitcode = 0
    except SystemExit:
      return
    except Exception as e:
      # TODO
      # _summarize(e, self.name, PRINT_LOCK)
      self._exception = e
      self._exitcode = 1


class Process:

  initializers = []

  def __init__(self, fn, *args, name=None, start=False):
    import cloudpickle
    name = name or fn.__name__
    fn = cloudpickle.dumps(fn)
    inits = cloudpickle.dumps(self.initializers)
    self._errqueue = mp.SimpleQueue()
    self._process = mp.Process(target=self._wrapper, name=name, args=(
        fn, name, args, PRINT_LOCK, self._errqueue, inits))
    start and self.start()

  def start(self):
    self._process.start()

  @property
  def name(self):
    return self._process.name

  @property
  def running(self):
    return self._process.is_alive()

  @property
  def pid(self):
    return self._process.pid

  @property
  def exitcode(self):
    return self._process.exitcode

  def check(self):
    if self._process.exitcode not in (None, 0):
      raise self._errqueue.get()

  def join(self):
    self.check()
    self._process.join()

  def terminate(self):
    self._process.terminate()

  @staticmethod
  def _wrapper(fn, name, args, lock, errqueue, inits):
    try:
      import cloudpickle
      for init in cloudpickle.loads(inits):
        init()
      fn = cloudpickle.loads(fn)
      fn(*args)
      sys.exit(0)
    except Exception as e:
      _summarize(e, name, lock)
      errqueue.put(e)
      sys.exit(1)


def run(workers):
  for worker in workers:
    if not worker.running:
      worker.start()
  while True:
    if all(x.exitcode == 0 for x in workers):
      print('All workers terminated successfully.')
      return
    for worker in workers:
      if worker.exitcode not in (None, 0):
        # Wait for everybody who wants to print their error messages.
        time.sleep(1)
        [x.terminate() for x in workers if x is not worker]
        msg = f'Terminated workers due to crash in {worker.name}.'
        raise RuntimeError(msg)
    time.sleep(0.1)


def _summarize(e, name, lock):
  summary = list(traceback.format_exception_only(e))[0].strip('\n')
  full = ''.join(traceback.format_exception(e)).strip('\n')
  msg = f"Exception in worker '{name}' ({summary}). "
  msg += 'Call check() to reraise in main process. '
  msg += f'Worker stack trace:\n{full}'
  with lock:
    embodied.print(msg, 'red')
  if sys.version_info.minor >= 11:
    e.add_note(f'\nWorker stack trace:\n\n{full}')
