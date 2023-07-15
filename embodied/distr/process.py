import sys

from . import utils


class Process:

  initializers = []

  def __init__(self, fn, *args, name=None, start=False, pass_running=False):
    import cloudpickle
    name = name or fn.__name__
    fn = cloudpickle.dumps(fn)
    inits = cloudpickle.dumps(self.initializers)
    self.errqueue = utils.mp.SimpleQueue()
    self.process = utils.mp.Process(target=self._wrapper, name=name, args=(
        fn, name, args, utils.PRINT_LOCK, self.errqueue, inits))
    self.started = False
    start and self.start()

  @property
  def name(self):
    return self.process.name

  @property
  def pid(self):
    return self.process.pid

  @property
  def alive(self):
    return self.process.is_alive()

  @property
  def exitcode(self):
    return self.process.exitcode

  def start(self):
    assert not self.started
    self.started = True
    self.process.start()

  def check(self):
    assert self.started
    if self.process.exitcode not in (None, 0):
      raise self.errqueue.get()

  def join(self, timeout=None):
    self.check()
    self.process.join(timeout)

  def terminate(self):
    if not self.alive:
      return
    self.process.terminate()

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
      print('exception in process')  # TODO
      utils.warn_remote_error(e, name, lock)
      errqueue.put(e)
      sys.exit(1)


class StoppableProcess(Process):

  def __init__(self, fn, *args, name=None, start=False):
    self.runflag = utils.mp.Event()
    def fn2(runflag, *args):
      assert runflag is not None
      is_running = runflag.is_set
      fn(is_running, *args)
    super().__init__(fn2, self.runflag, *args, name=name, start=start)

  def start(self):
    self.runflag.set()
    super().start()

  def stop(self, wait=True):
    self.runflag.clear()
    if wait is True:
      self.join()
    elif wait:
      self.join(wait)
