import multiprocessing as mp
import socket
import sys
import time
import traceback

import embodied
import psutil


_PRINT_LOCK = None
def get_print_lock():
  global _PRINT_LOCK
  if not _PRINT_LOCK:
    _PRINT_LOCK = mp.get_context().Lock()
  return _PRINT_LOCK


_PORTS = iter(range(5000, 8000))
def get_free_port():
  while True:
    port = next(_PORTS)
    if port_free(port):
      return port


def port_free(port):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    return s.connect_ex(('localhost', int(port)))


def run(workers, duration=None):
  try:

    for worker in workers:
      if not worker.started:
        try:
          worker.start()
        except Exception:
          print(f'Failed to start worker {worker.name}')
          raise

    start = time.time()
    while True:
      if duration and time.time() - start >= duration:
        print(f'Shutting down workers after {duration} seconds.')
        [x.kill() for x in workers]
        return
      if all(x.exitcode == 0 for x in workers):
        print('All workers terminated successfully.')
        return
      for worker in workers:
        if worker.exitcode not in (None, 0):
          time.sleep(0.1)  # Wait for workers to print their error messages.
          msg = f'Terminated workers due to crash in {worker.name}.'
          print(msg)
          worker.check()
          raise RuntimeError(msg)  # In case the check did not raise.
      time.sleep(0.1)

  finally:
    # Make sure all workers get stopped on shutdown. If some worker processes
    # survive program shutdown after an exception then ports may not be freeed
    # up. Even worse, clients of the new program execution could connect to
    # servers of the previous program execution that did not get cleaned up.
    [x.kill() for x in workers]


def kill_subprocs(parent=None):
  try:
    procs = list(psutil.Process(parent).children(recursive=True))
  except psutil.NoSuchProcess:
    return
  for proc in procs:
    try:
      proc.terminate()
    except psutil.NoSuchProcess:
      pass
  for proc in procs:
    try:
      proc.wait(timeout=0.1)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
      pass
  for proc in procs:
    try:
      proc.kill()
    except psutil.NoSuchProcess:
      pass
  for proc in procs:
    try:
      proc.wait(timeout=0.1)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
      pass
  for proc in procs:
    assert not proc_alive(proc.pid)


def kill_proc(pid):
  try:
    proc = psutil.Process(pid)
    proc.terminate()
    try:
      proc.wait(timeout=0.1)
    except psutil.TimeoutExpired:
      proc.kill()
      proc.wait(timeout=0.1)
  except psutil.NoSuchProcess:
    pass


def proc_alive(pid):
  try:
    return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
  except psutil.NoSuchProcess:
    return False


def warn_remote_error(e, name, lock=get_print_lock):
  lock = lock() if callable(lock) else lock
  typ, tb = type(e), e.__traceback__
  summary = list(traceback.format_exception_only(typ, e))[0].strip('\n')
  full = ''.join(traceback.format_exception(typ, e, tb)).strip('\n')
  msg = f"Exception in worker '{name}' ({summary}). "
  msg += 'Call check() to reraise in main process. '
  msg += f'Worker stack trace:\n{full}'
  with lock:
    embodied.print(msg, 'red')
  if sys.version_info.minor >= 11:
    e.add_note(f'\nWorker stack trace:\n\n{full}')


class Context:

  def __init__(self, predicate):
    self._predicate = predicate

  @property
  def running(self):
    return self._predicate()

  def __bool__(self):
    raise TypeError('Cannot convert Context to boolean.')
