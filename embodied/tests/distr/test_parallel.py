import pathlib
import queue
import sys
import time
import traceback

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

import embodied
import pytest


class TestThread:

  def test_terminate(self):
    def fn():
      while True:
        time.sleep(0.01)
    worker = embodied.distr.Thread(fn, start=True)
    worker.terminate()
    worker.join()

  def test_exitcode(self):
    worker = embodied.distr.Thread(lambda: None)
    assert worker.exitcode is None
    worker.start()
    worker.join()
    assert worker.exitcode == 0

  def test_exception(self):
    def fn1234(q):
      q.put(42)
      raise KeyError('foo')
    q = queue.SimpleQueue()
    worker = embodied.distr.Thread(fn1234, q, start=True)
    q.get()
    time.sleep(0.01)
    assert not worker.running
    assert worker.exitcode == 1
    with pytest.raises(KeyError) as info:
      worker.check()
    assert repr(info.value) == "KeyError('foo')"
    tb = ''.join(traceback.format_exception(info.value))
    assert 'Traceback' in tb
    assert ' File ' in tb
    assert 'fn1234' in tb
    assert "KeyError: 'foo'" in tb


class TestProcess:

  def test_terminate(self):
    def fn():
      while True:
        time.sleep(0.01)
    worker = embodied.distr.Process(fn, start=True)
    worker.terminate()
    worker.join()

  def test_exitcode(self):
    worker = embodied.distr.Process(lambda: None)
    assert worker.exitcode is None
    worker.start()
    worker.join()
    assert worker.exitcode == 0

  def test_exception(self):
    def fn1234(q):
      q.put(42)
      raise KeyError('foo')
    q = embodied.distr.mp.SimpleQueue()
    worker = embodied.distr.Process(fn1234, q, start=True)
    q.get()
    time.sleep(0.2)
    assert not worker.running
    assert worker.exitcode == 1
    with pytest.raises(KeyError) as info:
      worker.check()
    assert repr(info.value) == "KeyError('foo')"
    tb = ''.join(traceback.format_exception(info.value))
    assert 'Traceback' in tb
    assert ' File ' in tb
    assert 'fn1234' in tb
    assert "KeyError: 'foo'" in tb
