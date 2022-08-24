import functools
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import pytest


class TestWorker:

  @pytest.mark.parametrize('strategy', ['blocking', 'thread', 'process'])
  def test_consecutive_calls(self, strategy):
    worker = embodied.Worker(lambda state, x: x ** 2, strategy)
    for x in range(5):
      promise = worker(x=x)
      assert promise() == x ** 2
    worker.close()

  @pytest.mark.parametrize('strategy', ['blocking', 'thread', 'process'])
  def test_without_close(self, strategy):
    worker = embodied.Worker(lambda state, x: x ** 2, strategy)
    for x in range(5):
      promise = worker(x=x)
      assert promise() == x ** 2

  @pytest.mark.parametrize('strategy', ['blocking', 'thread', 'process'])
  def test_stateful(self, strategy):
    def triangle_number(state, x):
      current = x + state.get('last', 0)
      state['last'] = current
      return current
    worker = embodied.Worker(triangle_number, strategy)
    promises = [worker(x) for x in range(6)]
    results = [promise() for promise in promises]
    assert results == [0, 1, 3, 6, 10, 15]
