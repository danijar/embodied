import pathlib
import queue
import sys
import threading
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

import embodied
import numpy as np
import pytest

SERVERS = [
    embodied.distr.Server,
    embodied.distr.Server2,
]

PORTS = iter(range(5555, 6000))

def addresses(tcp=True, ipc=True):
  results = []
  tcp and results.append('tcp://localhost:{port}')
  ipc and results.append('ipc:///tmp/test-{port}')
  return results


class TestServer:

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_single_client(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    def function(data):
      assert data == {'foo': np.array(1)}
      return {'foo': 2 * data['foo']}
    server = Server(addr)
    server.bind('function', function)
    with server:
      client = embodied.distr.Client(addr, pings=0, maxage=1)
      client.connect(retry=False, timeout=1)
      future = client.function({'foo': np.array(1)})
      result = future.result()
      assert result['foo'] == 2

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_multiple_clients(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    server = Server(addr)
    server.bind('function', lambda data: data)
    with server:
      clients = []
      for i in range(10):
        client = embodied.distr.Client(addr, i, pings=0, maxage=1)
        client.connect()
        clients.append(client)
      futures = [
          client.function({'foo': i}) for i, client in enumerate(clients)]
      results = [future.result()['foo'] for future in futures]
      assert results == list(range(10))

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_multiple_methods(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    server = Server(addr)
    server.bind('add', lambda data: {'z': data['x'] + data['y']})
    server.bind('sub', lambda data: {'z': data['x'] - data['y']})
    with server:
      client = embodied.distr.Client(addr, pings=0, maxage=0.1)
      client.connect(retry=False, timeout=1)
      assert client.add({'x': 42, 'y': 13}).result()['z'] == 55
      assert client.sub({'x': 42, 'y': 13}).result()['z'] == 29

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_connect_before_server(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    server = Server(addr)
    server.bind('function', lambda data: {'foo': 2 * data['foo']})
    barrier = threading.Barrier(2)
    results = queue.SimpleQueue()
    def client():
      client = embodied.distr.Client(addr, pings=0, maxage=1)
      barrier.wait()
      client.connect(retry=False, timeout=1)
      future = client.function({'foo': np.array(1)})
      result = future.result()
      results.put(result)
    thread = embodied.distr.Thread(client, start=True)
    barrier.wait()
    time.sleep(0.2)
    with server:
      assert results.get()['foo'] == 2
    thread.join()

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_future_order(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    server = Server(addr)
    server.bind('function', lambda data: data)
    with server:
      client = embodied.distr.Client(addr, 0, pings=0, maxage=1)
      client.connect(retry=False, timeout=1)
      future1 = client.function({'foo': 1})
      future2 = client.function({'foo': 2})
      future3 = client.function({'foo': 3})
      assert future2.result()['foo'] == 2
      assert future1.result()['foo'] == 1
      assert future3.result()['foo'] == 3

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_future_cleanup(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    server = Server(addr)
    server.bind('function', lambda data: data)
    with server:
      client = embodied.distr.Client(addr, 0, pings=0, maxage=1)
      client.connect(retry=False, timeout=1)
      client.function({'foo': 1})
      client.function({'foo': 2})
      future3 = client.function({'foo': np.array(3)})
      assert future3.result()['foo'] == 3
      del future3
      assert not list(client.futures.keys())

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_ping_alive(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    def slow(data):
      time.sleep(0.1)
      return data
    server = Server(addr)
    server.bind('function', slow)
    with server:
      client = embodied.distr.Client(addr, pings=0.01, maxage=0.05)
      client.connect()
      assert client.function({'foo': 0}).result() == {'foo': 0}

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_ping_dead(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    def slow(data):
      time.sleep(0.2)
      return data
    server = Server(addr)
    server.bind('function', slow)
    with server:
      client = embodied.distr.Client(addr, pings=0.1, maxage=0.01)
      client.connect()
      with pytest.raises(embodied.distr.NotAliveError):
        client.function({'foo': 0}).result()

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_remote_error(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    def error(data):
      raise RuntimeError('foo')
    server = Server(addr)
    server.bind('function', error)
    with server:
      client = embodied.distr.Client(addr)
      client.connect()
      future = client.function({'bar': 0})
      with pytest.raises(embodied.distr.RemoteError) as info1:
        future.result()
      with pytest.raises(RuntimeError) as info2:
        server.check()
    assert repr(info1.value) == '''RemoteError("RuntimeError('foo')")'''
    assert repr(info2.value) == "RuntimeError('foo')"

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_logfn_ordered(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    rng = np.random.default_rng(0)
    completed = []
    logged = []
    def sometimes_wait(data):
      if rng.uniform() < 0.5:
        time.sleep(0.1)
      completed.append(data['i'])
      return data, data
    def logfn(data):
      logged.append(data['i'])
    server = Server(addr, workers=2)
    server.bind('function', sometimes_wait, logfn)
    with server:
      client = embodied.distr.Client(addr, pings=0, maxage=1)
      client.connect()
      futures = [client.function({'i': i}) for i in range(10)]
      results = [future.result()['i'] for future in futures]
    assert results == list(range(10))
    assert logged == list(range(10))
    assert completed != list(range(10))

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_connect_retry(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    results = []
    def client():
      try:
        client = embodied.distr.Client(addr)
        client.connect(retry=True, timeout=0.01)
        future = client.function({'foo': np.array(1)})
        results.append(future.result())
      except Exception as e:
        results.append(e)
    threading.Thread(target=client).start()
    time.sleep(0.2)
    server = Server(addr)
    server.bind('function', lambda data: data)
    with server:
      while not results:
        time.sleep(0.001)
    assert results == [{'foo': 1}]

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_shared_pool(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    def slow_function(data):
      time.sleep(0.1)
      return data
    def fast_function(data):
      time.sleep(0.01)
      return data
    server = Server(addr, workers=1)
    server.bind('slow_function', slow_function)
    server.bind('fast_function', fast_function)
    with server:
      client = embodied.distr.Client(addr)
      client.connect()
      slow_future = client.slow_function({'foo': 0})
      fast_future = client.fast_function({'foo': 0})
      assert not slow_future.done()
      fast_future.result()
      assert slow_future.done()

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  def test_separate_pools(self, Server, addr):
    addr = addr.format(port=next(PORTS))
    def slow_function(data):
      time.sleep(0.1)
      return data
    def fast_function(data):
      time.sleep(0.01)
      return data
    server = Server(addr)
    server.bind('slow_function', slow_function, workers=1)
    server.bind('fast_function', fast_function, workers=1)
    with server:
      client = embodied.distr.Client(addr)
      client.connect()
      slow_future = client.slow_function({'foo': 0})
      fast_future = client.fast_function({'foo': 0})
      fast_future.result()
      assert not slow_future.done()

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  @pytest.mark.parametrize('batch', (1, 2, 4))
  def test_batching_single(self, Server, addr, batch):
    addr = addr.format(port=next(PORTS))
    calls = [0]
    def function(data):
      assert set(data.keys()) == {'foo'}
      assert data['foo'].shape == (batch, 1)
      calls[0] += 1
      return data
    server = Server(addr)
    server.bind('function', function, batch=batch)
    with server:
      client = embodied.distr.Client(addr, pings=0, maxage=1)
      client.connect(retry=False, timeout=1)
      futures = [client.function({'foo': [i]}) for i in range(batch)]
      results = [future.result()['foo'][0] for future in futures]
      assert calls[0] == 1
      assert results == list(range(batch))

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('addr', addresses())
  @pytest.mark.parametrize('batch', (1, 2, 4))
  def test_batching_multiple(self, Server, addr, batch):
    addr = addr.format(port=next(PORTS))
    def function(data):
      return data
    server = Server(addr)
    server.bind('function', function, batch=batch)
    with server:
      clients = []
      for _ in range(3):
        client = embodied.distr.Client(addr, pings=0, maxage=1)
        client.connect(retry=False, timeout=1)
        clients.append(client)
      futures = ([], [], [])
      refs = ([], [], [])
      for n in range(batch):
        for i, client in enumerate(clients):
          futures[i].append(client.function({'foo': [i * n]}))
          refs[i].append(i * n)
      assert refs[0] == [x.result()['foo'][0] for x in futures[0]]
      assert refs[1] == [x.result()['foo'][0] for x in futures[1]]
      assert refs[2] == [x.result()['foo'][0] for x in futures[2]]

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('inner_addr', addresses())
  @pytest.mark.parametrize('outer_addr', addresses())
  def test_proxy(self, Server, inner_addr, outer_addr):
    inner_addr = inner_addr.format(port=next(PORTS))
    outer_addr = outer_addr.format(port=next(PORTS))
    proxy_client = embodied.distr.Client(inner_addr)
    proxy_server = Server(outer_addr)
    proxy_server.bind('function', lambda x: proxy_client.function(x).result())
    server = Server(inner_addr)
    server.bind('function', lambda data: {'foo': 2 * data['foo']})
    with server:
      proxy_client.connect(retry=False, timeout=1)
      with proxy_server:
        client = embodied.distr.Client(outer_addr, pings=0, maxage=1)
        client.connect(retry=False, timeout=1)
        futures = [client.function({'foo': 13}) for _ in range(10)]
        results = [future.result()['foo'] for future in futures]
        assert all(result == 26 for result in results)

  @pytest.mark.parametrize('Server', SERVERS)
  @pytest.mark.parametrize('inner_addr', addresses())
  @pytest.mark.parametrize('outer_addr', addresses())
  def test_proxy_batched(self, Server, inner_addr, outer_addr):
    inner_addr = inner_addr.format(port=next(PORTS))
    outer_addr = outer_addr.format(port=next(PORTS))
    proxy_client = embodied.distr.Client(inner_addr)
    proxy_server = Server(outer_addr, workers=2)
    proxy_server.bind('function', lambda x: proxy_client.function(x).result())
    server = Server(inner_addr)
    server.bind('function', lambda data: {'foo': 2 * data['foo']}, batch=2)
    with server:
      proxy_client.connect(retry=False, timeout=1)
      with proxy_server:
        client = embodied.distr.Client(outer_addr, pings=0, maxage=1)
        client.connect(retry=False, timeout=1)
        futures = [client.function({'foo': 13}) for _ in range(10)]
        results = [future.result()['foo'] for future in futures]
        print(results)
        assert all(result == 26 for result in results)


def repro1():
  Server = embodied.distr.Server2
  # Server = embodied.distr.Server
  inner_addr = 'tcp://localhost:5555'
  outer_addr = 'tcp://localhost:5556'
  inner_addr = inner_addr.format(port=next(PORTS))
  outer_addr = outer_addr.format(port=next(PORTS))
  proxy_client = embodied.distr.Client(inner_addr)
  proxy_server = Server(outer_addr, workers=2)
  proxy_server.bind('function', lambda x: proxy_client.function(x).result())
  server = Server(inner_addr)
  server.bind('function', lambda data: {'foo': 2 * data['foo']}, batch=2)
  with server:
    proxy_client.connect(retry=False, timeout=1)
    with proxy_server:
      client = embodied.distr.Client(outer_addr, pings=0, maxage=1)
      client.connect(retry=False, timeout=1)
      futures = [client.function({'foo': 13}) for _ in range(10)]
      results = [future.result()['foo'] for future in futures]
      print(results)
      assert all(result == 26 for result in results)


def repro2():
  # Server = embodied.distr.Server2
  Server = embodied.distr.Server
  inner_addr = 'tcp://localhost:5555'
  outer_addr = 'tcp://localhost:5556'
  inner_addr = inner_addr.format(port=next(PORTS))
  outer_addr = outer_addr.format(port=next(PORTS))
  proxy_client = embodied.distr.Client(inner_addr)
  proxy_server = Server(outer_addr, workers=3)
  proxy_server.bind('function', lambda x: proxy_client.function(x).result())
  server = Server(inner_addr)
  server.bind('function', lambda data: {'foo': 2 * data['foo']}, batch=2)
  with server:
    proxy_client.connect(retry=False, timeout=1)
    with proxy_server:
      client = embodied.distr.Client(outer_addr, pings=0, maxage=1)
      client.connect(retry=False, timeout=1)
      # futures = [client.function({'foo': 13}) for _ in range(10)]
      futures = [client.function({'foo': 13}) for _ in range(1)]
      results = [future.result()['foo'] for future in futures]
      print(results)
      assert all(result == 26 for result in results)

def repro3():
  # Server = embodied.distr.Server2
  Server = embodied.distr.Server
  # inner_addr = 'tcp://localhost:5555'
  # outer_addr = 'tcp://localhost:5556'
  inner_addr = 'ipc:///tmp/test-inner'
  outer_addr = 'ipc:///tmp/test-outer'
  inner_addr = inner_addr.format(port=next(PORTS))
  outer_addr = outer_addr.format(port=next(PORTS))

  def proxy():
    proxy_client = embodied.distr.Client(
        inner_addr, pings=1, maxage=2, name='ProxyInner')
    proxy_client.connect(retry=False, timeout=1)
    proxy_server = Server(outer_addr, workers=3, name='ProxyOuter')
    # proxy_server.bind('function', lambda x: proxy_client.function(x).result())
    def fwd(x):
      print('PROXY RECEIVED CALL')
      return proxy_client.function(x).result()
    proxy_server.bind('function', fwd)
    proxy_server.run()

  thread = embodied.distr.Thread(proxy, start=True)

  server = Server(inner_addr)
  def work(data):
    print('SERVER RECEIVED CALL')
    return {'foo': 2 * data['foo']}
  server.bind('function', work, batch=2)

  with server:
    client = embodied.distr.Client(outer_addr, pings=1, maxage=3)
    client.connect(retry=False, timeout=1)
    # futures = [client.function({'foo': 13}) for _ in range(10)]
    futures = [client.function({'foo': 13}) for _ in range(6)]
    results = [future.result()['foo'] for future in futures]
    print(results)
    assert all(result == 26 for result in results)

  thread.terminate()


def repro5():
  # Server = embodied.distr.Server2  # reproduces with either
  Server = embodied.distr.Server  # reproduces with either
  inner_addr = 'tcp://localhost:5555'
  outer_addr = 'tcp://localhost:5556'
  inner_addr = inner_addr.format(port=next(PORTS))
  outer_addr = outer_addr.format(port=next(PORTS))
  # inner_addr = 'ipc:///tmp/test-inner'
  # outer_addr = 'ipc:///tmp/test-outer'

  def proxy(outer_addr, inner_addr):
    proxy_client = embodied.distr.Client(
        inner_addr, pings=1, maxage=2, name='ProxyInner')
    proxy_client.connect(retry=False, timeout=1)
    proxy_server = Server(outer_addr, workers=3, name='ProxyOuter')
    # proxy_server.bind('function', lambda x: proxy_client.function(x).result())
    def fwd(x):
      print('PROXY RECEIVED CALL')
      return proxy_client.function(x).result()
    proxy_server.bind('function', fwd)
    proxy_server.run()

  # thread = embodied.distr.Thread(proxy, start=True)
  thread = embodied.distr.Process(
      proxy, outer_addr, inner_addr,
      start=True)

  server = Server(inner_addr)
  def work(data):
    print('SERVER RECEIVED CALL')
    return {'foo': 2 * data['foo']}
  server.bind('function', work, batch=1)

  with server:
    client = embodied.distr.Client(outer_addr, pings=1, maxage=3)
    client.connect(retry=False, timeout=1)
    # futures = [client.function({'foo': 13}) for _ in range(10)]
    futures = [client.function({'foo': 13}) for _ in range(6)]
    results = [future.result()['foo'] for future in futures]
    print(results)
    assert all(result == 26 for result in results)

  thread.terminate()

def repro4():
  # Server = embodied.distr.Server2  # reproduces with either
  Server = embodied.distr.Server  # reproduces with either
  inner_addr = 'tcp://localhost:5555'
  outer_addr = 'tcp://localhost:5556'
  inner_addr = inner_addr.format(port=next(PORTS))
  outer_addr = outer_addr.format(port=next(PORTS))
  # inner_addr = 'ipc:///tmp/test-inner'
  # outer_addr = 'ipc:///tmp/test-outer'

  # def proxy(outer_addr, inner_addr):
  #   proxy_client = embodied.distr.Client(
  #       inner_addr, pings=1, maxage=2, name='ProxyInner')
  #   proxy_client.connect(retry=False, timeout=1)
  #   proxy_server = Server(outer_addr, workers=3, name='ProxyOuter')
  #   # proxy_server.bind('function', lambda x: proxy_client.function(x).result())
  #   def fwd(x):
  #     print('PROXY RECEIVED CALL')
  #     return proxy_client.function(x).result()
  #   proxy_server.bind('function', fwd)
  #   proxy_server.run()

  # # thread = embodied.distr.Thread(proxy, start=True)
  # thread = embodied.distr.Process(
  #     proxy, outer_addr, inner_addr,
  #     start=True)

  server = Server(inner_addr)
  def work(data):
    print('SERVER RECEIVED CALL')
    return {'foo': 2 * data['foo']}
  server.bind('function', work, batch=1)

  with server:
    client = embodied.distr.Client(inner_addr, pings=1, maxage=3)
    client.connect(retry=False, timeout=1)
    # futures = [client.function({'foo': 13}) for _ in range(10)]
    futures = [client.function({'foo': 13}) for _ in range(6)]
    results = [future.result()['foo'] for future in futures]
    print(results)
    assert all(result == 26 for result in results)

  # thread.terminate()


if __name__ == '__main__':
  # let's "hope" this all might be a mac-specific zmq bug that happens when
  # connecting too many servers/clients together?
  # repro1()  # stochastic: sometimes errors
  # repro2()  # stochastic: sometimes hangs
  # repro3()  # stochastic: not enough values to unpack
  # repro4()
  repro5()
