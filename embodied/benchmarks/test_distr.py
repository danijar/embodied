import pathlib
import sys
import time
from collections import defaultdict

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np

PORTS = iter(range(5555, 6000))


class TestDistr:

  # def test_client_server_throughput(self, clients=64, batch=16, workers=4):
  def test_client_server_throughput(self, clients=32, batch=16, workers=4):
    addr = f'tcp://localhost:{next(PORTS)}'
    stats = defaultdict(int)
    barrier = embodied.distr.mp.Barrier(1 + clients)

    def client(addr, barrier):
      data = {
          'foo': np.zeros((64, 64, 3,), np.uint8),
          'bar': np.zeros((1024,), np.float32),
          'baz': np.zeros((), bool),
      }
      # data = {
      #     'foo': np.zeros((64, 64, 64, 3,), np.uint8),
      #     'bar': np.zeros((64, 1024,), np.float32),
      #     'baz': np.zeros((64,), bool),
      # }
      client = embodied.distr.Client(addr)
      client.connect()
      barrier.wait()
      while True:
        client.function(data).result()

    def workfn(data):
      time.sleep(0.002)
      return data, data

    def donefn(data):
      stats['batches'] += 1
      stats['frames'] += len(data['foo'])
      stats['nbytes'] += sum(x.nbytes for x in data.values())

    procs = [
        embodied.distr.Process(client, addr, barrier, start=True)
        for _ in range(clients)]
    # server = embodied.distr.Server(addr)  # 160bat / 2600frm / 0.04gib
    server = embodied.distr.Server2(addr)  # 400bat / 6400frm / 0.1gib
    server.bind('function', workfn, donefn, batch=batch, workers=4)
    with server:
      barrier.wait()
      start = time.time()
      while True:
        server.check()
        now = time.time()
        dur = now - start
        print(
            f'{stats["batches"]/dur:.2f} bat/s ' +
            f'{stats["frames"]/dur:.2f} frm/s ' +
            f'{stats["nbytes"]/dur/(1024**3):.2f} gib/s')
        stats.clear()
        start = now
        time.sleep(1)
    [x.terminate() for x in procs]

  def test_proxy_throughput(self, clients=4, batch=2, workers=4):
    inner_addr = f'tcp://localhost:{next(PORTS)}'
    outer_addr = f'tcp://localhost:{next(PORTS)}'
    stats = defaultdict(int)
    barrier = embodied.distr.mp.Barrier(1 + clients)

    def client(outer_addr, barrier):
      data = {
          'foo': np.zeros((64, 64, 3,), np.uint8),
          'bar': np.zeros((1024,), np.float32),
          'baz': np.zeros((), bool),
      }
      client = embodied.distr.Client(outer_addr)
      client.connect()
      barrier.wait()
      while True:
        client.function(data).result()

    procs = [
        embodied.distr.Process(client, outer_addr, barrier, start=True)
        for _ in range(clients)]

    def proxy(outer_addr, inner_addr):
      client = embodied.distr.Client(
          inner_addr, pings=0, maxage=0, name='ProxyInner')
      client.connect()
      # server = embodied.distr.Server2(outer_addr, name='ProxyOuter')
      server = embodied.distr.Server(outer_addr, name='ProxyOuter')
      def fwd(data):
        try:
          return client.function(data).result()
        except Exception as e:
          print(f'EXCEPTION IN PROXY: {e}')
          raise
      server.bind('function', fwd, workers=4)
      server.run()

    procs.append(embodied.distr.Process(
        proxy, outer_addr, inner_addr, start=True))

    def workfn(data):
      print('work')
      time.sleep(0.002)
      return data, data

    def donefn(data):
      stats['batches'] += 1
      stats['frames'] += len(data['foo'])
      stats['nbytes'] += sum(x.nbytes for x in data.values())

    # server = embodied.distr.Server(addr)  # 160bat / 2600frm / 0.04gib
    # server = embodied.distr.Server2(inner_addr)  # 400bat / 6400frm / 0.1gib
    server = embodied.distr.Server(inner_addr)  # 400bat / 6400frm / 0.1gib
    server.bind('function', workfn, donefn, batch=batch, workers=4)

    with server:

      barrier.wait()
      start = time.time()
      while True:
        server.check()
        now = time.time()
        dur = now - start
        print(
            f'{stats["batches"]/dur:.2f} bat/s ' +
            f'{stats["frames"]/dur:.2f} frm/s ' +
            f'{stats["nbytes"]/dur/(1024**3):.2f} gib/s')
        stats.clear()
        start = now
        time.sleep(1)
    [x.terminate() for x in procs]


if __name__ == '__main__':
  # TestDistr().test_client_server_throughput()
  TestDistr().test_proxy_throughput()
