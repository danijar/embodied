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
  # def test_client_server_throughput(self, clients=32, batch=16, workers=4):
  def test_batched_throughput(self, clients=16, batch=16, workers=4):
    addr = f'tcp://localhost:{next(PORTS)}'
    stats = defaultdict(int)
    barrier = embodied.distr.mp.Barrier(1 + clients)

    def client(context, addr, barrier):
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
      while context.running:
        client.function(data).result()

    def workfn(data):
      time.sleep(0.002)
      return data, data

    def donefn(data):
      stats['batches'] += 1
      stats['frames'] += len(data['foo'])
      stats['nbytes'] += sum(x.nbytes for x in data.values())

    procs = [
        embodied.distr.StoppableProcess(client, addr, barrier, start=True)
        for _ in range(clients)]

    # clients=32, batch=16, workers=4
    # server = embodied.distr.Server(addr)  # 160bat / 2600frm / 0.04gib
    # server = embodied.distr.Server2(addr)  # 400bat / 6400frm / 0.1gib

    # clients=16, batch=16, workers=4
    # server = embodied.distr.Server(addr)  # 140bat / 2200frm / 0.03gib
    server = embodied.distr.Server2(addr)  # 150bat / 2300frm / 0.04gib

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
    [x.stop() for x in procs]

  #############################################################################

  def test_proxy_throughput(self, clients=16, batch=16, workers=4):

    # TODO:
    # - make sure ctrl+c properly kills all processes
    # - also make sure ports are free at the beginning of the unittest

    def client(context, outer_addr, barrier):
      data = {
          'foo': np.zeros((64, 64, 3,), np.uint8),
          'bar': np.zeros((1024,), np.float32),
          'baz': np.zeros((), bool),
      }
      client = embodied.distr.Client(outer_addr)
      client.connect()
      barrier.wait()
      # while context:  # TODO: improve the API
      while context.running:
        client.function(data).result()

    def proxy(context, outer_addr, inner_addr, barrier):
      client = embodied.distr.Client(
          inner_addr, pings=0, maxage=0, name='ProxyInner')
      client.connect()
      server = embodied.distr.Server(
          outer_addr, errors=True, name='ProxyOuter')
      def function(data):
        return client.function(data).result()
      server.bind('function', function, workers=2 * batch)
      with server:
        barrier.wait()
        while context.running:
          server.check()
          time.sleep(0.1)

    def backend(context, inner_addr, barrier):
      stats = defaultdict(int)
      def workfn(data):
        time.sleep(0.002)
        return data, data
      def donefn(data):
        stats['batches'] += 1
        stats['frames'] += len(data['foo'])
        stats['nbytes'] += sum(x.nbytes for x in data.values())
      server = embodied.distr.Server(
          inner_addr, errors=True, name='Backend')
      server.bind('function', workfn, donefn, batch=batch, workers=4)
      with server:
        barrier.wait()
        start = time.time()
        while context.running:
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

    inner_addr = 'ipc:///tmp/test-inner'
    outer_addr = 'ipc:///tmp/test-outer'
    barrier = embodied.distr.mp.Barrier(2 + clients)
    procs = [
        embodied.distr.StoppableProcess(client, outer_addr, barrier)
        for _ in range(clients)]
    procs.append(embodied.distr.StoppableProcess(
        proxy, outer_addr, inner_addr, barrier))
    procs.append(embodied.distr.StoppableProcess(
        backend, inner_addr, barrier))
    embodied.distr.run(procs)


if __name__ == '__main__':
  # TestDistr().test_batched_throughput()
  TestDistr().test_proxy_throughput()
