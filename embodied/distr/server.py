import time
import concurrent.futures
from collections import deque, namedtuple

import numpy as np

from ..core import basics
from . import sockets
from . import parallel


Method = namedtuple('Method', 'name,workfn,logfn,pool,batch,queue')


class Server:

  def __init__(
      self, address, workers=1, name='Server', errors=True, ipv6=False):
    self.address = address
    self.workers = workers
    self.name = name
    self.errors = errors
    self.ipv6 = ipv6
    self.methods = {}
    self.default_pool = concurrent.futures.ThreadPoolExecutor(workers, 'work')
    self.log_pool = concurrent.futures.ThreadPoolExecutor(1, 'log')
    self.result_set = set()
    self.log_queue = deque()
    self.log_proms = deque()
    self.running = None
    self.loop = parallel.Thread(self._loop, name='loop')

  def bind(self, name, workfn, logfn=None, workers=0, batch=0):
    if workers:
      pool = concurrent.futures.ThreadPoolExecutor(workers, name)
    else:
      pool = self.default_pool
    self.methods[name] = Method(name, workfn, logfn, pool, batch, [])

  def start(self):
    self.running = True
    self.loop.start()

  def check(self):
    [not x.done() or x.result() for x in self.result_set.copy()]
    [not x.done() or x.result() for x in self.log_proms.copy()]
    self.loop.check()

  def close(self):
    self._print('Shutting down')
    self.running = False
    concurrent.futures.wait(self.result_set)
    concurrent.futures.wait(self.log_proms)
    self.loop.join()

  def run(self):
    try:
      self.start()
      while True:
        self.check()
        time.sleep(1)
    finally:
      self.close()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, type, value, traceback):
    self.close()

  def stats(self):
    return {}  # TODO

  def _loop(self):
    socket = sockets.ServerSocket(self.address, self.ipv6)
    self._print(f'Listening at {self.address}')

    while self.running:
      now = time.time()

      result = socket.receive()
      if result:
        addr, rid, name, payload = result
        method = self.methods.get(name, None)
        if method:

          if method.batch:
            method.queue.append((addr, rid, payload, now))
            if len(method.queue) == method.batch:

              addrs, rids, payloads, recvds = zip(*method.queue)
              future = method.pool.submit(
                  self._work_batch, method, addrs, rids, payloads, recvds)
              future.method = method
              future.addrs = addrs
              future.rids = rids
              self.result_set.add(future)
              self.log_queue.append(future)
              method.queue.clear()
          else:
            future = method.pool.submit(
                self._work, method, addr, rid, payload, now)
            future.method = method
            future.addr = addr
            future.rid = rid
            self.result_set.add(future)
            self.log_queue.append(future)

        else:
          socket.send_error(addr, rid, f'Unknown method {name}.')

      completed, self.result_set = concurrent.futures.wait(
          self.result_set, 0, concurrent.futures.FIRST_COMPLETED)
      for future in completed:
        try:

          method, *result = future.result()
          if method.batch:
            addrs, rids, payloads, logs, recvds = result
            for addr, rid, payload in zip(addrs, rids, payloads):
              socket.send_result(addr, rid, payload)
            result_times = [now - recvd for recvd in recvds]  # TODO
          else:
            addr, rid, payload, logs, recvd = result
            result_time = now - recvd  # TODO
            socket.send_result(addr, rid, payload)

        except Exception as e:
          if future.method.batch:
            for addr, rid in zip(future.addrs, future.rids):
              socket.send_error(addr, rid, repr(e))
          else:
            socket.send_error(future.addr, future.rid, repr(e))
          if self.errors:
            raise

      while self.log_queue and self.log_queue[0].done():
        try:
          result = self.log_queue.popleft().result()
          method, addr, rid, payload, logs, recvd = result
          if method.logfn:
            self.log_proms.append(self.log_pool.submit(method.logfn, logs))
        except Exception:
          pass

      while self.log_proms and self.log_proms[0].done():
        self.log_proms.popleft().result()

      time.sleep(0.0001)

    socket.close()

  def _work(self, method, addr, rid, payload, recvd):
    data = sockets.unpack(payload)
    if method.logfn:
      result, logs = method.workfn(data)
    else:
      result = method.workfn(data)
      logs = None
    payload = sockets.pack(result)
    return method, addr, rid, payload, logs, recvd

  def _work_batch(self, method, addrs, rids, payloads, recvds):
    datas = [sockets.unpack(x) for x in payloads]
    data = {
        k: np.stack([datas[i][k] for i in range(method.batch)])
        for k, v in datas[0].items()}
    if method.logfn:
      result, logs = method.workfn(data)
    else:
      result = method.workfn(data)
      logs = None
    results = [
        {k: v[i] for k, v in result.items()} for i in range(method.batch)]
    payloads = [sockets.pack(x) for x in results]
    return method, addrs, rids, payloads, logs, recvds

  def _print(self, text):
    basics.print_(f'[{self.name}] {text}')
