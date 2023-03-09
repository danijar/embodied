import ctypes
import queue as queuelib
import sys
import threading
import time
import traceback
import uuid

import numpy as np

from . import basics


class RemoteError(RuntimeError): pass
class ReconnectError(RuntimeError): pass


class Client:

  def __init__(self, address, ipv6=False, timeout=60):
    self.address = address
    self.ipv6 = ipv6
    self.timeout = timeout
    self.socket = None
    self.pending = False
    self.once = True
    self._connect()

  def __call__(self, data):
    assert isinstance(data, dict), type(data)
    if self.pending:
      self._receive()
    self.socket.send(basics.pack(data))
    self.once and print('Sent first request.')
    self.pending = True
    return self._receive

  def _connect(self):
    import zmq
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.REQ)
    self.socket.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)
    self.ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
    if self.timeout > 0:
      self.socket.RCVTIMEO = int(1000 * self.timeout)
    address = self._resolve(self.address)
    basics.print_(f'Client connecting to {address}', color='green')
    self.socket.connect(address)
    self.pending = False
    self.once = True

  def _resolve(self, address):
    return f'tcp://{address}'

  def _receive(self):
    import zmq
    start = time.time()  # TODO
    try:
      recieved = self.socket.recv()
      self.once and print('Received first response.')
    except zmq.Again:
      print(f'Request timed out ({time.time() - start:.2f}s), reconnecting.')
      self.socket.close(linger=0)
      self._connect()
      raise ReconnectError()
    except Exception as e:
      raise RuntimeError(f'Failed to receive data from server: {e}')
    result = basics.unpack(recieved)
    if result.get('type', 'data') == 'error':
      msg = result.get('message', None)
      raise RemoteError(f'Server responded with an error: {msg}')
    self.pending = False
    self.once = False
    return result


class Server:

  def __init__(self, port, function, ipv6=False):
    address = f'tcp://*:{port}'
    import zmq
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.REP)
    basics.print_(f'Server listening at {address}', color='green')
    ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
    self.socket.bind(address)
    self.function = function

  def run(self):
    while True:
      payload = self.socket.recv()
      inputs = basics.unpack(payload)
      assert isinstance(inputs, dict), type(inputs)
      try:
        result = self.function(inputs)
        assert isinstance(result, dict), type(result)
      except Exception as e:
        result = {'type': 'error', 'message': str(e)}
        self.socket.send(basics.pack(payload))
        raise
      payload = basics.pack(result)
      self.socket.send(payload)


class BatchServer:

  def __init__(self, port, batch, function, ipv6=False):
    address = f'tcp://*:{port}'
    import zmq
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.ROUTER)
    ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
    basics.print_(f'BatchServer listening at {address}', color='green')
    self.socket.bind(address)
    self.batch = batch
    self.function = function
    self.buffer = None
    self.once = True

  def run(self):
    while True:
      inputs, addresses = self._receive()
      results, addresses, exception = self._process(inputs, addresses)
      self._respond(results, addresses)
      if exception:
        raise exception
      self.once = False

  def _receive(self):
    addresses = []
    for i in range(self.batch):
      self.once and print(f'Waiting for request {i} of {self.batch}.')
      address, empty, payload = self.socket.recv_multipart()
      data = basics.unpack(payload)
      assert isinstance(data, dict), type(data)
      if self.buffer is None:
        self.buffer = {
            k: np.empty((self.batch, *v.shape), v.dtype)
            for k, v in data.items() if not isinstance(v, str)}
      for key, value in data.items():
        self.buffer[key][i] = value
      addresses.append(address)
    return self.buffer, addresses

  def _process(self, inputs, addresses):
      try:
        results = self.function(inputs, [x.hex() for x in addresses])
        assert isinstance(results, dict), type(results)
        for key, value in results.items():
          if not isinstance(value, str):
            assert len(value) == self.batch, (key, value.shape)
      except Exception as e:
        results = {'type': 'error', 'message': str(e)}
        return results, addresses, e
      return results, addresses, None

  def _respond(self, results, addresses):
    import zmq
    for i, address in enumerate(addresses):
      self.once and print(f'Sending response {i} of {self.batch}.')
      payload = basics.pack({
          k: v if isinstance(v, str) else v[i]
          for k, v in results.items()})
      try:
        self.socket.send_multipart([address, b'', payload], zmq.NOBLOCK)
      except zmq.Again:
        print('A client was not available to receive a response.')


class AsyncBatchServer:

  def __init__(self, port, workers, batch, function, ipv6=False):
    address = f'tcp://*:{port}'
    import zmq
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.ROUTER)
    basics.print_(f'BatchServer listening at {address}', color='green')
    ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
    self.socket.bind(address)
    self.workers = workers
    self.batch = batch
    self.function = function
    self.incoming = queuelib.Queue()
    self.outgoing = queuelib.Queue()
    self.lock = threading.Lock()

  def run(self):
    workers = []
    workers.append(Thread(self._receiver))
    workers.append(Thread(self._responder))
    for _ in range(self.workers):
      workers.append(Thread(self._processor))
    run(workers)

  def _receiver(self):
    while True:
      self._receive()

  def _processor(self):
    while True:
      self._process()

  def _responder(self):
    while True:
      self._respond()

  def _receive(self):
    import zmq
    inputs = None
    addresses = []
    for i in range(self.batch):
      while True:
        try:
          with self.lock:
            address, empty, payload = self.socket.recv_multipart(zmq.NOBLOCK)
          break
        except zmq.Again:
          time.sleep(0.001)
      data = basics.unpack(payload)
      assert isinstance(data, dict), type(data)
      if inputs is None:
        inputs = {
            k: np.empty((self.batch, *v.shape), v.dtype)
            for k, v in data.items() if not isinstance(v, str)}
      for key, value in data.items():
        inputs[key][i] = value
      addresses.append(address)
    self.incoming.put((inputs, addresses))

  def _process(self):
    inputs, addresses = self.incoming.get()
    results = self.function(inputs, [x.hex() for x in addresses])
    assert isinstance(results, dict), type(results)
    for key, value in results.items():
      if not isinstance(value, str):
        assert len(value) == self.batch, (key, value.shape)
    self.outgoing.put((results, addresses))

  def _respond(self):
    import zmq
    results, addresses = self.outgoing.get()
    payloads = [
        basics.pack({
            k: v if isinstance(v, str) else v[i]
            for k, v in results.items()})
        for i, _ in enumerate(addresses)]
    for address, payload in zip(addresses, payloads):
      try:
        with self.lock:
          self.socket.send_multipart([address, b'', payload], zmq.NOBLOCK)
      except zmq.Again:
        print('A client was not available to receive a response.')


class Thread(threading.Thread):

  lock = threading.Lock()

  def __init__(self, fn, *args, name=None):
    self.fn = fn
    self.exitcode = None
    name = name or fn.__name__
    super().__init__(target=self._wrapper, args=args, name=name, daemon=True)

  @property
  def running(self):
    return self.is_alive()

  def _wrapper(self, *args):
    try:
      self.fn(*args)
    except Exception:
      with self.lock:
        print('-' * 79)
        print(f'Exception in worker: {self.name}')
        print('-' * 79)
        print(''.join(traceback.format_exception(*sys.exc_info())))
        self.exitcode = 1
      raise
    self.exitcode = 0

  def terminate(self):
    if not self.is_alive():
      return
    if hasattr(self, '_thread_id'):
      thread_id = self._thread_id
    else:
      thread_id = [k for k, v in threading._active.items() if v is self][0]
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
    if result > 1:
      ctypes.pythonapi.PyThreadState_SetAsyncExc(
          ctypes.c_long(thread_id), None)
    print('Shut down worker:', self.name)


class Process:

  lock = None
  initializers = []

  def __init__(self, fn, *args, name=None):
    import multiprocessing
    import cloudpickle
    mp = multiprocessing.get_context('spawn')
    if Process.lock is None:
      Process.lock = mp.Lock()
    name = name or fn.__name__
    initializers = cloudpickle.dumps(self.initializers)
    args = (initializers,) + args
    self._process = mp.Process(
        target=self._wrapper, args=(Process.lock, fn, *args),
        name=name)

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

  def terminate(self):
    self._process.terminate()
    print('Shut down worker:', self.name)

  def _wrapper(self, lock, fn, *args):
    try:
      import cloudpickle
      initializers, *args = args
      for initializer in cloudpickle.loads(initializers):
        initializer()
      fn(*args)
    except Exception:
      with lock:
        print('-' * 79)
        print(f'Exception in worker: {self.name}')
        print('-' * 79)
        print(''.join(traceback.format_exception(*sys.exc_info())))
      raise


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
        raise RuntimeError(f'Stopped workers due to crash in {worker.name}.')
    time.sleep(0.1)
