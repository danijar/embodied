import time
import weakref
from functools import partial as bind

import numpy as np

from ..core import basics
from ..core import timer
from . import sockets


class Client:

  def __init__(
      self, address, identity=None, name='Client', ipv6=False, pings=10,
      maxage=60):
    if identity is None:
      identity = int(np.random.randint(2 ** 32))
    self.address = address
    self.identity = identity
    self.name = name
    self.resolved = None
    self.socket = sockets.ClientSocket(identity, ipv6, pings, maxage)
    self.futures = weakref.WeakValueDictionary()

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return bind(self.call, name)
    except AttributeError:
      raise ValueError(name)

  def close(self):
    return self.socket.close()

  @timer.section('client_connect')
  def connect(self, retry=True, timeout=10):
    while True:
      self.resolved = self._resolve(self.address)
      self._print(f'Connecting to {self.resolved}')
      try:
        self.socket.connect(self.resolved, timeout)
        self._print('Connection established')
        return
      except sockets.ProtocolError as e:
        self._print(f'Ignoring unexpected message: {e}')
      except sockets.ConnectError:
        pass
      if retry:
        continue
      else:
        # raise sockets.NotAliveError
        raise sockets.ConnectError

  @timer.section('client_call')
  def call(self, method, data):
    assert isinstance(data, dict)
    data = {k: np.asarray(v) for k, v in data.items()}
    data = sockets.pack(data)
    rid = self.socket.send_call(method, data)
    future = Future(self._receive, rid)
    self.futures[rid] = future
    return future

  @timer.section('client_receive')
  def _receive(self, rid):
    while rid in self.futures and not self.futures[rid].done():
      try:
        result = self.socket.receive()
        if result is not None:
          other, payload = result
          if other in self.futures:
            self.futures[other].set_result(sockets.unpack(payload))
      except sockets.NotAliveError:
        self._print('Server is not responding.')
        raise
      except sockets.RemoteError as e:
        # self._print('Received error response.')
        self._print(f'Received error response: {e.args[1]}')
        other = e.args[0]
        if other in self.futures:
          self.futures[other].set_error(sockets.RemoteError(e.args[1]))
      except sockets.ProtocolError as e:
        self._print(f'Ignoring unexpected message: {e}')
      time.sleep(0.001)

  @timer.section('client_resolve')
  def _resolve(self, address):
    protocol, address = address.split('://', 1)
    if address.startswith('/bns/'):
      assert self.ipv6, (address, self.ipv6)
      self._print(f'BNS address detected: {address}')
      from google3.chubby.python.public import pychubbyutil
      while True:
        try:
          address, port = pychubbyutil.ResolveBNSName(address)
          break
        except pychubbyutil.NoResultsError:
          self._print('BNS address not found, retrying')
          time.sleep(10)
      address = f'[{address}]:{port}'
      self._print(f'BNS address resolved to: {address}')
    return f'{protocol}://{address}'

  def _print(self, text):
    basics.print_(f'[{self.name}] {text}')


class Future:

  def __init__(self, waitfn, *args):
    self._waitfn = waitfn
    self._args = args
    self._status = 0
    self._result = None
    self._error = None

  def result(self):
    if self._status == 0:
      self._waitfn(*self._args)
    if self._status == 1:
      return self._result
    elif self._status == 2:
      raise self._error
    else:
      assert False

  def done(self):
    return self._status > 0

  def set_result(self, result):
    self._status = 1
    self._result = result

  def set_error(self, error):
    self._status = 2
    self._error = error
