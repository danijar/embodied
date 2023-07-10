import enum
import itertools
import pickle  # TODO: use msgpack instead
import time

import numpy as np
import zmq


class Type(enum.Enum):
  PING   = int(1).to_bytes(1, 'big')  # rid
  PONG   = int(2).to_bytes(1, 'big')  # rid
  CALL   = int(3).to_bytes(1, 'big')  # rid, text, payload
  RESULT = int(4).to_bytes(1, 'big')  # rid, payload
  ERROR  = int(5).to_bytes(1, 'big')  # rid, text


class NotAliveError(RuntimeError): pass
class RemoteError(RuntimeError): pass
class ProtocolError(RuntimeError): pass


class ClientSocket:

  def __init__(self, identity, ipv6=False, pings=10, maxage=60):
    self.socket = zmq.Context.instance().socket(zmq.DEALER)
    self.socket.setsockopt(zmq.IDENTITY, identity.to_bytes(16, 'big'))
    self.socket.setsockopt(zmq.IPV6, int(ipv6))
    self.socket.set_hwm(0)
    # TODO
    # self.poller = zmq.Poller()
    # self.poller.register(self.socket, zmq.POLLIN)
    self.pings = pings
    self.maxage = maxage
    self.connected = False
    self.last_response = 0
    self.last_pinged = 0
    self.addr = None
    self.rid = iter(itertools.count(0))
    self.running = True

  def connect(self, addr, timeout=10.0):
    self.disconnect()
    self.socket.connect(addr)
    self.addr = addr
    rid = next(self.rid).to_bytes(8, 'big')
    self.socket.send_multipart([Type.PING.value, rid])
    start = time.time()
    while True:
      try:
        parts = self.socket.recv_multipart(zmq.NOBLOCK, copy=False)
        self.last_response = time.time()
        typ, rid2, *args = [x.buffer for x in parts]
        if typ == Type.PONG.value and rid == rid2:
          self.connected = True
          return
        else:
          raise ProtocolError(Type(typ).name)
      except zmq.Again:
        pass
      if timeout and time.time() - start >= timeout:
        raise NotAliveError()
      time.sleep(0.01)

  def disconnect(self):
    if self.addr:
      self.socket.disconnect(self.addr)
      self.connected = False

  def receive(self):
    assert self.connected
    now = time.time()
    # TODO
    # if not self.poller.poll():
    #   return None
    try:
      self.last_listen = now
      parts = self.socket.recv_multipart(zmq.NOBLOCK, copy=False)
      self.last_response = now
    except zmq.Again:
      last_response_or_ping = max(self.last_response, self.last_pinged)
      if self.pings and now - last_response_or_ping >= self.pings:
        self.last_pinged = now
        self.send_ping()
      if self.maxage and self.last_listen - self.last_response >= self.maxage:
        raise NotAliveError(
            f'\nnow - last_response: {now - self.last_response:.3f}'
            f'\nnow - last_pinged:   {now - self.last_pinged:.3f}'
            f'\nnow - last_listen:   {now - self.last_listen:.3f}'
        )
      return None

    typ, rid, *args = [x.buffer for x in parts]
    rid = bytes(rid)
    if typ == Type.PING.value:
      assert not args
      self.socket.send_multipart([Type.PONG.value, rid])
      return None
    elif typ == Type.PONG.value:
      assert not args
      return None
    elif typ == Type.RESULT.value:
      payload = args
      return rid, payload
    elif typ == Type.ERROR.value:
      msgs = [str(x, 'utf-8') for x in args]
      raise RemoteError(rid, *msgs)
    else:
      raise ProtocolError(Type(typ).name)

  def send_call(self, name, payload):
    assert self.connected
    rid = next(self.rid).to_bytes(8, 'big')
    name = name.encode('utf-8')
    self.socket.send_multipart([Type.CALL.value, rid, name, *payload])
    return rid

  def send_ping(self):
    assert self.connected
    rid = next(self.rid).to_bytes(8, 'big')
    self.socket.send_multipart([Type.PING.value, rid])
    return rid

  def close(self):
    self.socket.close()


class ServerSocket:

  def __init__(self, addr, ipv6=False):
    assert any(addr.startswith(x) for x in ('tcp://', 'ipc://')), addr
    if addr.startswith('tcp://'):
      port = addr.split(':')[-1]
      addr = f'tcp://*:{port}'
    self.socket = zmq.Context.instance().socket(zmq.ROUTER)
    self.socket.setsockopt(zmq.IPV6, int(ipv6))
    self.socket.set_hwm(0)
    self.socket.bind(addr)
    # TODO
    # self.poller = zmq.Poller()
    # self.poller.register(self.socket, zmq.POLLIN)
    self.alive = {}
    self.rid = iter(itertools.count(0))

  def clients(self, maxage=float('inf')):
    now = time.time()
    return tuple(k for k, v in self.alive.items() if now - v <= maxage)

  def receive(self):
    # TODO
    # if not self.poller.poll():
    #   print('nope')
    #   return None
    now = time.time()
    try:
      parts = self.socket.recv_multipart(zmq.NOBLOCK, copy=False)
    except zmq.Again:
      return None
    addr, typ, rid, *args = [x.buffer for x in parts]
    addr = bytes(addr)
    self.alive[addr] = now
    if typ == Type.PING.value:
      assert not args
      self.socket.send_multipart([addr, Type.PONG.value, rid])
      return None
    elif typ == Type.PONG.value:
      assert not args
      return None
    elif typ == Type.CALL.value:
      method, *payload = args
      method = str(method, 'utf-8')
      return addr, rid, method, payload
    else:
      msg = f'Server received unexpected message of type {typ}'
      self.send_error(addr, rid, msg)
      return None

  def send_ping(self, addr):
    rid = next(self.rid).to_bytes(8, 'big')
    self.socket.send_multipart([addr, Type.PING.value, rid])
    return rid

  def send_result(self, addr, rid, payload):
    self.socket.send_multipart([addr, Type.RESULT.value, rid, *payload])

  def send_error(self, addr, rid, text):
    text = text.encode('utf-8')
    self.socket.send_multipart([addr, Type.ERROR.value, rid, text])

  def close(self):
    self.socket.close()


def pack(data):
  data = {k: np.asarray(v) for k, v in data.items()}
  dtypes, shapes, buffers = [], [], []
  items = sorted(data.items(), key=lambda x: x[0])
  keys, vals = zip(*items)
  dtypes = [v.dtype.str for v in vals]
  shapes = [v.shape for v in vals]
  buffers = [v.tobytes() for v in vals]
  meta = (keys, dtypes, shapes)
  payload = [pickle.dumps(meta), *buffers]
  return payload


def unpack(payload):
  meta, *buffers = payload
  keys, dtypes, shapes = pickle.loads(meta)
  vals = [
      np.frombuffer(b, d).reshape(s)
      for i, (d, s, b) in enumerate(zip(dtypes, shapes, buffers))]
  data = dict(zip(keys, vals))
  return data
