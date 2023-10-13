import threading
from collections import deque

import embodied
import numpy as np


class Atari(embodied.Env):

  LOCK = threading.Lock()
  WEIGHTS = np.array([0.299, 0.587, 1 - (0.299 + 0.587)])

  def __init__(
      self, name, repeat=4, size=(84, 84), gray=True, noops=0, lives='unused',
      sticky=True, actions='all', length=108000, pooling=2, aggregate='max',
      resize='pillow'):
    # atari_py = None
    # ale_py = None
    import ale_py
    import ale_py.roms as roms
    import ale_py.roms.utils as rom_utils

    assert lives in ('unused', 'discount', 'reset'), lives
    assert actions in ('all', 'needed'), actions
    assert resize in ('opencv', 'pillow'), resize
    assert aggregate in ('max', 'mean')
    assert pooling >= 1, pooling
    assert repeat >= 1, repeat
    if name == 'james_bond':
      name = 'jamesbond'

    self.repeat = repeat
    self.size = size
    self.gray = gray
    self.noops = noops
    self.lives = lives
    self.sticky = sticky
    self.length = length
    self.pooling = pooling
    self.aggregate = aggregate
    self.resize = resize

    with self.LOCK:
      # if atari_py:
      #   self.ale = atari_py.ALEInterface()
      #   self.ale.loadROM(atari_py.get_game_path(name))
      #   self.noop = 0
      # else:
      self.ale = ale_py.ALEInterface()
      self.ale.setLoggerMode(ale_py.LoggerMode.Error)
      self.ale.loadROM(getattr(roms, rom_utils.rom_id_to_name(name)))
      self.noop = ale_py.Action.NOOP

    self.ale.setFloat('repeat_action_probability', 0.25 if sticky else 0.0)
    self.actions = {
        'all': self.ale.getLegalActionSet,
        'needed': self.ale.getMinimalActionSet,
    }[actions]()

    W, H = self.ale.getScreenDims()
    C = 3  # C = 1 if self.gray else 3
    self.buffers = deque(
        [np.zeros((W, H, C), np.uint8) for _ in range(self.pooling)],
        maxlen=self.pooling)
    self.rng = np.random.default_rng()
    self.prevlives = None
    self.duration = None
    self.done = True

  @property
  def obs_space(self):
    C = 1 if self.gray else 3
    return {
        'image': embodied.Space(np.uint8, (*self.size, C)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, len(self.actions)),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self.done:
      self._reset()
      self.prevlives = self.ale.lives()
      self.duration = 0
      self.done = False
      return self._obs(0.0, is_first=True)
    reward = 0.0
    terminal = False
    last = False
    for repeat in range(self.repeat):
      reward += self.ale.act(action['action'])
      self.duration += 1
      if repeat >= self.repeat - self.pooling:
        self._render()
      if self.ale.game_over():
        terminal = True
        last = True
      if self.duration >= self.length:
        last = True
      lives = self.ale.lives()
      if self.lives == 'discount' and lives < self.prevlives:
        terminal = True
      if self.lives == 'reset' and lives < self.prevlives:
        terminal = True
        last = True
      self.prevlives = lives
      if terminal or last:
        break
    self.done = last
    obs = self._obs(reward, is_last=last, is_terminal=terminal)
    return obs

  def _reset(self):
    with self.LOCK:
      self.ale.reset_game()
    for _ in range(self.rng.integers(self.noops + 1)):
      self.ale.act(self.noop)
      if self.ale.game_over():
        with self.LOCK:
          self.ale.reset_game()
    self.prevlives = self.ale.lives()
    self._render(reset=True)

  def _render(self, reset=False):
    self.buffers.appendleft(self.buffers.pop())
    # if self.gray:
    #   self.ale.getScreenGrayscale(self.buffers[0])
    # else:
    self.ale.getScreenRGB(self.buffers[0])
    if reset:
      for i, dst in enumerate(self.buffers):
        if i > 0:
          np.copyto(self.buffers[0], dst)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    if self.aggregate == 'max':
      image = np.amax(self.buffers, 0)
    elif self.aggregate == 'mean':
      image = np.mean(self.buffers, 0).astype(np.uint8)
    if self.resize == 'opencv':
      import cv2
      image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
    elif self.resize == 'pillow':
      from PIL import Image
      # if self.gray:
      #   image = image[:, :, 0]
      image = Image.fromarray(image)
      image = image.resize(self.size, Image.BILINEAR)
      image = np.array(image)
      # if self.gray:
      #   image = image[:, :, None]
    if self.gray:
      image = (image * self.WEIGHTS).sum(-1).astype(image.dtype)[:, :, None]
    return dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last,
    )
