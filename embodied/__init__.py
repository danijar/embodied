__version__ = '1.2.0'

from .core import *

from . import envs
from . import replay
from . import run

try:
  from rich import traceback
  import numpy as np
  import jax

  traceback.install(
      # show_locals=True,
      suppress=[np, jax])

except ImportError:
  pass
