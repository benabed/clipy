"""Pure python clik implementation with jax support and candl integration."""

__author__ = ["K. Benabed","L. Balkenhol"]
__version__ = "0.15"
__description__ = "Pure python clik implementation with jax support and candl integration."

def version():
  return "clipy_"+__version__

import os
try:
    if globals().get("CLIPY_NOJAX",False) or os.environ.get("CLIPY_NOJAX",False):
        raise ImportError("nojax")
    from jax import jit
    import jax.numpy as jnp
    import jax.scipy as jsp
    hasjax = True
    try:
      from jax import config as jax_config
      jax_config.update("jax_enable_x64", True)
      jnp64 = jnp.float64
    except:
      # print TOTO in blod red and then TATA in regulartext
      print("\033[1mBEWARE: clipy would really prefer your jax to enable 64bits. however, your jax install does not...\nConsider all results potentially wrong!\033[0m")
      jnp64 = jnp.float32
    #should I believe jax that it can use float64 ?
    try:
      test=jnp.zeros(1,dtype=jnp64)
    except:
      print("\033[1mBEWARE: clipy would really prefer your jax to enable 64bits. however, your jax install does not...\nConsider all results potentially wrong!\033[0m")
      jnp64 = jnp.float32
except ImportError:
    import numpy as jnp
    import scipy as jsp
    jnp64 = jnp.float64
    hasjax = False
    # define jit decorator to do nothing
    #try:
    #  import numba as nmb
    #  def jit(func, **kwargs):
    #    return nmb.njit(func)

    #except ImportError:
    def jit(func, **kwargs):
      return func
    
from functools import partial

#oldskool
import numpy as nm

class clik_emul_error(Exception):
  pass

from . import minicldf as cldf

def split_cldf_namelist(jnpl):
  return jnpl.replace("\0"," ").split()

from .lkl import clik,clik_candl, clik_lensing

def _mini_searchsorted(a,x):
  lo = 0
  hi = len(a)
  while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
  return lo

import sys
if hasjax and sys.platform=="darwin":
  _searchsorted = _mini_searchsorted
else:
  _searchsorted = jnp.searchsorted

supported_list = [
  "plik_rd12_HM_v22_TT.clik",
  "plik_rd12_HM_v22b_TTTEEE.clik",
  ]

def check_all(plc_path=""):
  pass