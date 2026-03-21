from __future__ import annotations
import math
import numpy as np

try:
    from numba import cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False
    cuda = None  # type: ignore


def fspl_const(fc: float) -> float:
    """
    Frequency-dependent FSPL term (dB): 20·log10(fc) + 20·log10(4π/c).
    The kernel adds 20·log10(distance) to this to get the full FSPL.
    """
    c = 3e8
    return 20.0 * math.log10(fc) + 20.0 * math.log10(4.0 * math.pi / c)


def obs_arrays(obstacles):
    """
    Return (obs_min_np, obs_max_np) as float32 arrays for the GPU kernel.
    If there are no obstacles, returns a dummy obstacle far outside any domain.
    """
    if len(obstacles) > 0:
        obs_min = np.array([o.box_min for o in obstacles], dtype=np.float32)
        obs_max = np.array([o.box_max for o in obstacles], dtype=np.float32)
    else:
        obs_min = np.array([[-1e6, -1e6, -1e6]], dtype=np.float32)
        obs_max = np.array([[-1e6, -1e6, -1e6]], dtype=np.float32)
    return obs_min, obs_max
