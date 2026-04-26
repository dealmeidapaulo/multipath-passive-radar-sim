from __future__ import annotations
import math
import numpy as np

from numba import cuda



MATERIAL_MAP = {
    "concrete": 0,
    "glass": 1,
    "metal": 2,
    "wood": 3,
}

EPSILON_TABLE = np.array([
    7.0,   # concrete
    5.0,   # glass
    1e6,   # metal (aprox conductor)
    2.5,   # wood
], dtype=np.float32)



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
    Works for both Obstacle (AABB) and MeshObstacle (uses its AABB fallback).
    If there are no obstacles, returns a dummy obstacle far outside any domain.
    """
    if len(obstacles) > 0:
        obs_min = np.array([o.box_min for o in obstacles], dtype=np.float32)
        obs_max = np.array([o.box_max for o in obstacles], dtype=np.float32)
    else:
        obs_min = np.array([[-1e6, -1e6, -1e6]], dtype=np.float32)
        obs_max = np.array([[-1e6, -1e6, -1e6]], dtype=np.float32)
    return obs_min, obs_max


def obs_roughness_array(obstacles) -> np.ndarray:
    """
    Return float32[N_obs] of per-obstacle roughness values for the GPU kernel.
    Falls back to 0.0 for any obstacle missing the attribute.
    """
    if len(obstacles) == 0:
        return np.array([0.0], dtype=np.float32)
    return np.array(
        [float(getattr(o, "roughness", 0.0)) for o in obstacles],
        dtype=np.float32,
    )




def obs_eps_array(obstacles) -> np.ndarray:
    return np.array(
        [
            EPSILON_TABLE[MATERIAL_MAP.get(o.material, 0)]
            for o in obstacles
        ],
        dtype=np.float32,
    )