"""
simulation_v4/src/tracer.py
=================
Public API. Import everything from here; internal structure is opaque.

  from src.tracer import (
      make_street_scene,
      precompute, apply_uav, get_or_compute,
      get_covered_uav_spawn,
      to_dataframe, plot_from_static, plot_trajectory, make_frame_rays,
      StaticField, Scene, Ray, UAV,
  )
"""
# ── Scene ─────────────────────────────────────────────────────────────────────
from .core.scene.domain  import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from .core.scene.ray     import Ray
from .core.scene.streets import make_street_scene

# ── Core engine ───────────────────────────────────────────────────────────────
from .core.precompute.static_field import StaticField
from .core.precompute.precompute   import precompute
from .core.uav.apply_uav           import apply_uav
from .core.cache                   import get_or_compute, hash_scene, load_static, save_static

# ── IO ────────────────────────────────────────────────────────────────────────
from .outputs.observables import to_dataframe, extract
try:
    from .outputs.visualizer import (plot_trajectory, plot_from_static, make_frame_rays)
except ImportError:
    def plot_trajectory(*a, **kw):   raise ImportError("pip install plotly")
    def plot_from_static(*a, **kw):  raise ImportError("pip install plotly")
    def make_frame_rays(v,o,u):      return v + o + u

# ── Utility ───────────────────────────────────────────────────────────────────
import math
import numpy as np
from typing import Optional

def get_covered_uav_spawn(
    static,
    obstacles,
    uav_rad     : float = 1.0,
    min_segs    : int   = 10,
    min_z       : float = 10.0,
    max_z       : float = 60.0,
    max_attempts: int   = 500,
) -> np.ndarray:
    """
    Sample a UAV position that:
      (a) Is in a spatial-hash cell with >= min_segs path segments (covered zone).
      (b) Is in open air (no obstacle collision).
      (c) min_z <= z <= max_z.
    Falls back to unconstrained rejection sampling if no covered cell passes (b)+(c).
    """
    sh = static.spatial_hash; cs = sh.cell_size; bm = sh.box_min
    covered = np.where(sh.cell_counts >= min_segs)[0]
    rng = np.random.default_rng()

    def _clear(pos):
        for obs in obstacles:
            cx = max(float(obs.box_min[0]), min(float(pos[0]), float(obs.box_max[0])))
            cy = max(float(obs.box_min[1]), min(float(pos[1]), float(obs.box_max[1])))
            cz = max(float(obs.box_min[2]), min(float(pos[2]), float(obs.box_max[2])))
            if math.sqrt((cx-pos[0])**2+(cy-pos[1])**2+(cz-pos[2])**2) < uav_rad:
                return False
        return True

    if covered.size > 0:
        for _ in range(max_attempts):
            cid = int(rng.choice(covered))
            cz_ = cid // (sh.NX * sh.NY); rem = cid % (sh.NX * sh.NY)
            cy_ = rem // sh.NX;           cx_ = rem % sh.NX
            pos = np.array([float(bm[0])+(cx_+rng.random())*cs,
                             float(bm[1])+(cy_+rng.random())*cs,
                             float(bm[2])+(cz_+rng.random())*cs])
            if min_z <= pos[2] <= max_z and _clear(pos):
                return pos

    bx_max = float(bm[0]) + sh.NX * cs; by_max = float(bm[1]) + sh.NY * cs
    for _ in range(max_attempts):
        pos = np.array([rng.uniform(float(bm[0])+uav_rad, bx_max-uav_rad),
                         rng.uniform(float(bm[1])+uav_rad, by_max-uav_rad),
                         rng.uniform(min_z, max_z)])
        if _clear(pos):
            return pos
    raise RuntimeError("Could not find valid UAV spawn after max_attempts")


def trace(scene, seed: Optional[int] = None) -> list:
    """Backward-compatible wrapper: returns flat List[Ray]."""
    import copy
    static = precompute(scene, seed=seed)
    if scene.uav is None:
        return static.anchors
    vis, occ, bounces = apply_uav(static, scene.uav, scene)
    return vis + occ + bounces
