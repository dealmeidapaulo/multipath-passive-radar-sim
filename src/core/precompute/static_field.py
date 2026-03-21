from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Set
import numpy as np
from src.core.scene.ray import Ray
from .hash import SpatialHash


@dataclass
class StaticField:
    """
    Precomputed ray field (no UAV). All tensors on CPU.

    SoA layout (N_rays innermost) matches GPU coalesced access.

    pos_cpu      float32[N_max+2, N_rays, 3]  vertex positions
    dir_cpu      float32[N_max+2, N_rays, 3]  direction at each vertex
    step_powers  float32[N_max+2, N_rays]      power (dBm) at each vertex
    n_pts_cpu    int32[N_rays]                 valid vertex count
    reached_cpu  int32[N_rays]                 1 where ray reached RX
    tx_ids_cpu   int32[N_rays]                 transmitter_id per ray
    anchors      List[Ray]                     rays that reached RX (baseline)
    anchor_ids   Set[int]                      global indices of anchors
    spatial_hash SpatialHash
    fc           float                         carrier frequency (Hz)
    scene_ref    Scene
    """
    pos_cpu     : np.ndarray
    dir_cpu     : np.ndarray
    step_powers : np.ndarray
    n_pts_cpu   : np.ndarray
    reached_cpu : np.ndarray
    tx_ids_cpu  : np.ndarray
    anchors     : List[Ray]
    anchor_ids  : Set[int]
    spatial_hash: SpatialHash
    fc          : float
    scene_ref   : object


def fibonacci_dirs(n: int) -> np.ndarray:
    """Uniformly distributed unit directions on sphere (Fibonacci lattice). Returns (n,3) float32."""
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    i      = np.arange(n, dtype=np.float64)
    theta  = np.arccos(np.clip(1.0 - 2.0 * (i + 0.5) / n, -1.0, 1.0))
    phi    = 2.0 * math.pi * i / golden
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]).astype(np.float32)
