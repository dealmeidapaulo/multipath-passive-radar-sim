from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Set
import numpy as np
from src.core.scene.ray import Ray
from .hash import SpatialHash


@dataclass
class StaticField:
    """
    Precomputed ray field (no UAV, no Rx filtering).

    SoA layout (N_rays innermost) matches GPU coalesced access.

    pos_cpu      float32[N_max+2, N_rays, 3]   vertex positions
    dir_cpu      float32[N_max+2, N_rays, 3]   direction at each vertex
    step_powers  float32[N_max+2, N_rays]       power (dBm) at each vertex
    n_pts_cpu    int32[N_rays]                  valid vertex count per ray
    reached_cpu  int32[N_rays]                  1 where ray reached Rx
                                                (all zeros until apply_rx fills it)
    tx_ids_cpu   int32[N_rays]                  transmitter_id per ray
    anchors      List[Ray]                      rays that reached Rx
                                                (empty until apply_rx fills it)
    anchor_ids   Set[int]                       global indices of anchor rays
                                                (empty until apply_rx fills it)
    spatial_hash SpatialHash                    segment–cell index
    fc           float                          carrier frequency (Hz)
    scene_ref    Scene                          back-reference (no Rx)
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
    rx_ref      : object = None    # Receiver — set by apply_rx()


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
