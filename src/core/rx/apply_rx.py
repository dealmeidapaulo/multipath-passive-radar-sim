from __future__ import annotations
from dataclasses import replace
from typing import List, Set
import numpy as np
 
from src.core.scene.ray           import Ray
from src.core.precompute.static_field import StaticField
 
 
def apply_rx(static: StaticField, rx) -> StaticField:
    """
    Apply a Receiver to a StaticField.
    
    Returns
    -------
    New StaticField with reached_cpu, anchors, anchor_ids populated.
    All geometry arrays (pos_cpu, dir_cpu, etc.) are shared references.
    """
    rx_pos = np.asarray(rx.position, dtype=np.float64)
    rx_rad = float(rx.radius)
 
    N_total     = static.n_pts_cpu.shape[0]
    reached_cpu = np.zeros(N_total, dtype=np.int32)
 
    # ── 1. Spatial hash query ─────────────────────────────────────────────────
    #
    # query() expects a float64 position and a radius. It returns
    # (ray_id, seg_id) pairs whose bounding box overlaps the query sphere.
    candidates = static.spatial_hash.query(rx_pos, rx_rad)
 
    if not candidates:
        return _with_rx(static, rx, reached_cpu, [], set())
 
    # ── 2. Vectorised segment–sphere test ─────────────────────────────────────
    cands   = np.array(candidates, dtype=np.int32)  # (K, 2)
    rids    = cands[:, 0]
    sids    = cands[:, 1]
    n_valid = static.n_pts_cpu[rids]
    mask    = (sids + 1) < n_valid
    rids    = rids[mask]; sids = sids[mask]
 
    if rids.size == 0:
        return _with_rx(static, rx, reached_cpu, [], set())
 
    A  = static.pos_cpu[sids,     rids, :].astype(np.float64)   # (K, 3)
    B  = static.pos_cpu[sids + 1, rids, :].astype(np.float64)   # (K, 3)
    P  = rx_pos[np.newaxis, :]                                    # (1, 3)
 
    AB   = B - A                                         # (K, 3)
    AP   = P - A                                         # (K, 3)
    len2 = np.sum(AB * AB, axis=1)                       # (K,)
    denom = np.where(len2 > 1e-30, len2, 1.0)
    t     = np.clip(np.sum(AP * AB, axis=1) / denom, 0.0, 1.0)  # (K,)
    closest = A + t[:, np.newaxis] * AB                  # (K, 3)
    dist2   = np.sum((closest - P) ** 2, axis=1)         # (K,)
 
    hit_mask    = dist2 <= rx_rad ** 2
    hit_rids    = rids[hit_mask]
    hit_sids    = sids[hit_mask]
    hit_closest = closest[hit_mask]   # (K_hit, 3) — exact intersection points
 
    if hit_rids.size == 0:
        return _with_rx(static, rx, reached_cpu, [], set())
 
    # ── 3. Build anchors ──────────────────────────────────────────────────────
    rid_to_clip: dict = {}
    for k in range(len(hit_rids)):
        rid = int(hit_rids[k])
        sid = int(hit_sids[k])
        if rid not in rid_to_clip or sid < rid_to_clip[rid][0]:
            rid_to_clip[rid] = (sid, hit_closest[k].copy())
 
    unique_rids = np.unique(hit_rids)
    reached_cpu[unique_rids] = 1
 
    anchors   : List[Ray] = []
    anchor_ids: Set[int]  = set()
    fc = static.fc
 
    for rid in unique_rids:
        clip_sid, clip_pt = rid_to_clip[int(rid)]
        pts = [static.pos_cpu[j, rid, :].astype(np.float64) for j in range(clip_sid + 1)]
        pts.append(clip_pt)
        arr = static.dir_cpu[clip_sid, rid, :].astype(np.float64)
        r   = Ray(
            transmitter_id = int(static.tx_ids_cpu[rid]),
            points         = pts,
            arrival_dir    = arr,
            frequency      = float(fc),
            power_dbm      = float(static.step_powers[clip_sid, rid]),
        )
        r.is_uav_bounce = False
        r.doppler_shift = 0.0
        r.visible       = True
        anchors.append(r)
        anchor_ids.add(int(rid))
 
    return _with_rx(static, rx, reached_cpu, anchors, anchor_ids)
 
 
# ── Helper ────────────────────────────────────────────────────────────────────
 
def _with_rx(
    static     : StaticField,
    rx         : object,
    reached_cpu: np.ndarray,
    anchors    : List[Ray],
    anchor_ids : Set[int],
) -> StaticField:
    """Return a new StaticField with Rx fields filled; all other arrays shared."""
    return StaticField(
        pos_cpu     = static.pos_cpu,
        dir_cpu     = static.dir_cpu,
        step_powers = static.step_powers,
        n_pts_cpu   = static.n_pts_cpu,
        reached_cpu = reached_cpu,
        tx_ids_cpu  = static.tx_ids_cpu,
        anchors     = anchors,
        anchor_ids  = anchor_ids,
        spatial_hash= static.spatial_hash,
        fc          = static.fc,
        scene_ref   = static.scene_ref,
        rx_ref      = rx,
    )