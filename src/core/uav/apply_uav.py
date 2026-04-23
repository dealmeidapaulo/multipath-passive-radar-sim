from __future__ import annotations
import math
from typing import List, Set, Tuple

import numpy as np

try:
    from numba import cuda as _cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False; _cuda = None

from src.core.scene.ray         import Ray
from src.core.scene.propagation import (compute_sphere_rcs_bounce_gain,
                                        compute_scattered_doppler)
from src.core.gpu.kernels        import _HAS_CUDA
if _HAS_CUDA:
    from src.core.gpu.kernels import mini_trace_kernel
from src.core.gpu.utils import fspl_const, obs_arrays, obs_roughness_array


def apply_uav(static, uav, scene) -> Tuple[List[Ray], List[Ray], List[Ray]]:
    """
    Apply a UAV to a precomputed StaticField.

    static must have been processed by apply_rx() first so that
    static.anchors and static.anchor_ids are populated.

    Steps
    -----
    1. Spatial hash query → candidate (ray_id, seg_id) pairs near UAV.
    2. Vectorised sphere-hit filter (NumPy) → confirmed hits + blocked ray ids.
    3. Noise-floor filter (vectorised).
    4. Small Python loop over confirmed hits (≤ ~20) to build path prefixes.
    5. GPU batch: mini_trace_kernel traces all (N_hits × n_samples) post-UAV
       rays in one launch.
    6. Assemble Ray objects; split anchors into visible / occluded.

    Returns
    -------
    anchors_vis  : visible baseline paths
    anchors_occ  : occluded baseline paths (visible=False)
    uav_bounces  : new paths via UAV (is_uav_bounce=True)
    """
    uav_pos   = np.asarray(uav.position, dtype=np.float64)
    uav_vel   = np.asarray(uav.velocity, dtype=np.float64)
    uav_rad   = float(uav.radius)
    scene_ref = static.scene_ref
    uav_rough = float(scene_ref.uav_roughness)
    n_samp    = int(scene_ref.n_samples_uav)
    noise_f   = float(scene_ref.noise_floor_dbm) if scene_ref.use_physics else float('-inf')
    n_post    = max(4, int(scene_ref.n_max) // 2)
    # Receiver stored by apply_rx() in static.rx_ref
    rx_pos    = np.asarray(static.rx_ref.position, dtype=np.float64)
    rx_rad    = float(static.rx_ref.radius)
    fc        = static.fc
    rcs_gain  = compute_sphere_rcs_bounce_gain(uav_rad, fc)
    fc_c      = fspl_const(fc)

    # ── 1. Spatial hash query ─────────────────────────────────────────────────
    candidates = static.spatial_hash.query(uav_pos, uav_rad)

    if not candidates:
        # No candidates — all anchors visible, no bounces
        anchors_vis = []
        for ray in static.anchors:
            r = Ray(ray.transmitter_id, ray.points, ray.arrival_dir,
                    ray.frequency, ray.power_dbm)
            r.is_uav_bounce = False; r.doppler_shift = 0.0; r.visible = True
            anchors_vis.append(r)
        return anchors_vis, [], []

    # ── 2. Vectorised sphere-hit filter ──────────────────────────────────────
    cands_arr = np.array(candidates, dtype=np.int32)   # (K, 2)
    rids_k    = cands_arr[:, 0]
    sids_k    = cands_arr[:, 1]

    # Drop segments beyond ray end
    n_valid_k = static.n_pts_cpu[rids_k]
    mask_v    = (sids_k + 1) < n_valid_k
    rids_k    = rids_k[mask_v];  sids_k = sids_k[mask_v]
    if rids_k.size == 0:
        return _all_visible(static), [], []

    A = static.pos_cpu[sids_k,     rids_k, :].astype(np.float64)
    B = static.pos_cpu[sids_k + 1, rids_k, :].astype(np.float64)
    uav_p = uav_pos[np.newaxis, :]

    AB   = B - A
    AP   = uav_p - A
    len2 = np.sum(AB * AB, axis=1)
    denom = np.where(len2 > 1e-30, len2, 1.0)
    t     = np.clip(np.sum(AP * AB, axis=1) / denom, 0.0, 1.0)
    closest = A + t[:, np.newaxis] * AB
    dist2   = np.sum((closest - uav_p) ** 2, axis=1)
    hit_mask = dist2 <= uav_rad ** 2

    hit_rids = rids_k[hit_mask]
    hit_sids = sids_k[hit_mask]
    if hit_rids.size == 0:
        return _all_visible(static), [], []

    # ── 3. Noise-floor filter ─────────────────────────────────────────────────
    hit_segs_idx = hit_sids                                 # last valid segment index per hit
    seg_pwr = static.step_powers[hit_segs_idx, hit_rids]   # power at segment start
    above   = seg_pwr > noise_f
    hit_rids = hit_rids[above]; hit_sids = hit_sids[above]
    if hit_rids.size == 0:
        return _all_visible(static), [], []

    blocked_ids: Set[int] = set(hit_rids.tolist())

    # ── 4. Build per-hit geometry ─────────────────────────────────────────────
    hit_pts_list   : List[np.ndarray] = []
    v_in_list      : List[np.ndarray] = []
    n_uav_list     : List[np.ndarray] = []
    powers_list    : List[float]      = []
    pre_pts_list   : List[List[np.ndarray]] = []
    v_in_f64_list  : List[np.ndarray] = []

    seen_rids: Set[int] = set()
    for i in range(len(hit_rids)):
        rid = int(hit_rids[i]); sid = int(hit_sids[i])
        if rid in seen_rids:
            continue
        seen_rids.add(rid)

        # Segment direction (incoming toward UAV)
        A_pt = static.pos_cpu[sid,     rid, :].astype(np.float64)
        B_pt = static.pos_cpu[sid + 1, rid, :].astype(np.float64)
        seg_d = B_pt - A_pt
        seg_d /= np.linalg.norm(seg_d) + 1e-30

        # Exact hit point on UAV sphere
        t_hit = _seg_sphere_t(A_pt, seg_d, uav_pos, uav_rad)
        hit_pt = A_pt + t_hit * seg_d

        # Outward normal at hit point
        n_uav = (hit_pt - uav_pos) / (np.linalg.norm(hit_pt - uav_pos) + 1e-30)

        # Power at hit (segment start power minus FSPL to hit)
        pwr_seg = float(static.step_powers[sid, rid])
        dist_to_hit = np.linalg.norm(hit_pt - A_pt)
        if dist_to_hit > 1e-9:
            import math as _math
            pwr_seg -= 20.0 * _math.log10(dist_to_hit) + fc_c
        pwr_seg += rcs_gain

        # Path prefix (points up to segment start)
        n_prev = sid + 1
        pre_pts = [static.pos_cpu[j, rid, :].astype(np.float64) for j in range(n_prev)]
        pre_pts.append(hit_pt.copy())

        hit_pts_list.append(hit_pt.astype(np.float32))
        v_in_list.append(seg_d.astype(np.float32))
        n_uav_list.append(n_uav.astype(np.float32))
        powers_list.append(np.float32(pwr_seg))
        pre_pts_list.append(pre_pts)
        v_in_f64_list.append(seg_d.copy())

    # ── 5. GPU batch mini-trace ───────────────────────────────────────────────
    uav_bounces: List[Ray] = []

    if hit_pts_list and _HAS_CUDA:
        N_hits  = len(hit_pts_list)
        N_total = N_hits * n_samp
        TPB     = 256
        BPG     = max(1, (N_total + TPB - 1) // TPB)

        hit_pts_g = _cuda.to_device(np.stack(hit_pts_list).astype(np.float32))
        v_in_g    = _cuda.to_device(np.stack(v_in_list).astype(np.float32))
        n_uav_g   = _cuda.to_device(np.stack(n_uav_list).astype(np.float32))
        powers_g  = _cuda.to_device(np.array(powers_list, dtype=np.float32))

        obs_min_np, obs_max_np = obs_arrays(scene_ref.obstacles)
        obs_rough_np = obs_roughness_array(scene_ref.obstacles)
        obs_min_g    = _cuda.to_device(obs_min_np)
        obs_max_g    = _cuda.to_device(obs_max_np)
        obs_rough_g  = _cuda.to_device(obs_rough_np)
        bmin_g    = _cuda.to_device(np.asarray(scene_ref.box.box_min, dtype=np.float32))
        bmax_g    = _cuda.to_device(np.asarray(scene_ref.box.box_max, dtype=np.float32))
        rx_pos_g  = _cuda.to_device(rx_pos.astype(np.float32))

        rch_g  = _cuda.to_device(np.zeros(N_total, dtype=np.int32))
        pwr_g  = _cuda.device_array((N_total,),             dtype=np.float32)
        adir_g = _cuda.device_array((N_total, 3),           dtype=np.float32)
        sdir_g = _cuda.device_array((N_total, 3),           dtype=np.float32)
        pos_g  = _cuda.device_array((n_post+2, N_total, 3), dtype=np.float32)
        npts_g = _cuda.to_device(np.ones(N_total, dtype=np.int32))

        mini_trace_kernel[BPG, TPB](
            rch_g, pwr_g, adir_g, sdir_g, pos_g, npts_g,
            hit_pts_g, v_in_g, n_uav_g, powers_g,
            obs_min_g, obs_max_g, obs_rough_g, rx_pos_g, bmin_g, bmax_g,
            np.float32(rx_rad), np.int32(n_post), np.float32(noise_f),
            np.float32(uav_rough),
            np.int32(n_samp), np.float32(fc_c), np.int32(42),
        )
        _cuda.synchronize()

        rch_cpu  = rch_g.copy_to_host()
        pwr_cpu  = pwr_g.copy_to_host()
        adir_cpu = adir_g.copy_to_host()
        sdir_cpu = sdir_g.copy_to_host()
        pos_cpu  = pos_g.copy_to_host()
        npts_cpu = npts_g.copy_to_host()

        hit_done: Set[int] = set()
        for tid in range(N_total):
            if rch_cpu[tid] != 1: continue
            hit_id = tid // n_samp
            if hit_id in hit_done: continue
            hit_done.add(hit_id)

            n       = int(npts_cpu[tid])
            post    = [pos_cpu[j, tid, :].astype(np.float64) for j in range(n)]
            fin_pwr = float(pwr_cpu[tid])
            arr_dir = adir_cpu[tid].astype(np.float64)
            arr_dir = arr_dir / (np.linalg.norm(arr_dir) + 1e-30)

            d_sample = sdir_cpu[tid].astype(np.float64)
            doppler  = compute_scattered_doppler(uav_vel, v_in_f64_list[hit_id], d_sample, fc)

            all_pts = pre_pts_list[hit_id] + post[1:]
            r = Ray(transmitter_id=0, points=all_pts,
                    arrival_dir=arr_dir, frequency=float(fc), power_dbm=fin_pwr)
            r.is_uav_bounce = True; r.doppler_shift = doppler; r.visible = True
            uav_bounces.append(r)

    # ── 6. Split anchors ─────────────────────────────────────────────────────
    anchor_path_ids = sorted(static.anchor_ids)
    anchors_vis: List[Ray] = []; anchors_occ: List[Ray] = []

    for i, ray in enumerate(static.anchors):
        gid = anchor_path_ids[i] if i < len(anchor_path_ids) else -1
        r = Ray(transmitter_id=ray.transmitter_id, points=ray.points,
                arrival_dir=ray.arrival_dir, frequency=ray.frequency,
                power_dbm=ray.power_dbm)
        r.is_uav_bounce = False; r.doppler_shift = 0.0
        r.visible = gid not in blocked_ids
        (anchors_vis if r.visible else anchors_occ).append(r)

    return anchors_vis, anchors_occ, uav_bounces


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seg_sphere_t(
    A: np.ndarray, d: np.ndarray,
    center: np.ndarray, radius: float,
) -> float:
    """Return t ≥ 0 where ray A+t*d first hits sphere. Falls back to 0."""
    oc = A - center
    b  = 2.0 * np.dot(d, oc)
    c  = np.dot(oc, oc) - radius ** 2
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return 0.0
    sq = math.sqrt(disc)
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    if t1 >= 0.0: return t1
    if t2 >= 0.0: return t2
    return 0.0


def _all_visible(static) -> List[Ray]:
    """Return all anchors as visible (used when there are no candidates)."""
    result = []
    for ray in static.anchors:
        r = Ray(ray.transmitter_id, ray.points, ray.arrival_dir,
                ray.frequency, ray.power_dbm)
        r.is_uav_bounce = False; r.doppler_shift = 0.0; r.visible = True
        result.append(r)
    return result
