from __future__ import annotations
import math
from typing import List, Set, Tuple

import numpy as np

try:
    from numba import cuda as _cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False; _cuda = None

from src.core.scene.ray import Ray
from src.core.scene.propagation import (compute_fspl, compute_sphere_rcs_bounce_gain,
                                  compute_scattered_doppler)
from src.core.gpu.kernels import _HAS_CUDA
if _HAS_CUDA:
    from src.core.gpu.kernels import mini_trace_kernel
from src.core.gpu.utils import fspl_const, obs_arrays


def _sphere_hit_cpu(origin, direction, center, radius, t_min=1e-5):
    """Exact CPU sphere intersection. Returns t or None."""
    oc = origin - center
    a  = float(np.dot(direction, direction))
    b  = 2.0 * float(np.dot(oc, direction))
    c  = float(np.dot(oc, oc)) - radius**2
    disc = b*b - 4*a*c
    if disc < 0: return None
    sq = math.sqrt(disc)
    for t in ((-b-sq)/(2*a), (-b+sq)/(2*a)):
        if t > t_min: return float(t)
    return None


def apply_uav(static, uav, scene) -> Tuple[List[Ray], List[Ray], List[Ray]]:
    """
    Apply a UAV to a precomputed StaticField.

    Steps
    -----
    1. Spatial hash query → candidate (ray_id, seg_id) pairs near UAV.
    2. Exact sphere-hit test (CPU) → confirmed hits.
    3. Collect hit geometry arrays.
    4. GPU batch: mini_trace_kernel generates sample directions AND traces
       all (N_hits × n_samples) rays to RX in one kernel launch.
    5. Parse results; assemble Ray objects with correct Doppler.
    6. Split anchors into visible / occluded.

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
    roughness = float(getattr(scene_ref, 'roughness', 0.0))
    uav_rough = float(getattr(scene_ref, 'uav_roughness', 0.3))
    n_samp    = int(getattr(scene_ref, 'n_samples_uav', 8))
    use_phys  = bool(getattr(scene_ref, 'use_physics', True))
    noise_f   = float(scene_ref.noise_floor_dbm) if use_phys else float('-inf')
    n_post    = max(4, int(scene_ref.n_max) // 2)
    rx_pos    = np.asarray(scene_ref.receiver.position, dtype=np.float64)
    rx_rad    = float(scene_ref.receiver.radius)
    fc        = static.fc
    rcs_gain  = compute_sphere_rcs_bounce_gain(uav_rad, fc)
    fc_c      = fspl_const(fc)

    # ── 1. Spatial hash query ─────────────────────────────────────────────────
    candidates = static.spatial_hash.query(uav_pos, uav_rad)

    blocked_ids: Set[int] = set()

    # Accumulators for GPU batch
    hit_pts_list   : List[np.ndarray] = []   # float32[3] per hit
    v_in_list      : List[np.ndarray] = []   # float32[3] incoming dir
    n_uav_list     : List[np.ndarray] = []   # float32[3] outward normal
    powers_list    : List[float]       = []   # power at hit (after RCS)
    pre_pts_list   : List[List]        = []   # pre-UAV path points
    v_in_f64_list  : List[np.ndarray] = []   # float64 for Doppler

    # ── 2. Exact sphere-hit test ─────────────────────────────────────────────
    for (rid, sid) in candidates:
        n_valid = int(static.n_pts_cpu[rid])
        if sid + 1 >= n_valid:
            continue
        A  = static.pos_cpu[sid,   rid, :].astype(np.float64)
        B  = static.pos_cpu[sid+1, rid, :].astype(np.float64)
        AB = B - A; L = float(np.linalg.norm(AB))
        if L < 1e-9: continue
        seg_dir = AB / L

        t = _sphere_hit_cpu(A, seg_dir, uav_pos, uav_rad)
        if t is None or t >= L: continue

        # Hit confirmed
        blocked_ids.add(rid)
        hit_pt = A + t * seg_dir
        n_uav  = (hit_pt - uav_pos) / uav_rad   # outward normal (unit)

        power_at_seg = float(static.step_powers[sid, rid])
        power_at_seg -= compute_fspl(float(t), float(fc))
        power_at_seg += rcs_gain
        if power_at_seg <= noise_f:
            continue

        pre_pts = [static.pos_cpu[j, rid, :].astype(np.float64) for j in range(min(sid+1, n_valid))]
        pre_pts.append(hit_pt.copy())

        hit_pts_list.append(hit_pt.astype(np.float32))
        v_in_list.append(seg_dir.astype(np.float32))
        n_uav_list.append(n_uav.astype(np.float32))
        powers_list.append(np.float32(power_at_seg))
        pre_pts_list.append(pre_pts)
        v_in_f64_list.append(seg_dir.copy())    # float64 preserved for Doppler

    # ── 3. GPU batch mini-trace ───────────────────────────────────────────────
    uav_bounces: List[Ray] = []

    if hit_pts_list and _HAS_CUDA:
        N_hits  = len(hit_pts_list)
        N_total = N_hits * n_samp
        TPB     = 256
        BPG     = max(1, (N_total + TPB - 1) // TPB)

        hit_pts_g   = _cuda.to_device(np.stack(hit_pts_list).astype(np.float32))
        v_in_g      = _cuda.to_device(np.stack(v_in_list).astype(np.float32))
        n_uav_g     = _cuda.to_device(np.stack(n_uav_list).astype(np.float32))
        powers_g    = _cuda.to_device(np.array(powers_list, dtype=np.float32))

        obs_min_np, obs_max_np = obs_arrays(scene_ref.obstacles)
        obs_min_g = _cuda.to_device(obs_min_np)
        obs_max_g = _cuda.to_device(obs_max_np)
        bmin_g    = _cuda.to_device(np.asarray(scene_ref.box.box_min, dtype=np.float32))
        bmax_g    = _cuda.to_device(np.asarray(scene_ref.box.box_max, dtype=np.float32))
        rx_pos_g  = _cuda.to_device(rx_pos.astype(np.float32))

        rch_g     = _cuda.to_device(np.zeros(N_total, dtype=np.int32))
        pwr_g     = _cuda.device_array((N_total,),              dtype=np.float32)
        adir_g    = _cuda.device_array((N_total, 3),            dtype=np.float32)
        sdir_g    = _cuda.device_array((N_total, 3),            dtype=np.float32)
        pos_g     = _cuda.device_array((n_post+2, N_total, 3), dtype=np.float32)
        npts_g    = _cuda.to_device(np.ones(N_total, dtype=np.int32))

        mini_trace_kernel[BPG, TPB](
            rch_g, pwr_g, adir_g, sdir_g, pos_g, npts_g,
            hit_pts_g, v_in_g, n_uav_g, powers_g,
            obs_min_g, obs_max_g, rx_pos_g, bmin_g, bmax_g,
            np.float32(rx_rad), np.int32(n_post), np.float32(noise_f),
            np.float32(roughness), np.float32(uav_rough),
            np.int32(n_samp), np.float32(fc_c), np.int32(42),
        )
        _cuda.synchronize()

        rch_cpu  = rch_g.copy_to_host()
        pwr_cpu  = pwr_g.copy_to_host()
        adir_cpu = adir_g.copy_to_host()
        sdir_cpu = sdir_g.copy_to_host()    # outgoing dir from UAV per sample
        pos_cpu  = pos_g.copy_to_host()
        npts_cpu = npts_g.copy_to_host()

        # ── Assemble one Ray per hit (first successful sample) ─────────────
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

            d_sample = sdir_cpu[tid].astype(np.float64)   # for Doppler
            v_in_f64 = v_in_f64_list[hit_id]
            doppler  = compute_scattered_doppler(uav_vel, v_in_f64, d_sample, fc)

            all_pts = pre_pts_list[hit_id] + post[1:]
            r = Ray(transmitter_id=0, points=all_pts,
                    arrival_dir=arr_dir, frequency=float(fc), power_dbm=fin_pwr)
            r.is_uav_bounce = True; r.doppler_shift = doppler; r.visible = True
            uav_bounces.append(r)

    # ── 4. Split anchors ─────────────────────────────────────────────────────
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
