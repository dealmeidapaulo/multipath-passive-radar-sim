from __future__ import annotations
from typing import List, Optional, Set
import numpy as np

try:
    from numba import cuda as _cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False; _cuda = None

from src.core.scene.ray import Ray
from src.core.gpu.kernels import _HAS_CUDA
if _HAS_CUDA:
    from src.core.gpu.kernels import trace_all_kernel
from src.core.gpu.utils import fspl_const, obs_arrays
from .static_field import StaticField, fibonacci_dirs
from .hash         import build_spatial_hash


def precompute(
    scene,
    seed            : Optional[int] = None,
    batch_size      : int           = 0,
    threads_per_block: int          = 256,
    cell_size       : Optional[float] = None,
) -> StaticField:
    """
    Trace scene without UAV; build spatial hash; extract anchors.

    Parameters
    ----------
    scene      : Scene
    seed       : reproducibility seed
    batch_size : max rays per GPU launch (0 = all at once)
    cell_size  : spatial hash cell size in metres (default = uav_radius)
    """
    if not _HAS_CUDA:
        raise RuntimeError("Numba CUDA not available")
    if seed is not None:
        np.random.seed(seed)
    seed_val = int(seed) if seed is not None else 0

    roughness   = float(getattr(scene, 'roughness', 0.0))
    n_max       = int(scene.n_max)
    use_physics = bool(getattr(scene, 'use_physics', True))
    noise_floor = float(scene.noise_floor_dbm) if use_physics else float('-inf')
    uav_rad     = float(scene.uav.radius) if scene.uav is not None else 1.0
    cs          = cell_size if cell_size is not None else max(1.0, uav_rad)
    fc          = float(scene.transmitters[0].frequency)
    fc_c        = np.float32(fspl_const(fc))

    obs_min_np, obs_max_np = obs_arrays(scene.obstacles)
    box_min_np = np.asarray(scene.box.box_min, dtype=np.float32)
    box_max_np = np.asarray(scene.box.box_max, dtype=np.float32)
    rx_pos_np  = np.asarray(scene.receiver.position, dtype=np.float32)
    rx_rad     = np.float32(scene.receiver.radius)

    all_pos: List[np.ndarray] = []; all_dir: List[np.ndarray] = []
    all_sp : List[np.ndarray] = []; all_npts: List[np.ndarray] = []
    all_rch: List[np.ndarray] = []; all_txid: List[np.ndarray] = []
    global_offset = 0; anchor_ids: Set[int] = set(); anchors: List[Ray] = []

    for tx in scene.transmitters:
        tx_pos_np = np.asarray(tx.position, dtype=np.float32)
        init_pwr  = np.float32(tx.tx_power_dbm)
        dirs      = fibonacci_dirs(scene.n_rays)          # NO culling
        N_rays    = dirs.shape[0]
        _bs       = N_rays if batch_size <= 0 else batch_size

        tx_pos_l: List[np.ndarray] = []; tx_dir_l: List[np.ndarray] = []
        tx_sp_l : List[np.ndarray] = []; tx_npts_l: List[np.ndarray] = []
        tx_rch_l: List[np.ndarray] = []

        for b_idx, start in enumerate(range(0, N_rays, _bs)):
            batch = dirs[start:start+_bs]; NB = batch.shape[0]

            pos_g  = _cuda.device_array((n_max+2, NB, 3), dtype=np.float32)
            dir_g  = _cuda.device_array((n_max+2, NB, 3), dtype=np.float32)
            sp_g   = _cuda.device_array((n_max+2, NB),    dtype=np.float32)
            pwr_g  = _cuda.device_array((NB,),             dtype=np.float32)
            npts_g = _cuda.device_array((NB,),             dtype=np.int32)
            rch_g  = _cuda.device_array((NB,),             dtype=np.int32)
            npts_g.copy_to_device(np.ones(NB,  dtype=np.int32))
            rch_g.copy_to_device(np.zeros(NB,  dtype=np.int32))

            seed_off = np.int32(seed_val * 999983 + b_idx * 7919 + tx.tx_id * 31337)
            bpg      = (NB + threads_per_block - 1) // threads_per_block

            trace_all_kernel[bpg, threads_per_block](
                pos_g, dir_g, sp_g, pwr_g, npts_g, rch_g,
                _cuda.to_device(batch),
                _cuda.to_device(tx_pos_np),
                _cuda.to_device(obs_min_np), _cuda.to_device(obs_max_np),
                _cuda.to_device(rx_pos_np),
                _cuda.to_device(box_min_np), _cuda.to_device(box_max_np),
                rx_rad, np.int32(n_max), init_pwr,
                np.float32(noise_floor), np.float32(roughness), fc_c, seed_off,
            )
            _cuda.synchronize()

            tx_pos_l.append(pos_g.copy_to_host()); tx_dir_l.append(dir_g.copy_to_host())
            tx_sp_l.append(sp_g.copy_to_host());   tx_npts_l.append(npts_g.copy_to_host())
            tx_rch_l.append(rch_g.copy_to_host())

        pos_tx  = np.concatenate(tx_pos_l,  axis=1)
        dir_tx  = np.concatenate(tx_dir_l,  axis=1)
        sp_tx   = np.concatenate(tx_sp_l,   axis=1)
        npts_tx = np.concatenate(tx_npts_l, axis=0)
        rch_tx  = np.concatenate(tx_rch_l,  axis=0)
        txid_tx = np.full(N_rays, tx.tx_id, dtype=np.int32)

        for lid in np.where(rch_tx == 1)[0]:
            n   = int(npts_tx[lid])
            pts = [pos_tx[j, lid, :].astype(np.float64) for j in range(n)]
            arr = dir_tx[n-1, lid, :].astype(np.float64)
            r   = Ray(transmitter_id=tx.tx_id, points=pts, arrival_dir=arr,
                      frequency=float(fc), power_dbm=float(sp_tx[n-1, lid]))
            r.is_uav_bounce = False; r.doppler_shift = 0.0; r.visible = True
            anchors.append(r)
            anchor_ids.add(int(global_offset + lid))
        global_offset += N_rays

        all_pos.append(pos_tx); all_dir.append(dir_tx); all_sp.append(sp_tx)
        all_npts.append(npts_tx); all_rch.append(rch_tx); all_txid.append(txid_tx)

    pos_cpu  = np.concatenate(all_pos,  axis=1)
    dir_cpu  = np.concatenate(all_dir,  axis=1)
    sp_cpu   = np.concatenate(all_sp,   axis=1)
    npts_cpu = np.concatenate(all_npts, axis=0)
    rch_cpu  = np.concatenate(all_rch,  axis=0)
    txid_cpu = np.concatenate(all_txid, axis=0)

    sh = build_spatial_hash(pos_cpu, npts_cpu, box_min_np, box_max_np, cs, threads_per_block)

    return StaticField(
        pos_cpu=pos_cpu, dir_cpu=dir_cpu, step_powers=sp_cpu,
        n_pts_cpu=npts_cpu, reached_cpu=rch_cpu, tx_ids_cpu=txid_cpu,
        anchors=anchors, anchor_ids=anchor_ids,
        spatial_hash=sh, fc=fc, scene_ref=scene,
    )
