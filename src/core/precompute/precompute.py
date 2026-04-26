from __future__ import annotations
from typing import List, Optional, Callable
import numpy as np

try:
    from numba import cuda as _cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False; _cuda = None

from src.core.gpu.kernels import trace_all_kernel
from src.core.gpu.utils import fspl_const, obs_arrays, obs_roughness_array, obs_eps_array
from .static_field import StaticField, fibonacci_dirs
from .hash         import build_spatial_hash


def precompute(
    scene,
    seed            : Optional[int] = None,
    batch_size      : int           = 0,
    threads_per_block: int          = 256,
    cell_size       : Optional[float] = None,
    ray_dirs_fn     : Callable[[int], np.ndarray] = fibonacci_dirs,
) -> StaticField:
    """
    Trace scene without UAV or Rx; build spatial hash.

    Parameters
    ----------
    scene      : Scene  (no Receiver required — it is not part of Scene)
    seed       : reproducibility seed
    batch_size : max rays per GPU launch (0 = all at once)
    cell_size  : spatial hash cell size in metres (default 5.0)

    Returns
    -------
    StaticField with reached_cpu=zeros, anchors=[], anchor_ids=set().
    Call apply_rx(static, rx) to populate those fields.
    """
    if not _HAS_CUDA:
        raise RuntimeError("Numba CUDA not available")
    if seed is not None:
        np.random.seed(seed)
    seed_val = int(seed) if seed is not None else 0

    n_max       = int(scene.n_max)
    use_physics = bool(scene.use_physics)
    noise_floor = float(scene.noise_floor_dbm) if use_physics else float('-inf')
    cs          = cell_size if cell_size is not None else 5.0
    fc          = float(scene.transmitters[0].frequency)
    fc_c        = np.float32(fspl_const(fc))

    obs_min_np, obs_max_np = obs_arrays(scene.obstacles)
    obs_rough_np           = obs_roughness_array(scene.obstacles)
    obs_eps_np = obs_eps_array(scene.obstacles)    
    box_min_np = np.asarray(scene.box.box_min, dtype=np.float32)
    box_max_np = np.asarray(scene.box.box_max, dtype=np.float32)

    all_pos : List[np.ndarray] = []
    all_dir : List[np.ndarray] = []
    all_sp  : List[np.ndarray] = []
    all_npts: List[np.ndarray] = []
    all_txid: List[np.ndarray] = []

    for tx in scene.transmitters:
        tx_pos_np = np.asarray(tx.position, dtype=np.float32)
        init_pwr  = np.float32(tx.tx_power_dbm)
        dirs = ray_dirs_fn(scene.n_rays)
        N_rays    = dirs.shape[0]
        _bs       = N_rays if batch_size <= 0 else batch_size

        tx_pos_l : List[np.ndarray] = []
        tx_dir_l : List[np.ndarray] = []
        tx_sp_l  : List[np.ndarray] = []
        tx_npts_l: List[np.ndarray] = []

        for b_idx, start in enumerate(range(0, N_rays, _bs)):
            batch = dirs[start:start+_bs]; NB = batch.shape[0]

            pos_g  = _cuda.device_array((n_max+2, NB, 3), dtype=np.float32)
            dir_g  = _cuda.device_array((n_max+2, NB, 3), dtype=np.float32)
            sp_g   = _cuda.device_array((n_max+2, NB),    dtype=np.float32)
            pwr_g  = _cuda.device_array((NB,),             dtype=np.float32)
            npts_g = _cuda.device_array((NB,),             dtype=np.int32)
            npts_g.copy_to_device(np.ones(NB, dtype=np.int32))

            seed_off = np.int32(seed_val * 999983 + b_idx * 7919 + tx.tx_id * 31337)
            bpg      = (NB + threads_per_block - 1) // threads_per_block

            trace_all_kernel[bpg, threads_per_block](
                pos_g, dir_g, sp_g, pwr_g, npts_g,
                _cuda.to_device(batch),
                _cuda.to_device(tx_pos_np),
                _cuda.to_device(obs_min_np), _cuda.to_device(obs_max_np),
                _cuda.to_device(obs_rough_np),
                _cuda.to_device(obs_eps_np),
                _cuda.to_device(box_min_np), _cuda.to_device(box_max_np),
                np.int32(n_max), init_pwr,
                np.float32(noise_floor), fc_c, seed_off,
            )
            _cuda.synchronize()

            tx_pos_l.append(pos_g.copy_to_host())
            tx_dir_l.append(dir_g.copy_to_host())
            tx_sp_l.append(sp_g.copy_to_host())
            tx_npts_l.append(npts_g.copy_to_host())

        pos_tx  = np.concatenate(tx_pos_l,  axis=1)
        dir_tx  = np.concatenate(tx_dir_l,  axis=1)
        sp_tx   = np.concatenate(tx_sp_l,   axis=1)
        npts_tx = np.concatenate(tx_npts_l, axis=0)
        txid_tx = np.full(N_rays, tx.tx_id, dtype=np.int32)

        all_pos.append(pos_tx);  all_dir.append(dir_tx)
        all_sp.append(sp_tx);    all_npts.append(npts_tx)
        all_txid.append(txid_tx)

    pos_cpu  = np.concatenate(all_pos,  axis=1)
    dir_cpu  = np.concatenate(all_dir,  axis=1)
    sp_cpu   = np.concatenate(all_sp,   axis=1)
    npts_cpu = np.concatenate(all_npts, axis=0)
    txid_cpu = np.concatenate(all_txid, axis=0)

    sh = build_spatial_hash(pos_cpu, npts_cpu, box_min_np, box_max_np,
                             cs, threads_per_block)

    N_total = pos_cpu.shape[1]

    return StaticField(
        pos_cpu     = pos_cpu,
        dir_cpu     = dir_cpu,
        step_powers = sp_cpu,
        n_pts_cpu   = npts_cpu,
        reached_cpu = np.zeros(N_total, dtype=np.int32),  # filled by apply_rx
        tx_ids_cpu  = txid_cpu,
        anchors     = [],                                  # filled by apply_rx
        anchor_ids  = set(),                               # filled by apply_rx
        spatial_hash= sh,
        fc          = fc,
        scene_ref   = scene,
        rx_ref      = None,
    )
