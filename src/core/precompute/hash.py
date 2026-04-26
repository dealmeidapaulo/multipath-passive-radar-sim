from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Set, Tuple
import numpy as np

try:
    from numba import cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False
    cuda = None

from src.core.gpu.spatial_hash_kernels import count_kernel, fill_kernel


@dataclass
class SpatialHash:
    flat_ray_ids : np.ndarray   # int32[total]
    flat_seg_ids : np.ndarray   # int32[total]
    cell_offsets : np.ndarray   # int32[N_cells+1]
    cell_counts  : np.ndarray   # int32[N_cells]
    NX : int; NY : int; NZ : int
    cell_size : float
    box_min   : np.ndarray      # float32[3]

    def query(self, uav_pos: np.ndarray, uav_rad: float) -> List[Tuple[int, int]]:
        """Return (ray_id, seg_id) pairs whose bounding box overlaps the UAV sphere. O(k)."""
        cs = self.cell_size; bm = self.box_min
        r  = uav_rad + cs * 0.5
        lo_x = max(0,         int(math.floor((uav_pos[0]-r-bm[0])/cs)))
        hi_x = min(self.NX-1, int(math.floor((uav_pos[0]+r-bm[0])/cs)))
        lo_y = max(0,         int(math.floor((uav_pos[1]-r-bm[1])/cs)))
        hi_y = min(self.NY-1, int(math.floor((uav_pos[1]+r-bm[1])/cs)))
        lo_z = max(0,         int(math.floor((uav_pos[2]-r-bm[2])/cs)))
        hi_z = min(self.NZ-1, int(math.floor((uav_pos[2]+r-bm[2])/cs)))
        result: List[Tuple[int,int]] = []; seen: Set[Tuple[int,int]] = set()
        for cx in range(lo_x, hi_x+1):
            for cy in range(lo_y, hi_y+1):
                for cz in range(lo_z, hi_z+1):
                    cell_id = cx + cy*self.NX + cz*self.NX*self.NY
                    s = int(self.cell_offsets[cell_id]); e = int(self.cell_offsets[cell_id+1])
                    for i in range(s, e):
                        key = (int(self.flat_ray_ids[i]), int(self.flat_seg_ids[i]))
                        if key not in seen:
                            seen.add(key); result.append(key)
        return result

    @property
    def total_entries(self) -> int: return int(self.flat_ray_ids.shape[0])
    @property
    def n_cells(self) -> int: return self.NX * self.NY * self.NZ

    def coverage_stats(self):
        nc = self.cell_counts; nonempty = (nc > 0).sum()
        return float(nc.mean()), int(nc.max()), float(nonempty / len(nc))


def build_spatial_hash(pos_cpu: np.ndarray, n_pts_cpu: np.ndarray,
                       box_min: np.ndarray, box_max: np.ndarray,
                       cell_size: float, threads_per_block: int = 256) -> SpatialHash:
    """Two-pass GPU build: count then fill. Returns SpatialHash (all CPU)."""
    if not _HAS_CUDA:
        raise RuntimeError("Numba CUDA not available")
    N_rays  = int(pos_cpu.shape[1])
    box_min = np.asarray(box_min, dtype=np.float32)
    box_max = np.asarray(box_max, dtype=np.float32)
    NX = max(1, int(math.ceil(float(box_max[0]-box_min[0])/cell_size)))
    NY = max(1, int(math.ceil(float(box_max[1]-box_min[1])/cell_size)))
    NZ = max(1, int(math.ceil(float(box_max[2]-box_min[2])/cell_size)))
    N_cells = NX*NY*NZ; cs_inv = np.float32(1.0/cell_size)
    bpg = (N_rays + threads_per_block - 1) // threads_per_block

    pos_gpu    = cuda.to_device(pos_cpu)
    npts_gpu   = cuda.to_device(n_pts_cpu)
    bmin_gpu   = cuda.to_device(box_min)
    counts_gpu = cuda.to_device(np.zeros(N_cells, dtype=np.int32))

    count_kernel[bpg, threads_per_block](pos_gpu, npts_gpu, counts_gpu, cs_inv, bmin_gpu, NX, NY, NZ)
    cuda.synchronize()

    counts_cpu = counts_gpu.copy_to_host()
    offsets    = np.zeros(N_cells+1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts_cpu)
    total = int(offsets[-1])

    if total == 0:
        return SpatialHash(np.empty(0,dtype=np.int32), np.empty(0,dtype=np.int32),
                           offsets, counts_cpu, NX, NY, NZ, cell_size, box_min)

    fill_ptr_gpu  = cuda.to_device(offsets[:-1].copy())
    flat_rays_gpu = cuda.to_device(np.zeros(total, dtype=np.int32))
    flat_segs_gpu = cuda.to_device(np.zeros(total, dtype=np.int32))

    fill_kernel[bpg, threads_per_block](
        pos_gpu, npts_gpu, fill_ptr_gpu, flat_rays_gpu, flat_segs_gpu,
        cs_inv, bmin_gpu, NX, NY, NZ)
    cuda.synchronize()

    return SpatialHash(flat_rays_gpu.copy_to_host(), flat_segs_gpu.copy_to_host(),
                       offsets, counts_cpu, NX, NY, NZ, cell_size, box_min)
