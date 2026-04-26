from numba import cuda
import math


@cuda.jit(device=True, inline=True)
def _compute_cell_bounds(ax, ay, az, bx, by, bz, bmin, cs_inv, NX, NY, NZ):
    min_x = ax if ax < bx else bx
    max_x = ax if ax > bx else bx
    min_y = ay if ay < by else by
    max_y = ay if ay > by else by
    min_z = az if az < bz else bz
    max_z = az if az > bz else bz

    lo_x = int(math.floor((min_x - bmin[0]) * cs_inv))
    hi_x = int(math.floor((max_x - bmin[0]) * cs_inv))
    lo_y = int(math.floor((min_y - bmin[1]) * cs_inv))
    hi_y = int(math.floor((max_y - bmin[1]) * cs_inv))
    lo_z = int(math.floor((min_z - bmin[2]) * cs_inv))
    hi_z = int(math.floor((max_z - bmin[2]) * cs_inv))

    if lo_x < 0: lo_x = 0
    if lo_y < 0: lo_y = 0
    if lo_z < 0: lo_z = 0
    if hi_x >= NX: hi_x = NX - 1
    if hi_y >= NY: hi_y = NY - 1
    if hi_z >= NZ: hi_z = NZ - 1

    return lo_x, hi_x, lo_y, hi_y, lo_z, hi_z


@cuda.jit
def count_kernel(pos_out, n_pts, counts, cs_inv, bmin, NX, NY, NZ):
    ray_id = cuda.grid(1)
    if ray_id >= pos_out.shape[1]:
        return

    np_r = n_pts[ray_id]

    for seg_id in range(np_r - 1):
        ax = pos_out[seg_id,   ray_id, 0]
        ay = pos_out[seg_id,   ray_id, 1]
        az = pos_out[seg_id,   ray_id, 2]

        bx = pos_out[seg_id+1, ray_id, 0]
        by = pos_out[seg_id+1, ray_id, 1]
        bz = pos_out[seg_id+1, ray_id, 2]

        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = _compute_cell_bounds(
            ax, ay, az, bx, by, bz, bmin, cs_inv, NX, NY, NZ
        )

        for cx in range(lo_x, hi_x + 1):
            for cy in range(lo_y, hi_y + 1):
                for cz in range(lo_z, hi_z + 1):
                    idx = cx + cy*NX + cz*NX*NY
                    cuda.atomic.add(counts, idx, 1)


@cuda.jit
def fill_kernel(pos_out, n_pts, fill_ptr, flat_ray_ids, flat_seg_ids,
                cs_inv, bmin, NX, NY, NZ):
    ray_id = cuda.grid(1)
    if ray_id >= pos_out.shape[1]:
        return

    np_r = n_pts[ray_id]

    for seg_id in range(np_r - 1):
        ax = pos_out[seg_id,   ray_id, 0]
        ay = pos_out[seg_id,   ray_id, 1]
        az = pos_out[seg_id,   ray_id, 2]

        bx = pos_out[seg_id+1, ray_id, 0]
        by = pos_out[seg_id+1, ray_id, 1]
        bz = pos_out[seg_id+1, ray_id, 2]

        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = _compute_cell_bounds(
            ax, ay, az, bx, by, bz, bmin, cs_inv, NX, NY, NZ
        )

        for cx in range(lo_x, hi_x + 1):
            for cy in range(lo_y, hi_y + 1):
                for cz in range(lo_z, hi_z + 1):
                    idx = cx + cy*NX + cz*NX*NY
                    pos = cuda.atomic.add(fill_ptr, idx, 1)
                    flat_ray_ids[pos] = ray_id
                    flat_seg_ids[pos] = seg_id