from __future__ import annotations
import math
from numba import cuda


@cuda.jit(device=True, inline=True)
def _xorshift32(s):
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= (s >> 17) & 0xFFFFFFFF
    s ^= (s << 5)  & 0xFFFFFFFF
    return s & 0xFFFFFFFF

@cuda.jit(device=True, inline=True)
def _rand01(s):
    s = _xorshift32(s)
    return (s & 0xFFFFFF) * (1.0 / 16777216.0), s

@cuda.jit(device=True)
def _ray_aabb(ox, oy, oz, dx, dy, dz, bx0, by0, bz0, bx1, by1, bz1, t_min):
    INF = 1e30
    t_enter = -INF; t_exit = INF
    ent_ax = -1; ent_s = 0.0
    if math.fabs(dx) < 1e-12:
        if ox < bx0 or ox > bx1: return INF, 0.0, 0.0, 0.0
    else:
        inv = 1.0 / dx
        t1 = (bx0 - ox) * inv; t2 = (bx1 - ox) * inv
        if t1 > t2: t1, t2 = t2, t1
        if t1 > t_enter: t_enter = t1; ent_ax = 0; ent_s = dx
        if t2 < t_exit:  t_exit  = t2
    if math.fabs(dy) < 1e-12:
        if oy < by0 or oy > by1: return INF, 0.0, 0.0, 0.0
    else:
        inv = 1.0 / dy
        t1 = (by0 - oy) * inv; t2 = (by1 - oy) * inv
        if t1 > t2: t1, t2 = t2, t1
        if t1 > t_enter: t_enter = t1; ent_ax = 1; ent_s = dy
        if t2 < t_exit:  t_exit  = t2
    if math.fabs(dz) < 1e-12:
        if oz < bz0 or oz > bz1: return INF, 0.0, 0.0, 0.0
    else:
        inv = 1.0 / dz
        t1 = (bz0 - oz) * inv; t2 = (bz1 - oz) * inv
        if t1 > t2: t1, t2 = t2, t1
        if t1 > t_enter: t_enter = t1; ent_ax = 2; ent_s = dz
        if t2 < t_exit:  t_exit  = t2
    if t_enter > t_exit or t_exit < t_min: return INF, 0.0, 0.0, 0.0
    t_hit = t_enter if t_enter >= t_min else t_exit
    if t_hit < t_min: return INF, 0.0, 0.0, 0.0
    n_val = -1.0 if ent_s > 0.0 else 1.0
    nx_ = n_val if ent_ax == 0 else 0.0
    ny_ = n_val if ent_ax == 1 else 0.0
    nz_ = n_val if ent_ax == 2 else 0.0
    return t_hit, nx_, ny_, nz_

@cuda.jit(device=True)
def _ray_sphere(ox, oy, oz, dx, dy, dz, cx, cy, cz, r, t_min):
    ocx = ox - cx; ocy = oy - cy; ocz = oz - cz
    a = dx*dx + dy*dy + dz*dz
    b = 2.0*(ocx*dx + ocy*dy + ocz*dz)
    c = ocx*ocx + ocy*ocy + ocz*ocz - r*r
    disc = b*b - 4.0*a*c
    if disc < 0.0: return 1e30
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0*a); t2 = (-b + sq) / (2.0*a)
    if t1 > t_min: return t1
    if t2 > t_min: return t2
    return 1e30

@cuda.jit(device=True)
def _ray_floor(ox, oy, oz, dx, dy, dz, floor_z, bx0, by0, bx1, by1, t_min):
    if dz > -1e-10: return 1e30, 0.0, 0.0, 1.0
    t = (floor_z - oz) / dz
    if t < t_min: return 1e30, 0.0, 0.0, 1.0
    hx = ox + t*dx; hy = oy + t*dy
    if bx0 <= hx <= bx1 and by0 <= hy <= by1:
        return t, 0.0, 0.0, 1.0
    return 1e30, 0.0, 0.0, 1.0

@cuda.jit(device=True)
def _domain_exit(ox, oy, oz, dx, dy, dz, bx0, by0, bx1, by1, bz1, t_min):
    best = 1e30
    if dx > 1e-12:
        t = (bx1 - ox) / dx
        if t_min < t < best: best = t
    elif dx < -1e-12:
        t = (bx0 - ox) / dx
        if t_min < t < best: best = t
    if dy > 1e-12:
        t = (by1 - oy) / dy
        if t_min < t < best: best = t
    elif dy < -1e-12:
        t = (by0 - oy) / dy
        if t_min < t < best: best = t
    if dz > 1e-12:
        t = (bz1 - oz) / dz
        if t_min < t < best: best = t
    return best

@cuda.jit(device=True)
def _perturb_normal(nx, ny, nz, roughness, r1, r2):
    if roughness < 1e-5: return nx, ny, nz
    if math.fabs(nx) < 0.9: tx, ty, tz = 1.0, 0.0, 0.0
    else: tx, ty, tz = 0.0, 1.0, 0.0
    bx = ny*tz - nz*ty; by = nz*tx - nx*tz; bz = nx*ty - ny*tx
    inv = 1.0 / math.sqrt(bx*bx + by*by + bz*bz + 1e-30)
    bx *= inv; by *= inv; bz *= inv
    tx = by*nz - bz*ny; ty = bz*nx - bx*nz; tz = bx*ny - by*nx
    phi = 2.0 * math.pi * r1
    theta = roughness * math.sqrt(-2.0 * math.log(max(r2, 1e-10)))
    sin_t = math.sin(theta); cos_t = math.cos(theta)
    nx_p = cos_t*nx + sin_t*(math.cos(phi)*tx + math.sin(phi)*bx)
    ny_p = cos_t*ny + sin_t*(math.cos(phi)*ty + math.sin(phi)*by)
    nz_p = cos_t*nz + sin_t*(math.cos(phi)*tz + math.sin(phi)*bz)
    inv = 1.0 / math.sqrt(nx_p*nx_p + ny_p*ny_p + nz_p*nz_p + 1e-30)
    return nx_p*inv, ny_p*inv, nz_p*inv

@cuda.jit(device=True, inline=True)
def _reflect(dx, dy, dz, nx, ny, nz):
    dot = dx*nx + dy*ny + dz*nz
    rx = dx - 2.0*dot*nx; ry = dy - 2.0*dot*ny; rz = dz - 2.0*dot*nz
    inv = 1.0 / math.sqrt(rx*rx + ry*ry + rz*rz + 1e-30)
    return rx*inv, ry*inv, rz*inv

@cuda.jit(device=True, inline=True)
def _reflection_intensity(dx, dy, dz, nx, ny, nz, epsilon_r):
    cos_ti = -(dx*nx + dy*ny + dz*nz)
    if cos_ti < 0.0: cos_ti = -cos_ti
    if cos_ti > 1.0: cos_ti = 1.0
    sin2_ti = 1.0 - cos_ti * cos_ti
    val = epsilon_r - sin2_ti
    if val < 0.0: val = 0.0
    sqrt_term = math.sqrt(val)
    r_te = (cos_ti - sqrt_term) / (cos_ti + sqrt_term)
    R_te = r_te * r_te
    r_tm = (epsilon_r * cos_ti - sqrt_term) / (epsilon_r * cos_ti + sqrt_term)
    R_tm = r_tm * r_tm
    return 0.5 * (R_te + R_tm)

@cuda.jit(device=True)
def _bounce(dx, dy, dz, nx, ny, nz, roughness, epsilon_r, r1, r2):
    nx_p, ny_p, nz_p = _perturb_normal(nx, ny, nz, roughness, r1, r2)
    rx, ry, rz = _reflect(dx, dy, dz, nx_p, ny_p, nz_p)
    refl = _reflection_intensity(dx, dy, dz, nx_p, ny_p, nz_p, epsilon_r)
    return rx, ry, rz, refl

@cuda.jit
def trace_all_kernel(
    pos_out, dir_out, step_powers, power_out, n_pts_out,
    dirs_in, tx_pos, obs_min, obs_max, obs_roughness, obs_eps,
    box_min, box_max, n_max, init_power, noise_floor, fspl_c, seed_offset,
):
    ray_id = cuda.grid(1)
    if ray_id >= dirs_in.shape[0]: return

    n_obs = obs_min.shape[0]
    px = tx_pos[0]; py = tx_pos[1]; pz = tx_pos[2]
    dx = dirs_in[ray_id, 0]; dy = dirs_in[ray_id, 1]; dz = dirs_in[ray_id, 2]
    inv = 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz + 1e-30)
    dx *= inv; dy *= inv; dz *= inv
    power = init_power
    rng = ((seed_offset * 6364136223846793005 + ray_id * 1442695040888963407 + 1) & 0x7FFFFFFF)
    if rng == 0: rng = 1
    GROUND_EPS = 7.0

    pos_out[0, ray_id, 0] = px; pos_out[0, ray_id, 1] = py; pos_out[0, ray_id, 2] = pz
    dir_out[0, ray_id, 0] = dx; dir_out[0, ray_id, 1] = dy; dir_out[0, ray_id, 2] = dz
    step_powers[0, ray_id] = power
    n_pts = 1

    for bounce_idx in range(n_max):
        t_floor, floor_nx, floor_ny, floor_nz = _ray_floor(px, py, pz, dx, dy, dz, box_min[2], box_min[0], box_min[1], box_max[0], box_max[1], 1e-5)
        t_exit = _domain_exit(px, py, pz, dx, dy, dz, box_min[0], box_min[1], box_max[0], box_max[1], box_max[2], 1e-5)

        best_t = 1e30; best_nx = 0.0; best_ny = 0.0; best_nz = 0.0; best_obs_idx = -1; best_kind = 3
        for obs_idx in range(n_obs):
            t_obs, obs_nx, obs_ny, obs_nz = _ray_aabb(px, py, pz, dx, dy, dz, obs_min[obs_idx, 0], obs_min[obs_idx, 1], obs_min[obs_idx, 2], obs_max[obs_idx, 0], obs_max[obs_idx, 1], obs_max[obs_idx, 2], 1e-5)
            if t_obs < best_t:
                best_t = t_obs; best_nx = obs_nx; best_ny = obs_ny; best_nz = obs_nz; best_obs_idx = obs_idx; best_kind = 0
        if t_floor < best_t:
            best_t = t_floor; best_nx = floor_nx; best_ny = floor_ny; best_nz = floor_nz; best_obs_idx = -1; best_kind = 1
        if t_exit <= best_t:
            best_t = t_exit; best_kind = 2

        if best_kind == 2 or best_t >= 1e29:
            if best_t < 1e29:
                pos_out[n_pts, ray_id, 0] = px + best_t*dx; pos_out[n_pts, ray_id, 1] = py + best_t*dy; pos_out[n_pts, ray_id, 2] = pz + best_t*dz
                dir_out[n_pts, ray_id, 0] = dx; dir_out[n_pts, ray_id, 1] = dy; dir_out[n_pts, ray_id, 2] = dz
                step_powers[n_pts, ray_id] = power
                n_pts += 1
            power_out[ray_id] = power; n_pts_out[ray_id] = n_pts
            return

        current_eps = GROUND_EPS; current_rough = 0.0
        if best_kind == 0:
            current_eps = obs_eps[best_obs_idx]; current_rough = obs_roughness[best_obs_idx]

        hit_x = px + best_t*dx; hit_y = py + best_t*dy; hit_z = pz + best_t*dz
        dist = best_t if best_t > 1e-9 else 1e-9
        power -= 20.0 * math.log10(dist) + fspl_c
        r1, rng = _rand01(rng); r2, rng = _rand01(rng)
        dx_new, dy_new, dz_new, refl = _bounce(dx, dy, dz, best_nx, best_ny, best_nz, current_rough, current_eps, r1, r2)
        if refl < 1e-6: refl = 1e-6
        power += 10.0 * math.log10(refl)

        pos_out[n_pts, ray_id, 0] = hit_x; pos_out[n_pts, ray_id, 1] = hit_y; pos_out[n_pts, ray_id, 2] = hit_z
        dir_out[n_pts, ray_id, 0] = dx_new; dir_out[n_pts, ray_id, 1] = dy_new; dir_out[n_pts, ray_id, 2] = dz_new
        step_powers[n_pts, ray_id] = power
        n_pts += 1
        if power <= noise_floor:
            n_pts_out[ray_id] = n_pts; power_out[ray_id] = power
            return
        dx = dx_new; dy = dy_new; dz = dz_new
        px = hit_x + 1e-4 * dx; py = hit_y + 1e-4 * dy; pz = hit_z + 1e-4 * dz
    n_pts_out[ray_id] = n_pts; power_out[ray_id] = power

@cuda.jit
def mini_trace_kernel(
    reached_rx, power_out, arr_dir_out, sample_dir_out, pos_out, n_pts_out,
    hit_pts, v_in_dirs, n_uav_normals, init_powers,
    obs_min, obs_max, obs_roughness, obs_eps,
    rx_pos, box_min, box_max,
    rx_rad, n_post, noise_floor, uav_roughness, n_samp, fspl_c, seed_offset,
):
    tid = cuda.grid(1)
    if tid >= hit_pts.shape[0]: return
    ray_idx = tid // n_samp; samp_id = tid % n_samp; n_obs = obs_min.shape[0]
    px = hit_pts[ray_idx, 0]; py = hit_pts[ray_idx, 1]; pz = hit_pts[ray_idx, 2]
    nx = n_uav_normals[ray_idx, 0]; ny = n_uav_normals[ray_idx, 1]; nz = n_uav_normals[ray_idx, 2]
    vx = v_in_dirs[ray_idx, 0]; vy = v_in_dirs[ray_idx, 1]; vz = v_in_dirs[ray_idx, 2]
    power = init_powers[ray_idx]
    UAV_EPS = 3.0; GROUND_EPS = 7.0
    rng = ((seed_offset * 6364136223846793005 + tid * 1442695040888963407 + 1) & 0x7FFFFFFF)
    if rng == 0: rng = 1
    if samp_id == 0: dx, dy, dz, _ = _bounce(vx, vy, vz, nx, ny, nz, 0.0, UAV_EPS, 0.0, 0.0)
    else:
        r1, rng = _rand01(rng); r2, rng = _rand01(rng)
        dx, dy, dz, _ = _bounce(vx, vy, vz, nx, ny, nz, uav_roughness, UAV_EPS, r1, r2)
    sample_dir_out[tid, 0] = dx; sample_dir_out[tid, 1] = dy; sample_dir_out[tid, 2] = dz
    px += 1e-4 * dx; py += 1e-4 * dy; pz += 1e-4 * dz
    pos_out[0, tid, 0] = px; pos_out[0, tid, 1] = py; pos_out[0, tid, 2] = pz
    n_pts = 1

    for bounce_idx in range(n_post):
        t_rx = _ray_sphere(px, py, pz, dx, dy, dz, rx_pos[0], rx_pos[1], rx_pos[2], rx_rad, 1e-5)
        t_floor, floor_nx, floor_ny, floor_nz = _ray_floor(px, py, pz, dx, dy, dz, box_min[2], box_min[0], box_min[1], box_max[0], box_max[1], 1e-5)
        t_exit = _domain_exit(px, py, pz, dx, dy, dz, box_min[0], box_min[1], box_max[0], box_max[1], box_max[2], 1e-5)
        best_t = 1e30; best_nx = 0.0; best_ny = 0.0; best_nz = 0.0; best_obs_idx = -1; best_kind = 3
        for obs_idx in range(n_obs):
            t_obs, obs_nx, obs_ny, obs_nz = _ray_aabb(px, py, pz, dx, dy, dz, obs_min[obs_idx, 0], obs_min[obs_idx, 1], obs_min[obs_idx, 2], obs_max[obs_idx, 0], obs_max[obs_idx, 1], obs_max[obs_idx, 2], 1e-5)
            if t_obs < best_t:
                best_t = t_obs; best_nx = obs_nx; best_ny = obs_ny; best_nz = obs_nz; best_obs_idx = obs_idx; best_kind = 0
        if t_floor < best_t:
            best_t = t_floor; best_nx = floor_nx; best_ny = floor_ny; best_nz = floor_nz; best_obs_idx = -1; best_kind = 1
        if t_rx < best_t: best_t = t_rx; best_kind = 4
        if t_exit <= best_t: best_t = t_exit; best_kind = 2

        if best_kind == 4:
            dist = best_t if best_t > 1e-9 else 1e-9
            power -= 20.0 * math.log10(dist) + fspl_c
            pos_out[n_pts, tid, 0] = px + best_t*dx; pos_out[n_pts, tid, 1] = py + best_t*dy; pos_out[n_pts, tid, 2] = pz + best_t*dz
            n_pts += 1; power_out[tid] = power; n_pts_out[tid] = n_pts
            arr_dir_out[tid, 0] = dx; arr_dir_out[tid, 1] = dy; arr_dir_out[tid, 2] = dz
            if power > noise_floor: reached_rx[tid] = 1
            return
        if best_kind == 2 or best_t >= 1e29:
            if best_t < 1e29:
                pos_out[n_pts, tid, 0] = px + best_t*dx; pos_out[n_pts, tid, 1] = py + best_t*dy; pos_out[n_pts, tid, 2] = pz + best_t*dz
                n_pts += 1
            power_out[tid] = power; n_pts_out[tid] = n_pts
            return
        current_eps = GROUND_EPS; current_rough = 0.0
        if best_kind == 0:
            current_eps = obs_eps[best_obs_idx]; current_rough = obs_roughness[best_obs_idx]
        hit_x = px + best_t*dx; hit_y = py + best_t*dy; hit_z = pz + best_t*dz
        dist = best_t if best_t > 1e-9 else 1e-9
        power -= 20.0 * math.log10(dist) + fspl_c
        r1, rng = _rand01(rng); r2, rng = _rand01(rng)
        dx_new, dy_new, dz_new, refl = _bounce(dx, dy, dz, best_nx, best_ny, best_nz, current_rough, current_eps, r1, r2)
        if refl < 1e-6: refl = 1e-6
        power += 10.0 * math.log10(refl)
        pos_out[n_pts, tid, 0] = hit_x; pos_out[n_pts, tid, 1] = hit_y; pos_out[n_pts, tid, 2] = hit_z
        n_pts += 1
        if power <= noise_floor:
            n_pts_out[tid] = n_pts; power_out[tid] = power
            return
        dx = dx_new; dy = dy_new; dz = dz_new
        px = hit_x + 1e-4 * dx; py = hit_y + 1e-4 * dy; pz = hit_z + 1e-4 * dz
    n_pts_out[tid] = n_pts; power_out[tid] = power