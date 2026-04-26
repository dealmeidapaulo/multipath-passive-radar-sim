from __future__ import annotations
import math

try:
    from numba import cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False
    cuda = None  # type: ignore


if _HAS_CUDA:

    # ── RNG ───────────────────────────────────────────────────────────────────

    @cuda.jit(device=True, inline=True)
    def _xorshift32(s):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5)  & 0xFFFFFFFF
        return s & 0xFFFFFFFF

    @cuda.jit(device=True, inline=True)
    def _rand01(s):
        """Returns (uniform float in [0,1), new_state)."""
        s = _xorshift32(s)
        return (s & 0xFFFFFF) * (1.0 / 16777216.0), s

    # ── Geometry primitives ───────────────────────────────────────────────────

    @cuda.jit(device=True)
    def _ray_aabb(ox, oy, oz, dx, dy, dz,
                  bx0, by0, bz0, bx1, by1, bz1, t_min):
        """Slab method. Returns (t, nx, ny, nz) or (1e30, 0,0,0) on miss."""
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
        """Returns first positive t > t_min, or 1e30 on miss."""
        ocx = ox - cx; ocy = oy - cy; ocz = oz - cz
        a   = dx*dx + dy*dy + dz*dz
        b   = 2.0*(ocx*dx + ocy*dy + ocz*dz)
        c   = ocx*ocx + ocy*ocy + ocz*ocz - r*r
        disc = b*b - 4.0*a*c
        if disc < 0.0: return 1e30
        sq = math.sqrt(disc)
        t1 = (-b - sq) / (2.0*a); t2 = (-b + sq) / (2.0*a)
        if t1 > t_min: return t1
        if t2 > t_min: return t2
        return 1e30

    @cuda.jit(device=True)
    def _ray_floor(ox, oy, oz, dx, dy, dz,
                   floor_z, bx0, by0, bx1, by1, t_min):
        """Horizontal floor at floor_z. Returns (t, 0,0,1) or (1e30,…)."""
        if dz > -1e-10: return 1e30, 0.0, 0.0, 1.0
        t = (floor_z - oz) / dz
        if t < t_min: return 1e30, 0.0, 0.0, 1.0
        hx = ox + t*dx; hy = oy + t*dy
        if bx0 <= hx <= bx1 and by0 <= hy <= by1:
            return t, 0.0, 0.0, 1.0
        return 1e30, 0.0, 0.0, 1.0

    @cuda.jit(device=True)
    def _domain_exit(ox, oy, oz, dx, dy, dz,
                     bx0, by0, bx1, by1, bz1, t_min):
        """Nearest wall/ceiling exit. Returns t or 1e30."""
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

    # ── Bounce ───────────────────────────────────────────────────────────────

    @cuda.jit(device=True)
    def _perturb_normal(nx, ny, nz, roughness, r1, r2):


        if roughness < 1e-5:
            return nx, ny, nz

        if math.fabs(nx) < 0.9:
            tx, ty, tz = 1.0, 0.0, 0.0
        else:
            tx, ty, tz = 0.0, 1.0, 0.0

        # Gram-Schmidt
        bx = ny*tz - nz*ty
        by = nz*tx - nx*tz
        bz = nx*ty - ny*tx
        inv = 1.0 / math.sqrt(bx*bx + by*by + bz*bz + 1e-30)
        bx *= inv; by *= inv; bz *= inv

        tx = by*nz - bz*ny
        ty = bz*nx - bx*nz
        tz = bx*ny - by*nx

        phi = 2.0 * math.pi * r1
        theta = roughness * math.sqrt(-2.0 * math.log(max(r2,1e-10)))

        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        nx_p = cos_t*nx + sin_t*(math.cos(phi)*tx + math.sin(phi)*bx)
        ny_p = cos_t*ny + sin_t*(math.cos(phi)*ty + math.sin(phi)*by)
        nz_p = cos_t*nz + sin_t*(math.cos(phi)*tz + math.sin(phi)*bz)

        inv = 1.0 / math.sqrt(nx_p*nx_p + ny_p*ny_p + nz_p*nz_p + 1e-30)
        return nx_p*inv, ny_p*inv, nz_p*inv


    @cuda.jit(device=True, inline=True)
    def _reflect(dx, dy, dz, nx, ny, nz):
        dot = dx*nx + dy*ny + dz*nz
        rx = dx - 2.0*dot*nx
        ry = dy - 2.0*dot*ny
        rz = dz - 2.0*dot*nz

        inv = 1.0 / math.sqrt(rx*rx + ry*ry + rz*rz + 1e-30)
        return rx*inv, ry*inv, rz*inv

    @cuda.jit(device=True, inline=True)
    def _reflection_intensity(dx, dy, dz,
                            nx, ny, nz,
                            epsilon_r):

        cos_ti = -(dx*nx + dy*ny + dz*nz)
        if cos_ti < 0.0:
            cos_ti = -cos_ti
        if cos_ti > 1.0:
            cos_ti = 1.0

        sin2_ti = 1.0 - cos_ti * cos_ti
        sqrt_term = math.sqrt(epsilon_r - sin2_ti)

        # TE
        r_te = (cos_ti - sqrt_term) / (cos_ti + sqrt_term)
        R_te = r_te * r_te

        # TM
        r_tm = (epsilon_r * cos_ti - sqrt_term) / (epsilon_r * cos_ti + sqrt_term)
        R_tm = r_tm * r_tm

        return 0.5 * (R_te + R_tm)
    


    @cuda.jit(device=True)
    def _bounce(dx, dy, dz,
            nx, ny, nz,
            roughness,
            epsilon_r,
            r1, r2):

        nx_p, ny_p, nz_p = _perturb_normal(nx, ny, nz, roughness, r1, r2)

        rx, ry, rz = _reflect(dx, dy, dz, nx_p, ny_p, nz_p)

        refl = _reflection_intensity(dx, dy, dz, nx_p, ny_p, nz_p, epsilon_r)
        return rx, ry, rz, refl

    # ── trace_all_kernel ──────────────────────────────────────────────────────

@cuda.jit
def trace_all_kernel(
    pos_out, dir_out, step_powers, power_out, n_pts_out,
    dirs_in, tx_pos, obs_min, obs_max, obs_roughness,
    box_min, box_max, n_max, init_power, noise_floor, fspl_c, seed_offset,
):
    ray_id = cuda.grid(1)
    if ray_id >= dirs_in.shape[0]:
        return

    N_obs = obs_min.shape[0]

    px = tx_pos[0]; py = tx_pos[1]; pz = tx_pos[2]

    dx = dirs_in[ray_id, 0]
    dy = dirs_in[ray_id, 1]
    dz = dirs_in[ray_id, 2]

    inv = 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz + 1e-30)
    dx *= inv; dy *= inv; dz *= inv

    power = init_power

    rng = ((seed_offset * 6364136223846793005 +
           ray_id * 1442695040888963407 + 1) & 0x7FFFFFFF)
    if rng == 0:
        rng = 1

    EPSILON_CONCRETO = 7.0

    pos_out[0, ray_id, 0] = px
    pos_out[0, ray_id, 1] = py
    pos_out[0, ray_id, 2] = pz

    dir_out[0, ray_id, 0] = dx
    dir_out[0, ray_id, 1] = dy
    dir_out[0, ray_id, 2] = dz

    step_powers[0, ray_id] = power
    n_pts = 1

    for _bounce_i in range(n_max):

        t_fl, fnx, fny, fnz = _ray_floor(
            px, py, pz, dx, dy, dz,
            box_min[2], box_min[0], box_min[1],
            box_max[0], box_max[1], 1e-5
        )

        t_ex = _domain_exit(
            px, py, pz, dx, dy, dz,
            box_min[0], box_min[1],
            box_max[0], box_max[1], box_max[2], 1e-5
        )

        best_t = 1e30
        best_nx = 0.0; best_ny = 0.0; best_nz = 0.0
        best_obs_i = -1
        best_kind = 3

        for obs_i in range(N_obs):
            t_ob, onx, ony, onz = _ray_aabb(
                px, py, pz, dx, dy, dz,
                obs_min[obs_i,0], obs_min[obs_i,1], obs_min[obs_i,2],
                obs_max[obs_i,0], obs_max[obs_i,1], obs_max[obs_i,2],
                1e-5
            )
            if t_ob < best_t:
                best_t = t_ob
                best_nx = onx; best_ny = ony; best_nz = onz
                best_obs_i = obs_i
                best_kind = 0

        if t_fl < best_t:
            best_t = t_fl
            best_nx = fnx; best_ny = fny; best_nz = fnz
            best_obs_i = -1
            best_kind = 1

        if t_ex <= best_t:
            best_t = t_ex
            best_kind = 2

        if best_kind == 2 or best_t >= 1e29:
            if best_t < 1e29:
                pos_out[n_pts, ray_id, 0] = px + best_t*dx
                pos_out[n_pts, ray_id, 1] = py + best_t*dy
                pos_out[n_pts, ray_id, 2] = pz + best_t*dz

                dir_out[n_pts, ray_id, 0] = dx
                dir_out[n_pts, ray_id, 1] = dy
                dir_out[n_pts, ray_id, 2] = dz

                step_powers[n_pts, ray_id] = power
                n_pts += 1

            power_out[ray_id] = power
            n_pts_out[ray_id] = n_pts
            return

        hx = px + best_t*dx
        hy = py + best_t*dy
        hz = pz + best_t*dz

        dist = best_t if best_t > 1e-9 else 1e-9
        power -= 20.0 * math.log10(dist) + fspl_c

        surf_roughness = obs_roughness[best_obs_i] if best_obs_i >= 0 else 0.0

        r1, rng = _rand01(rng)
        r2, rng = _rand01(rng)

        dx_new, dy_new, dz_new, refl = _bounce(
            dx, dy, dz,
            best_nx, best_ny, best_nz,
            surf_roughness,
            EPSILON_CONCRETO,
            r1, r2
        )

        if refl < 1e-6:
            refl = 1e-6

        power += 10.0 * math.log10(refl)

        pos_out[n_pts, ray_id, 0] = hx
        pos_out[n_pts, ray_id, 1] = hy
        pos_out[n_pts, ray_id, 2] = hz

        dir_out[n_pts, ray_id, 0] = dx_new
        dir_out[n_pts, ray_id, 1] = dy_new
        dir_out[n_pts, ray_id, 2] = dz_new

        step_powers[n_pts, ray_id] = power
        n_pts += 1

        if power <= noise_floor:
            n_pts_out[ray_id] = n_pts
            power_out[ray_id] = power
            return

        dx = dx_new; dy = dy_new; dz = dz_new
        px = hx + 1e-4 * dx
        py = hy + 1e-4 * dy
        pz = hz + 1e-4 * dz

    n_pts_out[ray_id] = n_pts
    power_out[ray_id] = power

   

# ── mini_trace_kernel ─────────────────────────────────────────────────────

@cuda.jit
def mini_trace_kernel(
    reached_rx,
    power_out,
    arr_dir_out,
    sample_dir_out,
    pos_out,
    n_pts_out,
    hit_pts,
    v_in_dirs,
    n_uav_normals,
    init_powers,
    obs_min, obs_max,
    obs_roughness,
    rx_pos, box_min, box_max,
    rx_rad, n_post, noise_floor, uav_roughness, n_samp, fspl_c, seed_offset,
):
    tid = cuda.grid(1)
    N_hits  = hit_pts.shape[0]
    N_total = N_hits * n_samp
    if tid >= N_total:
        return

    hit_id  = tid // n_samp
    samp_id = tid % n_samp

    hx = hit_pts[hit_id, 0]; hy = hit_pts[hit_id, 1]; hz = hit_pts[hit_id, 2]
    vx = v_in_dirs[hit_id, 0]; vy = v_in_dirs[hit_id, 1]; vz = v_in_dirs[hit_id, 2]
    nx = n_uav_normals[hit_id, 0]; ny = n_uav_normals[hit_id, 1]; nz = n_uav_normals[hit_id, 2]

    EPSILON_CONCRETO = 7.0

    rng = ((seed_offset * 6364136223846793005 + tid * 1442695040888963407 + 1) & 0x7FFFFFFF)
    if rng == 0:
        rng = 1

    if samp_id == 0:
        dx, dy, dz, _ = _bounce(vx, vy, vz, nx, ny, nz, 0.0, EPSILON_CONCRETO, 0.0, 0.0)
    else:
        r1, rng = _rand01(rng)
        r2, rng = _rand01(rng)
        dx, dy, dz, _ = _bounce(vx, vy, vz, nx, ny, nz, uav_roughness, EPSILON_CONCRETO, r1, r2)

    sample_dir_out[tid, 0] = dx
    sample_dir_out[tid, 1] = dy
    sample_dir_out[tid, 2] = dz

    px = hx + 1e-4*dx
    py = hy + 1e-4*dy
    pz = hz + 1e-4*dz

    power = init_powers[hit_id]
    N_obs = obs_min.shape[0]

    pos_out[0, tid, 0] = px
    pos_out[0, tid, 1] = py
    pos_out[0, tid, 2] = pz
    n_pts = 1

    for _bounce_i in range(n_post):

        t_rx = _ray_sphere(px, py, pz, dx, dy, dz,
                        rx_pos[0], rx_pos[1], rx_pos[2], rx_rad, 1e-5)

        t_fl, fnx, fny, fnz = _ray_floor(
            px, py, pz, dx, dy, dz,
            box_min[2], box_min[0], box_min[1],
            box_max[0], box_max[1], 1e-5
        )

        t_ex = _domain_exit(
            px, py, pz, dx, dy, dz,
            box_min[0], box_min[1],
            box_max[0], box_max[1], box_max[2], 1e-5
        )

        best_t = 1e30
        best_nx = 0.0; best_ny = 0.0; best_nz = 0.0
        best_obs_i = -1
        best_kind = 3

        for obs_i in range(N_obs):
            t_ob, onx, ony, onz = _ray_aabb(
                px, py, pz, dx, dy, dz,
                obs_min[obs_i,0], obs_min[obs_i,1], obs_min[obs_i,2],
                obs_max[obs_i,0], obs_max[obs_i,1], obs_max[obs_i,2],
                1e-5
            )
            if t_ob < best_t:
                best_t = t_ob
                best_nx = onx; best_ny = ony; best_nz = onz
                best_obs_i = obs_i
                best_kind = 0

        if t_fl < best_t:
            best_t = t_fl
            best_nx = fnx; best_ny = fny; best_nz = fnz
            best_obs_i = -1
            best_kind = 1

        if t_ex <= best_t:
            best_t = t_ex
            best_kind = 2

        if t_rx < best_t and t_rx < 1e29:
            dist = t_rx if t_rx > 1e-9 else 1e-9
            power -= 20.0 * math.log10(dist) + fspl_c

            hx2 = px + t_rx*dx
            hy2 = py + t_rx*dy
            hz2 = pz + t_rx*dz

            pos_out[n_pts, tid, 0] = hx2
            pos_out[n_pts, tid, 1] = hy2
            pos_out[n_pts, tid, 2] = hz2
            n_pts += 1

            power_out[tid] = power
            n_pts_out[tid] = n_pts

            arr_dir_out[tid, 0] = dx
            arr_dir_out[tid, 1] = dy
            arr_dir_out[tid, 2] = dz

            if power > noise_floor:
                reached_rx[tid] = 1
            return

        if best_kind == 2 or best_t >= 1e29:
            if best_t < 1e29:
                pos_out[n_pts, tid, 0] = px + best_t*dx
                pos_out[n_pts, tid, 1] = py + best_t*dy
                pos_out[n_pts, tid, 2] = pz + best_t*dz
                n_pts += 1

            power_out[tid] = power
            n_pts_out[tid] = n_pts
            return

        hx2 = px + best_t*dx
        hy2 = py + best_t*dy
        hz2 = pz + best_t*dz

        dist = best_t if best_t > 1e-9 else 1e-9
        power -= 20.0 * math.log10(dist) + fspl_c

        surf_roughness = obs_roughness[best_obs_i] if best_obs_i >= 0 else 0.0

        r1, rng = _rand01(rng)
        r2, rng = _rand01(rng)

        dx, dy, dz, refl = _bounce(
            dx, dy, dz,
            best_nx, best_ny, best_nz,
            surf_roughness,
            EPSILON_CONCRETO,
            r1, r2
        )

        if refl < 1e-6:
            refl = 1e-6

        power += 10.0 * math.log10(refl)

        pos_out[n_pts, tid, 0] = hx2
        pos_out[n_pts, tid, 1] = hy2
        pos_out[n_pts, tid, 2] = hz2
        n_pts += 1

        if power <= noise_floor:
            n_pts_out[tid] = n_pts
            power_out[tid] = power
            return

        px = hx2 + 1e-4*dx
        py = hy2 + 1e-4*dy
        pz = hz2 + 1e-4*dz

    n_pts_out[tid] = n_pts
    power_out[tid] = power
