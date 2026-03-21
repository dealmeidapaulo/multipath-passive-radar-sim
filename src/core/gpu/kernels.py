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

    # ── Scatter — SINGLE implementation used by both kernels ─────────────────

    @cuda.jit(device=True)
    def _scatter(dx, dy, dz, nx, ny, nz, roughness, r1, r2):
        """
        Specular + cosine-weighted Lambertian blend.
        r1, r2 ∈ [0,1) from the kernel RNG.
        Returns normalised outgoing direction (ox, oy, oz).
        Physics identical to v2 scatter_ray.
        """
        # Specular
        dot = dx*nx + dy*ny + dz*nz
        sx = dx - 2.0*dot*nx
        sy = dy - 2.0*dot*ny
        sz = dz - 2.0*dot*nz
        inv = 1.0 / math.sqrt(sx*sx + sy*sy + sz*sz + 1e-30)
        sx *= inv; sy *= inv; sz *= inv

        if roughness < 1e-5:
            return sx, sy, sz

        # Cosine-weighted hemisphere sample
        phi   = 2.0 * math.pi * r1
        sin_t = math.sqrt(r2)
        cos_t = math.sqrt(1.0 - r2)

        # ONB (Gram-Schmidt)
        if math.fabs(nx) < 0.9:
            hx = 1.0; hy = 0.0; hz = 0.0
        else:
            hx = 0.0; hy = 1.0; hz = 0.0
        tx = hy*nz - hz*ny; ty = hz*nx - hx*nz; tz = hx*ny - hy*nx
        inv = 1.0 / math.sqrt(tx*tx + ty*ty + tz*tz + 1e-30)
        tx *= inv; ty *= inv; tz *= inv
        bx = ny*tz - nz*ty; by = nz*tx - nx*tz; bz = nx*ty - ny*tx

        dfx = sin_t*math.cos(phi)*tx + sin_t*math.sin(phi)*bx + cos_t*nx
        dfy = sin_t*math.cos(phi)*ty + sin_t*math.sin(phi)*by + cos_t*ny
        dfz = sin_t*math.cos(phi)*tz + sin_t*math.sin(phi)*bz + cos_t*nz

        ro = roughness
        ox = (1.0-ro)*sx + ro*dfx
        oy = (1.0-ro)*sy + ro*dfy
        oz = (1.0-ro)*sz + ro*dfz
        inv = 1.0 / math.sqrt(ox*ox + oy*oy + oz*oz + 1e-30)
        ox *= inv; oy *= inv; oz *= inv

        if ox*nx + oy*ny + oz*nz <= 0.0:
            inv = 1.0 / math.sqrt(dfx*dfx + dfy*dfy + dfz*dfz + 1e-30)
            return dfx*inv, dfy*inv, dfz*inv
        return ox, oy, oz

    # ── trace_all_kernel ─────────────────────────────────────────────────────

    @cuda.jit
    def trace_all_kernel(
        pos_out,       # float32[N_max+2, N_rays, 3]
        dir_out,       # float32[N_max+2, N_rays, 3]
        step_powers,   # float32[N_max+2, N_rays]
        power_out,     # float32[N_rays]
        n_pts_out,     # int32[N_rays]
        reached_rx,    # int32[N_rays]
        dirs_in,       # float32[N_rays, 3]
        tx_pos,        # float32[3]
        obs_min,       # float32[N_obs, 3]
        obs_max,       # float32[N_obs, 3]
        rx_pos,        # float32[3]
        box_min,       # float32[3]
        box_max,       # float32[3]
        rx_rad, n_max, init_power, noise_floor, roughness, fspl_c, seed_offset,
    ):
        ray_id = cuda.grid(1)
        if ray_id >= dirs_in.shape[0]: return
        N_obs = obs_min.shape[0]

        px = tx_pos[0]; py = tx_pos[1]; pz = tx_pos[2]
        dx = dirs_in[ray_id, 0]; dy = dirs_in[ray_id, 1]; dz = dirs_in[ray_id, 2]
        inv = 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz + 1e-30)
        dx *= inv; dy *= inv; dz *= inv

        power = init_power
        rng = ((seed_offset * 6364136223846793005 + ray_id * 1442695040888963407 + 1) & 0x7FFFFFFF)
        if rng == 0: rng = 1

        pos_out[0, ray_id, 0] = px; pos_out[0, ray_id, 1] = py; pos_out[0, ray_id, 2] = pz
        dir_out[0, ray_id, 0] = dx; dir_out[0, ray_id, 1] = dy; dir_out[0, ray_id, 2] = dz
        step_powers[0, ray_id] = power
        n_pts = 1

        for _bounce in range(n_max):
            t_rx = _ray_sphere(px, py, pz, dx, dy, dz,
                               rx_pos[0], rx_pos[1], rx_pos[2], rx_rad, 1e-5)
            t_fl, fnx, fny, fnz = _ray_floor(
                px, py, pz, dx, dy, dz,
                box_min[2], box_min[0], box_min[1], box_max[0], box_max[1], 1e-5)
            t_ex = _domain_exit(px, py, pz, dx, dy, dz,
                                box_min[0], box_min[1], box_max[0], box_max[1], box_max[2], 1e-5)

            best_t = 1e30; best_nx = 0.0; best_ny = 0.0; best_nz = 0.0; best_kind = 3
            for obs_i in range(N_obs):
                t_ob, onx, ony, onz = _ray_aabb(
                    px, py, pz, dx, dy, dz,
                    obs_min[obs_i,0], obs_min[obs_i,1], obs_min[obs_i,2],
                    obs_max[obs_i,0], obs_max[obs_i,1], obs_max[obs_i,2], 1e-5)
                if t_ob < best_t:
                    best_t = t_ob; best_nx = onx; best_ny = ony; best_nz = onz; best_kind = 0
            if t_fl < best_t:
                best_t = t_fl; best_nx = fnx; best_ny = fny; best_nz = fnz; best_kind = 1
            if t_ex <= best_t:
                best_t = t_ex; best_kind = 2

            # RX reached
            if t_rx < best_t and t_rx < 1e29:
                dist = t_rx if t_rx > 1e-9 else 1e-9
                power -= 20.0 * math.log10(dist) + fspl_c
                hx = px + t_rx*dx; hy = py + t_rx*dy; hz = pz + t_rx*dz
                pos_out[n_pts, ray_id, 0] = hx; pos_out[n_pts, ray_id, 1] = hy; pos_out[n_pts, ray_id, 2] = hz
                dir_out[n_pts, ray_id, 0] = dx; dir_out[n_pts, ray_id, 1] = dy; dir_out[n_pts, ray_id, 2] = dz
                step_powers[n_pts, ray_id] = power
                n_pts += 1
                power_out[ray_id] = power; n_pts_out[ray_id] = n_pts
                if power > noise_floor: reached_rx[ray_id] = 1
                return

            # Domain exit or no hit
            if best_kind == 2 or best_t >= 1e29:
                if best_t < 1e29:
                    pos_out[n_pts, ray_id, 0] = px + best_t*dx
                    pos_out[n_pts, ray_id, 1] = py + best_t*dy
                    pos_out[n_pts, ray_id, 2] = pz + best_t*dz
                    n_pts += 1
                power_out[ray_id] = power; n_pts_out[ray_id] = n_pts
                return

            # Surface hit
            hx = px + best_t*dx; hy = py + best_t*dy; hz = pz + best_t*dz
            dist = best_t if best_t > 1e-9 else 1e-9
            power -= 20.0 * math.log10(dist) + fspl_c

            # Wall attenuation Normal(0.1, 0.03) via Box-Muller
            r1, rng = _rand01(rng); r2, rng = _rand01(rng)
            if r1 < 1e-10: r1 = 1e-10
            gauss = math.sqrt(-2.0 * math.log(r1)) * math.cos(2.0 * math.pi * r2)
            refl = 0.1 + 0.03 * gauss
            if refl < 1e-6: refl = 1e-6
            if refl > 1.0:  refl = 1.0
            power += 10.0 * math.log10(refl)

            pos_out[n_pts, ray_id, 0] = hx; pos_out[n_pts, ray_id, 1] = hy; pos_out[n_pts, ray_id, 2] = hz
            step_powers[n_pts, ray_id] = power

            if power <= noise_floor:
                n_pts += 1; n_pts_out[ray_id] = n_pts; power_out[ray_id] = power; return

            r1, rng = _rand01(rng); r2, rng = _rand01(rng)
            dx, dy, dz = _scatter(dx, dy, dz, best_nx, best_ny, best_nz, roughness, r1, r2)
            dir_out[n_pts, ray_id, 0] = dx; dir_out[n_pts, ray_id, 1] = dy; dir_out[n_pts, ray_id, 2] = dz
            n_pts += 1
            px = hx + 1e-4*dx; py = hy + 1e-4*dy; pz = hz + 1e-4*dz

        n_pts_out[ray_id] = n_pts; power_out[ray_id] = power

    # ── mini_trace_kernel ─────────────────────────────────────────────────────

    @cuda.jit
    def mini_trace_kernel(
        # Outputs — indexed by tid = hit_id * n_samp + samp_id
        reached_rx,      # int32[N_total]
        power_out,       # float32[N_total]
        arr_dir_out,     # float32[N_total, 3]   arrival dir at RX
        sample_dir_out,  # float32[N_total, 3]   outgoing dir from UAV (for Doppler)
        pos_out,         # float32[n_post+2, N_total, 3]
        n_pts_out,       # int32[N_total]
        # Inputs — hit geometry, one entry per UAV hit
        hit_pts,         # float32[N_hits, 3]
        v_in_dirs,       # float32[N_hits, 3]   incoming direction TO UAV
        n_uav_normals,   # float32[N_hits, 3]   outward normal at UAV surface
        init_powers,     # float32[N_hits]       power at hit (after FSPL + RCS)
        # Scene
        obs_min, obs_max,
        rx_pos, box_min, box_max,
        # Scalars
        rx_rad, n_post, noise_floor, roughness, uav_roughness, n_samp, fspl_c, seed_offset,
    ):
        """
        One thread per (hit × sample).
        Sample 0: specular reflection.
        Samples 1..n_samp-1: diffuse scatter using kernel RNG (uav_roughness).
        Direction generation and post-UAV tracing happen entirely on GPU —
        no scatter_ray on CPU is needed.
        """
        tid = cuda.grid(1)
        N_hits  = hit_pts.shape[0]
        N_total = N_hits * n_samp
        if tid >= N_total: return

        hit_id  = tid // n_samp
        samp_id = tid % n_samp

        hx = hit_pts[hit_id, 0]; hy = hit_pts[hit_id, 1]; hz = hit_pts[hit_id, 2]
        vx = v_in_dirs[hit_id, 0]; vy = v_in_dirs[hit_id, 1]; vz = v_in_dirs[hit_id, 2]
        nx = n_uav_normals[hit_id, 0]; ny = n_uav_normals[hit_id, 1]; nz = n_uav_normals[hit_id, 2]

        # ── Outgoing direction ────────────────────────────────────────────────
        if samp_id == 0:
            # Specular
            dot = vx*nx + vy*ny + vz*nz
            dx = vx - 2.0*dot*nx; dy = vy - 2.0*dot*ny; dz = vz - 2.0*dot*nz
            inv = 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz + 1e-30)
            dx *= inv; dy *= inv; dz *= inv
        else:
            # Diffuse — deterministic per thread
            rng0 = ((seed_offset * 6364136223846793005 + tid * 1442695040888963407 + 1) & 0x7FFFFFFF)
            if rng0 == 0: rng0 = 1
            r1, rng0 = _rand01(rng0); r2, _ = _rand01(rng0)
            dx, dy, dz = _scatter(vx, vy, vz, nx, ny, nz, uav_roughness, r1, r2)

        # Store outgoing direction for Doppler (CPU reads this after kernel)
        sample_dir_out[tid, 0] = dx; sample_dir_out[tid, 1] = dy; sample_dir_out[tid, 2] = dz

        # ── Trace from UAV hit point toward RX ────────────────────────────────
        px = hx + 1e-4*dx; py = hy + 1e-4*dy; pz = hz + 1e-4*dz
        power = init_powers[hit_id]
        N_obs = obs_min.shape[0]

        rng = ((seed_offset * 6364136223846793005 + (tid + 99991) * 1442695040888963407 + 1) & 0x7FFFFFFF)
        if rng == 0: rng = 1

        pos_out[0, tid, 0] = px; pos_out[0, tid, 1] = py; pos_out[0, tid, 2] = pz
        n_pts = 1

        for _bounce in range(n_post):
            t_rx = _ray_sphere(px, py, pz, dx, dy, dz,
                               rx_pos[0], rx_pos[1], rx_pos[2], rx_rad, 1e-5)
            t_fl, fnx, fny, fnz = _ray_floor(
                px, py, pz, dx, dy, dz,
                box_min[2], box_min[0], box_min[1], box_max[0], box_max[1], 1e-5)
            t_ex = _domain_exit(px, py, pz, dx, dy, dz,
                                box_min[0], box_min[1], box_max[0], box_max[1], box_max[2], 1e-5)

            best_t = 1e30; best_nx = 0.0; best_ny = 0.0; best_nz = 0.0; best_kind = 3
            for obs_i in range(N_obs):
                t_ob, onx, ony, onz = _ray_aabb(
                    px, py, pz, dx, dy, dz,
                    obs_min[obs_i,0], obs_min[obs_i,1], obs_min[obs_i,2],
                    obs_max[obs_i,0], obs_max[obs_i,1], obs_max[obs_i,2], 1e-5)
                if t_ob < best_t:
                    best_t = t_ob; best_nx = onx; best_ny = ony; best_nz = onz; best_kind = 0
            if t_fl < best_t:
                best_t = t_fl; best_nx = fnx; best_ny = fny; best_nz = fnz; best_kind = 1
            if t_ex <= best_t:
                best_t = t_ex; best_kind = 2

            # RX reached
            if t_rx < best_t and t_rx < 1e29:
                dist = t_rx if t_rx > 1e-9 else 1e-9
                power -= 20.0 * math.log10(dist) + fspl_c
                hx2 = px + t_rx*dx; hy2 = py + t_rx*dy; hz2 = pz + t_rx*dz
                pos_out[n_pts, tid, 0] = hx2; pos_out[n_pts, tid, 1] = hy2; pos_out[n_pts, tid, 2] = hz2
                n_pts += 1
                power_out[tid] = power; n_pts_out[tid] = n_pts
                arr_dir_out[tid, 0] = dx; arr_dir_out[tid, 1] = dy; arr_dir_out[tid, 2] = dz
                if power > noise_floor: reached_rx[tid] = 1
                return

            if best_kind == 2 or best_t >= 1e29:
                if best_t < 1e29:
                    pos_out[n_pts, tid, 0] = px + best_t*dx
                    pos_out[n_pts, tid, 1] = py + best_t*dy
                    pos_out[n_pts, tid, 2] = pz + best_t*dz
                    n_pts += 1
                power_out[tid] = power; n_pts_out[tid] = n_pts
                return

            hx2 = px + best_t*dx; hy2 = py + best_t*dy; hz2 = pz + best_t*dz
            dist = best_t if best_t > 1e-9 else 1e-9
            power -= 20.0 * math.log10(dist) + fspl_c

            r1, rng = _rand01(rng); r2, rng = _rand01(rng)
            if r1 < 1e-10: r1 = 1e-10
            gauss = math.sqrt(-2.0 * math.log(r1)) * math.cos(2.0 * math.pi * r2)
            refl = 0.1 + 0.03 * gauss
            if refl < 1e-6: refl = 1e-6
            if refl > 1.0:  refl = 1.0
            power += 10.0 * math.log10(refl)

            pos_out[n_pts, tid, 0] = hx2; pos_out[n_pts, tid, 1] = hy2; pos_out[n_pts, tid, 2] = hz2
            n_pts += 1

            if power <= noise_floor:
                n_pts_out[tid] = n_pts; power_out[tid] = power; return

            r1, rng = _rand01(rng); r2, rng = _rand01(rng)
            dx, dy, dz = _scatter(dx, dy, dz, best_nx, best_ny, best_nz, roughness, r1, r2)
            px = hx2 + 1e-4*dx; py = hy2 + 1e-4*dy; pz = hz2 + 1e-4*dz

        n_pts_out[tid] = n_pts; power_out[tid] = power

    # ── Spatial-hash kernels ─────────────────────────────────────────────────

    @cuda.jit
    def _count_kernel(pos_out, n_pts, counts, cs_inv, bmin, NX, NY, NZ):
        """Pass 1: count segment–cell overlaps. One thread per ray."""
        ray_id = cuda.grid(1)
        if ray_id >= pos_out.shape[1]: return
        np_r = n_pts[ray_id]
        for seg_id in range(np_r - 1):
            ax = pos_out[seg_id,   ray_id, 0]; ay = pos_out[seg_id,   ray_id, 1]; az = pos_out[seg_id,   ray_id, 2]
            bx = pos_out[seg_id+1, ray_id, 0]; by = pos_out[seg_id+1, ray_id, 1]; bz = pos_out[seg_id+1, ray_id, 2]
            min_x = ax if ax < bx else bx; max_x = ax if ax > bx else bx
            min_y = ay if ay < by else by; max_y = ay if ay > by else by
            min_z = az if az < bz else bz; max_z = az if az > bz else bz
            lo_x = int(math.floor((min_x - bmin[0]) * cs_inv)); hi_x = int(math.floor((max_x - bmin[0]) * cs_inv))
            lo_y = int(math.floor((min_y - bmin[1]) * cs_inv)); hi_y = int(math.floor((max_y - bmin[1]) * cs_inv))
            lo_z = int(math.floor((min_z - bmin[2]) * cs_inv)); hi_z = int(math.floor((max_z - bmin[2]) * cs_inv))
            if lo_x < 0: lo_x = 0
            if lo_y < 0: lo_y = 0
            if lo_z < 0: lo_z = 0
            if hi_x >= NX: hi_x = NX - 1
            if hi_y >= NY: hi_y = NY - 1
            if hi_z >= NZ: hi_z = NZ - 1
            for cx in range(lo_x, hi_x + 1):
                for cy in range(lo_y, hi_y + 1):
                    for cz in range(lo_z, hi_z + 1):
                        cuda.atomic.add(counts, cx + cy*NX + cz*NX*NY, 1)

    @cuda.jit
    def _fill_kernel(pos_out, n_pts, fill_ptr, flat_ray_ids, flat_seg_ids,
                     cs_inv, bmin, NX, NY, NZ):
        """Pass 2: write (ray_id, seg_id) into flat arrays."""
        ray_id = cuda.grid(1)
        if ray_id >= pos_out.shape[1]: return
        np_r = n_pts[ray_id]
        for seg_id in range(np_r - 1):
            ax = pos_out[seg_id,   ray_id, 0]; ay = pos_out[seg_id,   ray_id, 1]; az = pos_out[seg_id,   ray_id, 2]
            bx = pos_out[seg_id+1, ray_id, 0]; by = pos_out[seg_id+1, ray_id, 1]; bz = pos_out[seg_id+1, ray_id, 2]
            min_x = ax if ax < bx else bx; max_x = ax if ax > bx else bx
            min_y = ay if ay < by else by; max_y = ay if ay > by else by
            min_z = az if az < bz else bz; max_z = az if az > bz else bz
            lo_x = int(math.floor((min_x - bmin[0]) * cs_inv)); hi_x = int(math.floor((max_x - bmin[0]) * cs_inv))
            lo_y = int(math.floor((min_y - bmin[1]) * cs_inv)); hi_y = int(math.floor((max_y - bmin[1]) * cs_inv))
            lo_z = int(math.floor((min_z - bmin[2]) * cs_inv)); hi_z = int(math.floor((max_z - bmin[2]) * cs_inv))
            if lo_x < 0: lo_x = 0
            if lo_y < 0: lo_y = 0
            if lo_z < 0: lo_z = 0
            if hi_x >= NX: hi_x = NX - 1
            if hi_y >= NY: hi_y = NY - 1
            if hi_z >= NZ: hi_z = NZ - 1
            for cx in range(lo_x, hi_x + 1):
                for cy in range(lo_y, hi_y + 1):
                    for cz in range(lo_z, hi_z + 1):
                        pos = cuda.atomic.add(fill_ptr, cx + cy*NX + cz*NX*NY, 1)
                        flat_ray_ids[pos] = ray_id
                        flat_seg_ids[pos] = seg_id
