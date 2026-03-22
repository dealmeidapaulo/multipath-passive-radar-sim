"""
Usage
-----
    python script/batch_runner.py --trajectories 20 --frames_per_traj 5
    python script/batch_runner.py --trajectories 100 --rays 50000 --out_csv dataset.csv
"""
import sys, pathlib, argparse, time, os
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.core.scene.streets   import make_street_scene
from src.core.cache           import get_or_compute
from src.core.uav.apply_uav  import apply_uav
from src.tracer               import get_covered_uav_spawn
from src.outputs.observables  import to_dataframe

_CACHE_DIR = _ROOT / "cache"


def check_uav_collision(uav_pos, uav_rad, obstacles):
    for obs in obstacles:
        cx = max(obs.box_min[0], min(uav_pos[0], obs.box_max[0]))
        cy = max(obs.box_min[1], min(uav_pos[1], obs.box_max[1]))
        cz = max(obs.box_min[2], min(uav_pos[2], obs.box_max[2]))
        d  = float(np.sqrt((cx-uav_pos[0])**2+(cy-uav_pos[1])**2+(cz-uav_pos[2])**2))
        if d < uav_rad:
            normal = np.zeros(3)
            dx = min(abs(uav_pos[0]-obs.box_min[0]),abs(uav_pos[0]-obs.box_max[0]))
            dy = min(abs(uav_pos[1]-obs.box_min[1]),abs(uav_pos[1]-obs.box_max[1]))
            dz = min(abs(uav_pos[2]-obs.box_min[2]),abs(uav_pos[2]-obs.box_max[2]))
            m = min(dx,dy,dz)
            if m==dx:   normal[0] = 1. if uav_pos[0]>(obs.box_min[0]+obs.box_max[0])/2 else -1.
            elif m==dy: normal[1] = 1. if uav_pos[1]>(obs.box_min[1]+obs.box_max[1])/2 else -1.
            else:       normal[2] = 1. if uav_pos[2]>(obs.box_min[2]+obs.box_max[2])/2 else -1.
            return True, normal
    return False, None


def main():
    p = argparse.ArgumentParser(description="multipath-passive-radar-sim — Batch Dataset Generator")

    p.add_argument("--out_csv",         type=str,   default="batch_dataset.csv")
    p.add_argument("--trajectories",    type=int,   default=100)
    p.add_argument("--frames_per_traj", type=int,   default=5)
    p.add_argument("--rays",            type=int,   default=50_000)
    p.add_argument("--bounces",         type=int,   default=10)
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--domain_x",        type=float, default=150.)
    p.add_argument("--domain_y",        type=float, default=150.)
    p.add_argument("--bld_height",      type=float, default=15.)
    p.add_argument("--tall_frac",       type=float, default=0.25)
    p.add_argument("--tx_x",            type=float, default=80.)
    p.add_argument("--tx_y",            type=float, default=80.)
    p.add_argument("--tx_z",            type=float, default=35.)
    p.add_argument("--rx_x",            type=float, default=0.)
    p.add_argument("--rx_y",            type=float, default=0.)
    p.add_argument("--rx_z",            type=float, default=20.)
    p.add_argument("--uav_rad",         type=float, default=1.)
    p.add_argument("--dt",              type=float, default=1.)
    p.add_argument("--freq",            type=float, default=700e6)
    p.add_argument("--tx_power",        type=float, default=250.)
    p.add_argument("--temp",            type=float, default=20.)
    p.add_argument("--bw",              type=float, default=0.5e6)
    p.add_argument("--roughness",       type=float, default=0.5)
    p.add_argument("--dyn_range",       type=float, default=50.)
    p.add_argument("--cell_size",       type=float, default=5.)

    args = p.parse_args()

    print("\n" + "="*52)
    print("  multipath-passive-radar-sim — Batch Generator (v4 GPU)")
    print("="*52)
    print(f"  {args.trajectories} trajectories × {args.frames_per_traj} frames")
    print(f"  rays={args.rays:,}  seed={args.seed}  csv={args.out_csv}")

    # ── Scene (built once) ────────────────────────────────────────────────────
    scene = make_street_scene(
        domain_x=args.domain_x, domain_y=args.domain_y,
        tx_pos=(args.tx_x, args.tx_y, args.tx_z), tx_power_w=args.tx_power,
        rx_pos=(args.rx_x, args.rx_y, args.rx_z),
        uav_pos=(args.tx_x, args.tx_y, 25.),   # placeholder
        uav_vel=(0., 0., 0.), uav_radius=args.uav_rad,
        frequency=args.freq, n_rays=args.rays, n_max=args.bounces,
        seed=args.seed, bld_height=args.bld_height, tall_fraction=args.tall_frac,
    )
    scene.use_physics    = True
    scene.roughness      = args.roughness
    scene.bandwidth_hz   = args.bw
    scene.temperature_c  = args.temp
    scene.receiver.radius = 5.
    scene.uav_roughness  = 0.4
    scene.n_samples_uav  = 8

    print(f"  TX={scene.transmitters[0].tx_power_dbm:.1f}dBm  "
          f"noise={scene.noise_floor_dbm:.1f}dBm  "
          f"buildings={len(scene.obstacles)}")

    # ── Precompute ONCE (cached) ──────────────────────────────────────────────
    print("\n[*] Precompute — checking cache...")
    static = get_or_compute(scene, seed=args.seed, cell_size=args.cell_size,
                             cache_dir=_CACHE_DIR, verbose=True)
    print(f"  anchors={len(static.anchors)}  hash={static.spatial_hash.total_entries:,}")

    sim_params = dict(
        domain_x=args.domain_x, domain_y=args.domain_y,
        bld_height=args.bld_height, tall_frac=args.tall_frac,
        seed=args.seed, roughness=args.roughness,
        temp=args.temp, bw=args.bw, tx_power=args.tx_power,
        enable_dr=False, agc=False, dyn_range=args.dyn_range,
        tx_pos_x=args.tx_x, tx_pos_y=args.tx_y, tx_pos_z=args.tx_z,
        rx_pos_x=args.rx_x, rx_pos_y=args.rx_y, rx_pos_z=args.rx_z,
    )

    # ── Trajectory loop ───────────────────────────────────────────────────────
    t_total = time.time()
    total_rows = 0

    for traj_idx in range(args.trajectories):
        rng = np.random.default_rng(args.seed * 10000 + traj_idx)

        # Smart spawn — covered zone guarantees UAV-bounce candidates
        try:
            spawn = get_covered_uav_spawn(
                static, scene.obstacles,
                uav_rad=args.uav_rad, min_segs=20,
                min_z=10., max_z=min(50., args.domain_x/4),
            )
        except RuntimeError:
            spawn = np.array([rng.uniform(5, args.domain_x-5),
                              rng.uniform(5, args.domain_y-5),
                              rng.uniform(15., 40.)])

        vel = np.array([rng.uniform(-5.,5.), rng.uniform(-5.,5.),
                        rng.uniform(-0.5,0.5)])
        spd = float(np.linalg.norm(vel))
        pos = spawn.copy()

        traj_dfs = []
        for step in range(args.frames_per_traj):
            proposed = pos + vel * args.dt
            is_col, normal = check_uav_collision(proposed, args.uav_rad, scene.obstacles)
            if is_col:
                v_r = vel - 2.*np.dot(vel,normal)*normal
                v_r += rng.uniform(-1.,1.,3)
                v_r[2] *= 0.5
                spd_new = float(np.linalg.norm(v_r))
                if spd_new > 1e-6: v_r = v_r*(spd/spd_new)
                vel = v_r
                proposed = pos + vel * args.dt
            pos = proposed
            scene.uav.position = pos.copy()
            scene.uav.velocity = vel.copy()

            vis, occ, bounces = apply_uav(static, scene.uav, scene)
            frame_rays = vis + occ + bounces
            df = to_dataframe(frame_rays,
                              instance_id=f"sim_{args.seed}_traj{traj_idx}",
                              time_step=step, uav=scene.uav, params=sim_params)
            traj_dfs.append(df)

        traj_df = pd.concat(traj_dfs, ignore_index=True)
        exists  = os.path.isfile(args.out_csv)
        traj_df.to_csv(args.out_csv, mode='a', header=not exists, index=False)
        total_rows += len(traj_df)

        if (traj_idx+1) % max(1, args.trajectories//10) == 0 or traj_idx == 0:
            elapsed = time.time()-t_total
            rate    = (traj_idx+1)/elapsed
            eta     = (args.trajectories-traj_idx-1)/max(rate,1e-6)
            print(f"  traj {traj_idx+1:4d}/{args.trajectories}  "
                  f"rows={len(traj_df):4d}  "
                  f"total={total_rows:,}  "
                  f"rate={rate:.1f}traj/s  ETA={eta:.0f}s")

    print(f"\n  Done in {time.time()-t_total:.1f}s  "
          f"Total rows: {total_rows:,}  →  {args.out_csv}")


if __name__ == "__main__":
    main()
