"""
Usage
-----
    python script/main.py --physics --rays 100000 --frames 5
    python script/main.py --physics --rays 200000 --frames 10 --roughness 0.3
    python script/main.py --physics --no_plot --out_csv results.csv
"""
import sys, pathlib, argparse, time, os
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.core.scene.streets   import make_street_scene
from src.core.scene.domain    import Obstacle
from src.core.cache           import get_or_compute
from src.core.uav.apply_uav  import apply_uav
from src.outputs.visualizer   import plot_from_static, make_frame_rays
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
            dx = min(abs(uav_pos[0]-obs.box_min[0]), abs(uav_pos[0]-obs.box_max[0]))
            dy = min(abs(uav_pos[1]-obs.box_min[1]), abs(uav_pos[1]-obs.box_max[1]))
            dz = min(abs(uav_pos[2]-obs.box_min[2]), abs(uav_pos[2]-obs.box_max[2]))
            m = min(dx,dy,dz)
            if m==dx:   normal[0] = 1. if uav_pos[0]>(obs.box_min[0]+obs.box_max[0])/2 else -1.
            elif m==dy: normal[1] = 1. if uav_pos[1]>(obs.box_min[1]+obs.box_max[1])/2 else -1.
            else:       normal[2] = 1. if uav_pos[2]>(obs.box_min[2]+obs.box_max[2])/2 else -1.
            return True, normal
    return False, None


def main():
    p = argparse.ArgumentParser(description="multipath-passive-radar-sim — Urban UAV Trajectory")

    # Geometry
    p.add_argument("--rays",      type=int,   default=100_000)
    p.add_argument("--bounces",   type=int,   default=10)
    p.add_argument("--domain_x",  type=float, default=200.)
    p.add_argument("--domain_y",  type=float, default=200.)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--bld_height",type=float, default=30.)
    p.add_argument("--tall_frac", type=float, default=0.35)
    p.add_argument("--simple",    action="store_true")

    # TX / RX
    p.add_argument("--tx_x",  type=float, default=150.); p.add_argument("--tx_y",  type=float, default=150.)
    p.add_argument("--tx_z",  type=float, default=55.)
    p.add_argument("--rx_x",  type=float, default=10.);  p.add_argument("--rx_y",  type=float, default=10.)
    p.add_argument("--rx_z",  type=float, default=35.)
    p.add_argument("--radius",type=float, default=15.)

    # UAV
    p.add_argument("--uav_x",  type=float, default=91.); p.add_argument("--uav_y",  type=float, default=50.)
    p.add_argument("--uav_z",  type=float, default=65.)
    p.add_argument("--uav_rad",type=float, default=1.)
    p.add_argument("--uav_vx", type=float, default=3.);  p.add_argument("--uav_vy", type=float, default=1.)
    p.add_argument("--uav_vz", type=float, default=0.)
    p.add_argument("--frames", type=int,   default=5)
    p.add_argument("--dt",     type=float, default=1.)

    # Physics
    p.add_argument("--physics",    action="store_true")
    p.add_argument("--tx_power",   type=float, default=50.)
    p.add_argument("--freq",       type=float, default=700e6)
    p.add_argument("--temp",       type=float, default=30.)
    p.add_argument("--bw",         type=float, default=20e6)
    p.add_argument("--roughness",  type=float, default=0.0)
    p.add_argument("--enable_dr",  action="store_true")
    p.add_argument("--dyn_range",  type=float, default=50.)
    p.add_argument("--agc",        action="store_true")

    # Output
    p.add_argument("--no_plot",    action="store_true")
    p.add_argument("--out_csv",    type=str, default="")
    p.add_argument("--cell_size",  type=float, default=5.)

    args = p.parse_args()

    print("\n" + "="*52)
    print("  multipath-passive-radar-sim — Kinematic UAV Trajectory (v4 GPU)")
    print("="*52)

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene = make_street_scene(
        domain_x=args.domain_x, domain_y=args.domain_y,
        tx_pos=(args.tx_x, args.tx_y, args.tx_z), tx_power_w=args.tx_power,
        rx_pos=(args.rx_x, args.rx_y, args.rx_z),
        uav_pos=(args.uav_x, args.uav_y, args.uav_z),
        uav_vel=(args.uav_vx, args.uav_vy, args.uav_vz),
        uav_radius=args.uav_rad, frequency=args.freq,
        n_rays=args.rays, n_max=args.bounces,
        seed=args.seed, bld_height=args.bld_height, tall_fraction=args.tall_frac,
    )
    scene.use_physics    = args.physics
    scene.roughness      = args.roughness
    scene.bandwidth_hz   = args.bw
    scene.temperature_c  = args.temp
    scene.receiver.radius = args.radius
    scene.uav_roughness  = 0.4
    scene.n_samples_uav  = 8

    if args.simple:
        scene.obstacles = [
            Obstacle(np.array([40.,40.,0.]), np.array([60.,100.,40.])),
            Obstacle(np.array([100.,40.,0.]), np.array([120.,100.,40.])),
        ]

    if args.physics:
        print(f"  TX={scene.transmitters[0].tx_power_dbm:.1f}dBm  "
              f"noise={scene.noise_floor_dbm:.1f}dBm  "
              f"budget={scene.transmitters[0].tx_power_dbm-scene.noise_floor_dbm:.0f}dB")

    # ── Precompute (cached) ───────────────────────────────────────────────────
    print(f"\n[*] Precompute ({args.rays:,} rays) — checking cache...")
    static = get_or_compute(scene, seed=args.seed, cell_size=args.cell_size,
                             cache_dir=_CACHE_DIR, verbose=True)
    print(f"  anchors={len(static.anchors)}  hash={static.spatial_hash.total_entries:,}")

    # Dynamic range baseline floor
    fixed_dr_floor = -np.inf
    if args.physics and args.enable_dr and static.anchors:
        bl_max = max(r.power_dbm for r in static.anchors)
        fixed_dr_floor = bl_max - args.dyn_range
        print(f"  DR floor (fixed): {fixed_dr_floor:.1f} dBm")

    sim_params = dict(
        domain_x=args.domain_x, domain_y=args.domain_y,
        bld_height=args.bld_height, tall_frac=args.tall_frac,
        seed=args.seed, roughness=args.roughness,
        temp=args.temp, bw=args.bw, tx_power=args.tx_power,
        enable_dr=args.enable_dr, agc=args.agc, dyn_range=args.dyn_range,
        tx_pos_x=args.tx_x, tx_pos_y=args.tx_y, tx_pos_z=args.tx_z,
        rx_pos_x=args.rx_x, rx_pos_y=args.rx_y, rx_pos_z=args.rx_z,
    )

    # ── Per-frame loop ────────────────────────────────────────────────────────
    uav_pos = np.array([args.uav_x, args.uav_y, args.uav_z])
    uav_vel = np.array([args.uav_vx, args.uav_vy, args.uav_vz])
    uav_spd = float(np.linalg.norm(uav_vel[:2]))

    fv_list, fo_list, fu_list, us_list, vel_list = [], [], [], [], []
    dfs = []
    t0  = time.time()

    print()
    for step in range(args.frames):
        # Kinematics with collision
        proposed = uav_pos + uav_vel * args.dt
        is_col, normal = check_uav_collision(proposed, args.uav_rad, scene.obstacles)
        if is_col:
            v_r = uav_vel - 2.*np.dot(uav_vel,normal)*normal
            v_r[0] += np.random.uniform(-1.5,1.5)
            v_r[1] += np.random.uniform(-1.5,1.5)
            v_r[2] = 0.
            spd_new = float(np.linalg.norm(v_r[:2]))
            if spd_new > 1e-6: v_r = v_r*(uav_spd/spd_new)
            uav_vel = v_r
            proposed = uav_pos + uav_vel * args.dt
        uav_pos = proposed
        scene.uav.position = uav_pos.copy()
        scene.uav.velocity = uav_vel.copy()

        vis, occ, bounces = apply_uav(static, scene.uav, scene)

        # Dynamic range filter
        if args.physics and args.enable_dr:
            active = vis + bounces
            if active:
                dr_floor = (max(r.power_dbm for r in active) - args.dyn_range
                            if args.agc else fixed_dr_floor)
                vis     = [r for r in vis     if r.power_dbm >= dr_floor]
                bounces = [r for r in bounces if r.power_dbm >= dr_floor]
                dr_occ  = [r for r in (vis+bounces) if r.power_dbm < dr_floor]
                occ     = occ + dr_occ

        frame_rays = vis + occ + bounces
        fv_list.append(vis); fo_list.append(occ); fu_list.append(bounces)
        us_list.append(uav_pos.copy()); vel_list.append(uav_vel.copy())

        df = to_dataframe(frame_rays, instance_id=f"sim_{args.seed}_{step}",
                          time_step=step, uav=scene.uav, params=sim_params)
        dfs.append(df)

        print(f"  frame {step+1}/{args.frames}: "
              f"pos=({uav_pos[0]:.1f},{uav_pos[1]:.1f},{uav_pos[2]:.1f})  "
              f"vis={len(vis)}  occ={len(occ)}  UAV_bounces={len(bounces)}")

    print(f"\n  Total: {time.time()-t0:.2f}s")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if args.out_csv and dfs:
        master = pd.concat(dfs, ignore_index=True)
        exists = os.path.isfile(args.out_csv)
        master.to_csv(args.out_csv, mode='a', header=not exists, index=False)
        print(f"  Saved {len(master)} rows → {args.out_csv}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        fig = plot_from_static(
            scene, fv_list, fo_list, fu_list, us_list,
            uav_vels=vel_list, dt=args.dt,
            title=f"multipath-passive-radar-sim main.py — {args.freq/1e6:.0f}MHz",
        )
        fig.show()


if __name__ == "__main__":
    main()
