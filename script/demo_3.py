"""
Usage
---
    python script/demo_doppler.py
    python script/demo_doppler.py --frames 10 --out_csv doppler_sweep.csv
"""
import sys, pathlib, time
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd

from src.core.scene.domain   import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.cache          import get_or_compute
from src.core.uav.apply_uav  import apply_uav
from src.outputs.visualizer  import plot_from_static
from src.outputs.observables import to_dataframe

_CACHE_DIR = _ROOT / "cache"

TX_POS = np.array([10.,  10.,  50.])
RX_POS = np.array([140., 140., 15.])
N_RAYS = 100_000

# UAV describes an arc: starts at (40,40,45) moving +X,
# curves through (75,75,45) moving +Y, ends at (110,110,45) moving -X
def uav_state(step: int, n_frames: int):
    """Parametric arc — 180° sweep of velocity direction."""
    t     = step / max(n_frames - 1, 1)
    angle = np.pi * t                           # 0 → π
    vel   = np.array([np.cos(angle), np.sin(angle), 0.]) * 12.
    # Position: arc centered at (75,75)
    radius = 35.
    cx, cy = 75., 75.
    px = cx + radius * np.cos(np.pi/2 - angle)
    py = cy - radius * np.sin(np.pi/2 - angle) + radius
    pos = np.array([px, py, 45.])
    return pos, vel


def make_scene(uav_pos, uav_vel) -> Scene:
    box = Box(np.zeros(3), np.array([150., 150., 80.]))
    tx  = Transmitter(TX_POS, 700e6, tx_power_w=500., tx_id=0)
    rx  = Receiver(RX_POS, radius=15.)
    uav = UAV(uav_pos.copy(), uav_vel.copy(), radius=3.0)

    # Two buildings — off-diagonal, don't block TX→RX LOS
    obs = [
        Obstacle(np.array([20., 70., 0.]), np.array([55.,  120., 35.])),
        Obstacle(np.array([95., 30., 0.]), np.array([130.,  80., 30.])),
    ]

    scene = Scene(box=box, transmitters=[tx], receiver=rx,
                  uav=uav, obstacles=obs, n_rays=N_RAYS, n_max=6)
    scene.use_physics    = True
    scene.roughness      = 0.4
    scene.uav_roughness  = 0.5
    scene.n_samples_uav  = 32
    scene.bandwidth_hz   = 0.5e6
    scene.temperature_c  = 20.
    return scene


def main():
    p = argparse.ArgumentParser(description="Demo 3 — Doppler sweep")
    p.add_argument("--frames",  type=int, default=8)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--out_csv", type=str, default="")
    args = p.parse_args()

    print("\n" + "="*52)
    print("  Demo 3 — Doppler Sweep (velocity estimation)")
    print("="*52)

    pos0, vel0 = uav_state(0, args.frames)
    scene  = make_scene(pos0, vel0)

    print(f"\n[*] Precompute — checking cache...")
    static = get_or_compute(scene, seed=13, cell_size=4.,
                            cache_dir=_CACHE_DIR, verbose=True)
    print(f"  anchors={len(static.anchors)}")

    fv, fo, fu, us, vl = [], [], [], [], []
    dfs = []
    t0  = time.time()
    all_dops = []

    print(f"\n  {'frame':>5}  {'angle°':>7}  {'vx':>7} {'vy':>7}  "
          f"{'bounces':>8}  {'f_D_mean Hz':>12}")
    print("  " + "-"*60)

    for step in range(args.frames):
        pos, vel = uav_state(step, args.frames)
        angle_deg = np.degrees(np.arctan2(vel[1], vel[0]))
        scene.uav.position = pos.copy()
        scene.uav.velocity = vel.copy()

        vis, occ, bounces = apply_uav(static, scene.uav, scene)
        fv.append(vis); fo.append(occ); fu.append(bounces)
        us.append(pos.copy()); vl.append(vel.copy())

        dops = [r.doppler_shift for r in bounces]
        all_dops.extend(dops)
        mean_d = np.mean(dops) if dops else float('nan')
        print(f"  {step+1:>5}  {angle_deg:>7.1f}  "
              f"{vel[0]:>7.2f} {vel[1]:>7.2f}  "
              f"{len(bounces):>8}  {mean_d:>12.3f}")

        if args.out_csv:
            df = to_dataframe(vis+occ+bounces, instance_id=f"doppler_{step}",
                              time_step=step, uav=scene.uav)
            dfs.append(df)

    if all_dops:
        print(f"\n  f_D range: [{min(all_dops):+.3f}, {max(all_dops):+.3f}] Hz")
        print(f"  Δf_D = {max(all_dops)-min(all_dops):.3f} Hz  "
              f"(velocity sweep signature)")
    print(f"  {time.time()-t0:.2f}s total")

    if args.out_csv and dfs:
        pd.concat(dfs).to_csv(args.out_csv, index=False)
        print(f"  → {args.out_csv}")

    if not args.no_plot:
        fig = plot_from_static(scene, fv, fo, fu, us, uav_vels=vl, dt=1.0,
                               title="Demo 3 — Doppler Sweep (f_D colored by velocity direction)")
        fig.show()


if __name__ == "__main__":
    main()
