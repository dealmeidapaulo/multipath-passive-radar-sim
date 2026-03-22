"""
script/demo_bounces.py
=======================
SCENARIO 1 — Open corridor: UAV bounce detection demo.

Geometry
--------
  Domain 200×100×80 m, NO buildings.
  TX=(5, 50, 25)  — 500 W, 700 MHz
  RX=(195, 50, 25) — radius 12m
  UAV flies along the corridor midline (y=50, z=25 ≡ TX-RX axis).

Usage
-----
  python script/demo_bounces.py
  python script/demo_bounces.py --frames 15 --no_plot --out_csv bounces.csv
"""
import sys, pathlib, time
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd

from src.core.scene.domain   import Scene, Box, Transmitter, Receiver, UAV
from src.core.cache          import get_or_compute
from src.core.uav.apply_uav import apply_uav
from src.outputs.visualizer  import plot_from_static
from src.outputs.observables import to_dataframe

_CACHE_DIR = _ROOT / "cache"


def make_scene(n_rays: int = 100_000) -> Scene:
    box = Box(np.zeros(3), np.array([200., 100., 80.]))
    tx  = Transmitter(np.array([5.,  50., 25.]), 700e6, tx_power_w=500., tx_id=0)
    rx  = Receiver(np.array([195., 50., 25.]), radius=12.)
    uav = UAV(np.array([30.,  50., 25.]), np.array([5., 0., 0.]), radius=1.0)
    scene = Scene(box=box, transmitters=[tx], receiver=rx, uav=uav,
                  obstacles=[], n_rays=n_rays, n_max=6)
    scene.use_physics   = True
    scene.roughness     = 0.0
    scene.bandwidth_hz  = 20e6
    scene.temperature_c = 20.
    scene.uav_roughness = 0.4
    scene.n_samples_uav = 16   # more samples → more bounce candidates
    return scene


def main():
    p = argparse.ArgumentParser(description="Demo: UAV bounce detection in open corridor")
    p.add_argument("--rays",    type=int,   default=100_000)
    p.add_argument("--frames",  type=int,   default=12)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--out_csv", type=str,   default="")
    args = p.parse_args()

    print("\n" + "="*58)
    print("  Demo 1 — Open Corridor: UAV Bounce Detection")
    print("="*58)

    scene  = make_scene(args.rays)
    static = get_or_compute(scene, seed=42, cell_size=4.,
                             cache_dir=_CACHE_DIR, verbose=True)

    print(f"  anchors={len(static.anchors)}  "
          f"TX={scene.transmitters[0].tx_power_dbm:.0f}dBm  "
          f"noise={scene.noise_floor_dbm:.1f}dBm\n")

    fv, fo, fu, us, vl = [], [], [], [], []
    dfs = []; uav_pos = scene.uav.position.copy()
    vel = scene.uav.velocity.copy(); t0 = time.time()

    for step in range(args.frames):
        scene.uav.position = uav_pos.copy()
        scene.uav.velocity = vel.copy()
        vis, occ, bounces = apply_uav(static, scene.uav, scene)
        fv.append(vis); fo.append(occ); fu.append(bounces)
        us.append(uav_pos.copy()); vl.append(vel.copy())

        dops = [f"{r.doppler_shift:+.3f}" for r in bounces]
        pwrs = [f"{r.power_dbm:.1f}" for r in bounces]
        print(f"  frame {step+1:2d} | UAV x={uav_pos[0]:.0f}m | "
              f"vis={len(vis):3d}  occ={len(occ):2d}  bounces={len(bounces):2d}"
              + (f"  f_D={dops}Hz  P={pwrs}dBm" if bounces else ""))

        if args.out_csv:
            df = to_dataframe(vis+occ+bounces,
                              instance_id=f"demo1_{step}", time_step=step,
                              uav=scene.uav)
            dfs.append(df)
        uav_pos = uav_pos + vel

    print(f"\n  {time.time()-t0:.2f}s total")
    if args.out_csv and dfs:
        pd.concat(dfs, ignore_index=True).to_csv(args.out_csv, index=False)
        print(f"  Saved → {args.out_csv}")

    if not args.no_plot:
        fig = plot_from_static(scene, fv, fo, fu, us, uav_vels=vl, dt=1.,
                               title="Demo 1 — Open Corridor UAV Bounces")
        fig.show()


if __name__ == "__main__":
    main()
