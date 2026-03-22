"""
Usage
---
    python script/demo_occlusion.py
    python script/demo_occlusion.py --frames 7 --no_plot
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

# ── Scene geometry ────────────────────────────────────────────────────────────
# TX and RX on the same horizontal axis (y=75).
# Two buildings form a partial canyon, creating clear LOS paths.
# UAV starts at y=20 (outside LOS cone) and moves toward y=130 (far side).
# It crosses the TX→RX diagonal around frame 3-4.
#
# TX=(10,75,40) → RX=(140,75,15): LOS at y=75 for all x
# UAV at (75, y_start, 45) moving in +Y → crosses LOS at y≈75

TX_POS    = np.array([10.,  75., 40.])
RX_POS    = np.array([140., 75., 15.])
UAV_START = np.array([70.,  10., 40.])   # starts south of LOS
UAV_VEL   = np.array([0.,   15., 0.])    # moves north (+Y) crossing LOS at y≈75
N_RAYS    = 100_000


def make_scene() -> Scene:
    box = Box(np.zeros(3), np.array([150., 150., 80.]))
    tx  = Transmitter(TX_POS, 700e6, tx_power_w=500., tx_id=0)
    rx  = Receiver(RX_POS, radius=15.)
    uav = UAV(UAV_START.copy(), UAV_VEL.copy(), radius=3.0)

    # Two buildings flanking the corridor — NOT blocking the TX→RX LOS
    obs = [
        Obstacle(np.array([30.,  0.,  0.]), np.array([65.,  50., 35.])),
        Obstacle(np.array([85.,  0.,  0.]), np.array([120., 50., 30.])),
        Obstacle(np.array([30.,  100.,0.]), np.array([65.,  150.,32.])),
        Obstacle(np.array([85.,  100.,0.]), np.array([120., 150.,28.])),
    ]

    scene = Scene(box=box, transmitters=[tx], receiver=rx,
                  uav=uav, obstacles=obs, n_rays=N_RAYS, n_max=8)
    scene.use_physics    = True
    scene.roughness      = 0.4
    scene.uav_roughness  = 0.5
    scene.n_samples_uav  = 32
    scene.bandwidth_hz   = 0.5e6
    scene.temperature_c  = 20.
    return scene


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


def main():
    p = argparse.ArgumentParser(description="Demo 2 — UAV occlusion crossing LOS")
    p.add_argument("--frames",  type=int,   default=8)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--out_csv", type=str,   default="")
    args = p.parse_args()

    print("\n" + "="*52)
    print("  Demo 2 — UAV Occlusion (crossing LOS)")
    print("="*52)
    print(f"  TX={TX_POS}  RX={RX_POS}")
    print(f"  UAV starts at y={UAV_START[1]:.0f}, crosses LOS (y=75) at frame ≈4")

    scene  = make_scene()
    print(f"\n[*] Precompute — checking cache...")
    static = get_or_compute(scene, seed=7, cell_size=4.,
                            cache_dir=_CACHE_DIR, verbose=True)
    print(f"  anchors={len(static.anchors)}")

    # Verify LOS is clear
    n_vis0 = len([r for r in static.anchors])
    print(f"  baseline anchors={n_vis0}  (will decrease as UAV crosses LOS)")

    pos = UAV_START.copy()
    vel = UAV_VEL.copy()
    fv, fo, fu, us, vl = [], [], [], [], []
    dfs = []
    t0  = time.time()

    print(f"\n  {'frame':>5}  {'uav_y':>7}  {'vis':>5}  {'occ':>5}  {'bounces':>8}  {'f_D Hz':>10}")
    print("  " + "-"*55)

    for step in range(args.frames):
        pos = UAV_START + vel * step
        scene.uav.position = pos.copy()
        scene.uav.velocity = vel.copy()

        vis, occ, bounces = apply_uav(static, scene.uav, scene)
        fv.append(vis); fo.append(occ); fu.append(bounces)
        us.append(pos.copy()); vl.append(vel.copy())

        dops = [r.doppler_shift for r in bounces]
        dop_str = f"[{min(dops):+.2f},{max(dops):+.2f}]" if dops else "—"
        rel_y = pos[1]
        marker = " ← LOS crossing" if abs(rel_y - 75) < 10 else ""
        print(f"  {step+1:>5}  {rel_y:>7.1f}  {len(vis):>5}  {len(occ):>5}  "
              f"{len(bounces):>8}  {dop_str:>10}{marker}")

        if args.out_csv:
            df = to_dataframe(vis+occ+bounces, instance_id=f"occ_{step}",
                              time_step=step, uav=scene.uav)
            dfs.append(df)

    print(f"\n  {time.time()-t0:.2f}s total")

    if args.out_csv and dfs:
        pd.concat(dfs).to_csv(args.out_csv, index=False)
        print(f"  → {args.out_csv}")

    if not args.no_plot:
        fig = plot_from_static(scene, fv, fo, fu, us, uav_vels=vl, dt=1.0,
                               title="Demo 2 — UAV Occlusion: crossing the LOS")
        fig.show()


if __name__ == "__main__":
    main()
