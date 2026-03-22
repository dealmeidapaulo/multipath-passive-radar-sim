"""
Usage
-----
    python script/multi_tx.py
    python script/multi_tx.py --frames 8 --rays 50000 --no_plot
"""
import sys, pathlib, argparse, time
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.core.scene.domain    import Scene, Box, Transmitter, Receiver, UAV, Obstacle
from src.core.cache           import get_or_compute
from src.core.uav.apply_uav  import apply_uav
from src.outputs.visualizer   import plot_from_static
from src.outputs.observables  import to_dataframe

_CACHE_DIR = _ROOT / "cache"


def make_two_tx_scene(
    domain   = (200., 200., 100.),
    freq     = 700e6,
    n_rays   = 50_000,
    n_max    = 8,
    seed     = 42,
    roughness= 0.3,
) -> Scene:
    """
    Symmetric corridor: TX-A left, TX-B right, RX in the centre.
    Four buildings flank the corridor.
    """
    dx, dy, dz = domain
    box  = Box(np.zeros(3), np.array([dx, dy, dz]))
    tx_a = Transmitter(np.array([10.,  dy/2, 40.]), freq, tx_power_w=250., tx_id=0)
    tx_b = Transmitter(np.array([dx-10.,dy/2, 40.]), freq, tx_power_w=250., tx_id=1)
    rx   = Receiver(np.array([dx/2, dy/2, 20.]), radius=10.)
    uav  = UAV(np.array([dx/2, dy/2, 45.]), np.array([4., 2., 0.]), radius=1.0)
    obs  = [
        Obstacle(np.array([30.,  10., 0.]), np.array([80.,  40., 30.])),
        Obstacle(np.array([30.,  dy-40., 0.]), np.array([80.,  dy-10., 28.])),
        Obstacle(np.array([dx-80.,10., 0.]), np.array([dx-30.,40.,  25.])),
        Obstacle(np.array([dx-80.,dy-40.,0.]), np.array([dx-30.,dy-10.,22.])),
    ]
    scene = Scene(box=box, transmitters=[tx_a, tx_b], receiver=rx,
                  uav=uav, obstacles=obs, n_rays=n_rays, n_max=n_max)
    scene.use_physics    = True
    scene.roughness      = roughness
    scene.bandwidth_hz   = 0.5e6
    scene.temperature_c  = 20.
    scene.uav_roughness  = 0.4
    scene.n_samples_uav  = 8
    return scene


def main():
    p = argparse.ArgumentParser(description="multipath-passive-radar-sim — Two TX scenario")
    p.add_argument("--rays",       type=int,   default=50_000)
    p.add_argument("--frames",     type=int,   default=5)
    p.add_argument("--dt",         type=float, default=1.)
    p.add_argument("--cell_size",  type=float, default=5.)
    p.add_argument("--roughness",  type=float, default=0.3)
    p.add_argument("--no_plot",    action="store_true")
    p.add_argument("--out_csv",    type=str,   default="")
    args = p.parse_args()

    print("\n" + "="*52)
    print("  multipath-passive-radar-sim — Two-TX Scenario")
    print("="*52)

    scene  = make_two_tx_scene(n_rays=args.rays, roughness=args.roughness)
    print(f"  TX-A: {scene.transmitters[0].position}  "
          f"TX-B: {scene.transmitters[1].position}")
    print(f"  RX  : {scene.receiver.position}")
    print(f"  UAV : {scene.uav.position}  vel={scene.uav.velocity}")

    print("\n[*] Precompute — checking cache...")
    static = get_or_compute(scene, seed=42, cell_size=args.cell_size,
                             cache_dir=_CACHE_DIR, verbose=True)

    ids    = [r.transmitter_id for r in static.anchors]
    print(f"  anchors total={len(static.anchors)}  "
          f"TX-A={ids.count(0)}  TX-B={ids.count(1)}")

    uav_start = scene.uav.position.copy()
    vel       = scene.uav.velocity.copy()

    fv, fo, fu, us, vl = [], [], [], [], []
    dfs = []
    t0  = time.time()

    for step in range(args.frames):
        pos = uav_start + vel * step * args.dt
        scene.uav.position = pos.copy()
        scene.uav.velocity = vel.copy()

        vis, occ, bounces = apply_uav(static, scene.uav, scene)
        fv.append(vis); fo.append(occ); fu.append(bounces)
        us.append(pos.copy()); vl.append(vel.copy())

        dops = [r.doppler_shift for r in bounces]
        print(f"  frame {step+1}: vis={len(vis)}  UAV_bounces={len(bounces)}"
              + (f"  f_D=[{min(dops):+.3f},{max(dops):+.3f}]Hz" if dops else ""))

        if dfs is not None:
            df = to_dataframe(vis+occ+bounces,
                              instance_id=f"multitx_{step}", time_step=step,
                              uav=scene.uav)
            dfs.append(df)

    print(f"\n  {time.time()-t0:.2f}s total")

    if args.out_csv and dfs:
        pd.concat(dfs, ignore_index=True).to_csv(args.out_csv, index=False)
        print(f"  Saved → {args.out_csv}")

    if not args.no_plot:
        fig = plot_from_static(
            scene, fv, fo, fu, us, uav_vels=vl, dt=args.dt,
            title="multipath-passive-radar-sim — Two TX (700 MHz each)",
        )
        fig.show()


if __name__ == "__main__":
    main()
