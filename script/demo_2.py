"""
script/demo_coverage.py
========================
SCENARIO 4 — Coverage grid sweep: detectability map.

What this does
--------------
  Sweeps the UAV over a 6×6 grid of positions at fixed height.
  For each position: runs apply_uav and records whether any UAV
  bounce was detected above the noise floor.
  Output: a matplotlib heatmap showing "detectable" vs "shadow" zones.

Usage
-----
  python script/demo_coverage.py
  python script/demo_coverage.py --grid 8 --no_plot --save coverage_sweep.png
"""
import sys, pathlib, time
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np

from src.core.scene.domain   import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.cache          import get_or_compute
from src.core.uav.apply_uav import apply_uav
from src.outputs.visualizer  import plot_from_static

_CACHE_DIR = _ROOT / "cache"


def make_scene(n_rays: int = 100_000) -> Scene:
    box = Box(np.zeros(3), np.array([160., 120., 60.]))
    tx  = Transmitter(np.array([5.,  60., 35.]), 700e6, tx_power_w=500., tx_id=0)
    rx  = Receiver(np.array([155., 60., 20.]), radius=12.)
    obs = [
        Obstacle(np.array([30.,  10., 0.]), np.array([65.,  45., 30.])),
        Obstacle(np.array([30.,  75., 0.]), np.array([65., 110., 28.])),
        Obstacle(np.array([95.,  10., 0.]), np.array([130., 45., 32.])),
        Obstacle(np.array([95.,  75., 0.]), np.array([130.,110., 25.])),
    ]
    uav = UAV(np.array([80., 60., 35.]), np.array([0., 0., 0.]), radius=1.0)
    scene = Scene(box=box, transmitters=[tx], receiver=rx, uav=uav,
                  obstacles=obs, n_rays=n_rays, n_max=8)
    scene.use_physics   = True
    scene.roughness     = 0.3
    scene.bandwidth_hz  = 20e6
    scene.temperature_c = 20.
    scene.uav_roughness = 0.4
    scene.n_samples_uav = 16
    return scene


def main():
    p = argparse.ArgumentParser(description="Demo: coverage grid sweep")
    p.add_argument("--rays",    type=int,   default=100_000)
    p.add_argument("--grid",    type=int,   default=6,
                   help="Grid size (NxN positions swept)")
    p.add_argument("--height",  type=float, default=35.,
                   help="UAV height for the sweep (metres)")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--save",    type=str,   default="",
                   help="Save heatmap PNG to this path")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        raise ImportError("pip install matplotlib")

    print("\n" + "="*58)
    print(f"  Demo 4 — Coverage Grid Sweep ({args.grid}×{args.grid} positions)")
    print("="*58)

    scene  = make_scene(args.rays)
    static = get_or_compute(scene, seed=42, cell_size=4.,
                             cache_dir=_CACHE_DIR, verbose=True)
    print(f"  anchors={len(static.anchors)}  buildings={len(scene.obstacles)}\n")

    # Build grid
    xs = np.linspace(20, 140, args.grid)
    ys = np.linspace(10, 110, args.grid)
    n_bounces = np.zeros((args.grid, args.grid), dtype=int)
    max_power  = np.full((args.grid, args.grid), -np.inf)

    t0 = time.time()
    total = args.grid * args.grid
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            pos = np.array([x, y, args.height])
            # Skip positions inside buildings
            in_bld = any(
                np.all(pos >= obs.box_min) and np.all(pos <= obs.box_max)
                for obs in scene.obstacles
            )
            if in_bld:
                n_bounces[i, j] = -1   # mark as building
                continue

            scene.uav.position = pos
            scene.uav.velocity = np.zeros(3)
            _, _, bounces = apply_uav(static, scene.uav, scene)
            n_bounces[i, j] = len(bounces)
            if bounces:
                max_power[i, j] = max(r.power_dbm for r in bounces)

            done = i * args.grid + j + 1
            print(f"  [{done:3d}/{total}]  pos=({x:.0f},{y:.0f},{args.height:.0f})  "
                  f"bounces={len(bounces)}"
                  + (f"  P_max={max_power[i,j]:.1f}dBm" if bounces else ""))

    print(f"\n  Sweep done in {time.time()-t0:.1f}s")
    detected = np.sum(n_bounces > 0)
    total_open = np.sum(n_bounces >= 0)
    print(f"  Detectable positions: {detected}/{total_open} "
          f"({100*detected/max(total_open,1):.0f}%)")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a12')
    for ax in axes:
        ax.set_facecolor('#0a0a12')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#555')

    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    # Left: bounce count
    disp_b = np.where(n_bounces >= 0, n_bounces.astype(float), np.nan)
    cmap_b = plt.cm.YlOrRd.copy(); cmap_b.set_bad('#111')
    im0 = axes[0].imshow(disp_b.T, origin='lower', extent=extent,
                          cmap=cmap_b, aspect='equal', vmin=0)
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.set_label('UAV bounce count', color='white')
    cb0.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color='white')

    # Right: max power
    disp_p = np.where(np.isfinite(max_power), max_power, np.nan)
    cmap_p = plt.cm.plasma.copy(); cmap_p.set_bad('#1a1a2a')
    im1 = axes[1].imshow(disp_p.T, origin='lower', extent=extent,
                          cmap=cmap_p, aspect='equal')
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.set_label('Max echo power (dBm)', color='white')
    cb1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white')

    tx_p = scene.transmitters[0].position
    rx_p = scene.receiver.position
    for ax in axes:
        ax.plot(tx_p[0], tx_p[1], 'r^', ms=10, label='TX')
        ax.plot(rx_p[0], rx_p[1], 'cs', ms=10, label='RX')
        for obs in scene.obstacles:
            bx = obs.box_min[0]; by = obs.box_min[1]
            ax.add_patch(plt.Rectangle((bx, by),
                                        obs.box_max[0]-bx, obs.box_max[1]-by,
                                        linewidth=1.5, edgecolor='#88bbff',
                                        facecolor='rgba(0,0,0,0)', alpha=0.85))
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.legend(facecolor='#111', labelcolor='white', fontsize=9)

    axes[0].set_title(f'UAV Echo Count  (z={args.height}m)', color='white')
    axes[1].set_title(f'Max Echo Power  (z={args.height}m)', color='white')
    plt.suptitle(f'Demo 4 — Coverage Sweep  ({args.grid}×{args.grid} grid)',
                 color='white', fontsize=13)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight', facecolor='#0a0a12')
        print(f"  Saved → {args.save}")

    if not args.no_plot:
        plt.show()

    # ── 3D visualisation at best position ─────────────────────────────────────
    if not args.no_plot and np.any(n_bounces > 0):
        best_ij = np.unravel_index(np.argmax(np.where(n_bounces>=0, n_bounces, 0)),
                                   n_bounces.shape)
        best_pos = np.array([xs[best_ij[0]], ys[best_ij[1]], args.height])
        print(f"\n  Best position: {best_pos}  bounces={n_bounces[best_ij]}")
        scene.uav.position = best_pos
        scene.uav.velocity = np.array([2., 1., 0.])   # small velocity for Doppler
        vis, occ, bounces = apply_uav(static, scene.uav, scene)
        fig3d = plot_from_static(
            scene, [vis], [occ], [bounces], [best_pos],
            uav_vels=[scene.uav.velocity], dt=1.,
            title=f"Demo 4 — Best Detection Position {best_pos.astype(int)}",
        )
        fig3d.show()


if __name__ == "__main__":
    main()
