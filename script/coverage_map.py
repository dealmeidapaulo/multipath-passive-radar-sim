"""
Two-subplot coverage map from a precomputed StaticField.

Left  — TX ray coverage: number of segments per spatial-hash cell
         (shows where UAV spawns will have candidates).
Right — Anchor power: max power_dbm of anchors passing through each XY cell
         at a given height slice (shows best RX placement).

Usage
-----
    python script/coverage_map.py
    python script/coverage_map.py --height 35 --save coverage.png
"""
import sys, pathlib, argparse
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    raise ImportError("pip install matplotlib")

from script.scene_setup import get_scene, get_static


def make_coverage_map(static, scene, height_m: float = 35.0):
    """
    Returns two 2D arrays (NX×NY) for the XY grid at ~height_m:

    ray_density[i,j]  — number of spatial-hash entries in cell (i,j) at height slice
    anchor_power[i,j] — max power_dbm of anchors whose midpoint falls in (i,j)
                        or -inf if no anchor passes through
    """
    sh  = static.spatial_hash
    cs  = sh.cell_size
    bm  = sh.box_min

    # Height slice index
    cz  = max(0, min(sh.NZ-1, int((height_m - float(bm[2])) / cs)))

    # ── Ray density from spatial hash ─────────────────────────────────────────
    density = np.zeros((sh.NX, sh.NY), dtype=np.float32)
    for cx in range(sh.NX):
        for cy in range(sh.NY):
            cell_id = cx + cy*sh.NX + cz*sh.NX*sh.NY
            density[cx, cy] = float(sh.cell_counts[cell_id])

    # ── Anchor power map ──────────────────────────────────────────────────────
    pwr_map = np.full((sh.NX, sh.NY), -np.inf, dtype=np.float32)
    for ray in static.anchors:
        pts = np.array(ray.points)
        mid = pts.mean(axis=0)
        cxi = max(0, min(sh.NX-1, int((mid[0]-float(bm[0]))/cs)))
        cyi = max(0, min(sh.NY-1, int((mid[1]-float(bm[1]))/cs)))
        pwr_map[cxi, cyi] = max(pwr_map[cxi, cyi], ray.power_dbm)

    return density, pwr_map


def main():
    p = argparse.ArgumentParser(description="Passive Radar Sim — Coverage maps")
    p.add_argument("--height",    type=float, default=35.,
                   help="Z height slice for the coverage maps (metres)")
    p.add_argument("--save",      type=str,   default="",
                   help="Save figure to this path (e.g. coverage.png)")
    p.add_argument("--rays",      type=int,   default=0,
                   help="Override n_rays (0 = use scene_setup default)")
    args = p.parse_args()

    print("Loading scene and StaticField...")
    kw = {} if args.rays == 0 else dict(n_rays=args.rays)
    scene  = get_scene(**kw)
    static = get_static(**kw, verbose=True)

    print(f"Building coverage maps at height={args.height}m...")
    density, pwr_map = make_coverage_map(static, scene, args.height)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a12')
    for ax in axes:
        ax.set_facecolor('#0a0a12')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#555')

    sh = static.spatial_hash
    cs = sh.cell_size; bm = sh.box_min
    extent = [float(bm[0]), float(bm[0])+sh.NX*cs,
              float(bm[1]), float(bm[1])+sh.NY*cs]

    # ── Left: ray density ────────────────────────────────────────────────────
    im0 = axes[0].imshow(density.T, origin='lower', extent=extent,
                          cmap='viridis', aspect='equal',
                          norm=mcolors.LogNorm(vmin=1, vmax=density.max()+1))
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.set_label('Segments per cell (log)', color='white')
    cb0.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color='white')

    # TX and RX markers
    for tx in scene.transmitters:
        axes[0].plot(tx.position[0], tx.position[1], 'r^', ms=9, label=f"TX{tx.tx_id}")
    axes[0].plot(scene.receiver.position[0], scene.receiver.position[1],
                 'cs', ms=9, label='RX')
    # Building footprints
    for obs in scene.obstacles:
        bx = obs.box_min[0]; by = obs.box_min[1]
        bw = obs.box_max[0]-bx; bh = obs.box_max[1]-by
        axes[0].add_patch(plt.Rectangle((bx,by),bw,bh,
                                         linewidth=1.5, edgecolor='#88bbff',
                                         facecolor='none', alpha=0.85))
    axes[0].set_title(f'TX Ray Coverage  (z≈{args.height}m)', color='white')
    axes[0].set_xlabel('X (m)', color='white'); axes[0].set_ylabel('Y (m)', color='white')
    axes[0].legend(facecolor='#111', labelcolor='white', fontsize=9)

    # ── Right: anchor power ───────────────────────────────────────────────────
    pwr_disp = np.where(np.isfinite(pwr_map), pwr_map, np.nan)
    vmin = np.nanmin(pwr_disp) if np.any(np.isfinite(pwr_map)) else -100.
    cmap_pwr = plt.cm.plasma.copy()
    cmap_pwr.set_bad('#1a1a2a')   # NaN cells → dark navy, building outlines visible
    im1  = axes[1].imshow(pwr_disp.T, origin='lower', extent=extent,
                           cmap=cmap_pwr, aspect='equal',
                           vmin=vmin, vmax=np.nanmax(pwr_disp)+1)
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.set_label('Max anchor power (dBm)', color='white')
    cb1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white')

    for tx in scene.transmitters:
        axes[1].plot(tx.position[0], tx.position[1], 'r^', ms=9)
    axes[1].plot(scene.receiver.position[0], scene.receiver.position[1], 'cs', ms=9)
    for obs in scene.obstacles:
        bx = obs.box_min[0]; by = obs.box_min[1]
        bw = obs.box_max[0]-bx; bh = obs.box_max[1]-by
        axes[1].add_patch(plt.Rectangle((bx,by),bw,bh,
                                         linewidth=1.5, edgecolor='#88bbff',
                                         facecolor='none', alpha=0.85))
    axes[1].set_title(f'Anchor Power Map  (z≈{args.height}m)', color='white')
    axes[1].set_xlabel('X (m)', color='white'); axes[1].set_ylabel('Y (m)', color='white')

    plt.suptitle(f'Passive Radar Sim Coverage Maps — {args.height}m slice', color='white', fontsize=13)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight', facecolor='#0a0a12')
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()