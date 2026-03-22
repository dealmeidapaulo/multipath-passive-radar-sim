"""
As a SCRIPT:
    python script/precomputed_scene.py [--rays N] [--cell_size F] [--seed S]
    Generates (or loads from cache) the StaticField and prints a summary.

As an IMPORTABLE MODULE:
    from script.precomputed_scene import get_scene, get_static
    scene  = get_scene()
    static = get_static()   # loads from cache if available

Default scenario
----------------
    City  : 200×200m, building_height=30m, tall_fraction=0.35
    TX    : (150, 150, 55) — 50 W — 700 MHz
    RX    : (10, 10, 35)  — radius 15m
    Rays  : 100_000
    Seed  : 42
"""
import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np

from src.core.scene.streets         import make_street_scene
from src.core.precompute.static_field import StaticField
from src.core.cache                 import get_or_compute

_CACHE_DIR = _ROOT / "cache"

# ── Default scene parameters ──────────────────────────────────────────────────
_DEFAULTS = dict(
    domain_x=200., domain_y=200., domain_z=120.,
    tx_pos=(150., 150., 55.),
    rx_pos=(10.,  10.,  35.),
    uav_pos=(80., 80., 40.),
    uav_vel=(3., 1., 0.),
    uav_radius=1.0,
    frequency=700e6,
    tx_power_w=50.,
    n_rays=100_000,
    n_max=10,
    seed=42,
    bld_height=30.,
    tall_fraction=0.35,
)

_PHYSICS = dict(
    use_physics=True,
    roughness=0.0,
    bandwidth_hz=20e6,
    temperature_c=30.,
    uav_roughness=0.4,
    n_samples_uav=8,
)

_RX_RADIUS  = 15.0
_CELL_SIZE  = 5.0


def get_scene(**overrides) -> object:
    """
    Return a Scene with default parameters, optionally overridden.

    Example
    -------
    scene = get_scene(n_rays=50_000, roughness=0.3)
    """
    params = {**_DEFAULTS, **{k: v for k, v in overrides.items()
                               if k in _DEFAULTS}}
    scene = make_street_scene(**params)
    phys  = {**_PHYSICS, **{k: v for k, v in overrides.items()
                             if k in _PHYSICS}}
    for k, v in phys.items():
        setattr(scene, k, v)
    scene.receiver.radius = overrides.get('rx_radius', _RX_RADIUS)
    return scene


def get_static(
    cell_size : float = _CELL_SIZE,
    seed      : int   = _DEFAULTS['seed'],
    verbose   : bool  = True,
    **scene_overrides,
) -> StaticField:
    """
    Return a StaticField for the default scene, using cache when available.

    Example
    -------
    static = get_static(n_rays=200_000, verbose=True)
    """
    scene = get_scene(**scene_overrides, seed=seed)
    return get_or_compute(scene, seed=seed, cell_size=cell_size,
                          cache_dir=_CACHE_DIR, verbose=verbose)


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute and cache default scene")
    parser.add_argument("--rays",      type=int,   default=_DEFAULTS['n_rays'])
    parser.add_argument("--cell_size", type=float, default=_CELL_SIZE)
    parser.add_argument("--seed",      type=int,   default=_DEFAULTS['seed'])
    parser.add_argument("--roughness", type=float, default=_PHYSICS['roughness'])
    args = parser.parse_args()

    print("\n" + "="*56)
    print("  multipath-passive-radar-sim — Precomputed Scene Generator")
    print("="*56)
    print(f"  n_rays    : {args.rays:,}")
    print(f"  cell_size : {args.cell_size}m")
    print(f"  seed      : {args.seed}")
    print(f"  roughness : {args.roughness}")
    print(f"  cache_dir : {_CACHE_DIR}")

    static = get_static(
        cell_size=args.cell_size,
        seed=args.seed,
        n_rays=args.rays,
        roughness=args.roughness,
        verbose=True,
    )

    sh = static.spatial_hash
    print(f"\n  anchors      : {len(static.anchors)}")
    print(f"  hash_entries : {sh.total_entries:,}")
    print(f"  cells        : {sh.NX}×{sh.NY}×{sh.NZ} = {sh.n_cells:,}")
    mean_s, max_s, frac = sh.coverage_stats()
    print(f"  per-cell     : mean={mean_s:.0f}  max={max_s}  nonempty={100*frac:.1f}%")
    print("\n  Done. StaticField ready — use get_static() to load it.")
