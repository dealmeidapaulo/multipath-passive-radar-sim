"""
benchmark/bench_obstacle_count.py
===================================
Measures precompute wall-clock time as a function of obstacle count.

Starting from a fixed set of base obstacles, each obstacle is progressively
subdivided along its longest axis into k equal pieces (split factor k).
Total building volume stays constant; only N_obs grows.

This isolates the cost of the inner per-obstacle loop inside trace_all_kernel
(O(N_rays × N_max × N_obs) intersection tests) from geometry changes.

Scene (fixed)
-------------
  Domain    : 200 x 200 x 100 m
  TX        : (0, 0, 80) — 500 W — 700 MHz
  RX        : (100, 100, 20) — radius 10 m
  n_rays    : 100,000 (fixed)
  Roughness : 0.0

Base obstacles (5 walls)
------------------------
  Four perimeter walls + one central divider, placed away from the TX.
  Each wall is subdivided k times along its longest axis, yielding
  N_obs = 5 × k obstacles with identical total volume.

Usage
-----
  python benchmark/bench_obstacle_count.py
  python benchmark/bench_obstacle_count.py --reps 5 --output-dir /tmp/results
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── Repository root ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain import Box, Obstacle, Receiver, Scene, Transmitter
from src.core.precompute.precompute import precompute

# ── Defaults ──────────────────────────────────────────────────────────────────
#  Split factors: N_obs = N_BASE_WALLS × k
SPLIT_FACTORS = [1, 2, 5, 10, 25, 50, 100, 200, 400]
N_RAYS        = 100_000
N_REPS        = 25
MAX_BOUNCES   = 8
CELL_SIZE     = 5.0
RESULTS_DIR   = _HERE / "results"

# Five base walls (bmin, bmax).  Placed at x > 100 m so they don't sit
# directly on the TX–RX baseline, keeping anchor count interpretable.
_BASE_WALLS: list[tuple[list[float], list[float]]] = [
    ([110.0,  2.0, 0.0], [190.0,  8.0, 30.0]),   # south wall
    ([110.0, 92.0, 0.0], [190.0, 98.0, 30.0]),   # north wall
    ([110.0,  8.0, 0.0], [116.0, 92.0, 30.0]),   # west wall
    ([184.0,  8.0, 0.0], [190.0, 92.0, 30.0]),   # east wall
    ([148.0,  8.0, 0.0], [152.0, 92.0, 25.0]),   # central divider
]


# ══════════════════════════════════════════════════════════════════════════════
# Scene
# ══════════════════════════════════════════════════════════════════════════════

def subdivide(bmin: list[float], bmax: list[float], k: int) -> list[Obstacle]:
    """
    Split one AABB into k equal pieces along its longest axis.
    Returns a list of k Obstacle objects with the same total volume.
    """
    dims   = [bmax[i] - bmin[i] for i in range(3)]
    axis   = int(np.argmax(dims))
    step   = dims[axis] / k
    pieces = []
    for i in range(k):
        lo = list(bmin); hi = list(bmax)
        lo[axis] = bmin[axis] + i * step
        hi[axis] = bmin[axis] + (i + 1) * step
        pieces.append(Obstacle(box_min=np.array(lo), box_max=np.array(hi)))
    return pieces


def build_obstacles(k: int) -> list[Obstacle]:
    """Return 5 × k obstacles from the base walls, each split k times."""
    obs = []
    for bmin, bmax in _BASE_WALLS:
        obs.extend(subdivide(bmin, bmax, k))
    return obs


def build_scene(k: int) -> Scene:
    scene = Scene(
        box          = Box(np.zeros(3), np.array([200., 200., 100.])),
        transmitters = [Transmitter(
                            position   = np.array([0., 0., 80.]),
                            frequency  = 700e6,
                            tx_power_w = 500.0,
                            tx_id      = 0,
                        )],
        receiver     = Receiver(position=np.array([100., 100., 20.]), radius=10.0),
        obstacles    = build_obstacles(k),
        n_rays       = N_RAYS,
        n_max        = MAX_BOUNCES,
    )
    scene.roughness     = 0.0
    scene.use_physics   = True
    scene.bandwidth_hz  = 20e6
    scene.temperature_c = 20.0
    return scene


# ══════════════════════════════════════════════════════════════════════════════
# System info
# ══════════════════════════════════════════════════════════════════════════════

def _gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            encoding="utf-8", stderr=subprocess.DEVNULL,
        )
        return out.strip().split("\n")[0]
    except Exception:
        return "unavailable"


def system_info() -> dict[str, Any]:
    return {
        "python"  : platform.python_version(),
        "platform": platform.platform(),
        "cpu"     : platform.processor() or platform.machine(),
        "gpu"     : _gpu_info(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def warmup() -> None:
    """Trigger Numba JIT compilation before timing begins."""
    precompute(build_scene(k=1), seed=0, cell_size=CELL_SIZE)


def measure_one(k: int, n_reps: int, seed: int = 42) -> dict[str, Any]:
    """
    Run precompute n_reps times for a given split factor k.
    Returns a result record with timing statistics, obstacle count, and anchors.
    """
    scene   = build_scene(k)
    n_obs   = len(scene.obstacles)
    times: list[float] = []
    n_anchors: int = 0

    for rep in range(n_reps):
        t0    = time.perf_counter()
        field = precompute(scene, seed=seed + rep, cell_size=CELL_SIZE)
        times.append(time.perf_counter() - t0)
        if rep == 0:
            n_anchors = len(field.anchors)

    times_ms = [round(t * 1e3, 3) for t in times]
    return {
        "split_factor": k,
        "n_obs"       : n_obs,
        "n_anchors"   : n_anchors,
        "median_ms"   : round(float(np.median(times_ms)), 3),
        "mean_ms"     : round(float(np.mean(times_ms)),   3),
        "std_ms"      : round(float(np.std(times_ms)),    3),
        "min_ms"      : round(float(np.min(times_ms)),    3),
        "max_ms"      : round(float(np.max(times_ms)),    3),
        "times_ms"    : times_ms,
    }


def run(split_factors: list[int], n_reps: int) -> list[dict[str, Any]]:
    print("warmup (split=1) ...", flush=True)
    warmup()

    results = []
    for k in split_factors:
        rec = measure_one(k, n_reps)
        results.append(rec)
        print(
            f"  split={rec['split_factor']:>4}  "
            f"n_obs={rec['n_obs']:>5}  "
            f"median={rec['median_ms']:>8.1f} ms  "
            f"anchors={rec['n_anchors']:>5}",
            flush=True,
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def save(results: list[dict], output_dir: Path, n_reps: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    payload = {
        "metadata": {
            "benchmark"     : "bench_obstacle_count",
            "timestamp_utc" : datetime.now(timezone.utc).isoformat(),
            "n_reps"        : n_reps,
            "max_bounces"   : MAX_BOUNCES,
            "cell_size_m"   : CELL_SIZE,
            "scene": {
                "domain_m"         : [200, 200, 100],
                "tx_position"      : [0, 0, 80],
                "tx_power_w"       : 500,
                "tx_freq_hz"       : 700e6,
                "rx_position"      : [100, 100, 20],
                "rx_radius_m"      : 10,
                "fixed_n_rays"     : N_RAYS,
                "roughness"        : 0.0,
                "n_base_walls"     : len(_BASE_WALLS),
                "subdivision_axis" : "longest",
                "note"             : "N_obs = n_base_walls × split_factor; total volume constant",
            },
            "system": system_info(),
        },
        "results": results,
    }

    path = output_dir / f"bench_obstacle_count_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark: precompute time vs obstacle count")
    p.add_argument("--reps",       type=int,  default=N_REPS)
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    results = run(SPLIT_FACTORS, args.reps)
    path    = save(results, args.output_dir, args.reps)
    print(f"saved → {path}")
