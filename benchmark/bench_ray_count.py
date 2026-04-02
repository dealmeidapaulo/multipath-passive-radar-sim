"""
benchmark/bench_ray_count.py
=============================
Measures precompute wall-clock time as a function of ray count.

The scene geometry is fixed; only n_rays varies across runs.
A single JIT warmup call (N_WARMUP rays) is executed before timing begins.
Each configuration is repeated N_REPS times; the median is recorded.

Scene (fixed)
-------------
  Domain    : 200 x 200 x 100 m
  TX        : (0, 0, 80) — 500 W — 700 MHz
  RX        : (100, 100, 20) — radius 10 m
  Obstacles : 20 blocks, 10 x 10 x 30 m, uniform grid at x ∈ [110, 190]
  Roughness : 0.0

Usage
-----
  python benchmark/bench_ray_count.py
  python benchmark/bench_ray_count.py --reps 5 --output-dir /tmp/results
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
RAY_SWEEP   = [10_000, 30_000, 60_000, 100_000, 250_000, 500_000, 750_000]
N_REPS      = 25
N_WARMUP    = 5_000
MAX_BOUNCES = 8
CELL_SIZE   = 5.0
RESULTS_DIR = _HERE / "results"


# ══════════════════════════════════════════════════════════════════════════════
# Scene
# ══════════════════════════════════════════════════════════════════════════════

def _obstacles() -> list[Obstacle]:
    """20 blocks (4 cols × 5 rows) placed at x > 110 m, away from TX→RX axis."""
    obs = []
    for col in range(4):
        for row in range(5):
            x0 = 110.0 + col * 20.0
            y0 =  10.0 + row * 36.0
            obs.append(Obstacle(
                box_min=np.array([x0,       y0,      0.0]),
                box_max=np.array([x0 + 10., y0 + 10., 30.0]),
            ))
    return obs


def build_scene(n_rays: int) -> Scene:
    scene = Scene(
        box          = Box(np.zeros(3), np.array([200., 200., 100.])),
        transmitters = [Transmitter(
                            position   = np.array([0., 0., 80.]),
                            frequency  = 700e6,
                            tx_power_w = 500.0,
                            tx_id      = 0,
                        )],
        receiver     = Receiver(position=np.array([100., 100., 20.]), radius=10.0),
        obstacles    = _obstacles(),
        n_rays       = n_rays,
        n_max        = MAX_BOUNCES,
    )
    scene.roughness    = 0.0
    scene.use_physics  = True
    scene.bandwidth_hz = 20e6
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
    precompute(build_scene(N_WARMUP), seed=0, cell_size=CELL_SIZE)


def measure_one(n_rays: int, n_reps: int, seed: int = 42) -> dict[str, Any]:
    """
    Run precompute n_reps times for the given ray count.
    Returns a result record with timing statistics and anchor count.
    """
    scene = build_scene(n_rays)
    times: list[float] = []
    n_anchors: int = 0

    for rep in range(n_reps):
        t0     = time.perf_counter()
        field  = precompute(scene, seed=seed + rep, cell_size=CELL_SIZE)
        times.append(time.perf_counter() - t0)
        if rep == 0:
            n_anchors = len(field.anchors)

    times_ms = [round(t * 1e3, 3) for t in times]
    return {
        "n_rays"    : n_rays,
        "n_anchors" : n_anchors,
        "median_ms" : round(float(np.median(times_ms)), 3),
        "mean_ms"   : round(float(np.mean(times_ms)),   3),
        "std_ms"    : round(float(np.std(times_ms)),    3),
        "min_ms"    : round(float(np.min(times_ms)),    3),
        "max_ms"    : round(float(np.max(times_ms)),    3),
        "times_ms"  : times_ms,
    }


def run(ray_sweep: list[int], n_reps: int) -> list[dict[str, Any]]:
    print(f"warmup ({N_WARMUP:,} rays) ...", flush=True)
    warmup()

    results = []
    for n in ray_sweep:
        rec = measure_one(n, n_reps)
        results.append(rec)
        print(
            f"  n_rays={rec['n_rays']:>7,}  "
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
            "benchmark"       : "bench_ray_count",
            "timestamp_utc"   : datetime.now(timezone.utc).isoformat(),
            "n_reps"          : n_reps,
            "n_warmup_rays"   : N_WARMUP,
            "max_bounces"     : MAX_BOUNCES,
            "cell_size_m"     : CELL_SIZE,
            "scene": {
                "domain_m"    : [200, 200, 100],
                "tx_position" : [0, 0, 80],
                "tx_power_w"  : 500,
                "tx_freq_hz"  : 700e6,
                "rx_position" : [100, 100, 20],
                "rx_radius_m" : 10,
                "n_obstacles" : 20,
                "roughness"   : 0.0,
            },
            "system"          : system_info(),
        },
        "results": results,
    }

    path = output_dir / f"bench_ray_count_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark: precompute time vs n_rays")
    p.add_argument("--reps",       type=int,          default=N_REPS)
    p.add_argument("--output-dir", type=Path,         default=RESULTS_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    results = run(RAY_SWEEP, args.reps)
    path    = save(results, args.output_dir, args.reps)
    print(f"saved → {path}")
