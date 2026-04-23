"""
benchmark/bench_ray_count.py
==============================
Throughput Analysis: Ray Density vs. Computational Latency

This benchmark measures the scalability of the ray-tracing engine by 
sweeping the number of emitted rays (N_rays). It uses the standardized 
Marsupial RF scene to ensure physical consistency with other benchmarks.

Standardized Scene (Ref: bench_obstacle_count.py)
-------------------------------------------------
  Domain    : 200 x 200 x 100 m
  TX        : (1, 1, 80) — 500 W — 700 MHz
  RX        : (100, 100, 20) — radius 10 m
  Obstacles : 36 building blocks (Uniform grid)
"""

import sys
import time
import pathlib
import argparse
import csv
import numpy as np
from datetime import datetime

# Environment Setup
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain import Scene, Box, Obstacle, Transmitter, Receiver
from src.core.precompute.precompute import precompute

# ── Configuration ────────────────────────────────────────────────────────────
RAY_COUNTS  = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
N_REPS      = 3
RESULTS_DIR = _ROOT / "benchmark" / "results"

def make_standard_scene(n_rays: int) -> Scene:
    """Generates the Marsupial RF standard urban scene."""
    box = Box(np.zeros(3), np.array([200., 200., 100.]))
    # Consistency Fix: Using 500W and standard position
    tx  = Transmitter(np.array([1., 1., 80.]), 700e6, tx_power_w=500.0, tx_id=0)
    rx  = Receiver(np.array([100., 100., 20.]), radius=10.0)
    
    # Building grid (Scenario standard)
    obs = []
    cols, rows = 6, 6
    for i in range(cols):
        for j in range(rows):
            x0, y0 = 110.0 + i*15.0, 5.0 + j*30.0
            obs.append(Obstacle(
                box_min=np.array([x0, y0, 0.]),
                box_max=np.array([x0+10., y0+10., 30.])
            ))
            
    scene = Scene(box=box, transmitters=[tx], receiver=rx, 
                  obstacles=obs, n_rays=n_rays, n_max=8)
    for obs in scene.obstacles: obs.roughness = 0.0
    return scene

def run_benchmark():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"bench_ray_count_{stamp}.csv"
    
    W = 80
    print("=" * W)
    print(f"  Ray Tracing Throughput Benchmark | {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Configuration: 500W TX @ (0,0,80), 36 Obstacles, 8 Max Bounces")
    print("-" * W)
    print(f"{'N_Rays':>12} | {'Median (ms)':>15} | {'Throughput (MRay/s)':>22} | {'Anchors':>10}")
    print("-" * W)

    rows = []
    
    # JIT Warmup
    precompute(make_standard_scene(10000), seed=42)

    for n in RAY_COUNTS:
        scene = make_standard_scene(n)
        latencies = []
        anchor_counts = []
        
        for rep in range(N_REPS):
            t0 = time.perf_counter()
            static = precompute(scene, seed=42 + rep)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            if rep == 0:
                anchor_counts.append(len(static.anchors))
            
        t_med = np.median(latencies)
        m_rays_per_sec = (n / (t_med / 1000.0)) / 1e6
        anchors = anchor_counts[0]
        
        print(f"{n:12,d} | {t_med:15.2f} | {m_rays_per_sec:22.2f} | {anchors:10,d}")
        
        rows.append({
            "n_rays": n,
            "t_ms": round(t_med, 4),
            "throughput_mrays": round(m_rays_per_sec, 2),
            "anchors": anchors
        })

    # Export results
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_rays", "t_ms", "throughput_mrays", "anchors"])
        writer.writeheader()
        writer.writerows(rows)
    
    print("-" * W)
    print(f"Results archived at: {csv_path}")
    print("=" * W)

if __name__ == "__main__":
    run_benchmark()