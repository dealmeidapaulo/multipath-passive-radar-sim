"""
benchmark/bench_obstacle_count.py
===================================
Analyzes precompute performance by varying the number of obstacles.
Uses a representative "Enclosed Room" geometry with a central divider.
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

# Add repository root to sys.path
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.scene.domain import Scene
from src.core.scene.domain import Transmitter
from src.core.scene.domain import Receiver
from src.core.scene.domain import Box
from src.core.scene.domain import Obstacle
from src.core.precompute.precompute import precompute

# --- Configuration ------------------------------------------------------------
N_RAYS = 250_000
N_REPS = 5
SPLIT_FACTORS = [1, 5, 10, 25, 50, 100, 200]
RESULTS_DIR = HERE / "results"

# Representative Layout: 100x100 domain, central wall between TX and RX
BASE_WALLS = [
    ((0.0, 0.0, 0.0), (100.0, 2.0, 50.0)),    # Bottom Wall
    ((0.0, 98.0, 0.0), (100.0, 100.0, 50.0)), # Top Wall
    ((0.0, 2.0, 0.0), (2.0, 98.0, 50.0)),     # Left Wall
    ((98.0, 2.0, 0.0), (100.0, 98.0, 50.0)),  # Right Wall
    ((49.0, 2.0, 0.0), (51.0, 98.0, 40.0)),   # Center Divider
]

def get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        )
        return out.strip().split("\n")[0]
    except Exception:
        return "Unknown_GPU"

def generate_subdivided_walls(splits: int):
    obstacles = []
    for bmin, bmax in BASE_WALLS:
        dims = [bmax[i] - bmin[i] for i in range(3)]
        axis = np.argmax(dims) # Split along longest dimension
        step = dims[axis] / splits
        for i in range(splits):
            m_min, m_max = list(bmin), list(bmax)
            m_min[axis] = bmin[axis] + i * step
            m_max[axis] = bmin[axis] + (i + 1) * step
            obstacles.append(Obstacle(box_min=m_min, box_max=m_max))
    return obstacles

def run_benchmark() -> dict:
    """Executes the benchmark and returns a results dictionary."""
    gpu_name = get_gpu_name()
    timestamp = datetime.now().astimezone().isoformat()
    
    box = Box(box_min=[0, 0, 0], box_max=[100, 100, 50])
    tx = Transmitter(tx_id=1, position=[75, 50, 20], tx_power_dbm=57.0, frequency=700e6)
    rx = Receiver(position=[25, 50, 20], radius=5.0)
    
    scene = Scene(box=box, transmitters=[tx], receiver=rx, obstacles=[])
    scene.n_rays = N_RAYS
    scene.n_max = 4
    for obs in scene.obstacles: obs.roughness = 0.0

    # JIT Warmup
    scene.obstacles = generate_subdivided_walls(1)
    precompute(scene, seed=42)

    results_data = []
    for s in SPLIT_FACTORS:
        scene.obstacles = generate_subdivided_walls(s)
        n_obs = len(scene.obstacles)
        times = []
        anchors = 0
        
        for _ in range(N_REPS):
            t0 = time.time()
            static = precompute(scene, seed=42)
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)
            if _ == 0:
                anchors = len(static.anchors)
                
        results_data.append({
            "n_obs": int(n_obs),
            "splits": int(s),
            "times_ms": [round(t, 3) for t in times],
            "median_ms": round(float(np.median(times)), 3),
            "anchors": int(anchors)
        })

    return {
        "metadata": {
            "benchmark": "bench_obstacle_count",
            "date": timestamp,
            "gpu": gpu_name,
            "fixed_n_rays": N_RAYS,
            "bounces_max": scene.n_max
        },
        "results": results_data
    }

def print_report(summary: dict, filepath: Path):
    m = summary["metadata"]
    print("\nBENCHMARK REPORT: OBSTACLE COUNT SCALING")
    print(f"Device: {m['gpu']}")
    print(f"Rays  : {m['fixed_n_rays']:,} | Bounces: {m['bounces_max']}")
    print("-" * 80)
    print(f"{'N_obs':>8} | {'Splits':>8} | {'Median (ms)':>12} | {'Range [Min - Max] (ms)':>25} | {'Anchors':>8}")
    print("-" * 80)
    for r in summary["results"]:
        med = f"{r['median_ms']:.2f}"
        rng = f"[{min(r['times_ms']):.1f} - {max(r['times_ms']):.1f}]"
        print(f"{r['n_obs']:>8} | {r['splits']:>8} | {med:>12} | {rng:>25} | {r['anchors']:>8}")
    print("-" * 80)
    print(f"Output: {filepath}\n")

if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    data = run_benchmark()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = RESULTS_DIR / f"bench_obs_count_{ts}.json"
    path.write_text(json.dumps(data, indent=2))
    print_report(data, path)