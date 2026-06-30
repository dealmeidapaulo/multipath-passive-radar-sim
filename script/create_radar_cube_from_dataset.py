from __future__ import annotations
import sys
import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass
import argparse
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CACHE_DIR = Path("cache")

from src.core.precompute.precompute import precompute
from src.core.cache import save_scene
from src.core.scene.domain import Receiver
from script.urban_scene_gen import generate_urban_scene
from src.core.rx.apply_rx import apply_rx


@dataclass
class UrbanConfig:
    seed: int = 42

    domain_x: float = 500.0
    domain_y: float = 500.0

    block_size_x: float = 80.0
    block_size_y: float = 80.0

    street_width: float = 15.0
    alley_width: float = 7.0

    block_noise: float = 0.15
    arterial_prob: float = 0.20

    residential_frac: float = 0.65

    parcel_step_res: float = 20.0
    parcel_step_com: float = 40.0

    setback: float = 1.2
    density: float = 0.80

    h_res_mu: float = 5.0
    h_res_sigma: float = 1.5
    h_res_min: float = 3.0
    h_res_max: float = 12.0

    h_com_mu: float = 22.0
    h_com_sigma: float = 12.0
    h_com_min: float = 8.0
    h_com_max: float = 60.0

    tower_prob: float = 0.05
    lshape_prob: float = 0.15
    complex_prob: float = 0.05

    n_transmitters: int = 3
    tx_frequency: float = 1800e6
    tx_power_w: float = 40.0
    tx_height_offset: float = 20.0

    n_rays: int = 250_000 
    batch_size: int = 100_000
    n_max: int = 10

    use_physics: bool = True
    temperature_c: float = 20.0
    bandwidth_hz: float = 8e6


def get_scene_hash(cfg: UrbanConfig) -> str:
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def build_ground_mask(scene, resolution=2.0):
    db = scene.box
    xs = np.arange(db.box_min[0], db.box_max[0], resolution)
    ys = np.arange(db.box_min[1], db.box_max[1], resolution)

    mask = np.ones((len(xs), len(ys)), dtype=bool)

    for obs in scene.obstacles:
        xmin, ymin, _ = obs.box_min
        xmax, ymax, _ = obs.box_max

        ix = (xs >= xmin) & (xs <= xmax)
        iy = (ys >= ymin) & (ys <= ymax)
        
        mask[np.ix_(ix, iy)] = False

    return xs, ys, mask


def sample_rx_positions(xs, ys, mask, n_samples=10, rng=None):
    rng = rng or np.random.default_rng()
    valid_indices = np.argwhere(mask)
    
    if len(valid_indices) < n_samples:
        raise ValueError("Not enough free positions to place RX.")

    chosen_idx = rng.choice(len(valid_indices), size=n_samples, replace=False)
    
    rx_list = []
    for i, j in valid_indices[chosen_idx]:
        rx_list.append(np.array([xs[i], ys[j], 2.5]))

    return rx_list


def extract_ray_features(ray, tx, scene_id, iteration_id, rx_pos, iter_seed, static_time_s, rx_time_s):
    if len(ray.points) < 2:
        return None
        
    last_bounce = ray.points[-2]
    n_bounces = len(ray.points) - 2

    lobe_id = ray.transmitter_id % 3
    lobe_theta_rad = (2 * np.pi / 3) * lobe_id

    return {
        "scene_id": str(scene_id),
        "iteration_id": int(iteration_id),
        "static_seed": int(iter_seed),
        "static_compute_time_s": float(static_time_s),
        "rx_compute_time_s": float(rx_time_s),
        "tau_s": float(ray.delay()),
        "azimuth_rad": float(ray.azimuth()),
        "elevation_rad": float(ray.elevation()),
        "doppler_hz": float(getattr(ray, 'doppler_shift', 0.0)),
        "freq_hz": float(ray.frequency),
        "power_dbm": float(ray.power_dbm),
        "n_bounces": int(n_bounces),
        "last_bounce_x": float(last_bounce[0]),
        "last_bounce_y": float(last_bounce[1]),
        "last_bounce_z": float(last_bounce[2]),
        "rx_pos_x": float(rx_pos[0]),
        "rx_pos_y": float(rx_pos[1]),
        "rx_pos_z": float(rx_pos[2]),
        "tx_pos_x": float(tx.position[0]),
        "tx_pos_y": float(tx.position[1]),
        "tx_pos_z": float(tx.position[2]),
        "tx_power_w": float(tx.tx_power_w),
        "tx_freq_hz": float(tx.frequency),
        "tx_lobe_theta_rad": float(lobe_theta_rad)
    }


def generate_dataset(base_seed: int, n_realizations: int, n_rx_per_realization: int):
    cfg = UrbanConfig(seed=base_seed)
    scene = generate_urban_scene(cfg)
    scene_id = get_scene_hash(cfg)
    
    out_dir = CACHE_DIR / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_scene(scene, out_dir / f"scene_{scene_id}.json")
    
    xs, ys, mask = build_ground_mask(scene, resolution=2.0)
    dataset_rows = []

    print(f"Starting generation for scene {scene_id} ({n_realizations} realizations)...")

    for i in range(n_realizations):
        iter_seed = base_seed + i + 1000 
        iter_rng = np.random.default_rng(iter_seed)
        
        print(f"  -> Computing realization {i+1}/{n_realizations} [Seed: {iter_seed}]")

        t0 = time.perf_counter()
        static_field = precompute(scene, seed=iter_seed, batch_size=cfg.batch_size)
        static_time_s = time.perf_counter() - t0

        rx_positions = sample_rx_positions(xs, ys, mask, n_samples=n_rx_per_realization, rng=iter_rng)

        for rx_pos in rx_positions:
            rx = Receiver(position=rx_pos, radius=10.0)
            
            t1 = time.perf_counter()
            static_rx = apply_rx(static_field, rx)
            rx_time_s = time.perf_counter() - t1
            
            rx_rays = getattr(static_rx, 'anchors', [])
            
            for ray in rx_rays:
                base_tx_idx = ray.transmitter_id // 3
                tx = scene.transmitters[base_tx_idx]
                
                features = extract_ray_features(ray, tx, scene_id, i, rx_pos, iter_seed, static_time_s, rx_time_s)
                if features:
                    dataset_rows.append(features)
                    
        del static_field

    if dataset_rows:
        df = pd.DataFrame(dataset_rows)
        parquet_path = out_dir / f"dataset_{scene_id}.parquet"
        df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"\n[OK] Generated {len(df)} rays. Dataset saved to: {parquet_path}")
    else:
        print("\n[WARNING] No rays reached the RX in any iteration.")




def main():
    parser = argparse.ArgumentParser(
        description="Generate a passive radar dataset from a synthetic urban scene."
    )

    parser.add_argument(
        "--scene-seed",
        type=int,
        default=None,
        help="Seed used to generate the urban scene. If omitted, a random seed is used."
    )

    parser.add_argument(
        "--n-realizations",
        type=int,
        default=20,
        help="Number of ray-tracing realizations per scene."
    )

    parser.add_argument(
        "--n-rx",
        type=int,
        default=50,
        help="Number of receiver positions per realization."
    )

    args = parser.parse_args()

    if args.scene_seed is None:
        args.scene_seed = int(np.random.SeedSequence().entropy)

    print(f"[INFO] Scene seed: {args.scene_seed}")

    generate_dataset(
        base_seed=args.scene_seed,
        n_realizations=args.n_realizations,
        n_rx_per_realization=args.n_rx,
    )

if __name__ == "__main__":
    main()
