from __future__ import annotations
import sys
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CACHE_DIR = Path("cache")

from src.core.precompute.precompute import precompute
from src.core.cache import save_scene, save_static
from src.core.scene.domain import Receiver
from script.urban_scene_gen_01 import generate_urban_scene, UrbanConfig

from src.core.rx.apply_rx import apply_rx


def scene_hash(cfg: UrbanConfig) -> str:
    s = json.dumps(cfg.__dict__, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def build_ground_mask(scene, resolution=10.0):
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


def sample_rx(xs, ys, mask, n=10, rng=None):
    rng = rng or np.random.default_rng()

    valid = np.argwhere(mask)

    idx = rng.choice(len(valid), size=n, replace=False)
    points = valid[idx]

    rx_list = []
    for i, j in points:
        rx_list.append(np.array([xs[i], ys[j], 2.5]))

    return rx_list


def main():
    cfg = UrbanConfig(seed=int(np.random.randint(0, 1e9)))
    rng = np.random.default_rng(cfg.seed)

    scene = generate_urban_scene(cfg)
    sid = scene_hash(cfg)

    scene_path = CACHE_DIR /f"precomputed_static_fields//sc_{sid}.json"
    static_path = CACHE_DIR / f"precomputed_static_fields//st_{sid}.npz"
    out_dir = CACHE_DIR / "observables"
    out_dir.mkdir(parents=True, exist_ok=True)

    _cols = [
        "instance_id",
        "time_step",
        "tau_s",
        "azimuth_rad",
        "elevation_rad",
        "doppler_hz",
        "freq_hz",
        "power_dbm",
        "n_bounces",
        "last_bounce_x",
        "last_bounce_y",
        "last_bounce_z",
        "rx_pos_x",
        "rx_pos_y",
        "rx_pos_z",
        "tx_pos_x",
        "tx_pos_y",
        "tx_pos_z",
        "tx_power_w",
        "tx_freq_hz",
    ]

    save_scene(scene, scene_path)

    static = precompute(scene)
    save_static(static, static_path)

    # Ground mask
    xs, ys, mask = build_ground_mask(scene)
    # Sample RX positions
    rx_positions = sample_rx(xs, ys, mask, n=20, rng=rng)
    rows = []


    for i, pos in enumerate(rx_positions):
        rx = Receiver(position=pos, radius=10.0)
        static_rx = apply_rx(static, rx)

        rx_rays = static_rx.anchors

        if len(rx_rays) == 0:
            del static_rx
            continue


        for ray in rx_rays:

            tx = scene.transmitters[ray.transmitter_id]

            last_bounce = ray.points[-2]

            rows.append({
                "instance_id": sid,
                "time_step": i,

                "tau_s": ray.delay(),

                "azimuth_rad": ray.azimuth(),
                "elevation_rad": ray.elevation(),

                "doppler_hz": ray.doppler_shift,

                "freq_hz": ray.frequency,
                "power_dbm": ray.power_dbm,

                "n_bounces": ray.n_bounces,

                "last_bounce_x": float(last_bounce[0]),
                "last_bounce_y": float(last_bounce[1]),
                "last_bounce_z": float(last_bounce[2]),

                "rx_pos_x": float(pos[0]),
                "rx_pos_y": float(pos[1]),
                "rx_pos_z": float(pos[2]),

                "tx_pos_x": float(tx.position[0]),
                "tx_pos_y": float(tx.position[1]),
                "tx_pos_z": float(tx.position[2]),

                "tx_power_w": float(tx.tx_power_w),
                "tx_freq_hz": float(tx.frequency),
            })


        del rx_rays
        del static_rx

    df = pd.DataFrame(rows)
    file_path = out_dir / f"{sid}.csv"
    df[_cols].to_csv(file_path, index=False)


if __name__ == "__main__":
    main()