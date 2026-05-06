from __future__ import annotations
import sys
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import cuda
import gc

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CACHE_DIR = Path("cache")

from src.core.precompute.precompute import precompute
from src.core.cache import save_scene, save_static, load_static
from src.core.scene.observables import to_dataframe, to_parquet
from src.core.scene.domain import Receiver
from script.urban_scene_gen import generate_urban_scene, UrbanConfig

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
    "theta_rad",
    "phi_rad",
    "f_D",
    "power_dbm",
    "rx_pos_x",
    "rx_pos_y",
    "rx_pos_z",
    "tx_pos_x",
    "tx_pos_y",
    "tx_pos_z"
    ]

    save_scene(scene, scene_path)

    static = precompute(scene)
    save_static(static, static_path)

    # Ground mask
    xs, ys, mask = build_ground_mask(scene)
    # Sample RX positions
    rx_positions = sample_rx(xs, ys, mask, n=20, rng=rng)

    del static
    cuda.close()
    cuda.current_context().deallocations.clear()

    for i, pos in enumerate(rx_positions):
    
        print(i)
        rx = Receiver(position=pos, radius=10.0)
        static = load_static(static_path,scene)
        static_rx = apply_rx(static, rx)

        rx_rays = static_rx.anchors

        if len(rx_rays) == 0:
            continue

       
        df = to_dataframe(
            rx_rays,
            instance_id=sid,
            time_step=i,
            uav=None,
            params={
                "rx_pos_x": float(pos[0]),
                "rx_pos_y": float(pos[1]),
                "rx_pos_z": float(pos[2]),
                "domain_x": cfg.domain_x,
                "domain_y": cfg.domain_y,
                "seed": cfg.seed,
            }
        )


        file_path = out_dir / f"{sid}_rx{i}.csv"
        df[_cols].to_csv(file_path, index=False)

        del static_rx
        del rx_rays
        cuda.current_context().deallocations.clear()
        gc.collect()

if __name__ == "__main__":
    main()