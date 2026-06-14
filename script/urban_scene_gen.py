from __future__ import annotations

import argparse
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain import Box, Obstacle, Scene, Transmitter

@dataclass
class UrbanConfig:
    seed: int = 42

    domain_x: float = 500.0
    domain_y: float = 500.0

    block_size_x: float = 80.0
    block_size_y: float = 80.0

    street_width: float = 15.0

    block_noise: float = 0.15
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
    tx_frequency: float = 700e6
    tx_power_w: float = 500.0
    tx_height_offset: float = 30.0

    n_rays: int = 250_000
    batch_size: int = 100_000
    n_max: int = 10

    use_physics: bool = True
    temperature_c: float = 20.0
    bandwidth_hz: float = 8e6


_MATS = {
    "concrete": (0.55, 0.07),
    "brick": (0.80, 0.06),
    "glass": (0.05, 0.02),
    "metal": (0.18, 0.05),
    "asphalt": (0.88, 0.04),
    "stone": (0.72, 0.05),
    "glass_tint": (0.08, 0.03),
}

_COM_MATS = ["concrete", "glass", "brick", "metal", "glass_tint"]
_COM_WEIGHTS = [0.35, 0.22, 0.18, 0.12, 0.13]

_RES_MATS = ["brick", "concrete", "stone"]
_RES_WEIGHTS = [0.55, 0.30, 0.15]


def _sample_mat(rng, mats: list, weights: list) -> Tuple[str, float]:
    w = np.array(weights, dtype=float)
    idx = rng.choice(len(mats), p=w / w.sum())
    m = mats[idx]
    mu, sigma = _MATS[m]
    return m, float(np.clip(rng.normal(mu, sigma), 0.01, 0.99))


def _obs(x0, y0, z0, x1, y1, z1, mat: str, roughness: float) -> Obstacle:
    return Obstacle(
        box_min=np.array([min(x0, x1), min(y0, y1), min(z0, z1)], dtype=float),
        box_max=np.array([max(x0, x1), max(y0, y1), max(z0, z1)], dtype=float),
        material=mat,
        roughness=roughness,
    )


def _bldg_box(x0, y0, x1, y1, h, cfg, rng, is_residential) -> List[Obstacle]:
    mats = _RES_MATS if is_residential else _COM_MATS
    wts = _RES_WEIGHTS if is_residential else _COM_WEIGHTS
    m, r = _sample_mat(rng, mats, wts)
    return [_obs(x0, y0, 0, x1, y1, h, m, r)]


def _bldg_tower(x0, y0, x1, y1, cfg, rng) -> List[Obstacle]:
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
    total_h = float(np.clip(np.exp(rng.normal(np.log(max(cfg.h_com_mu * 1.2, 1)), 0.5)), cfg.h_com_min, cfg.h_com_max * 1.3))
    
    n_bands = rng.integers(2, 5)
    z_cuts = np.sort(rng.uniform(total_h * 0.15, total_h * 0.85, n_bands - 1))
    zs = np.concatenate([[0.0], z_cuts, [total_h]])
    
    obs = []
    for k in range(n_bands):
        shrink = k * rng.uniform(0.07, 0.20)
        sx = max(hx * (1 - shrink), 1.5)
        sy = max(hy * (1 - shrink), 1.5)
        m, r = _sample_mat(rng, _COM_MATS, _COM_WEIGHTS)
        obs.append(_obs(cx - sx, cy - sy, zs[k], cx + sx, cy + sy, zs[k + 1], m, r))
    return obs


def _bldg_lshape(x0, y0, x1, y1, cfg, rng) -> List[Obstacle]:
    dx, dy = x1 - x0, y1 - y0
    fx = rng.uniform(0.45, 0.65)
    fy = rng.uniform(0.45, 0.65)
    
    m1, r1 = _sample_mat(rng, _COM_MATS, _COM_WEIGHTS)
    m2, r2 = _sample_mat(rng, _COM_MATS, _COM_WEIGHTS)
    
    h1 = float(np.clip(rng.normal(cfg.h_com_mu, cfg.h_com_sigma), cfg.h_com_min, cfg.h_com_max))
    h2 = float(np.clip(rng.normal(cfg.h_com_mu * 0.7, cfg.h_com_sigma * 0.5), cfg.h_com_min, cfg.h_com_max))
    
    return [
        _obs(x0, y0, 0, x0 + dx * fx, y1, h1, m1, r1),
        _obs(x0 + dx * fx, y0, 0, x1, y0 + dy * fy, h2, m2, r2),
    ]


def _buildings_for_parcel(x0: float, y0: float, x1: float, y1: float, is_residential: bool, cfg: UrbanConfig, rng: np.random.Generator) -> List[Obstacle]:
    s = cfg.setback
    bx0, by0 = x0 + s, y0 + s
    bx1, by1 = x1 - s, y1 - s

    # Strictly prevent degenerate (microscopic) buildings
    if bx1 - bx0 < 3.0 or by1 - by0 < 3.0:
        return []

    if is_residential:
        h = float(np.clip(rng.normal(cfg.h_res_mu, cfg.h_res_sigma), cfg.h_res_min, cfg.h_res_max))
        return _bldg_box(bx0, by0, bx1, by1, h, cfg, rng, True)

    roll = rng.random()
    if roll < cfg.tower_prob:
        return _bldg_tower(bx0, by0, bx1, by1, cfg, rng)
    if roll < cfg.tower_prob + cfg.lshape_prob:
        return _bldg_lshape(bx0, by0, bx1, by1, cfg, rng)
    
    h = float(np.clip(rng.normal(cfg.h_com_mu, cfg.h_com_sigma), cfg.h_com_min, cfg.h_com_max))
    return _bldg_box(bx0, by0, bx1, by1, h, cfg, rng, False)


def _place_transmitters(buildings: List[Obstacle], cfg: UrbanConfig, rng: np.random.Generator) -> List[Transmitter]:
    if not buildings:
        return [
            Transmitter(
                position=np.array([
                    cfg.domain_x * (0.5 + 0.35 * np.cos(k * 2 * np.pi / cfg.n_transmitters)),
                    cfg.domain_y * (0.5 + 0.35 * np.sin(k * 2 * np.pi / cfg.n_transmitters)),
                    30.0 + cfg.tx_height_offset,
                ]),
                frequency=cfg.tx_frequency,
                tx_power_w=cfg.tx_power_w,
                tx_id=k,
            )
            for k in range(cfg.n_transmitters)
        ]

    bldg_h = np.array([b.box_max[2] for b in buildings])
    sorted_indices = np.argsort(bldg_h)[::-1]
    
    txs: List[Transmitter] = []
    for k in range(cfg.n_transmitters):
        # Select among the tallest buildings to avoid all TXs clustered together
        b = buildings[sorted_indices[k % len(sorted_indices)]]
        txs.append(
            Transmitter(
                position=np.array([
                    float((b.box_min[0] + b.box_max[0]) / 2),
                    float((b.box_min[1] + b.box_max[1]) / 2),
                    float(b.box_max[2] + cfg.tx_height_offset),
                ]),
                frequency=cfg.tx_frequency,
                tx_power_w=cfg.tx_power_w,
                tx_id=k,
            )
        )
    return txs


def generate_urban_scene(cfg: UrbanConfig) -> Scene:
    rng = np.random.default_rng(cfg.seed)
    W, H = cfg.domain_x, cfg.domain_y

    pitch_x = cfg.block_size_x + cfg.street_width
    pitch_y = cfg.block_size_y + cfg.street_width

    nx_blocks = int(W / pitch_x) + 1
    ny_blocks = int(H / pitch_y) + 1

    all_obs: List[Obstacle] = []

    # 1. Generate Non-Overlapping Streets
    rough_asphalt = _MATS["asphalt"][0]
    
    # Vertical streets
    for i in range(nx_blocks):
        sx0 = i * pitch_x
        sx1 = sx0 + cfg.street_width
        if sx0 < W:
            all_obs.append(_obs(sx0, 0, -0.1, min(sx1, W), H, 0.05, "asphalt", rough_asphalt))

    # Horizontal streets (Fill gaps between vertical streets to avoid Z-fighting)
    for j in range(ny_blocks):
        sy0 = j * pitch_y
        sy1 = sy0 + cfg.street_width
        if sy0 < H:
            for i in range(nx_blocks):
                hx0 = i * pitch_x + cfg.street_width
                hx1 = min((i + 1) * pitch_x, W)
                if hx0 < W:
                    all_obs.append(_obs(hx0, sy0, -0.1, hx1, min(sy1, H), 0.05, "asphalt", rough_asphalt))

    # 2. Generate Buildings strictly inside Block parcels
    buildings: List[Obstacle] = []
    
    for i in range(nx_blocks):
        for j in range(ny_blocks):
            # Block boundaries strictly between streets
            bx0 = i * pitch_x + cfg.street_width
            by0 = j * pitch_y + cfg.street_width
            
            # Sub-divide block to avoid a single giant homogeneous mass, but strictly clamp to allocated area
            noise_w = rng.uniform(0.6, 1.0)
            noise_h = rng.uniform(0.6, 1.0)

            bw = cfg.block_size_x * noise_w
            bh = cfg.block_size_y * noise_h

            bx1 = min(bx0 + bw, W - 1.0)
            by1 = min(by0 + bh, H - 1.0)

            # Prevent processing degenerate allocations
            if bx1 - bx0 < 10 or by1 - by0 < 10:
                continue

            is_residential = (rng.random() < cfg.residential_frac)
            step = cfg.parcel_step_res if is_residential else cfg.parcel_step_com

            px_coords = np.arange(bx0, bx1 - step * 0.2, step)
            py_coords = np.arange(by0, by1 - step * 0.2, step)

            for px in px_coords:
                for py in py_coords:
                    if rng.random() > cfg.density:
                        continue
                    
                    px_end = min(px + step, bx1)
                    py_end = min(py + step, by1)
                    
                    bldg = _buildings_for_parcel(px, py, px_end, py_end, is_residential, cfg, rng)
                    buildings.extend(bldg)

    # Final cleanup to absolutely prevent any degenerate geometry going into the GPU
    valid_buildings = []
    for b in buildings:
        dx, dy, dz = b.box_max[0]-b.box_min[0], b.box_max[1]-b.box_min[1], b.box_max[2]-b.box_min[2]
        if dx > 1.0 and dy > 1.0 and dz > 1.0:
            valid_buildings.append(b)

    all_obs.extend(valid_buildings)
    txs = _place_transmitters(valid_buildings, cfg, rng)

    return Scene(
        box=Box(box_min=np.array([0.0, 0.0, -0.5]), box_max=np.array([W, H, cfg.h_com_max * 1.5])),
        obstacles=all_obs,
        transmitters=txs,
        n_rays=cfg.n_rays,
        n_max=cfg.n_max,
        use_physics=cfg.use_physics,
        temperature_c=cfg.temperature_c,
        bandwidth_hz=cfg.bandwidth_hz,
    )


def save_scene(scene: Scene, filepath: Path) -> None:
    data = {
        "box": {
            "box_min": list(map(float, scene.box.box_min)),
            "box_max": list(map(float, scene.box.box_max)),
        },
        "transmitters": [
            {
                "position": list(map(float, tx.position)),
                "frequency": float(tx.frequency),
                "tx_power_w": float(tx.tx_power_w),
                "tx_id": int(tx.tx_id),
            }
            for tx in scene.transmitters
        ],
        "obstacles": [
            {
                "box_min": list(map(float, o.box_min)),
                "box_max": list(map(float, o.box_max)),
                "roughness": float(o.roughness),
                "material": str(o.material),
            }
            for o in scene.obstacles
        ],
        "params": {
            "n_rays": scene.n_rays,
            "n_max": scene.n_max,
            "use_physics": scene.use_physics,
            "temperature_c": scene.temperature_c,
            "bandwidth_hz": scene.bandwidth_hz,
        },
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2))
    print(f"[urban_gen] Saved -> {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--domain-x", type=float, default=600.0)
    p.add_argument("--domain-y", type=float, default=600.0)
    p.add_argument("--block-size-x", type=float, default=80.0)
    p.add_argument("--block-size-y", type=float, default=80.0)
    p.add_argument("--street-width", type=float, default=12.0)
    p.add_argument("--n-tx", type=int, default=3)
    p.add_argument("--cache-dir", type=Path, default=Path("./cache"))
    args = p.parse_args(argv)

    cfg = UrbanConfig(
        seed=args.seed,
        domain_x=args.domain_x,
        domain_y=args.domain_y,
        block_size_x=args.block_size_x,
        block_size_y=args.block_size_y,
        street_width=args.street_width,
        n_transmitters=args.n_tx,
    )

    scene = generate_urban_scene(cfg)
    save_scene(scene, args.cache_dir / "scenes" / f"urban_{cfg.seed}.json")


if __name__ == "__main__":
    main()