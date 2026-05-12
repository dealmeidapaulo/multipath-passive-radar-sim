#!/usr/bin/env python3

"""
python script/urban_scene_gen.py --plot

Main CLI parameters:
    --seed
    --domain / --domain-x / --domain-y
    --block-size / --block-size-x / --block-size-y
    --street-width
    --density
    --residential-frac
    --n-tx
    --tx-freq
    --tx-power
    --n-rays
    --n-max
    --save
    --plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    tx_frequency: float = 700e6
    tx_power_w: float = 250.0
    tx_height_offset: float = 3.0

    n_rays: int = 150_000
    n_max: int = 6

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
        box_min=np.array(
            [min(x0, x1), min(y0, y1), min(z0, z1)],
            dtype=float,
        ),
        box_max=np.array(
            [max(x0, x1), max(y0, y1), max(z0, z1)],
            dtype=float,
        ),
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

    total_h = float(
        np.clip(
            np.exp(
                rng.normal(
                    np.log(max(cfg.h_com_mu * 1.2, 1)),
                    0.5,
                )
            ),
            cfg.h_com_min,
            cfg.h_com_max * 1.3,
        )
    )

    n_bands = rng.integers(2, 5)

    z_cuts = np.sort(
        rng.uniform(total_h * 0.15, total_h * 0.85, n_bands - 1)
    )

    zs = np.concatenate([[0.0], z_cuts, [total_h]])

    obs = []

    for k in range(n_bands):

        shrink = k * rng.uniform(0.07, 0.20)

        sx = max(hx * (1 - shrink), 1.5)
        sy = max(hy * (1 - shrink), 1.5)

        m, r = _sample_mat(rng, _COM_MATS, _COM_WEIGHTS)

        obs.append(
            _obs(
                cx - sx,
                cy - sy,
                zs[k],
                cx + sx,
                cy + sy,
                zs[k + 1],
                m,
                r,
            )
        )

    return obs


def _bldg_lshape(x0, y0, x1, y1, cfg, rng) -> List[Obstacle]:

    dx, dy = x1 - x0, y1 - y0

    fx = rng.uniform(0.45, 0.65)
    fy = rng.uniform(0.45, 0.65)

    m1, r1 = _sample_mat(rng, _COM_MATS, _COM_WEIGHTS)
    m2, r2 = _sample_mat(rng, _COM_MATS, _COM_WEIGHTS)

    h1 = float(
        np.clip(
            rng.normal(cfg.h_com_mu, cfg.h_com_sigma),
            cfg.h_com_min,
            cfg.h_com_max,
        )
    )

    h2 = float(
        np.clip(
            rng.normal(
                cfg.h_com_mu * 0.7,
                cfg.h_com_sigma * 0.5,
            ),
            cfg.h_com_min,
            cfg.h_com_max,
        )
    )

    return [
        _obs(
            x0,
            y0,
            0,
            x0 + dx * fx,
            y1,
            h1,
            m1,
            r1,
        ),
        _obs(
            x0 + dx * fx,
            y0,
            0,
            x1,
            y0 + dy * fy,
            h2,
            m2,
            r2,
        ),
    ]


def _bldg_complex(x0, y0, x1, y1, cfg, rng) -> List[Obstacle]:

    dx, dy = x1 - x0, y1 - y0

    n = rng.integers(2, 5)

    obs = []

    if dx >= dy:

        if dx < 6 * (n - 1):

            return _bldg_box(
                x0,
                y0,
                x1,
                y1,
                float(
                    np.clip(
                        rng.normal(
                            cfg.h_com_mu,
                            cfg.h_com_sigma,
                        ),
                        cfg.h_com_min,
                        cfg.h_com_max,
                    )
                ),
                cfg,
                rng,
                False,
            )

        cuts = np.sort(rng.uniform(x0 + 4, x1 - 4, n - 1))

        xs = np.concatenate([[x0], cuts, [x1]])

        for k in range(len(xs) - 1):

            if xs[k + 1] - xs[k] < 3:
                continue

            h = float(
                np.clip(
                    rng.normal(
                        cfg.h_com_mu,
                        cfg.h_com_sigma,
                    ),
                    cfg.h_com_min,
                    cfg.h_com_max,
                )
            )

            m, r = _sample_mat(rng, _COM_MATS, _COM_WEIGHTS)

            obs.append(
                _obs(
                    xs[k],
                    y0,
                    0,
                    xs[k + 1],
                    y1,
                    h,
                    m,
                    r,
                )
            )

    else:

        if dy < 6 * (n - 1):

            return _bldg_box(
                x0,
                y0,
                x1,
                y1,
                float(
                    np.clip(
                        rng.normal(
                            cfg.h_com_mu,
                            cfg.h_com_sigma,
                        ),
                        cfg.h_com_min,
                        cfg.h_com_max,
                    )
                ),
                cfg,
                rng,
                False,
            )

        cuts = np.sort(rng.uniform(y0 + 4, y1 - 4, n - 1))

        ys = np.concatenate([[y0], cuts, [y1]])

        for k in range(len(ys) - 1):

            if ys[k + 1] - ys[k] < 3:
                continue

            h = float(
                np.clip(
                    rng.normal(
                        cfg.h_com_mu,
                        cfg.h_com_sigma,
                    ),
                    cfg.h_com_min,
                    cfg.h_com_max,
                )
            )

            m, r = _sample_mat(rng, _COM_MATS, _COM_WEIGHTS)

            obs.append(
                _obs(
                    x0,
                    ys[k],
                    0,
                    x1,
                    ys[k + 1],
                    h,
                    m,
                    r,
                )
            )

    return obs


def _buildings_for_parcel(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    is_residential: bool,
    cfg: UrbanConfig,
    rng: np.random.Generator,
) -> List[Obstacle]:

    s = cfg.setback

    bx0, by0 = x0 + s, y0 + s
    bx1, by1 = x1 - s, y1 - s

    if bx1 - bx0 < 2.5 or by1 - by0 < 2.5:
        return []

    if is_residential:

        h = float(
            np.clip(
                rng.normal(
                    cfg.h_res_mu,
                    cfg.h_res_sigma,
                ),
                cfg.h_res_min,
                cfg.h_res_max,
            )
        )

        return _bldg_box(
            bx0,
            by0,
            bx1,
            by1,
            h,
            cfg,
            rng,
            True,
        )

    roll = rng.random()

    if roll < cfg.tower_prob:
        return _bldg_tower(bx0, by0, bx1, by1, cfg, rng)

    if roll < cfg.tower_prob + cfg.lshape_prob:
        return _bldg_lshape(bx0, by0, bx1, by1, cfg, rng)

    if roll < cfg.tower_prob + cfg.lshape_prob + cfg.complex_prob:
        return _bldg_complex(bx0, by0, bx1, by1, cfg, rng)

    h = float(
        np.clip(
            rng.normal(
                cfg.h_com_mu,
                cfg.h_com_sigma,
            ),
            cfg.h_com_min,
            cfg.h_com_max,
        )
    )

    return _bldg_box(
        bx0,
        by0,
        bx1,
        by1,
        h,
        cfg,
        rng,
        False,
    )


def _street_aabbs(
    bx_starts: np.ndarray,
    by_starts: np.ndarray,
    pitch_x: float,
    pitch_y: float,
    cfg: UrbanConfig,
    rng: np.random.Generator,
) -> List[Obstacle]:

    obs = []

    W, H = cfg.domain_x, cfg.domain_y

    sw = cfg.street_width

    ar_extra = sw * 0.5

    rough_asphalt = _MATS["asphalt"][0]

    x_centers = np.concatenate([
        [bx_s - sw / 2 for bx_s in bx_starts],
        [bx_starts[-1] + pitch_x - sw / 2],
    ])

    for xc in x_centers:

        is_arterial = rng.random() < cfg.arterial_prob

        w = sw + ar_extra if is_arterial else sw

        obs.append(
            _obs(
                xc - w / 2,
                0,
                -0.05,
                xc + w / 2,
                H,
                0.10,
                "asphalt",
                rough_asphalt,
            )
        )

    y_centers = np.concatenate([
        [by_s - sw / 2 for by_s in by_starts],
        [by_starts[-1] + pitch_y - sw /2],
    ])

    for yc in y_centers:

        is_arterial = rng.random() < cfg.arterial_prob

        w = sw + ar_extra if is_arterial else sw

        obs.append(
            _obs(
                0,
                yc - w / 2,
                -0.05,
                W,
                yc + w / 2,
                0.10,
                "asphalt",
                rough_asphalt,
            )
        )

    return obs


def _place_transmitters(
    buildings: List[Obstacle],
    cfg: UrbanConfig,
    rng: np.random.Generator,
) -> List[Transmitter]:

    if not buildings:

        return [
            Transmitter(
                position=np.array([
                    cfg.domain_x * (
                        0.5 + 0.35 * np.cos(k * 2 * np.pi / cfg.n_transmitters)
                    ),
                    cfg.domain_y * (
                        0.5 + 0.35 * np.sin(k * 2 * np.pi / cfg.n_transmitters)
                    ),
                    30.0 + cfg.tx_height_offset,
                ]),
                frequency=cfg.tx_frequency,
                tx_power_w=cfg.tx_power_w,
                tx_id=k,
            )
            for k in range(cfg.n_transmitters)
        ]

    bldg_cx = np.array([
        (b.box_min[0] + b.box_max[0]) / 2
        for b in buildings
    ])

    bldg_cy = np.array([
        (b.box_min[1] + b.box_max[1]) / 2
        for b in buildings
    ])

    bldg_h = np.array([
        b.box_max[2]
        for b in buildings
    ])

    nx = int(np.ceil(np.sqrt(cfg.n_transmitters)))
    ny = int(np.ceil(cfg.n_transmitters / nx))

    cx_size = cfg.domain_x / nx
    cy_size = cfg.domain_y / ny

    global_best = buildings[int(np.argmax(bldg_h))]

    txs: List[Transmitter] = []

    for iy in range(ny):

        for ix in range(nx):

            if len(txs) >= cfg.n_transmitters:
                break

            x0c, x1c = ix * cx_size, (ix + 1) * cx_size
            y0c, y1c = iy * cy_size, (iy + 1) * cy_size

            mask = (
                (bldg_cx >= x0c)
                & (bldg_cx < x1c)
                & (bldg_cy >= y0c)
                & (bldg_cy < y1c)
            )

            if not mask.any():

                b = global_best

            else:

                best_local_idx = np.argmax(bldg_h[mask])

                local_indices = np.where(mask)[0]

                b = buildings[local_indices[best_local_idx]]

            jitter_x = rng.uniform(
                -cx_size * 0.05,
                cx_size * 0.05,
            )

            jitter_y = rng.uniform(
                -cy_size * 0.05,
                cy_size * 0.05,
            )

            txs.append(
                Transmitter(
                    position=np.array([
                        float((b.box_min[0] + b.box_max[0]) / 2 + jitter_x),
                        float((b.box_min[1] + b.box_max[1]) / 2 + jitter_y),
                        float(b.box_max[2] + cfg.tx_height_offset),
                    ]),
                    frequency=cfg.tx_frequency,
                    tx_power_w=cfg.tx_power_w,
                    tx_id=len(txs),
                )
            )

    return txs


def generate_urban_scene(cfg: UrbanConfig) -> Scene:

    rng = np.random.default_rng(cfg.seed)


    W, H = cfg.domain_x, cfg.domain_y


    pitch_x = cfg.block_size_x + cfg.street_width
    pitch_y = cfg.block_size_y + cfg.street_width

    offset = cfg.street_width

    bx_starts = np.arange(
        offset,
        W - cfg.block_size_x * 0.5,
        pitch_x,
    )

    by_starts = np.arange(
        offset,
        H - cfg.block_size_y * 0.5,
        pitch_y,
    )



    all_obs: List[Obstacle] = _street_aabbs(
        bx_starts,
        by_starts,
        pitch_x,
        pitch_y,
        cfg,
        rng,
    )


    buildings: List[Obstacle] = []

    for bx in bx_starts:

        for by in by_starts:

            noise_x = rng.normal(1.0, cfg.block_noise)
            noise_y = rng.normal(1.0, cfg.block_noise)

            bw = float(
                np.clip(
                    cfg.block_size_x * noise_x,
                    cfg.block_size_x * 0.5,
                    cfg.block_size_x * 1.4,
                )
            )

            bh = float(
                np.clip(
                    cfg.block_size_y * noise_y,
                    cfg.block_size_y * 0.5,
                    cfg.block_size_y * 1.4,
                )
            )

            bx_end = min(
                bx + bw,
                W - cfg.street_width,
            )

            by_end = min(
                by + bh,
                H - cfg.street_width,
            )

            if bx_end - bx < 10 or by_end - by < 10:
                continue

            is_residential = (
                rng.random() < cfg.residential_frac
            )

            step = (
                cfg.parcel_step_res
                if is_residential
                else cfg.parcel_step_com
            )

            px_coords = np.arange(
                bx,
                bx_end - step * 0.2,
                step,
            )

            py_coords = np.arange(
                by,
                by_end - step * 0.2,
                step,
            )

            for px in px_coords:

                for py in py_coords:

                    if rng.random() > cfg.density:
                        continue

                    px_end = min(px + step, bx_end)
                    py_end = min(py + step, by_end)

                    bldg = _buildings_for_parcel(
                        px,
                        py,
                        px_end,
                        py_end,
                        is_residential,
                        cfg,
                        rng,
                    )

                    buildings.extend(bldg)

    all_obs.extend(buildings)


    txs = _place_transmitters(
        buildings,
        cfg,
        rng,
    )



    return Scene(
        box=Box(
            box_min=np.array([0.0, 0.0, -0.5]),
            box_max=np.array([
                W,
                H,
                cfg.h_com_max * 1.5,
            ]),
        ),
        obstacles=all_obs,
        transmitters=txs,
        n_rays=cfg.n_rays,
        n_max=cfg.n_max,
        use_physics=cfg.use_physics,
        temperature_c=cfg.temperature_c,
        bandwidth_hz=cfg.bandwidth_hz,
    )


def plot_scene_3d(scene: Scene):

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    fig.patch.set_facecolor("#050816")

    ax.set_facecolor("#050816")

    all_faces = []

    for obs in scene.obstacles:

        mn = obs.box_min
        mx = obs.box_max

        if np.any(mx <= mn):
            continue

        x0, y0, z0 = mn
        x1, y1, z1 = mx

        v = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])

        faces = [
            [v[0], v[1], v[2], v[3]],
            [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]],
            [v[2], v[3], v[7], v[6]],
            [v[0], v[3], v[7], v[4]],
            [v[1], v[2], v[6], v[5]],
        ]

        all_faces.extend(faces)

    poly = Poly3DCollection(
        all_faces,
        facecolors=(0.1, 0.7, 0.5, 0.18),
        edgecolors=(0.2, 0.9, 1.0, 0.5),
        linewidths=0.25,
    )

    ax.add_collection3d(poly)

    if scene.transmitters:

        tx = np.array([
            t.position
            for t in scene.transmitters
        ])

        ax.scatter(
            tx[:, 0],
            tx[:, 1],
            tx[:, 2],
            c="#00ff88",
            s=80,
            depthshade=False,
            label="TX",
        )

    db = scene.box

    ax.set_xlim(
        db.box_min[0],
        db.box_max[0],
    )

    ax.set_ylim(
        db.box_min[1],
        db.box_max[1],
    )

    ax.set_zlim(
        0,
        db.box_max[2],
    )

    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")

    ax.tick_params(colors="white")

    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_title(
        f"Urban Scene ({len(scene.obstacles)} obstacles)",
        color="white",
    )

    plt.tight_layout()

    plt.show()


def save_scene(scene: Scene, filepath: Path) -> None:

    import json

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

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    filepath.write_text(
        json.dumps(data, indent=2)
    )

    kb = filepath.stat().st_size / 1024

    print(
        f"[urban_gen] Saved -> {filepath} ({kb:.1f} KB)"
    )


def _build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser()

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--domain",
        type=float,
        default=None,
    )

    p.add_argument(
        "--domain-x",
        type=float,
        default=600.0,
    )

    p.add_argument(
        "--domain-y",
        type=float,
        default=600.0,
    )

    p.add_argument(
        "--block-size",
        type=float,
        default=None,
    )

    p.add_argument(
        "--block-size-x",
        type=float,
        default=80.0,
    )

    p.add_argument(
        "--block-size-y",
        type=float,
        default=80.0,
    )

    p.add_argument(
        "--street-width",
        type=float,
        default=12.0,
    )

    p.add_argument(
        "--block-noise",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--arterial-prob",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--residential-frac",
        type=float,
        default=0.65,
    )

    p.add_argument(
        "--density",
        type=float,
        default=0.80,
    )

    p.add_argument(
        "--h-res",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--h-com",
        type=float,
        default=22.0,
    )

    p.add_argument(
        "--h-com-max",
        type=float,
        default=60.0,
    )

    p.add_argument(
        "--tower-prob",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--lshape-prob",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--complex-prob",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--n-tx",
        type=int,
        default=3,
    )

    p.add_argument(
        "--tx-freq",
        type=float,
        default=700e6,
    )

    p.add_argument(
        "--tx-power",
        type=float,
        default=250.0,
    )

    p.add_argument(
        "--tx-z-offset",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--n-rays",
        type=int,
        default=100_000,
    )

    p.add_argument(
        "--n-max",
        type=int,
        default=6,
    )

    p.add_argument(
        "--save",
        action="store_true",
    )

    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache"),
    )

    p.add_argument(
        "--output-name",
        type=str,
        default=None,
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    p.add_argument(
        "--plot",
        action="store_true",
    )

    return p


def main(argv=None) -> None:

    args = _build_parser().parse_args(argv)

    if args.domain is not None:
        args.domain_x = args.domain_y = args.domain

    if args.block_size is not None:
        args.block_size_x = args.block_size_y = args.block_size

    seed = (
        args.seed
        if args.seed >= 0
        else int(np.random.SeedSequence().entropy & 0xFFFFFFFF)
    )

    cfg = UrbanConfig(
        seed=seed,
        domain_x=args.domain_x,
        domain_y=args.domain_y,
        block_size_x=args.block_size_x,
        block_size_y=args.block_size_y,
        street_width=args.street_width,
        block_noise=args.block_noise,
        arterial_prob=args.arterial_prob,
        residential_frac=args.residential_frac,
        density=args.density,
        h_res_mu=args.h_res,
        h_com_mu=args.h_com,
        h_com_max=args.h_com_max,
        tower_prob=args.tower_prob,
        lshape_prob=args.lshape_prob,
        complex_prob=args.complex_prob,
        n_transmitters=args.n_tx,
        tx_frequency=args.tx_freq,
        tx_power_w=args.tx_power,
        tx_height_offset=args.tx_z_offset,
        n_rays=args.n_rays,
        n_max=args.n_max,
    )

    name = args.output_name or f"urban_{cfg.seed}"

    filepath = (
        args.cache_dir
        / "scenes"
        / f"{name}.json"
    )

    if filepath.exists() and not args.force and args.save:

        scene = None

    else:

        scene = generate_urban_scene(cfg)

        if args.save:
            save_scene(scene, filepath)

    if args.plot:

        if scene is None:
            scene = generate_urban_scene(cfg)

        plot_scene_3d(scene)


if __name__ == "__main__":
    main()