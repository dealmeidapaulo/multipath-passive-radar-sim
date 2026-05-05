from __future__ import annotations
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
from shapely.geometry import LineString, Polygon, box as shapely_box
from shapely.ops import polygonize, unary_union

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain import Scene, Obstacle, Transmitter, Box


@dataclass
class UrbanConfig:
    seed: int = 42
    domain_x: float = 400.0
    domain_y: float = 400.0
    urban_density: float = 0.7
    grid_spacing: float = 60.0
    street_width: float = 8.0
    min_block_area: float = 400.0
    building_height_mean: float = 20.0
    building_height_std: float = 8.0
    n_transmitters: int = 3
    tx_height_offset: float = 2.0
    n_rays: int = 5000
    n_max: int = 6


_MATERIALS = {
    "concrete": (0.5, 0.05),
    "brick": (0.8, 0.02),
    "glass": (0.1, 0.01),
}


def _get_random_material(rng):
    m = rng.choice(list(_MATERIALS.keys()))
    mu, sigma = _MATERIALS[m]
    return m, float(np.clip(rng.normal(mu, sigma), 0.05, 0.95))


def generate_street_network(cfg, rng):
    W, H = cfg.domain_x, cfg.domain_y
    angle = rng.uniform(0, math.pi / 2)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])

    segments = []
    for i in np.arange(-W, 2 * W, cfg.grid_spacing):
        p1 = R @ np.array([i, -H]) + [W / 2, H / 2]
        p2 = R @ np.array([i, 2 * H]) + [W / 2, H / 2]
        segments.append(LineString([p1, p2]))

        p3 = R @ np.array([-W, i]) + [W / 2, H / 2]
        p4 = R @ np.array([2 * W, i]) + [W / 2, H / 2]
        segments.append(LineString([p3, p4]))

    return segments


def build_street_geometry(network, cfg):
    street_polys = []

    for line in network:
        poly = line.buffer(cfg.street_width / 2, cap_style=2, join_style=2)
        if not poly.is_empty:
            street_polys.append(poly)

    return unary_union(street_polys)


def generate_urban_scene(cfg: UrbanConfig):
    rng = np.random.default_rng(cfg.seed)
    domain = shapely_box(0, 0, cfg.domain_x, cfg.domain_y)

    network = generate_street_network(cfg, rng)
    street_geom = build_street_geometry(network, cfg)

    buildable = domain.difference(street_geom)

    if buildable.geom_type == "Polygon":
        blocks = [buildable]
    else:
        blocks = list(buildable.geoms)

    obstacles = []

    for block in blocks:
        if block.area < cfg.min_block_area:
            continue

        minx, miny, maxx, maxy = block.bounds

        nx = max(1, int((maxx - minx) / 30))
        ny = max(1, int((maxy - miny) / 30))

        xs = np.linspace(minx, maxx, nx + 1)
        ys = np.linspace(miny, maxy, ny + 1)

        for i in range(nx):
            for j in range(ny):
                cell = shapely_box(xs[i], ys[j], xs[i + 1], ys[j + 1])
                inter = cell.intersection(block)

                if inter.area < 40 or rng.random() > cfg.urban_density:
                    continue

                b = inter.bounds

                h = float(
                    max(
                        4.0,
                        rng.normal(
                            cfg.building_height_mean,
                            cfg.building_height_std,
                        ),
                    )
                )

                mat, rough = _get_random_material(rng)

                obstacles.append(
                    Obstacle(
                        box_min=np.array([b[0], b[1], 0.0]),
                        box_max=np.array([b[2], b[3], h]),
                        material=mat,
                        roughness=rough,
                    )
                )

    txs = []
    if obstacles:
        valid = [
            o for o in obstacles
            if (o.box_max[2] - o.box_min[2]) > 1.0
        ]

        valid.sort(key=lambda x: x.box_max[2], reverse=True)

        for i in range(cfg.n_transmitters):
            b = valid[i % len(valid)]

            cx = 0.5 * (b.box_min[0] + b.box_max[0])
            cy = 0.5 * (b.box_min[1] + b.box_max[1])
            cz = b.box_max[2] + cfg.tx_height_offset

            txs.append(
                Transmitter(
                    position=np.array([cx, cy, cz]),
                    frequency=700e6,
                    tx_power_w=10.0,
                )
            )

    return Scene(
        box=Box(
            np.array([0, 0, 0]),
            np.array([cfg.domain_x, cfg.domain_y, 100.0]),
        ),
        obstacles=obstacles,
        transmitters=txs,
        n_rays=cfg.n_rays,
        n_max=cfg.n_max,
    )


def plot_scene_3d(scene: Scene):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for obs in scene.obstacles:
        if not (
            obs.box_max[0] > obs.box_min[0]
            and obs.box_max[1] > obs.box_min[1]
            and obs.box_max[2] > obs.box_min[2]
        ):
            continue

        mn, mx = obs.box_min, obs.box_max

        x = [mn[0], mx[0], mx[0], mn[0], mn[0], mx[0], mx[0], mn[0]]
        y = [mn[1], mn[1], mx[1], mx[1], mn[1], mn[1], mx[1], mx[1]]
        z = [mn[2], mn[2], mn[2], mn[2], mx[2], mx[2], mx[2], mx[2]]

        vertices = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
        ]

        faces = []
        for v in vertices:
            face = [(x[i], y[i], z[i]) for i in v]
            if len(face) == 4:
                faces.append(face)

        if not faces:
            continue

        poly3d = Poly3DCollection(
            faces,
            facecolors=(0.0, 0.8, 1.0, 0.15),  # cyan suave con transparencia
            edgecolors="cyan",
            linewidths=0.4,
            )
        ax.add_collection3d(poly3d)

    if scene.transmitters:
        tx_positions = np.array([tx.position for tx in scene.transmitters])
        ax.scatter(
            tx_positions[:, 0],
            tx_positions[:, 1],
            tx_positions[:, 2],
            color="red",
            s=50,
            label="TX",
        )

    db = scene.box
    ax.set_xlim(db.box_min[0], db.box_max[0])
    ax.set_ylim(db.box_min[1], db.box_max[1])
    ax.set_zlim(0, db.box_max[2])

    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.set_title(f"Urban Scene ({len(scene.obstacles)} buildings)", color="white")

    ax.tick_params(colors="white")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)

    if scene.transmitters:
        ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    cfg = UrbanConfig()
    scene = generate_urban_scene(cfg)
    print(
        f"Scene generated: {len(scene.obstacles)} buildings, {len(scene.transmitters)} TXs."
    )
    plot_scene_3d(scene)


if __name__ == "__main__":
    main()