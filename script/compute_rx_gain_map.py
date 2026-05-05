from pathlib import Path
from typing import List

import numpy as np

from src.core.cache import _load_registry, load_static, load_scene
from src.core.precompute.static_field import StaticField
from src.core.scene.domain import Scene

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def load_scene_by_index(idx: int = 0, cache_dir: Path = DEFAULT_CACHE_DIR) -> Scene:
    registry: List[dict] = _load_registry(cache_dir)

    entry = registry[idx]
    filepath = cache_dir / "scenes" / entry["scene_file"]

    return load_scene(filepath)



def load_static_by_index(scene: Scene, idx: int = 0, cache_dir: Path = DEFAULT_CACHE_DIR) -> StaticField:
    registry: List[dict] = _load_registry(cache_dir)

    entry = registry[idx]
    filepath = cache_dir / "precomputed_static_fields" / entry["filename"]

    return load_static(filepath, scene)


import numpy as np

def compute_rx_power(static, rx_pos, radius=1.0, tx_id=None):
    sh = static.spatial_hash

    ray_ids, seg_ids = sh.query(rx_pos, radius)

    if len(ray_ids) == 0:
        return -120.0  # ruido base

    total_mw = 0.0

    # ── 2. Iterar candidatos ─────────────────────────
    for rid, sid in zip(ray_ids, seg_ids):

        n = static.n_pts_cpu[rid]
        if sid >= n - 1:
            continue

        p0 = static.pos_cpu[sid, rid]
        p1 = static.pos_cpu[sid + 1, rid]

        # ── 3. proyección sobre segmento ──────────────
        v = p1 - p0
        w = rx_pos - p0

        t = np.dot(w, v) / (np.dot(v, v) + 1e-12)
        t = np.clip(t, 0.0, 1.0)

        closest = p0 + t * v
        dist = np.linalg.norm(rx_pos - closest)

        if dist > radius:
            continue

        p_dbm_0 = static.step_powers[sid, rid]
        p_dbm_1 = static.step_powers[sid + 1, rid]

        p_dbm = (1 - t) * p_dbm_0 + t * p_dbm_1

        p_dbm -= 20 * np.log10(dist + 1e-3)

        total_mw += dbm_to_mw(p_dbm)

    if total_mw == 0:
        return -120.0

    return mw_to_dbm(total_mw)