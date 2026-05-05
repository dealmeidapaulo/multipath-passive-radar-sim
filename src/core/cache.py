"""
Registry : /cache/field_registry.json
           Contains one entry per cached field with hash, filename, params.
Fields   : /cache/precomputed_static_fields/<name>.npz
Scenes   : /cache/precomputed_static_fields/<name>.json   (co-located with .npz)

Workflow
--------
  get_or_compute(scene, seed, cell_size, cache_dir)
    → checks field_registry.json for a matching hash
    → if found: loads and returns the cached StaticField
    → if not:   runs precompute(), saves scene + field to disk,
                updates registry, returns
"""
from __future__ import annotations

import hashlib, json, math, time
from datetime import datetime, timezone
from pathlib  import Path
from typing   import List, Optional, Set, Tuple

import numpy as np

from .scene.ray               import Ray
from .precompute.static_field import StaticField
from .precompute.hash         import SpatialHash
from .precompute.precompute   import precompute


# ── Hash ─────────────────────────────────────────────────────────────────────

def _scene_params(scene, seed: int, cell_size: float) -> dict:
    """Extract all parameters that determine the precompute output."""
    txs = [{"pos": [round(float(x), 8) for x in tx.position],
             "freq": round(float(tx.frequency), 2),
             "power_w": round(float(tx.tx_power_w), 6),
             "tx_id": int(tx.tx_id)}
            for tx in scene.transmitters]

    obs = sorted([
        (
            tuple(round(x, 4) for x in o.box_min),
            tuple(round(x, 4) for x in o.box_max),
            round(getattr(o, "roughness", 0.0), 6),
            str(getattr(o, "material", "concrete"))
        )
        for o in scene.obstacles
    ])

    return {
        "seed": int(seed),
        "cell_size": round(float(cell_size), 6),
        "n_rays": int(scene.n_rays),
        "n_max": int(scene.n_max),
        "use_physics": bool(scene.use_physics),
        "temperature_c": round(float(scene.temperature_c), 4),
        "bandwidth_hz": round(float(scene.bandwidth_hz), 2),
        "box_min": [round(float(x), 4) for x in scene.box.box_min],
        "box_max": [round(float(x), 4) for x in scene.box.box_max],
        "transmitters": txs,
        "obstacles": obs
    }


def hash_scene(scene, seed: int, cell_size: float) -> str:
    """16-char hex hash of scene parameters."""
    params = _scene_params(scene, seed, cell_size)
    blob   = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ── Registry ─────────────────────────────────────────────────────────────────

def _registry_path(cache_dir: Path) -> Path:
    return cache_dir / "field_registry.json"


def _load_registry(cache_dir: Path) -> List[dict]:
    p = _registry_path(cache_dir)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _save_registry(cache_dir: Path, entries: List[dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _registry_path(cache_dir).write_text(json.dumps(entries, indent=2))


def _find_entry(entries: List[dict], h: str) -> Optional[dict]:
    for e in entries:
        if e.get("hash") == h:
            return e
    return None


# ── Serialise / deserialise Scene ─────────────────────────────────────────────

def save_scene(scene, filepath: Path) -> None:
    """Serialise a Scene to JSON.  filepath is created (parents included)."""
    data = {
        "box": {
            "box_min": [float(x) for x in scene.box.box_min],
            "box_max": [float(x) for x in scene.box.box_max],
        },
        "transmitters": [
            {
                "position":  [float(x) for x in tx.position],
                "frequency": float(tx.frequency),
                "tx_power_w": float(tx.tx_power_w),
                "tx_id":     int(tx.tx_id),
            }
            for tx in scene.transmitters
        ],
        "obstacles": [
            {
                "box_min":   [float(x) for x in o.box_min],
                "box_max":   [float(x) for x in o.box_max],
                "roughness": float(getattr(o, "roughness", 0.0)),
                "material":  str(getattr(o, "material", "concrete")),
            }
            for o in scene.obstacles
        ],
        "params": {
            "n_rays":       scene.n_rays,
            "n_max":        scene.n_max,
            "use_physics":  scene.use_physics,
            "temperature_c": scene.temperature_c,
            "bandwidth_hz": scene.bandwidth_hz,
        },
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2))


def load_scene(filepath: Path):
    """Deserialise a Scene from JSON produced by save_scene()."""
    d = json.loads(filepath.read_text())

    from .scene.domain import Scene, Box, Transmitter, Obstacle

    box = Box(
        box_min=np.array(d["box"]["box_min"], dtype=np.float64),
        box_max=np.array(d["box"]["box_max"], dtype=np.float64),
    )

    transmitters = [
        Transmitter(
            position=np.array(tx["position"], dtype=np.float64),
            frequency=tx["frequency"],
            tx_power_w=tx["tx_power_w"],
            tx_id=tx["tx_id"],
        )
        for tx in d["transmitters"]
    ]

    obstacles = [
        Obstacle(
            box_min=np.array(o["box_min"], dtype=np.float64),
            box_max=np.array(o["box_max"], dtype=np.float64),
            roughness=o["roughness"],
            material=o["material"],
        )
        for o in d["obstacles"]
    ]

    params = d["params"]
    return Scene(
        box=box,
        transmitters=transmitters,
        obstacles=obstacles,
        n_rays=params["n_rays"],
        n_max=params["n_max"],
        use_physics=params["use_physics"],
        temperature_c=params["temperature_c"],
        bandwidth_hz=params["bandwidth_hz"],
    )


# ── Serialise / deserialise StaticField ───────────────────────────────────────

def save_static(static: StaticField, filepath: Path) -> None:
    """Save StaticField tensors to .npz  (scene_ref and rx fields are not serialised)."""
    sh = static.spatial_hash
    anchor_ids_global = np.array(sorted(static.anchor_ids), dtype=np.int64)
    anchor_tx_ids     = np.array([r.transmitter_id for r in static.anchors], dtype=np.int32)

    np.savez_compressed(
        str(filepath),
        pos_cpu           = static.pos_cpu,
        dir_cpu           = static.dir_cpu,
        step_powers       = static.step_powers,
        n_pts_cpu         = static.n_pts_cpu,
        reached_cpu       = static.reached_cpu,
        tx_ids_cpu        = static.tx_ids_cpu,
        anchor_ids_global = anchor_ids_global,
        anchor_tx_ids     = anchor_tx_ids,
        fc                = np.float64(static.fc),
        sh_flat_ray_ids   = sh.flat_ray_ids,
        sh_flat_seg_ids   = sh.flat_seg_ids,
        sh_cell_offsets   = sh.cell_offsets,
        sh_cell_counts    = sh.cell_counts,
        sh_NX             = np.int32(sh.NX),
        sh_NY             = np.int32(sh.NY),
        sh_NZ             = np.int32(sh.NZ),
        sh_cell_size      = np.float64(sh.cell_size),
        sh_box_min        = sh.box_min,
    )


def load_static(filepath: Path, scene) -> StaticField:
    """Load StaticField from .npz and attach scene_ref.

    scene must already be in memory (or loaded via load_scene first).
    For the scene-free path use load_from_cache().
    """
    d = np.load(str(filepath))

    sh = SpatialHash(
        flat_ray_ids = d['sh_flat_ray_ids'],
        flat_seg_ids = d['sh_flat_seg_ids'],
        cell_offsets = d['sh_cell_offsets'],
        cell_counts  = d['sh_cell_counts'],
        NX=int(d['sh_NX']), NY=int(d['sh_NY']), NZ=int(d['sh_NZ']),
        cell_size    = float(d['sh_cell_size']),
        box_min      = d['sh_box_min'],
    )

    pos_cpu  = d['pos_cpu']
    dir_cpu  = d['dir_cpu']
    sp_cpu   = d['step_powers']
    npts_cpu = d['n_pts_cpu']
    fc       = float(d['fc'])
    anchor_ids_global = set(d['anchor_ids_global'].tolist())
    anchor_tx_ids     = d['anchor_tx_ids']

    anchors: List[Ray] = []
    for i, gid in enumerate(sorted(anchor_ids_global)):
        lid = int(gid)
        n   = int(npts_cpu[lid])
        pts = [pos_cpu[j, lid, :].astype(np.float64) for j in range(n)]
        arr = dir_cpu[n - 1, lid, :].astype(np.float64)
        r   = Ray(
            transmitter_id = int(anchor_tx_ids[i]) if i < len(anchor_tx_ids) else 0,
            points         = pts,
            arrival_dir    = arr,
            frequency      = fc,
            power_dbm      = float(sp_cpu[n - 1, lid]),
        )
        r.is_uav_bounce = False
        r.doppler_shift = 0.0
        r.visible       = True
        anchors.append(r)

    return StaticField(
        pos_cpu      = pos_cpu,
        dir_cpu      = dir_cpu,
        step_powers  = sp_cpu,
        n_pts_cpu    = npts_cpu,
        reached_cpu  = d['reached_cpu'],
        tx_ids_cpu   = d['tx_ids_cpu'],
        anchors      = anchors,
        anchor_ids   = anchor_ids_global,
        spatial_hash = sh,
        fc           = fc,
        scene_ref    = scene,
    )


# ── Public API ────────────────────────────────────────────────────────────────

_DEFAULT_CACHE = Path(__file__).parent.parent.parent / "cache"


def get_or_compute(
    scene,
    seed              : int            = 42,
    cell_size         : Optional[float] = None,
    cache_dir         : Optional[Path]  = None,
    verbose           : bool            = True,
    **precompute_kwargs,
) -> StaticField:
    """
    Load a cached StaticField if parameters match; otherwise run precompute()
    and persist both the scene and the result.

    Parameters
    ----------
    scene       : Scene
    seed        : random seed passed to precompute()
    cell_size   : spatial hash cell size (default = 1.0)
    cache_dir   : root of the cache (default: simulation_v4/cache/)
    verbose     : print cache hit/miss info
    **precompute_kwargs : forwarded to precompute() (e.g. batch_size)
    """
    cs         = cell_size if cell_size is not None else 1.0
    cdir       = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    fields_dir = cdir / "precomputed_static_fields"
    fields_dir.mkdir(parents=True, exist_ok=True)

    h       = hash_scene(scene, seed, cs)
    entries = _load_registry(cdir)
    entry   = _find_entry(entries, h)

    if entry is not None:
        fpath = fields_dir / entry["filename"]
        if fpath.exists():
            if verbose:
                print(
                    f"[cache] HIT  {h}  "
                    f"({entry.get('precompute_time_s', '?'):.1f}s saved)"
                    f"  ← {entry['filename']}"
                )
            return load_static(fpath, scene)
        if verbose:
            print(f"[cache] MISS {h}  (file missing, recomputing)")

    # ── Not cached — run precompute ──────────────────────────────────────────
    if verbose:
        print(f"[cache] MISS {h}  — running precompute...")
    t0      = time.time()
    static  = precompute(scene, seed=seed, cell_size=cs, **precompute_kwargs)
    elapsed = time.time() - t0

    # Persist static field
    sf_fname = f"sf_{h}.npz"
    sf_path  = fields_dir / sf_fname
    save_static(static, sf_path)

    # Persist scene alongside the static field so it can be reloaded later
    # without the caller keeping it in memory.
    sc_fname = f"sc_{h}.json"
    sc_path  = fields_dir / sc_fname
    save_scene(scene, sc_path)

    params    = _scene_params(scene, seed, cs)
    new_entry = {
        "hash"              : h,
        "filename"          : sf_fname,
        "scene_filename"    : sc_fname,       # ← new: links field → scene
        "precompute_time_s" : round(elapsed, 2),
        "created_at"        : datetime.now(timezone.utc).isoformat(),
        "params_summary"    : {
            "n_rays"        : int(scene.n_rays),
            "n_max"         : int(scene.n_max),
            "seed"          : int(seed),
            "cell_size"     : round(cs, 3),
            "domain"        : [round(float(x), 1) for x in scene.box.box_max],
            "n_transmitters": len(scene.transmitters),
            "n_obstacles"   : len(scene.obstacles),
            "anchors"       : len(static.anchors),
        },
    }

    entries = [e for e in entries if e.get("hash") != h]
    entries.append(new_entry)
    _save_registry(cdir, entries)

    if verbose:
        print(f"[cache] SAVE {h}  → {sf_fname}, {sc_fname}  ({elapsed:.1f}s)")
    return static


def load_from_cache(
    h         : str,
    cache_dir : Optional[Path] = None,
    verbose   : bool           = True,
) -> Tuple:
    """
    Reconstruct (scene, static_field) from the registry using only the hash.

    No Scene object needs to be in memory beforehand.

    Returns
    -------
    (scene, static_field) — static_field is pre-apply_rx (as stored by
    get_or_compute). Call apply_rx(static_field, receiver) before apply_uav.

    Raises
    ------
    KeyError          : hash not found in registry
    ValueError        : entry predates scene persistence (no scene_filename) —
                        re-run get_or_compute() to refresh it
    FileNotFoundError : .npz or .json file missing from disk
    """
    cdir       = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    fields_dir = cdir / "precomputed_static_fields"

    entries = _load_registry(cdir)
    entry   = _find_entry(entries, h)

    if entry is None:
        raise KeyError(
            f"Hash '{h}' not found in registry at {_registry_path(cdir)}. "
            "Run get_or_compute() first."
        )

    sc_fname = entry.get("scene_filename")
    if sc_fname is None:
        raise ValueError(
            f"Registry entry for '{h}' has no 'scene_filename'. "
            "This entry was created before scene persistence was added. "
            "Re-run get_or_compute() to refresh it."
        )

    sc_path = fields_dir / sc_fname
    sf_path = fields_dir / entry["filename"]

    if not sc_path.exists():
        raise FileNotFoundError(f"Scene file not found: {sc_path}")
    if not sf_path.exists():
        raise FileNotFoundError(f"Static field file not found: {sf_path}")

    if verbose:
        print(f"[cache] LOAD {h}  ← {sc_fname} + {entry['filename']}")

    scene  = load_scene(sc_path)
    static = load_static(sf_path, scene)
    return scene, static


def list_cache(cache_dir: Optional[Path] = None) -> List[dict]:
    cdir    = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    entries = _load_registry(cdir)
    return sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True)