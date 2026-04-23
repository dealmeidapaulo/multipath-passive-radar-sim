from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import osmnx as ox
    import pyproj
    from shapely.geometry import shape, MultiPolygon, Polygon
    from shapely.ops import transform as shapely_transform
    _HAS_GEO = True
except ImportError:
    _HAS_GEO = False

from .domain import Box, MeshObstacle, Obstacle, Scene, Transmitter


# ── Material catalogue ────────────────────────────────────────────────────────

_MATERIAL_ROUGHNESS: Dict[str, float] = {
    "concrete"  : 0.10,
    "brick"     : 0.25,
    "glass"     : 0.05,
    "metal"     : 0.02,
    "wood"      : 0.40,
    "wet_ground": 0.30,
    "dry_ground": 0.50,
    "road"      : 0.35,
    "default"   : 0.15,
}

_OSM_TYPE_TO_MATERIAL: Dict[str, str] = {
    "residential": "brick",
    "apartments" : "concrete",
    "house"      : "brick",
    "detached"   : "brick",
    "commercial" : "concrete",
    "office"     : "glass",
    "retail"     : "glass",
    "industrial" : "metal",
    "warehouse"  : "metal",
    "school"     : "concrete",
    "university" : "concrete",
    "hospital"   : "concrete",
    "church"     : "brick",
    "cathedral"  : "brick",
    "yes"        : "concrete",
}

DEFAULT_HEIGHT_M  = 12.0
METRES_PER_LEVEL  = 3.0
ROAD_HEIGHT_M     = 0.15
MARGIN_FRACTION   = 0.15


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180) / 6) + 1
    hem  = "6" if lat >= 0 else "7"
    return f"EPSG:32{hem}{zone:02d}"


def _make_transformer(src: str, dst: str):
    return pyproj.Transformer.from_crs(src, dst, always_xy=True)


def _project_polygon(poly: Polygon, transformer) -> Polygon:
    return shapely_transform(transformer.transform, poly)


# ── Mesh extrusion ────────────────────────────────────────────────────────────

def _ring_to_array(coords) -> Optional[np.ndarray]:
    pts = np.array([(x, y) for x, y, *_ in coords], dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts if len(pts) >= 3 else None


def _extrude_polygon(
    ring_xy : np.ndarray,
    z_bottom: float,
    z_top   : float,
    origin  : np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrude a 2-D ring to a closed triangular mesh."""
    ring_xy = ring_xy - origin[np.newaxis, :]
    N = len(ring_xy)

    verts_bot = np.column_stack([ring_xy, np.full(N, z_bottom)])
    verts_top = np.column_stack([ring_xy, np.full(N, z_top)])
    vertices  = np.vstack([verts_bot, verts_top]).astype(np.float64)

    faces: List[List[int]] = []
    for i in range(N):
        j  = (i + 1) % N
        b0 = i;     b1 = j
        t0 = i + N; t1 = j + N
        faces.append([b0, b1, t0])
        faces.append([b1, t1, t0])
    for i in range(1, N - 1):
        faces.append([0, i + 1, i])
    for i in range(1, N - 1):
        faces.append([N, N + i, N + i + 1])

    return vertices, np.array(faces, dtype=np.int32)


def _building_height(tags: Dict[str, Any]) -> float:
    if "height" in tags:
        try: return float(tags["height"])
        except (ValueError, TypeError): pass
    if "building:levels" in tags:
        try: return float(tags["building:levels"]) * METRES_PER_LEVEL
        except (ValueError, TypeError): pass
    return DEFAULT_HEIGHT_M


def _building_material(tags: Dict[str, Any]) -> str:
    btype = str(tags.get("building", "yes")).lower().strip()
    return _OSM_TYPE_TO_MATERIAL.get(btype, "concrete")


def _polygons_from_geometry(geom) -> List[Polygon]:
    if isinstance(geom, Polygon):     return [geom]
    if isinstance(geom, MultiPolygon): return list(geom.geoms)
    return []


# ── OSM fetch ────────────────────────────────────────────────────────────────

def _fetch_buildings(lat: float, lon: float, dist_m: float) -> List[Dict[str, Any]]:
    if not _HAS_GEO:
        raise ImportError("pip install osmnx pyproj shapely")
    margin = int(dist_m * (1.0 + MARGIN_FRACTION))
    gdf = ox.features_from_point((lat, lon), dist=margin, tags={"building": True})
    return [{"geometry": row.geometry,
             "tags": {k: v for k, v in row.items() if k != "geometry"}}
            for _, row in gdf.iterrows() if row.geometry is not None]


# ── Public API ────────────────────────────────────────────────────────────────

def load_osm_obstacles(
    lat          : float,
    lon          : float,
    dist_m       : float,
    n_rays       : int   = 100_000,
    n_max        : int   = 8,
    domain_height: float = 100.0,
    temperature_c: float = 20.0,
    bandwidth_hz : float = 20e6,
) -> Scene:
    """
    Fetch buildings from OSM and return a Scene with only box + obstacles.

    scene.transmitters is empty.  No Receiver.  No UAV.

    Parameters
    ----------
    lat, lon      : centre of the area of interest (WGS84 decimal degrees)
    dist_m        : half-side of the bounding square (metres)
    n_rays, n_max : precompute defaults stored in Scene
    domain_height : vertical extent of the simulation Box
    """
    if not _HAS_GEO:
        raise ImportError("pip install osmnx pyproj shapely")

    utm_epsg   = _utm_epsg(lat, lon)
    wgs_to_utm = _make_transformer("EPSG:4326", utm_epsg)

    origin_x, origin_y = wgs_to_utm.transform(lon, lat)
    origin = np.array([origin_x, origin_y], dtype=np.float64)

    poly_tf = _make_transformer("EPSG:4326", utm_epsg)

    raw = _fetch_buildings(lat, lon, dist_m)
    obstacles: List[MeshObstacle] = []

    for item in raw:
        geom     = item["geometry"]
        tags     = item["tags"]
        height   = _building_height(tags)
        material = _building_material(tags)
        roughness = _MATERIAL_ROUGHNESS.get(material, _MATERIAL_ROUGHNESS["default"])

        for poly in _polygons_from_geometry(geom):
            try:
                poly_utm = _project_polygon(poly, poly_tf)
            except Exception:
                continue
            if not poly_utm.is_valid or poly_utm.area < 1.0:
                continue

            cx, cy = poly_utm.centroid.x, poly_utm.centroid.y
            if math.hypot(cx - origin_x, cy - origin_y) > dist_m * (1 + MARGIN_FRACTION):
                continue

            ring = _ring_to_array(poly_utm.exterior.coords)
            if ring is None:
                continue

            try:
                verts, faces = _extrude_polygon(ring, 0.0, height, origin)
            except Exception as exc:
                warnings.warn(f"Skipping building (extrusion failed): {exc}")
                continue

            if len(faces) == 0:
                continue

            obstacles.append(MeshObstacle(
                vertices  = verts,
                faces     = faces,
                roughness = roughness,
                material  = material,
            ))

    box = Box(
        box_min=np.array([-dist_m, -dist_m, 0.0]),
        box_max=np.array([ dist_m,  dist_m, domain_height]),
    )

    return Scene(
        box           = box,
        transmitters  = [],          # empty — caller adds TX
        obstacles     = obstacles,
        n_rays        = n_rays,
        n_max         = n_max,
        temperature_c = temperature_c,
        bandwidth_hz  = bandwidth_hz,
        use_physics   = True,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save_scene(scene: Scene, directory: str | Path) -> Path:
    """
    Persist a Scene to disk.

    Layout
    ------
    <directory>/
        scene.json
        meshes/obs_NNNN.npz   (one per MeshObstacle)
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meshes").mkdir(exist_ok=True)

    obs_meta: List[Dict[str, Any]] = []
    for i, obs in enumerate(scene.obstacles):
        if isinstance(obs, MeshObstacle):
            fname = f"obs_{i:04d}.npz"
            np.savez_compressed(
                out / "meshes" / fname,
                vertices=obs.vertices.astype(np.float64),
                faces=obs.faces.astype(np.int32),
            )
            obs_meta.append({
                "type": "mesh", "file": fname,
                "roughness": float(obs.roughness), "material": obs.material,
            })
        else:
            obs_meta.append({
                "type": "aabb",
                "box_min": obs.box_min.tolist(), "box_max": obs.box_max.tolist(),
                "roughness": float(obs.roughness), "material": obs.material,
            })

    meta: Dict[str, Any] = {
        "box_min"      : scene.box.box_min.tolist(),
        "box_max"      : scene.box.box_max.tolist(),
        "n_rays"       : scene.n_rays,
        "n_max"        : scene.n_max,
        "use_physics"  : scene.use_physics,
        "temperature_c": scene.temperature_c,
        "bandwidth_hz" : scene.bandwidth_hz,
        "uav_roughness": scene.uav_roughness,
        "n_samples_uav": scene.n_samples_uav,
        "transmitters" : [
            {"position": tx.position.tolist(), "frequency": tx.frequency,
             "tx_power_w": tx.tx_power_w, "tx_id": tx.tx_id}
            for tx in scene.transmitters
        ],
        "obstacles": obs_meta,
    }
    (out / "scene.json").write_text(json.dumps(meta, indent=2))
    return out


def load_scene(directory: str | Path) -> Scene:
    """Load a Scene previously saved with save_scene()."""
    out  = Path(directory)
    meta = json.loads((out / "scene.json").read_text())

    box = Box(box_min=np.array(meta["box_min"]),
              box_max=np.array(meta["box_max"]))

    transmitters = [
        Transmitter(position=np.array(tx["position"]), frequency=float(tx["frequency"]),
                    tx_power_w=float(tx["tx_power_w"]), tx_id=int(tx["tx_id"]))
        for tx in meta["transmitters"]
    ]

    obstacles: List[Any] = []
    for obs_d in meta["obstacles"]:
        if obs_d["type"] == "mesh":
            data = np.load(out / "meshes" / obs_d["file"])
            obstacles.append(MeshObstacle(
                vertices=data["vertices"], faces=data["faces"],
                roughness=float(obs_d["roughness"]),
                material=obs_d.get("material", "concrete"),
            ))
        else:
            obstacles.append(Obstacle(
                box_min=np.array(obs_d["box_min"]),
                box_max=np.array(obs_d["box_max"]),
                roughness=float(obs_d["roughness"]),
                material=obs_d.get("material", "concrete"),
            ))

    return Scene(
        box           = box,
        transmitters  = transmitters,
        obstacles     = obstacles,
        n_rays        = int(meta["n_rays"]),
        n_max         = int(meta["n_max"]),
        use_physics   = bool(meta["use_physics"]),
        temperature_c = float(meta["temperature_c"]),
        bandwidth_hz  = float(meta["bandwidth_hz"]),
        uav_roughness = float(meta.get("uav_roughness", 0.3)),
        n_samples_uav = int(meta.get("n_samples_uav", 8)),
    )
