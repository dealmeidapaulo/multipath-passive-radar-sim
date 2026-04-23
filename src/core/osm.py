"""
core/scene/osm.py
==================
OpenStreetMap → Scene pipeline.  
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import osmnx as ox
import pyproj
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shapely_transform


from .scene.domain import (
    Box, MeshObstacle, Obstacle, Receiver, Scene, Transmitter, UAV,
)


# ── Material catalogue ────────────────────────────────────────────────────────

# Roughness values: 0 = perfect mirror, 1 = fully Lambertian
_MATERIAL_ROUGHNESS: Dict[str, float] = {
    "concrete"  : 0.1,
    "brick"     : 0.25,
    "glass"     : 0.05,
    "metal"     : 0.02,
    "wood"      : 0.4,
    "wet_ground": 0.3,
    "dry_ground": 0.5,
    "road"      : 0.35,
    "default"   : 0.15,
}

# Map OSM building type tags → material label
_OSM_TYPE_TO_MATERIAL: Dict[str, str] = {
    "residential"   : "brick",
    "apartments"    : "concrete",
    "house"         : "brick",
    "detached"      : "brick",
    "commercial"    : "concrete",
    "office"        : "glass",
    "retail"        : "glass",
    "industrial"    : "metal",
    "warehouse"     : "metal",
    "school"        : "concrete",
    "university"    : "concrete",
    "hospital"      : "concrete",
    "church"        : "brick",
    "cathedral"     : "brick",
    "yes"           : "concrete",   # generic "building=yes"
}

DEFAULT_HEIGHT_M    = 12.0   # fallback when no height info in OSM
METRES_PER_LEVEL    = 3.0    # floor height estimate
ROAD_HEIGHT_M       = 0.15   # road slab thickness above ground
MARGIN_FRACTION     = 0.15   # extra fetch margin to avoid edge effects


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _utm_crs_from_latlon(lat: float, lon: float) -> str:
    """Return the EPSG code string for the UTM zone containing (lat, lon)."""
    zone   = int((lon + 180) / 6) + 1
    hem    = "6" if lat >= 0 else "7"   # 326xx N, 327xx S
    return f"EPSG:32{hem}{zone:02d}"


def _make_transformer(src_crs: str, dst_crs: str):
    """Return a pyproj Transformer (always_xy=True)."""
    return pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def _project_polygon(poly: Polygon, transformer) -> Polygon:
    """Project a Shapely polygon with a pyproj Transformer."""
    return shapely_transform(transformer.transform, poly)


# ── Mesh extrusion ────────────────────────────────────────────────────────────

def _extrude_polygon(
    ring_xy  : np.ndarray,   # (N, 2) exterior ring in metres, counter-clockwise
    z_bottom : float,
    z_top    : float,
    origin   : np.ndarray,   # (2,) local origin to subtract
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrude a 2-D ring into a closed triangular mesh.

    Returns
    -------
    vertices : float64[V, 3]
    faces    : int32[F, 3]
    """
    ring_xy = ring_xy - origin[np.newaxis, :]   # shift to local frame
    N = len(ring_xy)

    # Build vertex array: bottom ring then top ring
    verts_bot = np.column_stack([ring_xy, np.full(N, z_bottom)])
    verts_top = np.column_stack([ring_xy, np.full(N, z_top)])
    vertices  = np.vstack([verts_bot, verts_top]).astype(np.float64)

    faces: List[List[int]] = []

    # Side walls: each edge of the ring → 2 triangles
    for i in range(N):
        j  = (i + 1) % N
        b0 = i;     b1 = j          # bottom edge indices
        t0 = i + N; t1 = j + N      # corresponding top edge indices
        faces.append([b0, b1, t0])
        faces.append([b1, t1, t0])

    # Bottom cap: fan triangulation from vertex 0
    for i in range(1, N - 1):
        faces.append([0, i + 1, i])   # clockwise from outside = inward normal

    # Top cap
    for i in range(1, N - 1):
        faces.append([N, N + i, N + i + 1])

    return vertices, np.array(faces, dtype=np.int32)


def _ring_to_array(coords) -> Optional[np.ndarray]:
    """Convert a coordinate sequence to (N, 2) float64. Returns None if degenerate."""
    pts = np.array([(x, y) for x, y, *_ in coords], dtype=np.float64)
    # Drop closing duplicate
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    return pts


def _building_height(tags: Dict[str, Any]) -> float:
    """Derive building height (metres) from OSM tags."""
    if "height" in tags:
        try:
            return float(tags["height"])
        except (ValueError, TypeError):
            pass
    if "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * METRES_PER_LEVEL
        except (ValueError, TypeError):
            pass
    return DEFAULT_HEIGHT_M


def _building_material(tags: Dict[str, Any]) -> str:
    """Map OSM building tag to a material label."""
    btype = tags.get("building", "yes")
    if isinstance(btype, str):
        btype = btype.lower().strip()
        if btype in _OSM_TYPE_TO_MATERIAL:
            return _OSM_TYPE_TO_MATERIAL[btype]
    return "concrete"


# ── Road mesh ─────────────────────────────────────────────────────────────────

def _road_mesh_from_polygon(
    poly   : Polygon,
    origin : np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a flat slab mesh for a road polygon (flat extrusion from 0 to ROAD_HEIGHT_M).
    """
    ring = _ring_to_array(poly.exterior.coords)
    if ring is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return _extrude_polygon(ring, 0.0, ROAD_HEIGHT_M, origin)


# ── OSM fetch and conversion ──────────────────────────────────────────────────

def _fetch_buildings(
    lat: float, lon: float, dist_m: float
) -> List[Dict[str, Any]]:
    """
    Fetch building footprints from OSM within dist_m metres of (lat, lon).
    Returns a list of dicts with keys 'geometry' (Shapely), 'tags'.
    """

    margin = int(dist_m * (1.0 + MARGIN_FRACTION))
    gdf = ox.features_from_point((lat, lon), dist=margin, tags={"building": True})
    result = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        tags = {k: v for k, v in row.items() if k != "geometry"}
        result.append({"geometry": geom, "tags": tags})
    return result


def _fetch_roads(
    lat: float, lon: float, dist_m: float, lane_width_m: float = 3.5
) -> List[Polygon]:
    """
    Fetch road polygons (buffered LineStrings) from OSM.
    """

    margin = int(dist_m * (1.0 + MARGIN_FRACTION))
    try:
        gdf = ox.features_from_point((lat, lon), dist=margin, tags={"highway": True})
    except Exception:
        return []
    polys: List[Polygon] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        lanes = 1
        try:
            lanes = max(1, int(row.get("lanes", 1) or 1))
        except (ValueError, TypeError):
            pass
        width = lanes * lane_width_m
        try:
            buf = geom.buffer(width / 2.0)
            if isinstance(buf, Polygon):
                polys.append(buf)
        except Exception:
            pass
    return polys


def _polygons_from_geometry(geom) -> List[Polygon]:
    """Yield individual Polygon objects from a geometry (handles MultiPolygon)."""
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


# ── Public API ────────────────────────────────────────────────────────────────

def load_osm_scene(
    lat          : float,
    lon          : float,
    dist_m       : float,
    tx_configs   : List[Dict[str, Any]],
    rx_config    : Dict[str, Any],
    include_roads: bool  = False,
    n_rays       : int   = 100_000,
    n_max        : int   = 8,
    uav_config   : Optional[Dict[str, Any]] = None,
    domain_height: float = 100.0,
    temperature_c: float = 20.0,
    bandwidth_hz : float = 20e6,
) -> Scene:
    """
    Build a Scene from OpenStreetMap data.

    Parameters
    ----------
    lat, lon     : centre of the area of interest (decimal degrees, WGS84)
    dist_m       : half-side of the square bounding box (metres)
    tx_configs   : list of transmitter dicts:
                   {"position": [x,y,z], "frequency": Hz, "tx_power_w": W, "tx_id": int}
                   Positions are in local Cartesian metres relative to (lat,lon).
    rx_config    : receiver dict: {"position": [x,y,z], "radius": m}
    include_roads: if True, also fetch and extrude road polygons
    n_rays       : rays per TX for precompute
    n_max        : max bounces
    uav_config   : optional {"position":[x,y,z],"velocity":[vx,vy,vz],"radius":r}
    domain_height: Z extent of the Box in metres
    temperature_c: noise temperature
    bandwidth_hz : receiver bandwidth

    Returns
    -------
    Scene with MeshObstacle list, ready for precompute or save_scene().
    """

    utm_crs  = _utm_crs_from_latlon(lat, lon)
    wgs84_to_utm = _make_transformer("EPSG:4326", utm_crs)

    # Origin in UTM (centre of the AOI)
    origin_x, origin_y = wgs84_to_utm.transform(lon, lat)
    origin = np.array([origin_x, origin_y], dtype=np.float64)

    # ── Fetch buildings ───────────────────────────────────────────────────────
    raw_buildings = _fetch_buildings(lat, lon, dist_m)

    # Source CRS of the GeoDataFrame returned by osmnx is EPSG:4326
    src_crs = "EPSG:4326"
    poly_transformer = _make_transformer(src_crs, utm_crs)

    obstacles: List[MeshObstacle] = []

    for item in raw_buildings:
        geom = item["geometry"]
        tags = item["tags"]
        height   = _building_height(tags)
        material = _building_material(tags)
        roughness = _MATERIAL_ROUGHNESS.get(material, _MATERIAL_ROUGHNESS["default"])

        for poly in _polygons_from_geometry(geom):
            # Project to UTM
            try:
                poly_utm = _project_polygon(poly, poly_transformer)
            except Exception:
                continue
            if not poly_utm.is_valid or poly_utm.area < 1.0:
                continue

            ring = _ring_to_array(poly_utm.exterior.coords)
            if ring is None:
                continue

            # Filter by distance from origin
            cx, cy = poly_utm.centroid.x, poly_utm.centroid.y
            if math.hypot(cx - origin_x, cy - origin_y) > dist_m * (1 + MARGIN_FRACTION):
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

    # ── Fetch roads (optional) ────────────────────────────────────────────────
    if include_roads:
        road_polys = _fetch_roads(lat, lon, dist_m)
        for poly in road_polys:
            try:
                poly_utm = _project_polygon(poly, poly_transformer)
            except Exception:
                continue
            if not poly_utm.is_valid or poly_utm.area < 1.0:
                continue
            try:
                verts, faces = _road_mesh_from_polygon(poly_utm, origin)
            except Exception:
                continue
            if len(faces) == 0:
                continue
            obstacles.append(MeshObstacle(
                vertices  = verts,
                faces     = faces,
                roughness = _MATERIAL_ROUGHNESS["road"],
                material  = "road",
            ))

    # ── Domain box ────────────────────────────────────────────────────────────
    box = Box(
        box_min=np.array([-dist_m, -dist_m, 0.0]),
        box_max=np.array([ dist_m,  dist_m, domain_height]),
    )

    # ── Transmitters ──────────────────────────────────────────────────────────
    transmitters = []
    for i, cfg in enumerate(tx_configs):
        transmitters.append(Transmitter(
            position   = np.array(cfg["position"], dtype=float),
            frequency  = float(cfg.get("frequency", 700e6)),
            tx_power_w = float(cfg.get("tx_power_w", 500.0)),
            tx_id      = int(cfg.get("tx_id", i)),
        ))

    # ── Receiver ──────────────────────────────────────────────────────────────
    receiver = Receiver(
        position = np.array(rx_config["position"], dtype=float),
        radius   = float(rx_config.get("radius", 5.0)),
    )

    # ── UAV (optional) ────────────────────────────────────────────────────────
    uav = None
    if uav_config is not None:
        uav = UAV(
            position = np.array(uav_config["position"], dtype=float),
            velocity = np.array(uav_config.get("velocity", [0, 0, 0]), dtype=float),
            radius   = float(uav_config.get("radius", 0.5)),
        )

    scene = Scene(
        box          = box,
        transmitters = transmitters,
        receiver     = receiver,
        uav          = uav,
        obstacles    = obstacles,
        n_rays       = n_rays,
        n_max        = n_max,
        temperature_c = temperature_c,
        bandwidth_hz  = bandwidth_hz,
        use_physics   = True,
    )

    return scene


# ── Persistence ───────────────────────────────────────────────────────────────

def save_scene(scene: Scene, directory: str | Path) -> Path:
    """
    Persist a Scene to disk.

    Layout
    ------
    <directory>/
        scene.json          metadata (box, transmitters, receiver, obstacle list)
        meshes/
            obs_0000.npz    vertices + faces for MeshObstacle obstacles
            ...

    AABB Obstacle objects are stored inline in scene.json.
    MeshObstacle objects reference their .npz file.

    Returns the Path to the directory.
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
                "type"     : "mesh",
                "file"     : fname,
                "roughness": float(obs.roughness),
                "material" : obs.material,
            })
        else:  # Obstacle (AABB)
            obs_meta.append({
                "type"     : "aabb",
                "box_min"  : obs.box_min.tolist(),
                "box_max"  : obs.box_max.tolist(),
                "roughness": float(obs.roughness),
                "material" : obs.material,
            })

    meta: Dict[str, Any] = {
        "box_min"     : scene.box.box_min.tolist(),
        "box_max"     : scene.box.box_max.tolist(),
        "n_rays"      : scene.n_rays,
        "n_max"       : scene.n_max,
        "use_physics" : scene.use_physics,
        "temperature_c": scene.temperature_c,
        "bandwidth_hz" : scene.bandwidth_hz,
        "uav_roughness": scene.uav_roughness,
        "n_samples_uav": scene.n_samples_uav,
        "transmitters": [
            {
                "position"  : tx.position.tolist(),
                "frequency" : tx.frequency,
                "tx_power_w": tx.tx_power_w,
                "tx_id"     : tx.tx_id,
            }
            for tx in scene.transmitters
        ],
        "receiver": {
            "position": scene.receiver.position.tolist(),
            "radius"  : scene.receiver.radius,
        },
        "uav": None if scene.uav is None else {
            "position": scene.uav.position.tolist(),
            "velocity": scene.uav.velocity.tolist(),
            "radius"  : scene.uav.radius,
        },
        "obstacles": obs_meta,
    }

    (out / "scene.json").write_text(json.dumps(meta, indent=2))
    return out


def load_scene(directory: str | Path) -> Scene:
    """
    Load a Scene previously saved with save_scene().
    """
    out  = Path(directory)
    meta = json.loads((out / "scene.json").read_text())

    box = Box(
        box_min=np.array(meta["box_min"]),
        box_max=np.array(meta["box_max"]),
    )

    transmitters = [
        Transmitter(
            position   = np.array(tx["position"]),
            frequency  = float(tx["frequency"]),
            tx_power_w = float(tx["tx_power_w"]),
            tx_id      = int(tx["tx_id"]),
        )
        for tx in meta["transmitters"]
    ]

    rx_d    = meta["receiver"]
    receiver = Receiver(
        position=np.array(rx_d["position"]),
        radius  =float(rx_d["radius"]),
    )

    uav = None
    if meta.get("uav") is not None:
        u   = meta["uav"]
        uav = UAV(
            position=np.array(u["position"]),
            velocity=np.array(u["velocity"]),
            radius  =float(u["radius"]),
        )

    obstacles: List[Any] = []
    for obs_d in meta["obstacles"]:
        if obs_d["type"] == "mesh":
            data = np.load(out / "meshes" / obs_d["file"])
            obstacles.append(MeshObstacle(
                vertices  = data["vertices"],
                faces     = data["faces"],
                roughness = float(obs_d["roughness"]),
                material  = obs_d.get("material", "concrete"),
            ))
        else:  # aabb
            obstacles.append(Obstacle(
                box_min   = np.array(obs_d["box_min"]),
                box_max   = np.array(obs_d["box_max"]),
                roughness = float(obs_d["roughness"]),
                material  = obs_d.get("material", "concrete"),
            ))

    scene = Scene(
        box           = box,
        transmitters  = transmitters,
        receiver      = receiver,
        uav           = uav,
        obstacles     = obstacles,
        n_rays        = int(meta["n_rays"]),
        n_max         = int(meta["n_max"]),
        use_physics   = bool(meta["use_physics"]),
        temperature_c = float(meta["temperature_c"]),
        bandwidth_hz  = float(meta["bandwidth_hz"]),
        uav_roughness = float(meta.get("uav_roughness", 0.3)),
        n_samples_uav = int(meta.get("n_samples_uav", 8)),
    )
    return scene