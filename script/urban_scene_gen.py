#!/usr/bin/env python3
"""
urban_scene_gen.py
==================
Generador procedural de escenas urbanas basadas exclusivamente en AABBs,
compatible con el formato de cache del simulador TAMI7.

Uso básico:
    python urban_scene_gen.py --seed 42 --cache-dir ./cache

Opciones completas:
    python urban_scene_gen.py --help

──────────────────────────────────────────────────────────────────────────────
ANÁLISIS Y PROPUESTA DE GENERACIÓN DE CALLES
──────────────────────────────────────────────────────────────────────────────

Propuesta original (punto → punto → punto en la recta → ...):
  Construye una cadena de segmentos tipo "paseo del borracho". El grafo
  resultante es un árbol (sin ciclos), lo que impide extraer polígonos
  cerrados → no se pueden generar manzanas. Además, la cobertura espacial
  es altamente aleatoria: el grafo puede agruparse en una zona y dejar
  otras vacías.

Propuesta mejorada: GRILLA PERTURBADA + SUPERPOSICIÓN ORGÁNICA
──────────────────────────────────────────────────────────────────────────────

Fase 1 — Grilla rotada y perturbada (calles primarias)
  - Se define una grilla regular de N×M nodos con espaciado configurable.
  - A la grilla se le aplica una ROTACIÓN GLOBAL (ángulo sorteado por seed),
    de modo que el barrio no sea ortogonal al dominio.
  - Cada nodo se desplaza con ruido gaussiano (perturbación), rompiendo la
    regularidad y produciendo intersecciones no ortogonales.
  - Se conectan nodos vecinos (arriba, derecha, diagonal opcional).
  - Resultado: grafo planar con ciclos garantizados → manzanas extraíbles.

Fase 2 — Calles secundarias orgánicas
  - Se sortean pares de nodos distantes de la grilla y se los conecta.
  - Esto añade "diagonales urbanas" (similar a avenidas que cruzan barrios).

Fase 3 — Campo de dirección suave (variación de orientación entre barrios)
  - Se divide el dominio en sub-regiones y cada una tiene su propia rotación
    de grilla, interpolada suavemente con una función coseno.
  - Los bloques adyacentes tienen orientaciones similares; los bloques
    distantes pueden diferir ±45°. Esto emula ciudades reales donde distintos
    barrios tienen distintas orientaciones (ej. Buenos Aires: microcentro vs
    barrios del sur).


──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from shapely.geometry import (
    LineString, MultiLineString, MultiPolygon,
    Point, Polygon,
    box as shapely_box,
)
from shapely.ops import polygonize, unary_union


from src.core.scene.domain import Scene, Obstacle, Transmitter, Box


# ── Compatibilidad Shapely 1.x / 2.x ─────────────────────────────────────────

def _offset_line(line: LineString, dist: float) -> Optional[LineString]:
    """Offset a LineString a distancia `dist` (positivo = izquierda)."""
    try:
        result = line.offset_curve(dist)          # Shapely ≥ 2.0
    except AttributeError:
        result = line.parallel_offset(            # Shapely 1.x
            abs(dist), side="left" if dist > 0 else "right"
        )
    if result is None or result.is_empty:
        return None
    if result.geom_type == "MultiLineString":
        # Tomar el segmento más largo
        parts = list(result.geoms)
        result = max(parts, key=lambda g: g.length)
    return result if result.geom_type == "LineString" else None


# ═══════════════════════════════════════════════════════════════════════════════
#  DATACLASSES DE DOMINIO  (espejo de scene.domain para uso standalone)
# ═══════════════════════════════════════════════════════════════════════════════











# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UrbanConfig:
    # ── Identidad ──────────────────────────────────────────────────────────────
    seed: int = 42

    # ── Dominio ────────────────────────────────────────────────────────────────
    domain_x: float = 600.0   # [m] extensión este-oeste
    domain_y: float = 600.0   # [m] extensión norte-sur
    urban_density: float = 0.65  # fracción de lotes con edificio [0,1]

    # ── Red de calles ──────────────────────────────────────────────────────────
    grid_cols: int = 7          # nodos horizontales de la grilla base
    grid_rows: int = 7          # nodos verticales de la grilla base
    grid_perturb_frac: float = 0.20  # perturbación como fracción del espaciado
    grid_base_angle_deg: float = -999  # <0 → sorteado por seed
    direction_field_strength: float = 0.35  # variación angular entre barrios [rad]
    n_organic_streets: int = 6  # calles diagonales orgánicas adicionales
    street_connectivity: float = 0.85  # fracción de aristas de grilla activadas

    street_width_primary: float = 12.0   # [m]
    street_width_secondary: float = 7.0  # [m]
    street_height: float = 0.12          # [m] AABB de asfalto

    # ── Manzanas ──────────────────────────────────────────────────────────────
    min_block_area: float = 300.0   # [m²] manzanas más chicas se descartan

    # ── Parcelas ──────────────────────────────────────────────────────────────
    min_parcel_area: float = 60.0    # [m²]
    min_parcel_side: float = 6.0     # [m]
    max_parcel_area: float = 1800.0  # [m²] — si es mayor se fuerza split

    # ── Edificios ─────────────────────────────────────────────────────────────
    building_height_mean: float = 16.0    # [m]
    building_height_std: float = 12.0
    building_height_min: float = 3.0
    building_height_max: float = 80.0
    max_setback: float = 2.5      # retiro máximo desde borde de lote [m]
    tower_prob: float = 0.25      # probabilidad de edificio tipo torre
    complex_prob: float = 0.20    # probabilidad de complejo de bloques

    # ── Elementos opcionales ───────────────────────────────────────────────────
    add_sidewalks: bool = True
    sidewalk_width: float = 2.5
    sidewalk_height: float = 0.12

    add_trees: bool = True
    tree_spacing: float = 14.0   # [m] distancia media entre árboles
    tree_density: float = 0.5    # fracción de posibles árboles que se instancian

    add_obstacles: bool = True
    obstacle_spacing: float = 25.0  # [m] distancia media entre mobiliario
    obstacle_density: float = 0.4

    add_plazas: bool = True
    plaza_fraction: float = 0.06  # fracción de manzanas que son plaza

    # ── Iluminadores ─────────────────────────────────────────────────────────
    n_transmitters: int = 3
    tx_frequency: float = 700e6   # [Hz] DVB-T / LTE
    tx_power_w: float = 1000.0    # [W]
    tx_height: float = 45.0       # [m]
    tx_border_margin: float = 50.0  # [m] — TXs colocados en periferia

    # ── Parámetros de simulación ──────────────────────────────────────────────
    n_rays: int = 2000
    n_max: int = 5
    use_physics: bool = True
    temperature_c: float = 20.0
    bandwidth_hz: float = 8e6


# ═══════════════════════════════════════════════════════════════════════════════
#  BIBLIOTECA DE MATERIALES
# ═══════════════════════════════════════════════════════════════════════════════

#  Cada entrada: (roughness_mean, roughness_std, reflectivity)
_MATERIAL_PROPS: Dict[str, Tuple[float, float, float]] = {
    "concrete":   (0.60, 0.06, 0.30),
    "glass":      (0.05, 0.02, 0.72),
    "brick":      (0.78, 0.06, 0.22),
    "metal":      (0.18, 0.05, 0.62),
    "asphalt":    (0.88, 0.04, 0.08),
    "vegetation": (0.98, 0.02, 0.05),
    "stone":      (0.70, 0.06, 0.24),
    "glass_tint": (0.08, 0.03, 0.55),
}

# Pesos de sampling para fachadas de edificios
_BUILDING_MAT_WEIGHTS = [0.38, 0.18, 0.22, 0.08, 0.00, 0.00, 0.06, 0.08]
_BUILDING_MATS        = ["concrete", "glass", "brick", "metal",
                         "asphalt", "vegetation", "stone", "glass_tint"]


def _sample_material(rng: np.random.Generator,
                     mats: List[str] = _BUILDING_MATS,
                     weights: List[float] = _BUILDING_MAT_WEIGHTS
                     ) -> Tuple[str, float]:
    """Devuelve (material, roughness) sorteado."""
    idx = rng.choice(len(mats), p=np.array(weights) / sum(weights))
    mat = mats[idx]
    mu, sigma, _ = _MATERIAL_PROPS[mat]
    roughness = float(np.clip(rng.normal(mu, sigma), 0.01, 0.99))
    return mat, roughness


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE LA RED DE CALLES
# ═══════════════════════════════════════════════════════════════════════════════

def _rotation_matrix_2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _clip_segment(x0: float, y0: float, x1: float, y1: float,
                  W: float, H: float
                  ) -> Optional[Tuple[float, float, float, float]]:
    """Recorta un segmento al dominio [0,W]×[0,H]. Devuelve None si queda vacío."""
    domain = shapely_box(0, 0, W, H)
    line = LineString([(x0, y0), (x1, y1)])
    clipped = line.intersection(domain)
    if clipped.is_empty:
        return None
    if clipped.geom_type == "LineString":
        coords = list(clipped.coords)
        if len(coords) < 2:
            return None
        return (coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])
    return None


def generate_street_network(
    cfg: UrbanConfig,
    rng: np.random.Generator
) -> List[Tuple[float, float, float, float]]:
    """
    Genera la red de calles como lista de segmentos (x0,y0,x1,y1).

    Algoritmo: grilla perturbada + campo de dirección + calles orgánicas.
    Ver docstring del módulo para justificación detallada.
    """
    W, H = cfg.domain_x, cfg.domain_y
    cx, cy = W / 2, H / 2  # centro del dominio

    # ── Ángulo base de la grilla ───────────────────────────────────────────────
    if cfg.grid_base_angle_deg < -900:
        base_angle = rng.uniform(-math.pi / 4, math.pi / 4)
    else:
        base_angle = math.radians(cfg.grid_base_angle_deg)

    # ── Grilla de nodos ────────────────────────────────────────────────────────
    cols, rows = cfg.grid_cols, cfg.grid_rows
    sp_x = W / (cols - 1)
    sp_y = H / (rows - 1)

    # Campo de dirección: cada nodo tiene una rotación local suave
    def local_angle(i: int, j: int) -> float:
        # Variación coseno sobre el dominio
        fx = math.cos(math.pi * i / max(cols - 1, 1))
        fy = math.cos(math.pi * j / max(rows - 1, 1))
        return base_angle + cfg.direction_field_strength * 0.5 * (fx - fy)

    # Construir nodos con perturbación
    nodes: Dict[Tuple[int, int], np.ndarray] = {}
    perturb = cfg.grid_perturb_frac * min(sp_x, sp_y)
    for i in range(cols):
        for j in range(rows):
            # Posición base en grilla rotada localmente
            theta = local_angle(i, j)
            R = _rotation_matrix_2d(theta)
            local_xy = np.array([i * sp_x - cx, j * sp_y - cy])
            gx, gy = R @ local_xy + np.array([cx, cy])

            # Perturbación gaussiana
            dx, dy = rng.normal(0, perturb, size=2)
            x = float(np.clip(gx + dx, 0, W))
            y = float(np.clip(gy + dy, 0, H))
            nodes[(i, j)] = np.array([x, y])

    # ── Aristas de grilla (con subsampling para variedad) ─────────────────────
    segments: List[Tuple[float, float, float, float]] = []

    def add_edge(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        if a not in nodes or b not in nodes:
            return
        p, q = nodes[a], nodes[b]
        if np.linalg.norm(p - q) < 2.0:
            return  # segmento degenerado
        seg = _clip_segment(p[0], p[1], q[0], q[1], W, H)
        if seg is not None:
            segments.append(seg)

    for i in range(cols):
        for j in range(rows):
            # Eje horizontal
            if i + 1 < cols and rng.random() < cfg.street_connectivity:
                add_edge((i, j), (i + 1, j))
            # Eje vertical
            if j + 1 < rows and rng.random() < cfg.street_connectivity:
                add_edge((i, j), (i, j + 1))

    # ── Calles orgánicas diagonales ────────────────────────────────────────────
    node_list = list(nodes.values())
    organic_added = 0
    attempts = 0
    while organic_added < cfg.n_organic_streets and attempts < cfg.n_organic_streets * 10:
        attempts += 1
        idx_a = rng.integers(0, len(node_list))
        idx_b = rng.integers(0, len(node_list))
        if idx_a == idx_b:
            continue
        pa, pb = node_list[idx_a], node_list[idx_b]
        dist = float(np.linalg.norm(pa - pb))
        # Solo conectar nodos lo suficientemente distantes
        if dist < W * 0.25 or dist > W * 0.75:
            continue
        seg = _clip_segment(pa[0], pa[1], pb[0], pb[1], W, H)
        if seg is not None:
            segments.append(seg)
            organic_added += 1

    # ── Añadir borde del dominio (asegura polígonos cerrados) ─────────────────
    # El borde lo aporta shapely_box en extract_blocks; no hace falta
    # agregarlo aquí explícitamente.

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE MANZANAS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_blocks(
    segments: List[Tuple[float, float, float, float]],
    cfg: UrbanConfig
) -> List[Polygon]:
    """
    Extrae manzanas como polígonos de Shapely usando polygonize sobre la
    superposición de segmentos de calle y el contorno del dominio.
    """
    W, H = cfg.domain_x, cfg.domain_y
    domain = shapely_box(0, 0, W, H)

    lines = [LineString([(x0, y0), (x1, y1)]) for (x0, y0, x1, y1) in segments]
    lines.append(domain.boundary)

    merged = unary_union(lines)
    polys = list(polygonize(merged))

    valid = [
        p for p in polys
        if p.is_valid and not p.is_empty and p.area >= cfg.min_block_area
    ]
    return valid


# ═══════════════════════════════════════════════════════════════════════════════
#  SUBDIVISIÓN EN PARCELAS (BSP recursivo)
# ═══════════════════════════════════════════════════════════════════════════════

def _bsp_split_poly(poly: Polygon, cfg: UrbanConfig,
                    rng: np.random.Generator,
                    depth: int, max_depth: int,
                    out: List[Polygon]) -> None:
    """Divide recursivamente una manzana en lotes."""
    area = poly.area
    minx, miny, maxx, maxy = poly.bounds
    dx, dy = maxx - minx, maxy - miny

    # Criterio de parada
    too_small = area < cfg.min_parcel_area * 1.8
    too_deep  = depth >= max_depth
    small_enough = area <= cfg.max_parcel_area

    if (too_small or too_deep) or (small_enough and rng.random() < 0.4):
        if area >= cfg.min_parcel_area and dx >= cfg.min_parcel_side and dy >= cfg.min_parcel_side:
            out.append(poly)
        return

    # Elegir eje: el más largo, con leve sesgo aleatorio
    cut_x = (dx >= dy) if rng.random() < 0.8 else (dx < dy)

    try:
        if cut_x:
            margin = cfg.min_parcel_side
            if dx < 2 * margin:
                out.append(poly)
                return
            split_val = rng.uniform(minx + margin, maxx - margin)
            splitter = LineString([(split_val, miny - 1), (split_val, maxy + 1)])
        else:
            margin = cfg.min_parcel_side
            if dy < 2 * margin:
                out.append(poly)
                return
            split_val = rng.uniform(miny + margin, maxy - margin)
            splitter = LineString([(minx - 1, split_val), (maxx + 1, split_val)])

        from shapely.ops import split as shp_split
        result = shp_split(poly, splitter)
        parts = list(result.geoms) if hasattr(result, "geoms") else [result]

    except Exception:
        out.append(poly)
        return

    for part in parts:
        if part.is_valid and not part.is_empty:
            _bsp_split_poly(part, cfg, rng, depth + 1, max_depth, out)


def subdivide_block(block: Polygon, cfg: UrbanConfig,
                    rng: np.random.Generator) -> List[Polygon]:
    parcels: List[Polygon] = []
    _bsp_split_poly(block, cfg, rng, depth=0, max_depth=7, out=parcels)
    return parcels


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE EDIFICIOS
# ═══════════════════════════════════════════════════════════════════════════════

def _height_sample(cfg: UrbanConfig, rng: np.random.Generator,
                   scale: float = 1.0) -> float:
    """Muestrea altura con distribución log-normal truncada."""
    # Log-normal da cola larga hacia arriba (pocos rascacielos, muchos bajos)
    mu = math.log(max(cfg.building_height_mean * scale, 1.0))
    sigma = 0.6
    h = float(np.exp(rng.normal(mu, sigma)))
    return float(np.clip(h, cfg.building_height_min, cfg.building_height_max))


def _make_obstacle(x0, y0, z0, x1, y1, z1,
                   mat: str, roughness: float) -> Obstacle:
    return Obstacle(
        box_min=np.array([min(x0,x1), min(y0,y1), min(z0,z1)]),
        box_max=np.array([max(x0,x1), max(y0,y1), max(z0,z1)]),
        roughness=roughness,
        material=mat,
    )


def _building_single(x0, y0, x1, y1, cfg: UrbanConfig,
                     rng: np.random.Generator) -> List[Obstacle]:
    """Edificio tipo bloque simple."""
    h = _height_sample(cfg, rng)
    mat, roughness = _sample_material(rng)
    return [_make_obstacle(x0, y0, 0, x1, y1, h, mat, roughness)]


def _building_tower(x0, y0, x1, y1, cfg: UrbanConfig,
                    rng: np.random.Generator) -> List[Obstacle]:
    """Torre: base ancha + pisos que se estrechan hacia arriba."""
    obs = []
    cx, cy = (x0+x1)/2, (y0+y1)/2
    half_x, half_y = (x1-x0)/2, (y1-y0)/2

    total_h = _height_sample(cfg, rng, scale=2.0)
    n_bands = rng.integers(2, 5)
    z_cuts = np.sort(rng.uniform(0, total_h, n_bands - 1))
    zs = np.concatenate([[0.0], z_cuts, [total_h]])

    for k in range(n_bands):
        shrink_frac = k * rng.uniform(0.06, 0.18)
        hx = half_x * max(1 - shrink_frac, 0.3)
        hy = half_y * max(1 - shrink_frac, 0.3)
        if hx < 1.5 or hy < 1.5:
            break
        mat, roughness = _sample_material(rng)
        obs.append(_make_obstacle(
            cx-hx, cy-hy, zs[k],
            cx+hx, cy+hy, zs[k+1],
            mat, roughness
        ))
    return obs


def _building_complex(x0, y0, x1, y1, cfg: UrbanConfig,
                      rng: np.random.Generator) -> List[Obstacle]:
    """Complejo: varios bloques yuxtapuestos con alturas variadas."""
    obs = []
    n = rng.integers(2, 5)
    dx, dy = x1-x0, y1-y0

    # Dividir por el eje más largo
    if dx >= dy:
        cuts = np.sort(rng.uniform(x0, x1, n-1))
        xs = np.concatenate([[x0], cuts, [x1]])
        for k in range(n):
            if xs[k+1] - xs[k] < 3:
                continue
            h = _height_sample(cfg, rng)
            mat, roughness = _sample_material(rng)
            obs.append(_make_obstacle(xs[k], y0, 0, xs[k+1], y1, h, mat, roughness))
    else:
        cuts = np.sort(rng.uniform(y0, y1, n-1))
        ys = np.concatenate([[y0], cuts, [y1]])
        for k in range(n):
            if ys[k+1] - ys[k] < 3:
                continue
            h = _height_sample(cfg, rng)
            mat, roughness = _sample_material(rng)
            obs.append(_make_obstacle(x0, ys[k], 0, x1, ys[k+1], h, mat, roughness))
    return obs


def buildings_for_parcel(parcel: Polygon, cfg: UrbanConfig,
                         rng: np.random.Generator) -> List[Obstacle]:
    minx, miny, maxx, maxy = parcel.bounds
    dx, dy = maxx-minx, maxy-miny

    if dx < cfg.min_parcel_side or dy < cfg.min_parcel_side:
        return []

    # Retiro del borde
    setback = rng.uniform(0.3, cfg.max_setback)
    bx0, by0 = minx + setback, miny + setback
    bx1, by1 = maxx - setback, maxy - setback

    if bx1 - bx0 < 3 or by1 - by0 < 3:
        return []

    # Elegir tipo
    roll = rng.random()
    if roll < cfg.tower_prob:
        return _building_tower(bx0, by0, bx1, by1, cfg, rng)
    elif roll < cfg.tower_prob + cfg.complex_prob:
        return _building_complex(bx0, by0, bx1, by1, cfg, rng)
    else:
        return _building_single(bx0, by0, bx1, by1, cfg, rng)


# ═══════════════════════════════════════════════════════════════════════════════
#  CALLES COMO AABBs
# ═══════════════════════════════════════════════════════════════════════════════

def streets_to_aabbs(segments: List[Tuple[float, float, float, float]],
                     cfg: UrbanConfig,
                     rng: np.random.Generator) -> List[Obstacle]:
    """
    Aproxima cada segmento de calle como el AABB del buffer del segmento.
    Para calles diagonales esto sobreestima levemente el área, pero es
    correcto para ray tracing AABB-based.
    """
    obs = []
    threshold_primary = cfg.domain_x * 0.3

    for (x0, y0, x1, y1) in segments:
        length = math.hypot(x1-x0, y1-y0)
        if length < 2:
            continue
        is_primary = length > threshold_primary
        width = cfg.street_width_primary if is_primary else cfg.street_width_secondary

        line = LineString([(x0, y0), (x1, y1)])
        buffered = line.buffer(width / 2, cap_style=2)  # flat caps
        bx0, by0, bx1, by1 = buffered.bounds
        obs.append(Obstacle(
            box_min=np.array([bx0, by0, -0.05]),
            box_max=np.array([bx1, by1, cfg.street_height]),
            roughness=_MATERIAL_PROPS["asphalt"][0],
            material="asphalt",
        ))
    return obs


# ═══════════════════════════════════════════════════════════════════════════════
#  VEREDAS
# ═══════════════════════════════════════════════════════════════════════════════

def sidewalks_to_aabbs(segments: List[Tuple[float, float, float, float]],
                       cfg: UrbanConfig,
                       rng: np.random.Generator) -> List[Obstacle]:
    obs = []
    W, H = cfg.domain_x, cfg.domain_y
    threshold_primary = cfg.domain_x * 0.3

    for (x0, y0, x1, y1) in segments:
        length = math.hypot(x1-x0, y1-y0)
        if length < 4:
            continue
        is_primary = length > threshold_primary
        street_half = (cfg.street_width_primary if is_primary
                       else cfg.street_width_secondary) / 2

        line = LineString([(x0, y0), (x1, y1)])
        sw = cfg.sidewalk_width

        for side in [1, -1]:
            offset_dist = side * (street_half + sw / 2)
            off = _offset_line(line, offset_dist)
            if off is None:
                continue
            buffered = off.buffer(sw / 2, cap_style=2)
            bx0, by0, bx1, by1 = buffered.bounds
            bx0 = max(0.0, bx0); by0 = max(0.0, by0)
            bx1 = min(W, bx1);   by1 = min(H, by1)
            if bx1 - bx0 < 0.5 or by1 - by0 < 0.5:
                continue
            obs.append(Obstacle(
                box_min=np.array([bx0, by0, 0.0]),
                box_max=np.array([bx1, by1, cfg.sidewalk_height]),
                roughness=_MATERIAL_PROPS["stone"][0],
                material="stone",
            ))
    return obs


# ═══════════════════════════════════════════════════════════════════════════════
#  ÁRBOLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_trees(segments: List[Tuple[float, float, float, float]],
                   cfg: UrbanConfig,
                   rng: np.random.Generator) -> List[Obstacle]:
    """Árboles como stack tronco + capas de copa (AABBs)."""
    obs = []
    W, H = cfg.domain_x, cfg.domain_y
    threshold_primary = cfg.domain_x * 0.3

    for (x0, y0, x1, y1) in segments:
        length = math.hypot(x1-x0, y1-y0)
        if length < cfg.tree_spacing:
            continue

        is_primary = length > threshold_primary
        street_half = (cfg.street_width_primary if is_primary
                       else cfg.street_width_secondary) / 2

        n_trees = max(1, int(length / cfg.tree_spacing))
        ts = rng.uniform(0.05, 0.95, size=n_trees)

        dx, dy = x1-x0, y1-y0
        # Vector perpendicular unitario
        perp_x, perp_y = -dy / length, dx / length

        for t in ts:
            if rng.random() > cfg.tree_density:
                continue
            side = rng.choice([-1, 1])
            offset = street_half + cfg.sidewalk_width * 0.5
            tx = x0 + t*dx + side * offset * perp_x
            ty = y0 + t*dy + side * offset * perp_y
            if not (0.5 < tx < W-0.5 and 0.5 < ty < H-0.5):
                continue

            tr = rng.uniform(0.15, 0.35)  # radio de tronco
            th = rng.uniform(1.8, 4.5)    # altura de tronco

            # Tronco
            obs.append(Obstacle(
                box_min=np.array([tx-tr, ty-tr, 0.0]),
                box_max=np.array([tx+tr, ty+tr, th]),
                roughness=0.92,
                material="vegetation",
            ))
            # Capas de copa (2-3)
            n_layers = rng.integers(2, 4)
            for layer in range(n_layers):
                cr = rng.uniform(1.2, 2.8) * (1 - layer * 0.15)
                ch = rng.uniform(0.7, 1.4)
                cz = th + layer * ch * 0.65
                obs.append(Obstacle(
                    box_min=np.array([tx-cr, ty-cr, cz]),
                    box_max=np.array([tx+cr, ty+cr, cz+ch]),
                    roughness=0.98,
                    material="vegetation",
                ))
    return obs


# ═══════════════════════════════════════════════════════════════════════════════
#  MOBILIARIO URBANO
# ═══════════════════════════════════════════════════════════════════════════════

_FURNITURE = [
    # (sx, sy, sz, material, roughness)
    (0.45, 1.60, 0.50, "metal",    0.35),  # banco
    (1.50, 1.50, 2.80, "metal",    0.42),  # kiosco / parada
    (0.30, 0.30, 1.00, "concrete", 0.72),  # bolardo
    (0.25, 0.25, 3.50, "metal",    0.22),  # poste de luz
    (0.60, 0.60, 1.20, "metal",    0.30),  # contenedor de basura
    (0.80, 2.00, 0.10, "stone",    0.78),  # baldosa decorativa
]


def generate_urban_obstacles(segments: List[Tuple[float, float, float, float]],
                              cfg: UrbanConfig,
                              rng: np.random.Generator) -> List[Obstacle]:
    obs = []
    W, H = cfg.domain_x, cfg.domain_y
    threshold_primary = cfg.domain_x * 0.3

    for (x0, y0, x1, y1) in segments:
        length = math.hypot(x1-x0, y1-y0)
        if length < cfg.obstacle_spacing:
            continue

        is_primary = length > threshold_primary
        street_half = (cfg.street_width_primary if is_primary
                       else cfg.street_width_secondary) / 2

        n_items = max(1, int(length / cfg.obstacle_spacing))
        ts = rng.uniform(0.05, 0.95, size=n_items)
        dx, dy = x1-x0, y1-y0
        perp_x, perp_y = -dy / length, dx / length

        for t in ts:
            if rng.random() > cfg.obstacle_density:
                continue
            side = rng.choice([-1, 1])
            offset = street_half + 1.0
            ox = x0 + t*dx + side * offset * perp_x
            oy = y0 + t*dy + side * offset * perp_y
            if not (0.5 < ox < W-0.5 and 0.5 < oy < H-0.5):
                continue

            sx, sy, sz, mat, roughness = _FURNITURE[rng.integers(0, len(_FURNITURE))]
            obs.append(Obstacle(
                box_min=np.array([ox-sx/2, oy-sy/2, 0.0]),
                box_max=np.array([ox+sx/2, oy+sy/2, sz]),
                roughness=roughness,
                material=mat,
            ))
    return obs


# ═══════════════════════════════════════════════════════════════════════════════
#  ILUMINADORES (TRANSMISORES)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_transmitters(cfg: UrbanConfig,
                           rng: np.random.Generator) -> List[Transmitter]:
    """
    Coloca TXs en la periferia del dominio (borde externo), simulando
    torres de difusión DVB-T / LTE alejadas de la zona de interés.
    """
    W, H = cfg.domain_x, cfg.domain_y
    m = cfg.tx_border_margin
    txs = []

    # Candidatos en los 4 bordes
    border_candidates = [
        (rng.uniform(m, W-m), rng.uniform(0,   m    )),   # sur
        (rng.uniform(m, W-m), rng.uniform(H-m, H    )),   # norte
        (rng.uniform(0,   m), rng.uniform(m,   H-m  )),   # oeste
        (rng.uniform(W-m, W), rng.uniform(m,   H-m  )),   # este
    ]
    rng.shuffle(border_candidates)

    for k in range(min(cfg.n_transmitters, len(border_candidates))):
        bx, by = border_candidates[k]
        txs.append(Transmitter(
            position=np.array([bx, by, cfg.tx_height]),
            frequency=cfg.tx_frequency,
            tx_power_w=cfg.tx_power_w,
            tx_id=k,
        ))
    # Si se piden más TXs que bordes disponibles, rellenar con posiciones random
    extra = cfg.n_transmitters - len(txs)
    for k in range(extra):
        txs.append(Transmitter(
            position=np.array([
                rng.uniform(0, W),
                rng.uniform(0, H),
                cfg.tx_height
            ]),
            frequency=cfg.tx_frequency,
            tx_power_w=cfg.tx_power_w,
            tx_id=len(txs),
        ))
    return txs


# ═══════════════════════════════════════════════════════════════════════════════
#  ORQUESTADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def generate_urban_scene(cfg: UrbanConfig) -> Scene:
    rng = np.random.default_rng(cfg.seed)
    all_obs: List[Obstacle] = []

    def log(msg: str) -> None:
        print(f"[urban_gen] {msg}", flush=True)

    log(f"seed={cfg.seed} | dominio={cfg.domain_x:.0f}×{cfg.domain_y:.0f} m "
        f"| densidad={cfg.urban_density:.0%}")

    # ── 1. Red de calles ───────────────────────────────────────────────────────
    log("Generando red de calles (grilla perturbada + campo de dirección)…")
    segments = generate_street_network(cfg, rng)
    log(f"  → {len(segments)} segmentos de calle")

    # ── 2. AABBs de calles ────────────────────────────────────────────────────
    street_obs = streets_to_aabbs(segments, cfg, rng)
    all_obs.extend(street_obs)
    log(f"  → {len(street_obs)} AABBs de asfalto")

    # ── 3. Veredas ────────────────────────────────────────────────────────────
    if cfg.add_sidewalks:
        sw_obs = sidewalks_to_aabbs(segments, cfg, rng)
        all_obs.extend(sw_obs)
        log(f"  → {len(sw_obs)} AABBs de vereda")

    # ── 4. Extracción de manzanas ─────────────────────────────────────────────
    log("Extrayendo manzanas (polygonize)…")
    blocks = extract_blocks(segments, cfg)
    log(f"  → {len(blocks)} manzanas válidas (área ≥ {cfg.min_block_area:.0f} m²)")

    # ── 5. Seleccionar plazas ─────────────────────────────────────────────────
    plaza_idx: set = set()
    if cfg.add_plazas and len(blocks) > 0:
        n_plazas = max(1, int(len(blocks) * cfg.plaza_fraction))
        plaza_idx = set(
            rng.choice(len(blocks), size=min(n_plazas, len(blocks)),
                       replace=False).tolist()
        )
        log(f"  → {len(plaza_idx)} manzanas designadas como plazas")

    # ── 6. Subdivisión y edificios ────────────────────────────────────────────
    log("Subdividiendo manzanas y generando edificios…")
    n_buildings = 0
    for i, block in enumerate(blocks):
        if i in plaza_idx:
            continue
        parcels = subdivide_block(block, cfg, rng)
        for parcel in parcels:
            if rng.random() > cfg.urban_density:
                continue
            bldg = buildings_for_parcel(parcel, cfg, rng)
            all_obs.extend(bldg)
            n_buildings += len(bldg)
    log(f"  → {n_buildings} AABBs de edificios")

    # ── 7. Árboles ────────────────────────────────────────────────────────────
    if cfg.add_trees:
        tree_obs = generate_trees(segments, cfg, rng)
        all_obs.extend(tree_obs)
        log(f"  → {len(tree_obs)} AABBs de árboles")

    # ── 8. Mobiliario urbano ──────────────────────────────────────────────────
    if cfg.add_obstacles:
        furn_obs = generate_urban_obstacles(segments, cfg, rng)
        all_obs.extend(furn_obs)
        log(f"  → {len(furn_obs)} AABBs de mobiliario urbano")

    # ── 9. Transmisores ───────────────────────────────────────────────────────
    transmitters = generate_transmitters(cfg, rng)
    log(f"  → {len(transmitters)} transmisores "
        f"({cfg.tx_frequency/1e6:.0f} MHz, {cfg.tx_power_w:.0f} W)")

    log(f"Total AABBs en escena: {len(all_obs)}")

    scene_box = Box(
        box_min=np.array([0.0, 0.0, -0.5]),
        box_max=np.array([cfg.domain_x, cfg.domain_y,
                         cfg.building_height_max + 15.0]),
    )

    return Scene(
        box=scene_box,
        transmitters=transmitters,
        obstacles=all_obs,
        n_rays=cfg.n_rays,
        n_max=cfg.n_max,
        use_physics=cfg.use_physics,
        temperature_c=cfg.temperature_c,
        bandwidth_hz=cfg.bandwidth_hz,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SERIALIZACIÓN  (espejo exacto de cache.py::save_scene)
# ═══════════════════════════════════════════════════════════════════════════════

def save_scene(scene: Scene, filepath: Path) -> None:
    data = {
        "box": {
            "box_min": [float(x) for x in scene.box.box_min],
            "box_max": [float(x) for x in scene.box.box_max],
        },
        "transmitters": [
            {
                "position":   [float(x) for x in tx.position],
                "frequency":  float(tx.frequency),
                "tx_power_w": float(tx.tx_power_w),
                "tx_id":      int(tx.tx_id),
            }
            for tx in scene.transmitters
        ],
        "obstacles": [
            {
                "box_min":   [float(x) for x in o.box_min],
                "box_max":   [float(x) for x in o.box_max],
                "roughness": float(o.roughness),
                "material":  str(o.material),
            }
            for o in scene.obstacles
        ],
        "params": {
            "n_rays":        scene.n_rays,
            "n_max":         scene.n_max,
            "use_physics":   scene.use_physics,
            "temperature_c": scene.temperature_c,
            "bandwidth_hz":  scene.bandwidth_hz,
        },
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2))
    print(f"[urban_gen] Escena guardada → {filepath}  "
          f"({filepath.stat().st_size / 1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generador procedural de escenas urbanas AABB para TAMI7",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Identidad
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--output-name",        type=str,   default=None,
                   help="Nombre base del archivo (sin extensión). "
                        "Por defecto: urban_<seed>_<WxH>")

    # Dominio
    p.add_argument("--domain-x",           type=float, default=600.0,   metavar="M")
    p.add_argument("--domain-y",           type=float, default=600.0,   metavar="M")
    p.add_argument("--urban-density",      type=float, default=0.65,
                   help="Fracción de lotes con edificio [0,1]")

    # Calles
    p.add_argument("--grid-cols",          type=int,   default=7)
    p.add_argument("--grid-rows",          type=int,   default=7)
    p.add_argument("--grid-perturb-frac",  type=float, default=0.20)
    p.add_argument("--grid-base-angle",    type=float, default=-999,
                   help="Ángulo base de la grilla en grados. -999 = sorteado")
    p.add_argument("--n-organic-streets",  type=int,   default=6)
    p.add_argument("--street-width-primary",   type=float, default=12.0, metavar="M")
    p.add_argument("--street-width-secondary", type=float, default=7.0,  metavar="M")

    # Edificios
    p.add_argument("--height-mean",        type=float, default=16.0,    metavar="M")
    p.add_argument("--height-max",         type=float, default=80.0,    metavar="M")
    p.add_argument("--tower-prob",         type=float, default=0.25)
    p.add_argument("--complex-prob",       type=float, default=0.20)

    # Elementos opcionales
    p.add_argument("--no-sidewalks",  action="store_true")
    p.add_argument("--no-trees",      action="store_true")
    p.add_argument("--no-obstacles",  action="store_true")
    p.add_argument("--no-plazas",     action="store_true")
    p.add_argument("--tree-density",       type=float, default=0.5)
    p.add_argument("--obstacle-density",   type=float, default=0.4)
    p.add_argument("--plaza-fraction",     type=float, default=0.06)

    # Transmisores
    p.add_argument("--n-tx",              type=int,   default=3)
    p.add_argument("--tx-freq",           type=float, default=700e6,    metavar="Hz")
    p.add_argument("--tx-power",          type=float, default=1000.0,   metavar="W")
    p.add_argument("--tx-height",         type=float, default=45.0,     metavar="M")

    # Cache / salida
    p.add_argument("--cache-dir",         type=Path,  default=Path("./cache"))
    p.add_argument("--force",             action="store_true",
                   help="Regenerar aunque exista en caché")

    # Simulación
    p.add_argument("--n-rays",            type=int,   default=2000)
    p.add_argument("--n-max",             type=int,   default=5)
    p.add_argument("--bandwidth-hz",      type=float, default=8e6)

    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    cfg = UrbanConfig(
        seed=args.seed,
        domain_x=args.domain_x,
        domain_y=args.domain_y,
        urban_density=args.urban_density,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        grid_perturb_frac=args.grid_perturb_frac,
        grid_base_angle_deg=args.grid_base_angle,
        n_organic_streets=args.n_organic_streets,
        street_width_primary=args.street_width_primary,
        street_width_secondary=args.street_width_secondary,
        building_height_mean=args.height_mean,
        building_height_max=args.height_max,
        tower_prob=args.tower_prob,
        complex_prob=args.complex_prob,
        add_sidewalks=not args.no_sidewalks,
        add_trees=not args.no_trees,
        add_obstacles=not args.no_obstacles,
        add_plazas=not args.no_plazas,
        tree_density=args.tree_density,
        obstacle_density=args.obstacle_density,
        plaza_fraction=args.plaza_fraction,
        n_transmitters=args.n_tx,
        tx_frequency=args.tx_freq,
        tx_power_w=args.tx_power,
        tx_height=args.tx_height,
        n_rays=args.n_rays,
        n_max=args.n_max,
        bandwidth_hz=args.bandwidth_hz,
    )

    output_name = (
        args.output_name
        or f"urban_{cfg.seed}_{int(cfg.domain_x)}x{int(cfg.domain_y)}"
    )
    scene_file = f"{output_name}.json"
    filepath = args.cache_dir / "scenes" / scene_file

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if filepath.exists() and not args.force:
        print(f"[urban_gen] Cache hit: {filepath} — omitiendo generación. "
              f"Usar --force para regenerar.")
        return

    # ── Generar y guardar ──────────────────────────────────────────────────────
    scene = generate_urban_scene(cfg)
    save_scene(scene, filepath)


if __name__ == "__main__":
    main()
