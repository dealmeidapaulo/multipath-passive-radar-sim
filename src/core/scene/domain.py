from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np

C_LIGHT: float = 3e8
BOLTZMANN: float = 1.380649e-23


@dataclass
class Box:
    box_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    box_max: np.ndarray = field(default_factory=lambda: np.ones(3))

    def __post_init__(self):
        self.box_min = np.asarray(self.box_min, dtype=float)
        self.box_max = np.asarray(self.box_max, dtype=float)


@dataclass
class Obstacle:
    """
    Axis-aligned bounding box obstacle.
    roughness: 0 = perfect mirror, 1 = fully Lambertian.
    """
    box_min  : np.ndarray
    box_max  : np.ndarray
    roughness: float = 0.0
    material : str   = "concrete"

    def __post_init__(self):
        self.box_min = np.asarray(self.box_min, dtype=float)
        self.box_max = np.asarray(self.box_max, dtype=float)


@dataclass
class MeshObstacle:
    """
    Triangulated surface obstacle (e.g. extruded from OSM footprint).

    vertices : float64[V, 3]   3-D vertex positions in scene coordinates
    faces    : int32[F, 3]     triangle indices into vertices
    roughness: float           0 = perfect mirror, 1 = fully Lambertian
    material : str             label used for Fresnel parameters

    box_min / box_max are computed lazily from vertices so that existing
    code that accesses o.box_min still works (e.g. obs_arrays()).
    """
    vertices : np.ndarray
    faces    : np.ndarray
    roughness: float = 0.0
    material : str   = "concrete"

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces    = np.asarray(self.faces,    dtype=np.int32)

    @property
    def box_min(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def box_max(self) -> np.ndarray:
        return self.vertices.max(axis=0)


AnyObstacle = Union[Obstacle, MeshObstacle]


@dataclass
class Transmitter:
    position  : np.ndarray
    frequency : float
    tx_power_w: float = 50.0
    tx_id     : int   = 0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)

    @property
    def tx_power_dbm(self) -> float:
        return 10.0 * np.log10(self.tx_power_w * 1000.0)


@dataclass
class Receiver:
    """Standalone receiver. NOT part of Scene — passed directly to apply_rx()."""
    position: np.ndarray
    radius  : float = 2.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class UAV:
    position: np.ndarray
    velocity: np.ndarray
    radius  : float = 0.3

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)


@dataclass
class Scene:
    """
    Simulation domain: geometry and transmitters only.

    Receiver and UAV are NOT fields of Scene. They are applied to a
    precomputed StaticField via apply_rx() and apply_uav() respectively.
    """
    box          : Box
    transmitters : List[Transmitter]          = field(default_factory=list)
    n_max        : int                        = 10
    n_rays       : int                        = 60_000
    obstacles    : List[AnyObstacle]          = field(default_factory=list)

    # Thermal / link budget
    temperature_c : float = 30.0
    bandwidth_hz  : float = 20e6
    use_physics   : bool  = True

    # UAV sampling (used by apply_uav)
    uav_roughness : float = 0.3
    n_samples_uav : int   = 8

    @property
    def noise_floor_dbm(self) -> float:
        temp_k  = self.temperature_c + 273.15
        noise_w = BOLTZMANN * temp_k * self.bandwidth_hz
        return 10.0 * np.log10(noise_w * 1000.0)
