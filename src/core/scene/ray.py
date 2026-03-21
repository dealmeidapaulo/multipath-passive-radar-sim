from __future__ import annotations
from typing import List, Iterator, Tuple
import numpy as np

class Ray:
    def __init__(
        self,
        transmitter_id: int,
        points        : List[np.ndarray],
        arrival_dir   : np.ndarray,
        frequency     : float = 433e6,
        power_dbm     : float = 0.0,
    ):
        arrival_dir = np.asarray(arrival_dir, dtype=float)
        self.transmitter_id  : int               = transmitter_id
        self.points          : List[np.ndarray]  = [np.asarray(p, dtype=float) for p in points]
        self.arrival_dir     : np.ndarray        = arrival_dir / np.linalg.norm(arrival_dir)
        self.n_bounces       : int               = len(points) - 2
        self.frequency       : float             = frequency
        self.power_dbm       : float             = power_dbm

        self.is_uav_bounce   : bool              = False
        self.doppler_shift   : float             = 0.0
        self.visible         : bool              = True

    def total_length(self) -> float:
        return sum(float(np.linalg.norm(b - a)) for a, b in self.segments())

    def segments(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(len(self.points) - 1):
            yield self.points[i], self.points[i + 1]

    def delay(self, c: float = 3e8) -> float:
        return self.total_length() / c

    def azimuth(self) -> float:
        return float(np.arctan2(self.arrival_dir[1], self.arrival_dir[0]))

    def elevation(self) -> float:
        return float(np.arcsin(np.clip(self.arrival_dir[2], -1.0, 1.0)))
