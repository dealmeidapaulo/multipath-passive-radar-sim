from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .domain import Box, Obstacle, Transmitter, Receiver, UAV, Scene


def make_street_grid(
    domain_x     : float = 200.0,
    domain_y     : float = 200.0,
    block_w      : float = 22.0,
    block_d      : float = 22.0,
    street_w     : float = 10.0,
    street_d     : float = 10.0,
    bld_height   : float = 30.0,
    height_jitter: float = 10.0,
    tall_fraction: float = 0.35,
    tall_factor  : float = 3.0,
    tall_shrink  : float = 0.45,
    roughness    : float = 0.15,
    seed         : int   = 42,
) -> List[Obstacle]:
    """
    Generate a grid of AABB building obstacles with per-obstacle roughness.
    roughness is applied uniformly to all generated buildings.
    """
    rng  = np.random.default_rng(seed)
    blds : List[Obstacle] = []

    step_x = block_w + street_w
    step_y = block_d + street_d
    x = 0.0
    while x + block_w <= domain_x:
        y = 0.0
        while y + block_d <= domain_y:
            if rng.random() < tall_fraction:
                sx = block_w * (1 - tall_shrink) / 2
                sy = block_d * (1 - tall_shrink) / 2
                h  = bld_height * tall_factor + rng.uniform(0, bld_height * 0.8)
                blds.append(Obstacle(
                    box_min  = np.array([x+sx,         y+sy,         0.0]),
                    box_max  = np.array([x+block_w-sx, y+block_d-sy, h  ]),
                    roughness= roughness,
                    material = "concrete",
                ))
            else:
                h = bld_height + rng.uniform(-height_jitter, height_jitter)
                blds.append(Obstacle(
                    box_min  = np.array([x,         y,         0.0]),
                    box_max  = np.array([x+block_w, y+block_d, h  ]),
                    roughness= roughness,
                    material = "brick",
                ))
            y += step_y
        x += step_x
    return blds


def make_street_scene(
    domain_x     : float = 200.0,
    domain_y     : float = 200.0,
    domain_z     : float = 120.0,
    block_w      : float = 22.0,
    block_d      : float = 22.0,
    street_w     : float = 10.0,
    street_d     : float = 10.0,
    bld_height   : float = 30.0,
    height_jitter: float = 10.0,
    tall_fraction : float = 0.35,
    tall_factor  : float = 3.0,
    tall_shrink  : float = 0.45,
    tx_pos       : tuple = (90.0, 90.0, 35.0),
    tx_power_w   : float = 50.0,
    rx_pos       : tuple = None,
    uav_pos      : tuple = None,
    uav_vel      : tuple = (5.0, 0.0, 0.0),
    uav_radius   : float = 1.0,
    frequency    : float = 2.4e9,
    n_rays       : int   = 60_000,
    n_max        : int   = 4,
    roughness    : float = 0.15,
    seed         : int   = 42,
) -> Tuple[Scene, Receiver, UAV]:
    """
    Build a synthetic street-grid simulation scene.

    Returns
    -------
    scene    : Scene (no Receiver, no UAV — geometry + TX only)
    receiver : Receiver  standalone, pass to apply_rx()
    uav      : UAV       standalone, pass to apply_uav()

    Migration note
    --------------
    Previous code:
        scene = make_street_scene(...)
        static = precompute(scene, ...)
        vis, occ, bounces = apply_uav(static, scene.uav, scene)

    New code:
        scene, rx, uav = make_street_scene(...)
        static = precompute(scene, ...)
        static = apply_rx(static, rx)
        vis, occ, bounces = apply_uav(static, uav, scene)
    """
    step_x = block_w + street_w
    step_y = block_d + street_d
    cx_street = block_w + street_w / 2
    cy_street = block_d + street_d / 2

    if rx_pos is None:
        rx_pos = (91.0, 5.0, 1.5)
    if uav_pos is None:
        uav_pos = (cx_street, cy_street, bld_height / 2)

    buildings = make_street_grid(
        domain_x=domain_x, domain_y=domain_y,
        block_w=block_w, block_d=block_d,
        street_w=street_w, street_d=street_d,
        bld_height=bld_height, height_jitter=height_jitter,
        tall_fraction=tall_fraction, tall_factor=tall_factor,
        tall_shrink=tall_shrink, roughness=roughness, seed=seed,
    )

    scene = Scene(
        box          = Box(np.zeros(3), np.array([domain_x, domain_y, domain_z])),
        transmitters = [Transmitter(np.array(tx_pos, dtype=float), frequency,
                                    tx_power_w=tx_power_w, tx_id=0)],
        obstacles    = buildings,
        n_rays       = n_rays,
        n_max        = n_max,
        use_physics  = True,
    )

    receiver = Receiver(np.array(rx_pos, dtype=float), radius=2.0)
    uav      = UAV(np.array(uav_pos, dtype=float),
                   np.array(uav_vel, dtype=float), uav_radius)

    return scene, receiver, uav
