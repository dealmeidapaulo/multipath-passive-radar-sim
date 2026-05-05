from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .ray import Ray


def extract(rays: List[Ray], instance_id: str = "sim_0", time_step: int = 0,
            uav=None, params: dict = None) -> List[dict]:
    out = []
    p = params or {}
    for r in rays:
        out.append(dict(
            instance_id   = instance_id,
            time_step     = time_step,
            uav_present   = (uav is not None),
            uav_pos_x     = float(uav.position[0]) if uav else np.nan,
            uav_pos_y     = float(uav.position[1]) if uav else np.nan,
            uav_pos_z     = float(uav.position[2]) if uav else np.nan,
            uav_vel_x     = float(uav.velocity[0]) if uav else np.nan,
            uav_vel_y     = float(uav.velocity[1]) if uav else np.nan,
            uav_vel_z     = float(uav.velocity[2]) if uav else np.nan,
            tau_s         = r.delay(),
            theta_rad     = r.elevation(),
            phi_rad       = r.azimuth(),
            f_D           = r.doppler_shift,
            is_uav_bounce = getattr(r, 'is_uav_bounce', False),
            is_los        = (r.n_bounces == 0),
            visible       = int(getattr(r, 'visible', True)),
            power_dbm     = r.power_dbm,
            tx_pos_x      = p.get('tx_pos_x', np.nan),
            tx_pos_y      = p.get('tx_pos_y', np.nan),
            tx_pos_z      = p.get('tx_pos_z', np.nan),
            rx_pos_x      = p.get('rx_pos_x', np.nan),
            rx_pos_y      = p.get('rx_pos_y', np.nan),
            rx_pos_z      = p.get('rx_pos_z', np.nan),
            domain_x      = p.get('domain_x', 0.0),
            domain_y      = p.get('domain_y', 0.0),
            bld_height    = p.get('bld_height', 0.0),
            tall_frac     = p.get('tall_frac', 0.0),
            seed          = p.get('seed', 0),
            temp_c        = p.get('temp', 30.0),
            bw_hz         = p.get('bw', 20e6),
            tx_power_w    = p.get('tx_power', 50.0),
            enable_dr     = int(p.get('enable_dr', False)),
            agc_active    = int(p.get('agc', False)),
            dyn_range_db  = p.get('dyn_range', 0.0),
        ))
    return out


def to_dataframe(rays: List[Ray], instance_id: str = "sim_0", time_step: int = 0,
                 uav=None, params: dict = None) -> pd.DataFrame:
    return pd.DataFrame(extract(rays, instance_id, time_step, uav, params))


def to_parquet(df: pd.DataFrame, path: str, **kwargs) -> None:
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False, **kwargs)