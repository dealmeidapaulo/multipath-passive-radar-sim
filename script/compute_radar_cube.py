import numpy as np
import pandas as pd

"""
Compute a radar cube from the rays corresponding to a single transmitter and a single receiver position.

Required columns:
- tx_lobe_theta_rad
- tau_s
- elevation_rad
- azimuth_rad
- power_dbm

Returned radar cube:
    (3, n_elevation, n_azimuth, n_tau)
"""

def compute_radar_cube(
    df: pd.DataFrame,
    n_elevation: int,
    n_azimuth: int,
    n_tau: int,
    delta_t: float,
) -> tuple[np.ndarray, dict]:

    tau = df["tau_s"].values
    tau_min_s = df["tau_s"].min()


    tx_pos = (
    df["tx_pos_x"].iloc[0],
    df["tx_pos_y"].iloc[0],
    df["tx_pos_z"].iloc[0],
)

    rx_pos = (
        df["rx_pos_x"].iloc[0],
        df["rx_pos_y"].iloc[0],
        df["rx_pos_z"].iloc[0],
    )

    tau_idx = np.floor((tau - tau_min_s) / delta_t).astype(int)
    
    valid = (tau_idx >= 0) & (tau_idx < n_tau)
    if not np.any(valid):
        return np.zeros((3, n_elevation, n_azimuth, n_tau), dtype=np.float32), {}

    df_valid = df.iloc[valid]
    tau_idx = tau_idx[valid]

    elev = df_valid["elevation_rad"].values
    azim = df_valid["azimuth_rad"].values
    power_dbm = df_valid["power_dbm"].values
    lobes = df_valid["tx_lobe_theta_rad"].values

    unique_lobes = np.sort(np.unique(lobes))
    lobe_idx = np.searchsorted(unique_lobes, lobes)

    elev_edges = np.linspace(-np.pi / 2, np.pi / 2, n_elevation + 1)
    azim_edges = np.linspace(-np.pi, np.pi, n_azimuth + 1)

    elev_idx = np.clip(np.searchsorted(elev_edges, elev, side="right") - 1, 0, n_elevation - 1)
    azim_idx = np.clip(np.searchsorted(azim_edges, azim, side="right") - 1, 0, n_azimuth - 1)

    power_w = (10.0 ** (power_dbm / 10.0)) / 1000.0

    cube = np.zeros((3, n_elevation, n_azimuth, n_tau), dtype=np.float32)
    np.add.at(cube, (lobe_idx, elev_idx, azim_idx, tau_idx), power_w.astype(np.float32))

    metadata = {
        "tau_min_s": tau_min_s,
        "delta_t_s": delta_t,
        "elevation_edges": elev_edges,
        "azimuth_edges": azim_edges,
        "lobe_angles_rad": unique_lobes,
        "tx_pos_m": np.asarray(tx_pos, dtype=np.float64),
        "rx_pos_m": np.asarray(rx_pos, dtype=np.float64),
    }

    return cube, metadata