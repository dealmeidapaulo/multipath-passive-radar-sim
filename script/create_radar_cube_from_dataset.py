"""
Generate radar cubes from a dataset produced by `dataset_builder.py`.

Each radar cube corresponds to a unique (scene, iteration, transmitter, receiver)
combination and is saved as an individual `.npz` file.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from compute_radar_cube import compute_radar_cube


def process_dataset(input_path: Path, params: dict) -> dict[str, np.ndarray]:
    df = pd.read_parquet(input_path)

    radar_cubes = {}

    for (scene_id, iteration_id), scene_df in df.groupby(
        ["scene_id", "iteration_id"],
        sort=False,
    ):

        tx_groups = scene_df.groupby(
            ["tx_pos_x", "tx_pos_y", "tx_pos_z"],
            sort=False,
        )

        for tx_id, (_, tx_df) in enumerate(tx_groups):

            rx_groups = tx_df.groupby(
                ["rx_pos_x", "rx_pos_y", "rx_pos_z"],
                sort=False,
            )

            for rx_id, (_, rx_df) in enumerate(rx_groups):

                cube, metadata = compute_radar_cube(
                    rx_df,
                    n_tau=params["n_tau"],
                    n_elevation=params["n_elevation"],
                    n_azimuth=params["n_azimuth"],
                    delta_t=params["delta_t"],
                )

                key = (
                    f"scene_{scene_id}"
                    f"_iter_{iteration_id:04d}"
                    f"_tx_{tx_id}"
                    f"_rx_{rx_id}"
                )

                radar_cubes[key] = {
                    "cube": cube,
                    "metadata": metadata,
                }

    return radar_cubes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-elev", type=int, required=True)
    parser.add_argument("--n-azim", type=int, required=True)
    parser.add_argument("--n-tau", type=int, required=True)
    parser.add_argument("--delta-t", type=float, required=True)
    args = parser.parse_args()

    params = {
        "n_elevation": args.n_elev,
        "n_azimuth": args.n_azim,
        "n_tau": args.n_tau,
        "delta_t": args.delta_t,
    }

    radar_cubes = process_dataset(args.dataset, params)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key, result in radar_cubes.items():
        np.savez(
            args.out_dir / f"{key}.npz",
            cube=result["cube"],
            **result["metadata"],
        )

if __name__ == "__main__":
    main()