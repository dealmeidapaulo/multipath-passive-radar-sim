import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import gaussian_filter

# --------------------------------------------------
# Environment setup
# --------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.cache import load_scene
from src.core.scene.domain import Scene

# --------------------------------------------------
# Paths
# --------------------------------------------------

CACHE_DIR = _ROOT / "cache"
FIELDS_DIR = CACHE_DIR / "precomputed_static_fields"

# --------------------------------------------------
# Compute map
# --------------------------------------------------

def compute_2d_power_map(
    scene: Scene,
    data: dict,
    grid_res: int = 300,
    sub_steps: int = 5,
):
    t0 = time.time()

    pos_cpu = data["pos_cpu"]
    n_pts_cpu = data["n_pts_cpu"]
    sp_cpu = data["step_powers"]

    N_STEPS, N_RAYS, _ = pos_cpu.shape

    bmin = scene.box.box_min.astype(np.float32)
    bmax = scene.box.box_max.astype(np.float32)

    span_x = float(bmax[0] - bmin[0])
    span_y = float(bmax[1] - bmin[1])

    #
    # Subsegment interpolation
    #

    alphas = np.linspace(
        0.0,
        1.0,
        sub_steps,
    )[:, None, None, None]

    A = pos_cpu[:-1][None, ...]
    B = pos_cpu[1:][None, ...]

    interp_pts = (
        A
        + alphas * (B - A)
    )

    pos_orig = pos_cpu[
        0
    ][None, None, :, :]

    dist_to_p = np.linalg.norm(
        interp_pts - pos_orig,
        axis=3,
    )

    dist_to_p = np.maximum(
        dist_to_p,
        0.5,
    )

    dist_start = np.linalg.norm(
        A - pos_orig,
        axis=3,
    )

    dist_start = np.maximum(
        dist_start,
        0.5,
    )

    pwr_start = sp_cpu[:-1][None, ...]

    pwr_interp = (
        pwr_start
        - 20.0
        * np.log10(
            dist_to_p / dist_start
        )
    )

    #
    # Valid segments
    #

    step_idx = np.arange(
        N_STEPS - 1
    )[:, None]

    valid_mask = (
        step_idx
        < (
            n_pts_cpu[None, :]
            - 1
        )
    )[None, ...]

    valid_mask = np.repeat(
        valid_mask,
        sub_steps,
        axis=0,
    )

    pts_v = interp_pts[
        valid_mask
    ].reshape(-1, 3)

    pwr_v = pwr_interp[
        valid_mask
    ].ravel()

    noise_floor = getattr(
        scene,
        "noise_floor_dbm",
        -120.0,
    )

    mask = pwr_v > noise_floor

    pts_v = pts_v[mask]
    pwr_v = pwr_v[mask]

    #
    # Accumulate
    #

    cx = np.clip(
        (
            (
                pts_v[:, 0]
                - bmin[0]
            )
            / span_x
            * grid_res
        ).astype(np.int32),
        0,
        grid_res - 1,
    )

    cy = np.clip(
        (
            (
                pts_v[:, 1]
                - bmin[1]
            )
            / span_y
            * grid_res
        ).astype(np.int32),
        0,
        grid_res - 1,
    )

    grid_lin = np.zeros(
        (
            grid_res,
            grid_res,
        ),
        dtype=np.float32,
    )

    pwr_lin = np.power(
        10.0,
        pwr_v / 10.0,
    )

    np.maximum.at(
        grid_lin,
        (cx, cy),
        pwr_lin,
    )

    with np.errstate(
        divide="ignore"
    ):
        power_map = np.where(
            grid_lin > 0,
            10.0 * np.log10(
                grid_lin
            ),
            np.nan,
        )

    #
    # Smooth
    #

    p_min = np.nanpercentile(
        power_map,
        1,
    )

    p_max = np.nanmax(
        power_map
    )

    power_map_filled = np.where(
        np.isnan(power_map),
        p_min - 5.0,
        power_map,
    )

    power_map_smoothed = gaussian_filter(
        power_map_filled,
        sigma=0.8,
    )

    power_map_final = np.clip(
        power_map_smoothed,
        p_min,
        p_max,
    )

    extent = [
        bmin[0],
        bmax[0],
        bmin[1],
        bmax[1],
    ]

    print(
        f"[Compute] "
        f"{time.time()-t0:.2f}s"
    )

    return (
        power_map_final,
        extent,
        p_min,
        p_max,
    )


# --------------------------------------------------
# Plot
# --------------------------------------------------

def plot_power_map(
    power_map,
    scene,
    extent,
    p_min,
    p_max,
    scene_hash,
):

    fig, ax = plt.subplots(
        figsize=(12, 10),
    )

    im = ax.imshow(
        power_map.T,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=p_min,
        vmax=p_max,
    )

    #
    # Obstacles
    #

    for obs in scene.obstacles:

        w = (
            obs.box_max[0]
            - obs.box_min[0]
        )

        h = (
            obs.box_max[1]
            - obs.box_min[1]
        )

        ax.add_patch(
            plt.Rectangle(
                (
                    obs.box_min[0],
                    obs.box_min[1],
                ),
                w,
                h,
                fc="black",
                ec="cyan",
                lw=1,
                alpha=0.9,
            )
        )

    #
    # TX
    #

    for tx in scene.transmitters:

        ax.plot(
            tx.position[0],
            tx.position[1],
            "^",
            color="red",
            markersize=10,
        )

    ax.set_title(
        f"Coverage Map ({scene_hash})"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    plt.colorbar(
        im,
        ax=ax,
        label="Power [dBm]",
    )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ruta",
        required=True,
        type=str,
        help="Path to st_xxxxx.npz",
    )

    args = parser.parse_args()

    npz_path = Path(
        args.ruta
    )

    if not npz_path.exists():

        raise FileNotFoundError(
            npz_path
        )

    #
    # st_84adb0a9cd.npz
    # -> 84adb0a9cd
    #

    scene_hash = (
        npz_path.stem
        .split("_", 1)[1]
    )

    scene_path = (
        FIELDS_DIR
        / f"sc_{scene_hash}.json"
    )

    if not scene_path.exists():

        raise FileNotFoundError(
            f"Scene not found:\n{scene_path}"
        )

    print(
        f"Loading scene: "
        f"{scene_path.name}"
    )

    scene = load_scene(
        scene_path
    )

    print(
        f"Loading field: "
        f"{npz_path.name}"
    )

    t0 = time.time()

    data = np.load(
        npz_path,
        allow_pickle=True,
    )

    print(
        f"Loaded in "
        f"{time.time()-t0:.2f}s"
    )

    print(
        f"Rays: "
        f"{data['pos_cpu'].shape[1]:,}"
    )

    power_map, extent, p_min, p_max = (
        compute_2d_power_map(
            scene,
            data,
            grid_res=300,
            sub_steps=5,
        )
    )

    plot_power_map(
        power_map,
        scene,
        extent,
        p_min,
        p_max,
        scene_hash,
    )


if __name__ == "__main__":
    main()