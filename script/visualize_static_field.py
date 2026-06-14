#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


BG_COLOR = "#0a0a12"
CLR_STREET = "rgb(35,35,35)"
CLR_BUILDING = "rgb(70,120,200)"
CLR_WIRE = "rgba(0,255,255,0.4)"


def get_discrete_color(tx_idx: int, lobe_idx: int, alpha: float = 0.35) -> str:
    base_hues = [180, 0, 60, 300, 120, 240]
    hue = base_hues[tx_idx % len(base_hues)]
    lightness = 40 + (lobe_idx * 20)
    return f"hsla({hue},100%,{lightness}%,{alpha})"


def load_data(target_hash: str, cache_dir: Path):
    fields_dir = cache_dir / "precomputed_static_fields"

    npz_path = fields_dir / f"st_{target_hash}.npz"
    scene_path = fields_dir / f"sc_{target_hash}.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Static field not found: {npz_path}")

    scene_data = {}

    if scene_path.exists():
        with open(scene_path, "r") as f:
            scene_data = json.load(f)
    else:
        print(f"[WARNING] Scene file not found: {scene_path.name}")

    print(f"[INFO] Loading static field: {npz_path.name}")

    data = np.load(npz_path, allow_pickle=True)

    return data, scene_data


def build_visualization(
    data: np.lib.npyio.NpzFile,
    scene_data: dict,
    max_rays: int,
    hash_id: str,
):
    pos_cpu = data["pos_cpu"]
    n_pts_cpu = data["n_pts_cpu"]
    tx_ids = data["tx_ids_cpu"]

    print(f"[INFO] Loaded {len(n_pts_cpu)} ray candidates.")

    fig = go.Figure()

    base_trace_count = 0

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    x_b, y_b, z_b, i_b, j_b, k_b = [], [], [], [], [], []
    x_s, y_s, z_s, i_s, j_s, k_s = [], [], [], [], [], []
    x_w, y_w, z_w = [], [], []

    b_offset = 0
    s_offset = 0

    if "obstacles" in scene_data:

        for obs in scene_data["obstacles"]:

            mn = obs["box_min"]
            mx = obs["box_max"]

            is_street = (
                obs.get("material", "") == "asphalt"
                or mx[2] < 1.0
            )

            vx = [
                mn[0], mn[0], mx[0], mx[0],
                mn[0], mn[0], mx[0], mx[0]
            ]

            vy = [
                mn[1], mx[1], mx[1], mn[1],
                mn[1], mx[1], mx[1], mn[1]
            ]

            vz = [
                0, 0, 0, 0,
                mx[2], mx[2], mx[2], mx[2]
            ]

            i_f = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
            j_f = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
            k_f = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

            if is_street:

                x_s.extend(vx)
                y_s.extend(vy)
                z_s.extend(vz)

                i_s.extend([v + s_offset for v in i_f])
                j_s.extend([v + s_offset for v in j_f])
                k_s.extend([v + s_offset for v in k_f])

                s_offset += 8

            else:

                x_b.extend(vx)
                y_b.extend(vy)
                z_b.extend(vz)

                i_b.extend([v + b_offset for v in i_f])
                j_b.extend([v + b_offset for v in j_f])
                k_b.extend([v + b_offset for v in k_f])

                b_offset += 8

                p = [
                    (mn[0], mn[1], 0),
                    (mx[0], mn[1], 0),
                    (mx[0], mx[1], 0),
                    (mn[0], mx[1], 0),
                    (mn[0], mn[1], 0),

                    (mn[0], mn[1], mx[2]),
                    (mx[0], mn[1], mx[2]),
                    (mx[0], mx[1], mx[2]),
                    (mn[0], mx[1], mx[2]),
                    (mn[0], mn[1], mx[2]),

                    None,

                    (mx[0], mn[1], 0),
                    (mx[0], mn[1], mx[2]),

                    None,

                    (mx[0], mx[1], 0),
                    (mx[0], mx[1], mx[2]),

                    None,

                    (mn[0], mx[1], 0),
                    (mn[0], mx[1], mx[2]),
                ]

                for pt in p:

                    if pt is None:
                        x_w.append(None)
                        y_w.append(None)
                        z_w.append(None)
                    else:
                        x_w.append(pt[0])
                        y_w.append(pt[1])
                        z_w.append(pt[2])

                x_w.append(None)
                y_w.append(None)
                z_w.append(None)

    if x_s:
        fig.add_trace(
            go.Mesh3d(
                x=x_s,
                y=y_s,
                z=z_s,
                i=i_s,
                j=j_s,
                k=k_s,
                color=CLR_STREET,
                opacity=1.0,
                flatshading=True,
                hoverinfo="skip",
                name="Streets",
            )
        )
        base_trace_count += 1

    if x_b:
        fig.add_trace(
            go.Mesh3d(
                x=x_b,
                y=y_b,
                z=z_b,
                i=i_b,
                j=j_b,
                k=k_b,
                color=CLR_BUILDING,
                opacity=0.8,
                flatshading=True,
                hoverinfo="skip",
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.9,
                    specular=0.1,
                ),
                name="Buildings",
            )
        )
        base_trace_count += 1

        fig.add_trace(
            go.Scatter3d(
                x=x_w,
                y=y_w,
                z=z_w,
                mode="lines",
                line=dict(
                    color=CLR_WIRE,
                    width=2,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        base_trace_count += 1

    if "transmitters" in scene_data:

        tx_pos = np.array(
            [tx["position"] for tx in scene_data["transmitters"]]
        )

        fig.add_trace(
            go.Scatter3d(
                x=tx_pos[:, 0],
                y=tx_pos[:, 1],
                z=tx_pos[:, 2],
                mode="markers+text",
                marker=dict(
                    color="red",
                    size=8,
                    symbol="diamond",
                ),
                text=[
                    f"TX{i}"
                    for i in range(len(tx_pos))
                ],
                textposition="top center",
                name="Transmitters",
            )
        )

        base_trace_count += 1

    # ------------------------------------------------------------------
    # Rays
    # ------------------------------------------------------------------

    valid_indices = np.where(n_pts_cpu >= 2)[0]

    if len(valid_indices) > max_rays:
        valid_indices = np.random.choice(
            valid_indices,
            max_rays,
            replace=False,
        )

    print(f"[INFO] Rendering {len(valid_indices)} rays.")

    grouped = {}

    for idx in valid_indices:

        t_id = int(tx_ids[idx])

        tx_idx = t_id // 3
        lobe_idx = t_id % 3

        key = (tx_idx, lobe_idx)

        if key not in grouped:
            grouped[key] = []

        grouped[key].append(idx)

    ray_trace_specs = []

    for key in sorted(grouped.keys()):

        tx_idx, lobe_idx = key

        x_r, y_r, z_r = [], [], []

        for idx in grouped[key]:

            k = int(n_pts_cpu[idx])

            if k < 2:
                continue

            pts = pos_cpu[:k, idx, :]

            x_r.extend(pts[:, 0])
            x_r.append(None)

            y_r.extend(pts[:, 1])
            y_r.append(None)

            z_r.extend(pts[:, 2])
            z_r.append(None)

        if not x_r:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=x_r,
                y=y_r,
                z=z_r,
                mode="lines",
                line=dict(
                    color=get_discrete_color(
                        tx_idx,
                        lobe_idx,
                        alpha=0.35,
                    ),
                    width=1.5,
                ),
                name=f"TX {tx_idx} - Lobe {lobe_idx}",
            )
        )

        ray_trace_specs.append(key)

    # ------------------------------------------------------------------
    # Dropdown
    # ------------------------------------------------------------------

    dropdown_buttons = []

    total_traces = base_trace_count + len(ray_trace_specs)

    dropdown_buttons.append(
        dict(
            label="Show All",
            method="update",
            args=[{"visible": [True] * total_traces}],
        )
    )

    lobes = sorted(
        set(
            lobe
            for _, lobe in ray_trace_specs
        )
    )

    for lobe_val in lobes:

        visibility = (
            [True] * base_trace_count
            +
            [
                spec[1] == lobe_val
                for spec in ray_trace_specs
            ]
        )

        dropdown_buttons.append(
            dict(
                label=f"Lobe {lobe_val}",
                method="update",
                args=[{"visible": visibility}],
            )
        )

    axis_style = dict(
        showgrid=True,
        gridcolor="#333",
        zeroline=False,
        showticklabels=False,
        backgroundcolor=BG_COLOR,
        title="",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        title=dict(
            text=f"Static Field Visualization (Hash: {hash_id})"
        ),
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0,
        ),
        height=800,
        scene=dict(
            aspectmode="data",
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            camera=dict(
                eye=dict(
                    x=1.5,
                    y=1.5,
                    z=1.0,
                )
            ),
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="#1a1a24",
                bordercolor="#444",
                font=dict(color="#fff"),
                buttons=dropdown_buttons,
            )
        ],
    )

    print("[INFO] Launching interactive 3D visualization...")

    fig.show()


def main():

    parser = argparse.ArgumentParser(
        description="Static Field 3D Visualizer"
    )

    parser.add_argument(
        "--hash",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
    )

    parser.add_argument(
        "--max-rays",
        type=int,
        default=1500,
    )

    args = parser.parse_args()

    cache_path = Path(args.cache_dir).resolve()

    try:

        data, scene_data = load_data(
            args.hash,
            cache_path,
        )

        build_visualization(
            data,
            scene_data,
            args.max_rays,
            args.hash,
        )

    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")


if __name__ == "__main__":
    main()