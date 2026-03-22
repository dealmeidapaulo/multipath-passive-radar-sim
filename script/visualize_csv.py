"""
Load a saved CSV of observables and generate an interactive 3D figure.

The script reconstructs approximate ray endpoints from the observables
(τ, θ, φ) and TX/RX positions stored in the CSV. It does NOT need the
original scene or StaticField — useful for reviewing past runs.

Usage
-----
    python script/visualize.py results.csv
    python script/visualize.py results.csv --frame 2 --no_browser
"""
import sys, pathlib, argparse
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("pip install plotly")

C = 3e8
BG = '#0a0a12'


def _ray_endpoint_from_observables(row):
    """
    Approximate ray path: TX → midpoint → RX using delay and AoA.
    Returns (start, end) as np arrays.
    """
    tau   = float(row['tau_s'])
    theta = float(row['theta_rad'])
    phi   = float(row['phi_rad'])
    rx    = np.array([row['rx_pos_x'], row['rx_pos_y'], row['rx_pos_z']])
    # Arrival direction from (θ, φ)
    arr = np.array([
        np.cos(theta)*np.cos(phi),
        np.cos(theta)*np.sin(phi),
        np.sin(theta),
    ])
    total_len = tau * C
    start = rx - arr * total_len   # approximate TX position along arrival direction
    return start, rx


def main():
    p = argparse.ArgumentParser(description="multipath-passive-radar-sim — Visualise saved CSV")
    p.add_argument("csv",         type=str, help="Path to observables CSV")
    p.add_argument("--frame",     type=int, default=-1,
                   help="time_step to display (-1 = all frames, use slider)")
    p.add_argument("--no_browser",action="store_true",
                   help="Don't open browser; just show inline (Jupyter/Colab)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows, {df['time_step'].nunique()} frames")
    print(f"  instance_ids: {df['instance_id'].unique()[:5]}")
    print(f"  UAV bounces : {df['is_uav_bounce'].sum()}")

    steps  = sorted(df['time_step'].unique())
    if args.frame >= 0:
        steps = [args.frame]

    traces = []
    # Static background
    if 'rx_pos_x' in df.columns:
        rx = df[['rx_pos_x','rx_pos_y','rx_pos_z']].iloc[0].values
        traces.append(go.Scatter3d(
            x=[rx[0]], y=[rx[1]], z=[rx[2]],
            mode='markers+text', text=['RX'],
            marker=dict(size=8, color='#00FFFF'),
            textfont=dict(color='#00FFFF'), showlegend=False))

    for step in steps:
        sub = df[df['time_step']==step]
        vis_rows = sub[sub['visible']==1]
        occ_rows = sub[sub['visible']==0]
        uav_rows = sub[sub['is_uav_bounce']==True]

        # UAV position marker
        if 'uav_pos_x' in sub.columns and sub['uav_present'].iloc[0]:
            uav_pos = sub[['uav_pos_x','uav_pos_y','uav_pos_z']].iloc[0].values
            traces.append(go.Scatter3d(
                x=[uav_pos[0]], y=[uav_pos[1]], z=[uav_pos[2]],
                mode='markers', name=f'UAV t={step}',
                marker=dict(size=6, color='#FFA500'), showlegend=True))

        # Ray traces (approximate)
        def add_rays(rows, color, name, dash='solid', width=1):
            first = True
            for _, row in rows.iterrows():
                try:
                    start, end = _ray_endpoint_from_observables(row)
                    traces.append(go.Scatter3d(
                        x=[start[0],end[0]], y=[start[1],end[1]], z=[start[2],end[2]],
                        mode='lines', line=dict(color=color, width=width, dash=dash),
                        name=name, legendgroup=name,
                        showlegend=first,
                        hovertemplate=(f"τ={row['tau_s']*1e9:.1f}ns<br>"
                                       f"P={row['power_dbm']:.1f}dBm<br>"
                                       f"f_D={row['f_D']:+.3f}Hz<extra></extra>"),
                    ))
                    first = False
                except Exception:
                    pass

        add_rays(vis_rows, 'rgba(0,200,255,0.3)', 'Static rays')
        add_rays(occ_rows, 'rgba(255,255,255,0.4)', 'Occluded rays', dash='dot')
        add_rays(uav_rows, 'rgba(255,160,0,0.9)',  'UAV bounces', width=2)

    axis = dict(showgrid=False, zeroline=False, showticklabels=False,
                backgroundcolor=BG)
    fig = go.Figure(data=traces, layout=go.Layout(
        title=dict(text=f"multipath-passive-radar-sim — {pathlib.Path(args.csv).name}",
                   font=dict(color='white')),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color='white'),
        scene=dict(bgcolor=BG, xaxis=axis, yaxis=axis, zaxis=axis,
                   aspectmode='data'),
        legend=dict(x=1., y=0.95, bgcolor='rgba(20,20,30,0.8)',
                    itemclick='toggle'),
        height=600,
    ))

    renderer = None if args.no_browser else 'browser'
    if renderer:
        fig.show(renderer=renderer)
    else:
        fig.show()


if __name__ == "__main__":
    main()
