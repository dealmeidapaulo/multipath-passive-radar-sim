from __future__ import annotations
from typing import List, Optional
import numpy as np
import colorsys

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("pip install plotly")

from src.core.scene.domain import Scene
from src.core.scene.ray    import Ray

# ── Colour palette ────────────────────────────────────────────────────────────
BG           = '#0a0a12'
CLR_BUILDING = 'rgba(80,140,220,0.35)'
CLR_FLOOR    = 'rgba(50,160,70,0.18)'
CLR_TX       = '#FF4444'
CLR_RX       = '#00FFFF'
CLR_UAV      = '#FFA500'
CLR_STATIC   = 'rgba(0,220,255,{a})'     # cyan, alpha variable
CLR_OCC      = 'rgba(255,255,255,{a})'   # white dashed
CLR_AXES     = 'rgba(160,160,160,0.35)'

MAX_STATIC_PER_FRAME = 300   # WebGL cap — downsampled by stride


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _sphere_mesh(center, radius, color, opacity=0.85) -> go.Mesh3d:
    phi_g = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,phi_g,0],[1,phi_g,0],[-1,-phi_g,0],[1,-phi_g,0],
        [0,-1,phi_g],[0,1,phi_g],[0,-1,-phi_g],[0,1,-phi_g],
        [phi_g,0,-1],[phi_g,0,1],[-phi_g,0,-1],[-phi_g,0,1]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]], dtype=int)
    # one subdivision pass
    midpoints = {}; vl = list(verts); nf = []
    def _mid(a, b):
        k = (min(a,b), max(a,b))
        if k not in midpoints:
            m = (vl[a]+vl[b])/2; midpoints[k]=len(vl); vl.append(m/np.linalg.norm(m))
        return midpoints[k]
    for f in faces:
        a,b,c = int(f[0]),int(f[1]),int(f[2])
        ab,bc,ca = _mid(a,b),_mid(b,c),_mid(c,a)
        nf += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
    verts = np.array(vl); faces = np.array(nf, dtype=int)
    v = verts * radius + np.asarray(center, dtype=float)
    return go.Mesh3d(
        x=v[:,0], y=v[:,1], z=v[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color=color, opacity=opacity, flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=0.3),
        hoverinfo='skip', showlegend=False)


def _wireframes(obstacles, color=CLR_BUILDING, width=1) -> go.Scatter3d:
    xs, ys, zs = [], [], []
    for obs in obstacles:
        mn, mx = np.asarray(obs.box_min, float), np.asarray(obs.box_max, float)
        corners = np.array([
            mn, [mx[0],mn[1],mn[2]], [mx[0],mx[1],mn[2]], [mn[0],mx[1],mn[2]],
            [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]], mx, [mn[0],mx[1],mx[2]]])
        for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            xs += [corners[a,0],corners[b,0],None]
            ys += [corners[a,1],corners[b,1],None]
            zs += [corners[a,2],corners[b,2],None]
    return go.Scatter3d(x=xs,y=ys,z=zs, mode='lines',
                        line=dict(color=color,width=width),
                        hoverinfo='skip', showlegend=False)


def _floor_grid(box, n=8) -> go.Scatter3d:
    mn,mx,z = box.box_min, box.box_max, float(box.box_min[2])
    xs,ys,zs = [],[],[]
    for t in np.linspace(0,1,n+2):
        xv = float(mn[0]+t*(mx[0]-mn[0])); yv = float(mn[1]+t*(mx[1]-mn[1]))
        xs += [xv,xv,None,float(mn[0]),float(mx[0]),None]
        ys += [float(mn[1]),float(mx[1]),None,yv,yv,None]
        zs += [z,z,None,z,z,None]
    return go.Scatter3d(x=xs,y=ys,z=zs, mode='lines',
                        line=dict(color=CLR_FLOOR,width=1),
                        hoverinfo='skip', showlegend=False)


def _doppler_color(doppler: float, d_min: float, d_max: float) -> str:
    """Map Doppler value to HSV hue (blue=negative, red=positive)."""
    span = d_max - d_min if d_max > d_min else 1.0
    t    = (doppler - d_min) / span
    hue  = 0.67 * (1.0 - t)   # 0.67=blue … 0.0=red
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.9)"


# ── Frame builder ─────────────────────────────────────────────────────────────

def _build_frame_traces(
    vis_rays  : List[Ray],
    occ_rays  : List[Ray],
    uav_rays  : List[Ray],
    uav_pos   : np.ndarray,
    uav_vel   : Optional[np.ndarray],
    uav_radius: float,
    step      : int,
    dt        : float,
    d_min     : float,
    d_max     : float,
) -> List[go.BaseTraceType]:
    """Build all Scatter3d/Cone/Mesh3d traces for one frame."""
    traces = []

    # ── UAV sphere ────────────────────────────────────────────────────────────
    traces.append(_sphere_mesh(uav_pos, uav_radius, CLR_UAV, opacity=0.80))

    # ── UAV velocity cone ────────────────────────────────────────────────────
    if uav_vel is not None:
        spd = float(np.linalg.norm(uav_vel))
        if spd > 0.01:
            scale = min(uav_radius * 3.0, spd * 0.6)
            uv    = uav_vel / spd * scale
            traces.append(go.Cone(
                x=[float(uav_pos[0])], y=[float(uav_pos[1])],
                z=[float(uav_pos[2])],
                u=[float(uv[0])], v=[float(uv[1])], w=[float(uv[2])],
                sizeref=1.0, sizemode='absolute',
                colorscale=[[0,CLR_UAV],[1,CLR_UAV]],
                showscale=False, hoverinfo='skip',
                showlegend=False,
            ))

    # ── Static visible rays ───────────────────────────────────────────────────
    static_sample = vis_rays
    if len(vis_rays) > MAX_STATIC_PER_FRAME:
        stride = len(vis_rays) // MAX_STATIC_PER_FRAME
        static_sample = vis_rays[::stride][:MAX_STATIC_PER_FRAME]

    first_static = True
    for ray in static_sample:
        pts = np.array(ray.points)
        traces.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='lines',
            line=dict(color=CLR_STATIC.format(a=0.25), width=1),
            name='Static rays', legendgroup='static',
            showlegend=first_static,
            hovertemplate=(f"t={step*dt:.1f}s<br>"
                           f"τ={ray.delay()*1e9:.1f}ns<br>"
                           f"P={ray.power_dbm:.1f}dBm<extra></extra>"),
        ))
        first_static = False

    # ── Occluded rays (white dashed) ──────────────────────────────────────────
    first_occ = True
    for ray in occ_rays:
        pts = np.array(ray.points)
        traces.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='lines',
            line=dict(color=CLR_OCC.format(a=0.5), width=1, dash='dot'),
            name='Occluded rays', legendgroup='occluded',
            showlegend=first_occ,
            hovertemplate=f"t={step*dt:.1f}s<br>OCCLUDED<extra></extra>",
        ))
        first_occ = False

    # ── UAV-bounce rays (coloured by Doppler) ─────────────────────────────────
    first_uav = True
    for ray in uav_rays:
        pts  = np.array(ray.points)
        col  = _doppler_color(ray.doppler_shift, d_min, d_max)
        traces.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='lines',
            line=dict(color=col, width=3),
            name='UAV bounces', legendgroup='uav_bounces',
            showlegend=first_uav,
            hovertemplate=(f"t={step*dt:.1f}s<br>"
                           f"τ={ray.delay()*1e9:.1f}ns<br>"
                           f"f_D={ray.doppler_shift:+.3f}Hz<br>"
                           f"P={ray.power_dbm:.1f}dBm<extra></extra>"),
        ))
        first_uav = False

    return traces


# ── Public API ────────────────────────────────────────────────────────────────

def plot_from_static(
    scene       : Scene,
    frames_vis  : List[List[Ray]],
    frames_occ  : List[List[Ray]],
    frames_uav  : List[List[Ray]],
    uav_states  : List[np.ndarray],
    uav_vels    : Optional[List[np.ndarray]] = None,
    dt          : float = 1.0,
    title       : str   = "multipath-passive-radar-sim",
) -> go.Figure:
    """
    Build an interactive Plotly figure with:
      - Frame slider (one step per frame)
      - Play / Pause buttons
      - Clickable legend: Static rays | Occluded rays | UAV bounces
      - UAV velocity cone

    Parameters
    ----------
    scene       : Scene (for buildings, TX, RX)
    frames_vis  : list of anchors_vis per frame
    frames_occ  : list of anchors_occ per frame
    frames_uav  : list of uav_bounces per frame
    uav_states  : list of uav position arrays per frame
    uav_vels    : list of uav velocity arrays per frame (optional)
    dt          : time step in seconds
    title       : figure title
    """
    N = len(frames_vis)
    if uav_vels is None:
        uav_vels = [None] * N

    # Doppler range across all frames for consistent colour scale
    all_dops = [r.doppler_shift
                for fu in frames_uav for r in fu]
    d_min = min(all_dops) if all_dops else -1.0
    d_max = max(all_dops) if all_dops else  1.0

    uav_rad = float(scene.uav.radius) if scene.uav is not None else 1.0

    # ── Static scene elements (shared across all frames) ─────────────────────
    base_data: list = [_floor_grid(scene.box)]
    if scene.obstacles:
        base_data.append(_wireframes(scene.obstacles))

    for tx in scene.transmitters:
        p = tx.position
        base_data.append(_sphere_mesh(p, 2.0, CLR_TX, opacity=0.9))
        base_data.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]+4], mode='text',
            text=[f"TX {tx.tx_id}<br>{tx.frequency/1e6:.0f}MHz"],
            textfont=dict(color=CLR_TX, size=10),
            hoverinfo='skip', showlegend=False))

    rp = scene.receiver.position; rr = scene.receiver.radius
    base_data.append(_sphere_mesh(rp, rr, CLR_RX, opacity=0.60))
    base_data.append(go.Scatter3d(
        x=[rp[0]], y=[rp[1]], z=[rp[2]+rr+2], mode='text', text=['RX'],
        textfont=dict(color=CLR_RX, size=11),
        hoverinfo='skip', showlegend=False))

    # UAV path line
    path = np.array(uav_states)
    base_data.append(go.Scatter3d(
        x=path[:,0], y=path[:,1], z=path[:,2],
        mode='lines', line=dict(color=CLR_UAV, width=3, dash='dot'),
        hoverinfo='skip', showlegend=False))

    n_base = len(base_data)

    # ── Frame 0 initial traces ────────────────────────────────────────────────
    frame0_traces = _build_frame_traces(
        frames_vis[0], frames_occ[0], frames_uav[0],
        uav_states[0], uav_vels[0], uav_rad, 0, dt, d_min, d_max)

    all_data = base_data + frame0_traces

    # ── Plotly frames ─────────────────────────────────────────────────────────
    plotly_frames = []
    for step in range(N):
        ft = _build_frame_traces(
            frames_vis[step], frames_occ[step], frames_uav[step],
            uav_states[step], uav_vels[step], uav_rad,
            step, dt, d_min, d_max)
        n_vis   = len(frames_vis[step])
        n_occ   = len(frames_occ[step])
        n_uav   = len(frames_uav[step])
        d_frame = [r.doppler_shift for r in frames_uav[step]]
        stats   = (f"t={step*dt:.1f}s | "
                   f"vis={n_vis} occ={n_occ} UAV_bounces={n_uav}"
                   + (f" | f_D=[{min(d_frame):+.2f},{max(d_frame):+.2f}]Hz"
                      if d_frame else ""))
        plotly_frames.append(go.Frame(
            data=base_data + ft,
            name=str(step),
            layout=go.Layout(title_text=f"{title} — {stats}"),
        ))

    # ── Slider + buttons ──────────────────────────────────────────────────────
    sliders = [dict(
        active=0, pad=dict(t=50),
        steps=[dict(
            label=f"t={i*dt:.1f}s",
            method='animate',
            args=[[str(i)], dict(
                mode='immediate', frame=dict(duration=400, redraw=True),
                transition=dict(duration=100))],
        ) for i in range(N)],
    )]

    updatemenus = [dict(
        type='buttons', showactive=False,
        y=0, x=0.05, xanchor='right', yanchor='top',
        buttons=[
            dict(label='▶ Play',
                 method='animate',
                 args=[None, dict(frame=dict(duration=600, redraw=True),
                                  fromcurrent=True,
                                  transition=dict(duration=150))]),
            dict(label='⏸ Pause',
                 method='animate',
                 args=[[None], dict(mode='immediate',
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0))]),
        ],
    )]

    # ── Legend groups visibility ──────────────────────────────────────────────
    # Plotly legend click toggles the whole legendgroup automatically.
    # Users click "Static rays", "Occluded rays", "UAV bounces" in the legend.

    axis_cfg = dict(
        showgrid=False, zeroline=False, showticklabels=False,
        showspikes=False, title='',
        backgroundcolor=BG, gridcolor=CLR_AXES, linecolor=CLR_AXES,
    )

    fig = go.Figure(
        data=all_data,
        frames=plotly_frames,
        layout=go.Layout(
            title=dict(text=title, font=dict(color='white', size=13), x=0.01),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color='white'),
            scene=dict(
                bgcolor=BG,
                xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg,
                aspectmode='data',
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.7)),
            ),
            legend=dict(
                x=1.0, y=0.95,
                bgcolor='rgba(20,20,30,0.8)',
                bordercolor='rgba(180,180,180,0.4)',
                borderwidth=1,
                font=dict(size=11),
                itemclick='toggle',
                itemdoubleclick='toggleothers',
            ),
            sliders=sliders,
            updatemenus=updatemenus,
            margin=dict(l=0, r=0, t=50, b=60),
            height=620,
        ),
    )
    return fig


def plot_trajectory(
    scene           : Scene,
    trajectory_rays : List[List[Ray]],
    uav_states      : List[np.ndarray],
    dt              : float = 1.0,
    title           : str   = "UAV Trajectory",
) -> go.Figure:
    """
    Convenience wrapper that accepts the old-style flat List[Ray] per frame.
    Splits each frame into (vis, occ, uav_bounces) and calls plot_from_static.
    """
    fv, fo, fu = [], [], []
    for rays in trajectory_rays:
        v = [r for r in rays if getattr(r,'visible',True) and not r.is_uav_bounce]
        o = [r for r in rays if not getattr(r,'visible',True) and not r.is_uav_bounce]
        u = [r for r in rays if r.is_uav_bounce]
        fv.append(v); fo.append(o); fu.append(u)
    return plot_from_static(scene, fv, fo, fu, uav_states, dt=dt, title=title)


def make_frame_rays(
    anchors_vis: List[Ray],
    anchors_occ: List[Ray],
    uav_bounces: List[Ray],
) -> List[Ray]:
    """Flatten vis/occ/bounces into a single list (backward compat)."""
    return anchors_vis + anchors_occ + uav_bounces