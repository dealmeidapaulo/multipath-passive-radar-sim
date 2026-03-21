from __future__ import annotations
from typing import List
import numpy as np
import colorsys

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("pip install plotly")

from src.core.scene.domain import Scene
from src.core.scene.ray import Ray

BG, CLR_BUILDING, CLR_FLOOR, CLR_TX, CLR_RX, CLR_UAV, CLR_AXES = 'black', 'rgba(100,180,255,0.55)', 'rgba(60,180,80,0.25)', '#FF4444', '#00FFFF', '#FFA500', 'rgba(180,180,180,0.4)'

def _sphere_mesh(center, radius, color, opacity=0.85, n_subdivide=2) -> go.Mesh3d:
    phi_g = (1 + np.sqrt(5)) / 2
    verts = np.array([[-1,phi_g,0],[1,phi_g,0],[-1,-phi_g,0],[1,-phi_g,0],[0,-1,phi_g],[0,1,phi_g],[0,-1,-phi_g],[0,1,-phi_g],[phi_g,0,-1],[phi_g,0,1],[-phi_g,0,-1],[-phi_g,0,1]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    faces = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]], dtype=int)
    for _ in range(n_subdivide):
        midpoints = {}; verts_list = list(verts); new_faces = []
        def _mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                m = (verts_list[a] + verts_list[b]) / 2
                midpoints[key] = len(verts_list); verts_list.append(m / np.linalg.norm(m))
            return midpoints[key]
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        verts = np.array(verts_list); faces = np.array(new_faces, dtype=int)
    v = verts * radius + np.asarray(center, dtype=float)
    return go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color=color, opacity=opacity, flatshading=False, lighting=dict(ambient=0.5, diffuse=0.9, specular=0.3, roughness=0.4), hoverinfo='skip', showlegend=False)

def _assign_uav_colors(rays: List[Ray], opacity: float = 1.0) -> dict:
    hit = [r for r in rays if getattr(r, 'is_uav_bounce', False)]
    hit_sorted = sorted(hit, key=lambda r: getattr(r, 'doppler_shift', 0.0))
    return {id(r): f"rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{opacity})" for i, r in enumerate(hit_sorted) for rgb in [colorsys.hsv_to_rgb(i / max(len(hit_sorted), 1), 1.0, 1.0)]}

def _all_wireframes(obstacles, color, width=1) -> go.Scatter3d:
    xs, ys, zs = [], [], []
    for obs in obstacles:
        mn, mx = np.asarray(obs.box_min, float), np.asarray(obs.box_max, float)
        corners = np.array([mn, [mx[0],mn[1],mn[2]], [mx[0],mx[1],mn[2]], [mn[0],mx[1],mn[2]], [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]], mx, [mn[0],mx[1],mx[2]]])
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            xs += [corners[a,0], corners[b,0], None]; ys += [corners[a,1], corners[b,1], None]; zs += [corners[a,2], corners[b,2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color=color, width=width), hoverinfo='skip', showlegend=False)

def _floor_grid(box, n_lines: int = 8) -> go.Scatter3d:
    mn, mx, z = box.box_min, box.box_max, float(box.box_min[2])
    xs, ys, zs = [], [], []
    for t in np.linspace(0, 1, n_lines + 2):
        xv, yv = float(mn[0] + t * (mx[0] - mn[0])), float(mn[1] + t * (mx[1] - mn[1]))
        xs += [xv, xv, None, float(mn[0]), float(mx[0]), None]; ys += [float(mn[1]), float(mx[1]), None, yv, yv, None]; zs += [z, z, None, z, z, None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color=CLR_FLOOR, width=1), hoverinfo='skip', showlegend=False)

def plot_trajectory(scene: Scene, trajectory_rays: List[List[Ray]], uav_states: List[np.ndarray], dt: float, title: str = "UAV Trajectory") -> go.Figure:
    data: list = [_floor_grid(scene.box)]
    if scene.obstacles: data.append(_all_wireframes(scene.obstacles, color=CLR_BUILDING, width=1))

    for tx in scene.transmitters:
        p = tx.position
        data.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode='markers+text', marker=dict(size=9, color=CLR_TX, symbol='diamond', line=dict(color='white', width=1)), text=[f"TX  {tx.frequency/1e9:.2f} GHz"], textfont=dict(color='white', size=11), textposition='top center', name=f"TX{tx.tx_id}"))

    p, r = scene.receiver.position, getattr(scene.receiver, 'radius', 2.0)
    data.append(_sphere_mesh(p, r, color=CLR_RX, opacity=0.75))
    data.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2] + r + 2], mode='text', text=['RX'], textfont=dict(color=CLR_RX, size=12), hoverinfo='skip', showlegend=False))

    uav_path = np.array(uav_states)
    data.append(go.Scatter3d(x=uav_path[:,0], y=uav_path[:,1], z=uav_path[:,2], mode='lines', line=dict(color=CLR_UAV, width=4, dash='dot'), hoverinfo='skip', showlegend=False))

    # Safety Cap for WebGL
    MAX_STATIC_RAYS_DRAWN = 300 
    total_uav_hits = 0

    for step, (uav_pos, rays) in enumerate(zip(uav_states, trajectory_rays)):
        is_last = (step == len(uav_states) - 1)
        base_opacity = 0.9 if is_last else 0.15
        
        data.append(_sphere_mesh(uav_pos, scene.uav.radius, color=CLR_UAV, opacity=base_opacity))
        if is_last:
            data.append(go.Scatter3d(x=[uav_pos[0]], y=[uav_pos[1]], z=[uav_pos[2] + scene.uav.radius + 2], mode='text', text=[f"UAV (t={step*dt}s)"], textfont=dict(color='white', size=11), hoverinfo='skip', showlegend=False))

        uav_colors = _assign_uav_colors(rays, opacity=base_opacity)

        # Separate rays
        uav_rays = [r for r in rays if getattr(r, 'is_uav_bounce', False)]
        static_rays = [r for r in rays if not getattr(r, 'is_uav_bounce', False)]
        total_uav_hits += len(uav_rays)

        # Downsample static rays via stride to preserve Fibonacci angular distribution
        if len(static_rays) > MAX_STATIC_RAYS_DRAWN:
            stride = len(static_rays) // MAX_STATIC_RAYS_DRAWN
            static_rays = static_rays[::stride][:MAX_STATIC_RAYS_DRAWN]

        rays_to_draw = uav_rays + static_rays

        for ray in rays_to_draw:
            pts = np.array(ray.points)
            is_visible = getattr(ray, 'visible', True)
            is_uav = getattr(ray, 'is_uav_bounce', False)

            if not is_visible:
                # Occluded baseline ray (White dashed)
                data.append(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='lines', 
                    line=dict(color=f'rgba(255,255,255,{base_opacity*0.6})', width=1, dash='dot'),
                    hovertemplate=f"t={step*dt}s<br>OCCLUDED<extra></extra>", showlegend=False
                ))
            elif is_uav:
                # Dynamic Doppler bounce (Colored solid)
                color = uav_colors.get(id(ray), f'rgba(255,255,0,{base_opacity})')
                data.append(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='lines', 
                    line=dict(color=color, width=2 if is_last else 1),
                    hovertemplate=f"t={step*dt}s<br>nb={ray.n_bounces}<br>τ={ray.delay()*1e9:.2f} ns<br>f_D={ray.doppler_shift:+.3f} Hz<extra></extra>", showlegend=False
                ))
            else:
                # Visible static baseline ray (Cyan thin)
                data.append(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='lines', 
                    line=dict(color=f'rgba(0,255,255,{base_opacity*0.2})', width=1),
                    hovertemplate=f"t={step*dt}s<br>STATIC<extra></extra>", showlegend=False
                ))

    axis_cfg = dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False, title='', backgroundcolor=BG, gridcolor=CLR_AXES, linecolor=CLR_AXES)
    
    title_display = f"{title} | {len(uav_states)} frames | {total_uav_hits} UAV Hits | Display capped to {MAX_STATIC_RAYS_DRAWN} static rays/frame"
    
    return go.Figure(data=data, layout=go.Layout(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color='white'), title=dict(text=title_display, font=dict(color='white', size=13), x=0.01), scene=dict(bgcolor=BG, xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg, aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))), margin=dict(l=0, r=0, t=45, b=0)))

# Kept for backward compatibility
def plot(scene: Scene, rays: List[Ray], title: str = "Ray Tracer") -> go.Figure:
    return plot_trajectory(scene, [rays], [scene.uav.position], 0.0, title)


# ─────────────────────────────────────────────────────────────────────────────
# v3 helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_frame_rays(
    anchors_vis : "List[Ray]",
    anchors_occ : "List[Ray]",
    uav_bounces : "List[Ray]",
) -> "List[Ray]":
    """
    Flatten the three lists returned by apply_uav() into a single List[Ray]
    compatible with plot_trajectory().

    Occluded anchors have visible=False (drawn as white dashed).
    UAV bounces have is_uav_bounce=True (drawn coloured by Doppler).
    Visible anchors are static baseline (drawn cyan thin).
    """
    return anchors_vis + anchors_occ + uav_bounces


def plot_from_static(
    scene,
    frames_vis  : "List[List[Ray]]",
    frames_occ  : "List[List[Ray]]",
    frames_uav  : "List[List[Ray]]",
    uav_states  : "List[np.ndarray]",
    dt          : float = 1.0,
    title       : str   = "UAV Trajectory v3",
) -> "go.Figure":
    """
    Convenience wrapper: accepts the (vis, occ, bounces) tuples produced by
    apply_uav() for each frame and forwards them to plot_trajectory().

    Parameters
    ----------
    scene        : Scene
    frames_vis   : list of anchors_vis per frame
    frames_occ   : list of anchors_occ per frame
    frames_uav   : list of uav_bounces per frame
    uav_states   : list of uav position arrays per frame
    dt, title    : passed through to plot_trajectory
    """
    trajectory_rays = [
        make_frame_rays(v, o, u)
        for v, o, u in zip(frames_vis, frames_occ, frames_uav)
    ]
    return plot_trajectory(scene, trajectory_rays, uav_states, dt, title)
