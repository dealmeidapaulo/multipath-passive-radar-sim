import sys, pathlib, time
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain          import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav
from src.outputs.visualizer         import plot_trajectory, plot_from_static, make_frame_rays

CELL = "=" * 62


def make_scene(n_rays=20_000):
    box = Box(np.zeros(3), np.array([100., 100., 60.]))
    tx  = Transmitter(np.array([5.,  50., 30.]), 700e6, tx_power_w=500., tx_id=0)
    rx  = Receiver(np.array([95., 50., 20.]), radius=8.)
    obs = [Obstacle(np.array([20., 10., 0.]), np.array([50., 35., 30.])),
           Obstacle(np.array([20., 65., 0.]), np.array([50., 90., 28.]))]
    uav = UAV(np.array([55., 50., 35.]), np.array([1., 0., 0.]), radius=1.0)
    scene = Scene(box=box, transmitters=[tx], receiver=rx,
                  uav=uav, obstacles=obs, n_rays=n_rays, n_max=6)
    scene.use_physics=True; scene.roughness=0.3
    scene.bandwidth_hz=0.5e6; scene.temperature_c=20.
    scene.uav_roughness=0.4; scene.n_samples_uav=8
    return scene


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


# ── Precompute once ───────────────────────────────────────────────────────────
scene  = make_scene()
static = precompute(scene, seed=42, cell_size=10.)
print(f"Setup: anchors={len(static.anchors)}  hash={static.spatial_hash.total_entries:,}")

# Build 3-frame trajectory
fv_list, fo_list, fu_list, us_list = [], [], [], []
uav_start = scene.uav.position.copy()
for step in range(3):
    pos = uav_start + np.array([1., 0., 0.]) * step
    scene.uav.position = pos.copy(); scene.uav.velocity = np.array([1.,0.,0.])
    v, o, u = apply_uav(static, scene.uav, scene)
    fv_list.append(v); fo_list.append(o); fu_list.append(u); us_list.append(pos.copy())

# ── TEST 1: make_frame_rays ────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 1 — make_frame_rays\n{CELL}")
for step, (v, o, u) in enumerate(zip(fv_list, fo_list, fu_list)):
    frame = make_frame_rays(v, o, u)
    assert len(frame) == len(v)+len(o)+len(u), "FAIL: ray count mismatch"
    uav_count = sum(1 for r in frame if r.is_uav_bounce)
    occ_count = sum(1 for r in frame if not r.visible)
    assert uav_count == len(u),  "FAIL: UAV bounce count wrong"
    assert occ_count == len(o),  "FAIL: occluded count wrong"
    print(f"  frame {step+1}: total={len(frame)}  uav={uav_count}  occ={occ_count}")
print("  → PASS")

# ── TEST 2: plot_trajectory ────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 2 — plot_trajectory (3 frames)\n{CELL}")
traj_rays = [make_frame_rays(v,o,u) for v,o,u in zip(fv_list,fo_list,fu_list)]
t0  = time.time()
fig = plot_trajectory(scene, traj_rays, us_list, dt=1.0,
                      title="test_visualizer — 3 frames")
t_fig = time.time() - t0

import plotly.graph_objects as go
scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
mesh_traces    = [t for t in fig.data if isinstance(t, go.Mesh3d)]

print(f"  figure built: {_fmt(t_fig)}")
print(f"  Scatter3d traces: {len(scatter_traces)}")
print(f"  Mesh3d traces   : {len(mesh_traces)}  (TX sphere, RX sphere, UAV sphere×frames)")
assert len(scatter_traces) > 0, "FAIL: no Scatter3d traces"
assert len(mesh_traces)    > 0, "FAIL: no Mesh3d traces (spheres missing)"
print("  → PASS")

# ── TEST 3: plot_from_static ───────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 3 — plot_from_static wrapper\n{CELL}")
t0   = time.time()
fig2 = plot_from_static(scene, frames_vis=fv_list, frames_occ=fo_list,
                        frames_uav=fu_list, uav_states=us_list, dt=1.0,
                        title="test_visualizer — plot_from_static")
t_fig2 = time.time() - t0
print(f"  figure built: {_fmt(t_fig2)}")
assert len(fig2.data) == len(fig.data), \
    f"FAIL: plot_from_static gives {len(fig2.data)} traces vs {len(fig.data)}"
print("  → PASS")

# ── TEST 4: UAV-bounce hover template ─────────────────────────────────────────
print(f"\n{CELL}\nTEST 4 — Hover templates: τ e f_D en UAV-bounce traces\n{CELL}")
# New visualiser uses Plotly frames (slider) — UAV bounce traces live in
# fig.frames[n].data, not in fig.data (which is only the initial frame 0).
all_traces = list(fig.data) + [t for fr in fig.frames for t in fr.data]
uav_traces = [t for t in all_traces
              if isinstance(t, go.Scatter3d)
              and hasattr(t, 'hovertemplate')
              and t.hovertemplate
              and 'f_D' in str(t.hovertemplate)]
print(f"  UAV-bounce traces con f_D en hover: {len(uav_traces)}")
total_uav = sum(len(u) for u in fu_list)
if total_uav > 0:
    assert len(uav_traces) > 0, "FAIL: UAV bounces existentes pero sin hover f_D"
print("  → PASS")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")