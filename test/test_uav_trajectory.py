import sys, pathlib, time
import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain          import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav
from src.outputs.observables        import to_dataframe

C    = 3e8
CELL = "=" * 62


def make_scene() -> Scene:
    """
    Corredor horizontal: TX=(5,50,30) → RX=(95,50,20).
    Edificios al norte y sur del corredor, ninguno bloquea la diagonal.
    UAV empieza en (40,50,35) y avanza +1m/frame en X.
    """
    box = Box(np.zeros(3), np.array([100., 100., 60.]))
    tx  = Transmitter(np.array([5.,  50., 30.]), 700e6, tx_power_w=500., tx_id=0)
    rx  = Receiver(np.array([95., 50., 20.]),    radius=8.)
    obs = [
        Obstacle(np.array([20., 10., 0.]), np.array([50., 35., 30.])),
        Obstacle(np.array([20., 65., 0.]), np.array([50., 90., 28.])),
        Obstacle(np.array([60., 10., 0.]), np.array([80., 35., 25.])),
        Obstacle(np.array([60., 65., 0.]), np.array([80., 90., 22.])),
    ]
    uav = UAV(np.array([40., 50., 35.]), np.array([1., 0., 0.]), radius=1.0)
    scene = Scene(box=box, transmitters=[tx], receiver=rx,
                  uav=uav, obstacles=obs, n_rays=50_000, n_max=8)
    scene.use_physics   = True
    scene.roughness     = 0.3
    scene.bandwidth_hz  = 0.5e6
    scene.temperature_c = 20.
    scene.uav_roughness = 0.4
    scene.n_samples_uav = 8
    return scene


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


# ── TEST 1: Precompute ────────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 1 — Precompute 50k rayos, 5 m celdas\n{CELL}")
scene  = make_scene()
t0     = time.time()
static = precompute(scene, seed=42, cell_size=5.)
t_pre  = time.time() - t0

sh = static.spatial_hash
mean_s, max_s, frac = sh.coverage_stats()
print(f"  precompute  : {_fmt(t_pre)}")
print(f"  anchors     : {len(static.anchors)}")
print(f"  total_rays  : {static.n_pts_cpu.shape[0]:,}")
print(f"  hash_entries: {sh.total_entries:,}  cells={sh.n_cells:,}")
print(f"  per-cell    : mean={mean_s:.1f}  max={max_s}  nonempty={100*frac:.1f}%")
print(f"  noise_floor : {scene.noise_floor_dbm:.1f} dBm  "
      f"TX={scene.transmitters[0].tx_power_dbm:.1f} dBm")
assert len(static.anchors) > 0, "FAIL: no anchors — diagonal TX→RX debe ser visible"
assert sh.total_entries     > 0, "FAIL: hash vacío"
print("  → PASS")

# ── TEST 2: Trayectoria ───────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 2 — Trayectoria 5 frames (1 m/frame)\n{CELL}")
uav_start = scene.uav.position.copy()
vel       = scene.uav.velocity.copy()
noise_f   = float(scene.noise_floor_dbm)

all_dfs   = []
n_vis_prev = None
t_frames  = []

for step in range(5):
    pos = uav_start + vel * step
    scene.uav.position = pos.copy()
    scene.uav.velocity = vel.copy()

    t0 = time.time()
    vis, occ, bounces = apply_uav(static, scene.uav, scene)
    t_frames.append(time.time() - t0)

    # Geometría válida en todos los rayos
    for r in vis + occ + bounces:
        assert r.total_length() > 0, f"Frame {step}: longitud 0"
        assert abs(r.delay() - r.total_length()/C) < 1e-14, \
            f"Frame {step}: delay incorrecto"
        assert abs(float(np.linalg.norm(r.arrival_dir)) - 1.) < 1e-4, \
            f"Frame {step}: arrival_dir no unitario"

    # Bounces sobre noise floor
    for r in bounces:
        assert r.power_dbm > noise_f, \
            f"Frame {step}: bounce bajo noise floor ({r.power_dbm:.1f} < {noise_f:.1f})"

    # Anchors visibles no aumentan entre frames
    if n_vis_prev is not None:
        assert len(vis) <= n_vis_prev + 1, \
            f"Frame {step}: visible anchors aumentaron ({len(vis)} > {n_vis_prev})"
    n_vis_prev = len(vis)

    print(f"  frame {step+1}: pos={pos.round(1)}  "
          f"vis={len(vis)}  occ={len(occ)}  bounces={len(bounces)}  "
          f"t={_fmt(t_frames[-1])}")

    frame_rays = vis + occ + bounces
    if frame_rays:
        df = to_dataframe(frame_rays, instance_id=f"traj_{step}",
                          time_step=step, uav=scene.uav)
        all_dfs.append(df)

print(f"\n  tiempo medio por frame: {_fmt(sum(t_frames)/len(t_frames))}")
print("  → PASS (geometría + noise floor)")

# ── TEST 3: Doppler no trivial ────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 3 — Doppler con UAV en movimiento rápido\n{CELL}")
scene.uav.position = np.array([50., 50., 35.])
scene.uav.velocity = np.array([5., 3., 0.])
_, _, fast_bounces = apply_uav(static, scene.uav, scene)
if fast_bounces:
    dops = np.array([r.doppler_shift for r in fast_bounces])
    print(f"  n_bounces={len(fast_bounces)}  "
          f"f_D: min={dops.min():+.3f}  max={dops.max():+.3f}  std={dops.std():.3f} Hz")
    assert not np.all(dops == 0.), "FAIL: todos los Doppler son 0"
    print("  → PASS")
else:
    print("  SKIP — sin bounces en posición (50,50,35)")

# ── TEST 4: CSV consistente entre frames ──────────────────────────────────────
print(f"\n{CELL}\nTEST 4 — CSV consistente entre frames\n{CELL}")
if all_dfs:
    cols0 = set(all_dfs[0].columns)
    for i, df in enumerate(all_dfs[1:], 1):
        assert set(df.columns) == cols0, f"FAIL: columnas difieren en frame {i}"
    master = pd.concat(all_dfs, ignore_index=True)
    steps  = list(master["time_step"].unique())
    assert steps == sorted(steps), "FAIL: time_step no monotóno"
    print(f"  total_rows={len(master)}  frames={len(all_dfs)}  columnas={len(cols0)}")
    print("  → PASS")
else:
    print("  SKIP — sin datos CSV")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
