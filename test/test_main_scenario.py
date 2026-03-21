import sys, pathlib, time
import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.streets         import make_street_scene
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav
from src.core.cache                 import get_or_compute
from src.outputs.observables        import to_dataframe

CELL      = "=" * 62
N_RAYS    = 50_000
N_FRAMES  = 3
DYN_RANGE = 50.0     # dB — matches --dyn_range default in main.py


def make_scene():
    s = make_street_scene(
        domain_x=200., domain_y=200., domain_z=120.,
        tx_pos=(150., 150., 55.), tx_power_w=50.,
        rx_pos=(10.,  10.,  35.), uav_pos=(150., 153., 57.),
        uav_vel=(3., 1., 0.), uav_radius=1.0,
        frequency=700e6, n_rays=N_RAYS, n_max=8, seed=42,
        bld_height=30., tall_fraction=0.35,
    )
    s.use_physics=True; s.roughness=0.0; s.bandwidth_hz=20e6
    s.temperature_c=30.; s.receiver.radius=15.
    s.uav_roughness=0.4; s.n_samples_uav=8
    return s


def apply_dynamic_range(vis, occ, bounces, noise_floor, dyn_range):
    """
    Equivalent to main.py --enable_dr logic:
    Compute max_power over visible+bounces, then drop rays below that − dyn_range.
    Returns (vis_filtered, occ_filtered, bounces_filtered).
    """
    active = vis + bounces
    if not active:
        return vis, occ, bounces
    max_pwr  = max(r.power_dbm for r in active)
    dr_floor = max_pwr - dyn_range

    vis_f, occ_f, bou_f = [], list(occ), []
    for r in vis:
        if r.power_dbm >= dr_floor:
            vis_f.append(r)
        else:
            r.visible = False
            occ_f.append(r)
    for r in bounces:
        if r.power_dbm >= dr_floor:
            bou_f.append(r)
    return vis_f, occ_f, bou_f


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


# ── TEST 1: Precompute urban city ─────────────────────────────────────────────
print(f"\n{CELL}\nTEST 1 — Precompute escena main.py (50k rays, 200×200m)\n{CELL}")
scene  = make_scene()
t0     = time.time()
static = precompute(scene, seed=42, cell_size=5.)
t_pre  = time.time() - t0

print(f"  precompute : {_fmt(t_pre)}")
print(f"  anchors    : {len(static.anchors)}")
print(f"  buildings  : {len(scene.obstacles)}")
print(f"  TX={scene.transmitters[0].tx_power_dbm:.1f}dBm  "
      f"noise={scene.noise_floor_dbm:.1f}dBm  "
      f"budget={scene.transmitters[0].tx_power_dbm - scene.noise_floor_dbm:.0f}dB")
assert len(static.anchors) > 0, "FAIL: no anchors — urban geometry too dense?"
print("  → PASS")

# ── TEST 2: Per-frame loop with dynamic range filter ─────────────────────────
print(f"\n{CELL}\nTEST 2 — {N_FRAMES} frames con dynamic range filter ({DYN_RANGE}dB)\n{CELL}")
noise_f   = float(scene.noise_floor_dbm)
uav_start = scene.uav.position.copy()
vel       = scene.uav.velocity.copy()

master_dfs = []
for step in range(N_FRAMES):
    pos = uav_start + vel * step
    scene.uav.position = pos.copy(); scene.uav.velocity = vel.copy()

    vis, occ, bounces = apply_uav(static, scene.uav, scene)
    vis_f, occ_f, bou_f = apply_dynamic_range(vis, occ, bounces, noise_f, DYN_RANGE)

    frame_rays = vis_f + occ_f + bou_f
    df = to_dataframe(frame_rays, instance_id=f"main_{step}", time_step=step, uav=scene.uav)
    master_dfs.append(df)

    n_logged  = len(frame_rays)
    n_missing = sum(1 for r in frame_rays if not r.visible)
    n_echo    = len(bou_f)
    print(f"  frame {step+1}: ({n_logged} logged, {n_echo} UAV echos, {n_missing} missing)")

master = pd.concat(master_dfs, ignore_index=True)
print(f"\n  total rows: {len(master)}")

# All surviving rays are above (max_pwr - dyn_range)
vis_rows = master[master["visible"]==1]
if len(vis_rows) > 1:
    pwr_range = vis_rows["power_dbm"].max() - vis_rows["power_dbm"].min()
    assert pwr_range <= DYN_RANGE + 1.0, \
        f"FAIL: power range {pwr_range:.1f}dB > DYN_RANGE {DYN_RANGE}dB"
print("  → PASS")

# ── TEST 3: Cache round-trip ──────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 3 — Cache: precompute cacheado = live\n{CELL}")
CACHE_DIR = _ROOT / "cache"
s1 = get_or_compute(scene, seed=42, cell_size=5., cache_dir=CACHE_DIR, verbose=True)
assert len(s1.anchors) == len(static.anchors), \
    f"FAIL: cached anchors {len(s1.anchors)} ≠ live {len(static.anchors)}"
print("  → PASS")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
