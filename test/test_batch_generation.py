import sys, pathlib, time, os
import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain          import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.scene.streets         import make_street_scene
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav
from src.outputs.observables        import to_dataframe

CELL = "=" * 62
N_TRAJ   = 3
N_FRAMES = 3
N_RAYS   = 30_000


def make_scene():
    s = make_street_scene(
        domain_x=150., domain_y=150., domain_z=80.,
        tx_pos=(80., 80., 35.), tx_power_w=250.,
        rx_pos=(0.,  0.,  20.), uav_pos=(40., 40., 25.),
        uav_vel=(3., 2., 0.), uav_radius=1.0,
        frequency=700e6, n_rays=N_RAYS, n_max=6, seed=0,
        bld_height=15., tall_fraction=0.25,
    )
    s.use_physics=True; s.roughness=0.5; s.bandwidth_hz=0.5e6
    s.temperature_c=20.; s.receiver.radius=5.
    s.uav_roughness=0.4; s.n_samples_uav=8
    return s


def check_col(pos, rad, obstacles):
    for obs in obstacles:
        cx=max(obs.box_min[0],min(pos[0],obs.box_max[0]))
        cy=max(obs.box_min[1],min(pos[1],obs.box_max[1]))
        cz=max(obs.box_min[2],min(pos[2],obs.box_max[2]))
        if float(np.sqrt((cx-pos[0])**2+(cy-pos[1])**2+(cz-pos[2])**2)) < rad:
            return True
    return False


def safe_spawn(domain_x, domain_y, uav_rad, obstacles, rng):
    for _ in range(500):
        pos = np.array([rng.uniform(uav_rad, domain_x-uav_rad),
                        rng.uniform(uav_rad, domain_y-uav_rad),
                        rng.uniform(10., 50.)])
        if not check_col(pos, uav_rad, obstacles):
            return pos
    raise RuntimeError("No safe spawn")


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


# ── Setup ─────────────────────────────────────────────────────────────────────
scene  = make_scene()
OUT_CSV = str(_ROOT / "cache" / "test_batch_out.csv")
if os.path.exists(OUT_CSV):
    os.remove(OUT_CSV)

print(f"\n{CELL}\nTEST — Batch generation ({N_TRAJ} traj × {N_FRAMES} frames)\n{CELL}")
print(f"  rays={N_RAYS:,}  TX={scene.transmitters[0].tx_power_dbm:.1f}dBm  "
      f"noise={scene.noise_floor_dbm:.1f}dBm")

t0     = time.time()
static = precompute(scene, seed=0, cell_size=5.)
t_pre  = time.time() - t0
print(f"  precompute: {_fmt(t_pre)}  anchors={len(static.anchors)}")

sim_params = dict(
    domain_x=150., domain_y=150., bld_height=15., tall_frac=0.25,
    seed=0, roughness=0.5, temp=20., bw=0.5e6, tx_power=250.,
    enable_dr=False, agc=False, dyn_range=50.,
    tx_pos_x=80., tx_pos_y=80., tx_pos_z=35.,
    rx_pos_x=0.,  rx_pos_y=0.,  rx_pos_z=20.,
)

total_rows    = 0
total_bounces = 0
t_start       = time.time()

for traj_idx in range(N_TRAJ):
    rng       = np.random.default_rng(traj_idx * 1000)
    pos       = safe_spawn(150., 150., float(scene.uav.radius), scene.obstacles, rng)
    vel       = np.array([rng.uniform(-4.,4.), rng.uniform(-4.,4.), rng.uniform(-0.5,0.5)])
    traj_dfs  = []

    for step in range(N_FRAMES):
        pos = pos + vel * 1.0   # simple kinematic, no collision for speed
        pos = np.clip(pos, [1.,1.,5.], [149.,149.,75.])
        scene.uav.position = pos.copy()
        scene.uav.velocity = vel.copy()

        vis, occ, bounces = apply_uav(static, scene.uav, scene)
        frame_rays = vis + occ + bounces
        total_bounces += len(bounces)

        df = to_dataframe(frame_rays,
                          instance_id=f"traj{traj_idx}",
                          time_step=step,
                          uav=scene.uav,
                          params=sim_params)
        traj_dfs.append(df)

    traj_df = pd.concat(traj_dfs, ignore_index=True)
    file_exists = os.path.isfile(OUT_CSV)
    traj_df.to_csv(OUT_CSV, mode='a', header=not file_exists, index=False)
    total_rows += len(traj_df)
    print(f"  traj {traj_idx+1}: spawn={pos.round(1)}  rows={len(traj_df)}")

t_total = time.time() - t_start
print(f"\n  total_rows={total_rows:,}  total_bounces={total_bounces}  "
      f"time={_fmt(t_total)}")

# Validate CSV
df_full = pd.read_csv(OUT_CSV)
assert len(df_full) == total_rows, \
    f"FAIL: CSV rows {len(df_full)} ≠ expected {total_rows}"

# No duplicate headers
assert df_full.columns.tolist() == list(df_full.columns), \
    "FAIL: duplicate headers in CSV"

# tau_s all positive
assert (df_full["tau_s"] > 0).all(), "FAIL: tau_s <= 0"

# Each instance_id has monotone time_step
for tid in df_full["instance_id"].unique():
    sub   = df_full[df_full["instance_id"]==tid]["time_step"].tolist()
    assert sub == sorted(sub), f"FAIL: time_step not monotone for {tid}"

os.remove(OUT_CSV)
print(f"\n  CSV validated and removed")
print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
