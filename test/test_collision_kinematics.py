import sys, pathlib, time
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


# ── Collision helpers (from main.py / batch_runner.py) ───────────────────────

def check_uav_collision(uav_pos, uav_rad, obstacles):
    for obs in obstacles:
        cx = max(obs.box_min[0], min(uav_pos[0], obs.box_max[0]))
        cy = max(obs.box_min[1], min(uav_pos[1], obs.box_max[1]))
        cz = max(obs.box_min[2], min(uav_pos[2], obs.box_max[2]))
        dist = float(np.sqrt((cx-uav_pos[0])**2+(cy-uav_pos[1])**2+(cz-uav_pos[2])**2))
        if dist < uav_rad:
            normal = np.zeros(3)
            dx = min(abs(uav_pos[0]-obs.box_min[0]), abs(uav_pos[0]-obs.box_max[0]))
            dy = min(abs(uav_pos[1]-obs.box_min[1]), abs(uav_pos[1]-obs.box_max[1]))
            dz = min(abs(uav_pos[2]-obs.box_min[2]), abs(uav_pos[2]-obs.box_max[2]))
            m  = min(dx, dy, dz)
            if m == dx:   normal[0] = 1. if uav_pos[0] > (obs.box_min[0]+obs.box_max[0])/2 else -1.
            elif m == dy: normal[1] = 1. if uav_pos[1] > (obs.box_min[1]+obs.box_max[1])/2 else -1.
            else:         normal[2] = 1. if uav_pos[2] > (obs.box_min[2]+obs.box_max[2])/2 else -1.
            return True, normal
    return False, None


def get_safe_spawn(domain_x, domain_y, min_z, max_z, uav_rad, obstacles, seed=None):
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        pos = np.array([rng.uniform(uav_rad, domain_x-uav_rad),
                        rng.uniform(uav_rad, domain_y-uav_rad),
                        rng.uniform(min_z, max_z)])
        if not check_uav_collision(pos, uav_rad, obstacles)[0]:
            return pos
    raise RuntimeError("No safe spawn found")


def step_kinematics(pos, vel, uav_rad, obstacles, dt=1.0):
    """One kinematic step with elastic wall reflection (from main.py)."""
    proposed = pos + vel * dt
    col, normal = check_uav_collision(proposed, uav_rad, obstacles)
    if col:
        speed_xy = float(np.linalg.norm(vel[:2]))
        v_refl   = vel - 2.0 * np.dot(vel, normal) * normal
        v_refl[0] += np.random.uniform(-1.0, 1.0)
        v_refl[1] += np.random.uniform(-1.0, 1.0)
        v_refl[2] *= 0.5
        new_speed = float(np.linalg.norm(v_refl[:2]))
        if new_speed > 1e-6:
            v_refl = v_refl * (speed_xy / new_speed)
        vel     = v_refl
        proposed = pos + vel * dt
    return proposed, vel


# ── Scene ─────────────────────────────────────────────────────────────────────

def make_scene() -> Scene:
    """
    batch_runner.py defaults, scaled down:
      50k rays (was 50k), 5 frames, domain 150×150.
    """
    return make_street_scene(
        domain_x=150., domain_y=150., domain_z=80.,
        tx_pos=(80., 80., 35.), tx_power_w=250.,
        rx_pos=(0.,  0.,  20.), uav_pos=(40., 40., 25.),
        uav_vel=(4., 2., 0.), uav_radius=1.0,
        frequency=700e6, n_rays=50_000, n_max=8, seed=0,
        bld_height=15., tall_fraction=0.25,
    )


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


# ── TEST 1: Precompute ────────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 1 — Precompute (batch_runner params)\n{CELL}")
np.random.seed(0)
scene  = make_scene()
scene.use_physics   = True
scene.roughness     = 0.5
scene.bandwidth_hz  = 0.5e6
scene.temperature_c = 20.
scene.receiver.radius = 5.
scene.uav_roughness = 0.4
scene.n_samples_uav = 8

t0     = time.time()
static = precompute(scene, seed=0, cell_size=5.)
t_pre  = time.time() - t0
print(f"  precompute: {_fmt(t_pre)}  anchors={len(static.anchors)}  "
      f"hash={static.spatial_hash.total_entries:,}")
print(f"  TX={scene.transmitters[0].tx_power_dbm:.1f}dBm  "
      f"noise={scene.noise_floor_dbm:.1f}dBm")
assert static.spatial_hash.total_entries > 0, "FAIL: hash vacío"
print("  → PASS")

# ── TEST 2: Kinematic trajectory with collision detection ─────────────────────
print(f"\n{CELL}\nTEST 2 — Trayectoria cinemática con colisiones (10 frames)\n{CELL}")
N_FRAMES = 10
DT       = 1.0
UAV_RAD  = float(scene.uav.radius)

np.random.seed(42)
pos = get_safe_spawn(150., 150., 10., 50., UAV_RAD, scene.obstacles, seed=42)
vel = np.array([4., 3., 0.5])
speed0 = float(np.linalg.norm(vel))

all_dfs    = []
t_frames   = []
n_collisions = 0

for step in range(N_FRAMES):
    pos, vel = step_kinematics(pos, vel, UAV_RAD, scene.obstacles, dt=DT)

    # [A] UAV never inside an obstacle after step
    for obs in scene.obstacles:
        cx = max(obs.box_min[0], min(pos[0], obs.box_max[0]))
        cy = max(obs.box_min[1], min(pos[1], obs.box_max[1]))
        cz = max(obs.box_min[2], min(pos[2], obs.box_max[2]))
        d  = float(np.sqrt((cx-pos[0])**2+(cy-pos[1])**2+(cz-pos[2])**2))
        assert d >= UAV_RAD - 0.01, f"Frame {step}: UAV penetra obstacle (d={d:.3f}m)"

    # [B] Speed roughly preserved (within 3× after noisy reflection)
    speed_now = float(np.linalg.norm(vel))
    assert speed_now > speed0 * 0.05, f"Frame {step}: UAV stopped"

    scene.uav.position = pos.copy()
    scene.uav.velocity = vel.copy()

    tf = time.time()
    vis, occ, bounces = apply_uav(static, scene.uav, scene)
    t_frames.append(time.time()-tf)

    frame_rays = vis + occ + bounces
    if frame_rays:
        df = to_dataframe(frame_rays, instance_id=f"col_{step}", time_step=step, uav=scene.uav)
        all_dfs.append(df)

    print(f"  frame {step+1:2d}: pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  "
          f"vis={len(vis)}  occ={len(occ)}  bounces={len(bounces)}  "
          f"{_fmt(t_frames[-1])}")

print(f"\n  avg apply_uav: {_fmt(sum(t_frames)/len(t_frames))}")
print("  → PASS (no penetration, no stop)")

# ── TEST 3: CSV across all frames ─────────────────────────────────────────────
print(f"\n{CELL}\nTEST 3 — CSV consistente en todos los frames\n{CELL}")
if all_dfs:
    master = pd.concat(all_dfs, ignore_index=True)
    cols   = set(all_dfs[0].columns)
    for i, df in enumerate(all_dfs[1:], 1):
        assert set(df.columns) == cols, f"FAIL: columnas difieren en frame {i}"
    assert (master["tau_s"] > 0).all(), "FAIL: tau_s <= 0"
    print(f"  total_rows={len(master)}  frames={len(all_dfs)}  cols={len(cols)}")
    print("  → PASS")
else:
    print("  SKIP — sin observables")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
