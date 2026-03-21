import sys, pathlib, os, math
import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain          import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav
from src.outputs.observables        import to_dataframe, extract

CELL = "=" * 62

EXPECTED_COLS = [
    "instance_id","time_step","uav_present",
    "uav_pos_x","uav_pos_y","uav_pos_z",
    "uav_vel_x","uav_vel_y","uav_vel_z",
    "tau_s","theta_rad","phi_rad","f_D",
    "is_uav_bounce","is_los","visible","power_dbm",
    "tx_pos_x","tx_pos_y","tx_pos_z",
    "rx_pos_x","rx_pos_y","rx_pos_z",
    "domain_x","domain_y","bld_height","tall_frac","seed",
    "roughness","temp_c","bw_hz","tx_power_w",
    "enable_dr","agc_active","dyn_range_db",
]

SIM_PARAMS = dict(
    domain_x=100., domain_y=60., bld_height=20., tall_frac=0.,
    seed=42, roughness=0.0, temp=20., bw=0.5e6, tx_power=500.,
    enable_dr=False, agc=False, dyn_range=50.,
    tx_pos_x=5., tx_pos_y=30., tx_pos_z=25.,
    rx_pos_x=95., rx_pos_y=30., rx_pos_z=15.,
)


def make_scene(n_rays=25_000):
    box = Box(np.zeros(3), np.array([100., 60., 50.]))
    tx  = Transmitter(np.array([5., 30., 25.]), 700e6, tx_power_w=500., tx_id=0)
    rx  = Receiver(np.array([95., 30., 15.]), radius=6.)
    uav = UAV(np.array([50., 30., 35.]), np.array([3., 1., 0.]), radius=1.0)
    scene = Scene(box=box, transmitters=[tx], receiver=rx,
                  uav=uav, obstacles=[], n_rays=n_rays, n_max=4)
    scene.use_physics=True; scene.roughness=0.0
    scene.bandwidth_hz=0.5e6; scene.temperature_c=20.
    scene.uav_roughness=0.4; scene.n_samples_uav=8
    return scene


# ── Setup ─────────────────────────────────────────────────────────────────────
scene  = make_scene()
static = precompute(scene, seed=42, cell_size=5.)
vis, occ, bounces = apply_uav(static, scene.uav, scene)
all_rays = vis + occ + bounces
print(f"Setup: anchors={len(static.anchors)}  "
      f"vis={len(vis)}  occ={len(occ)}  bounces={len(bounces)}")
assert len(all_rays) > 0, "Need rays to test observables"

df = to_dataframe(all_rays, instance_id="schema_test", time_step=0,
                  uav=scene.uav, params=SIM_PARAMS)

# ── TEST 1: All 35 columns, correct order ─────────────────────────────────────
print(f"\n{CELL}\nTEST 1 — 35 columnas en el orden correcto\n{CELL}")
actual = df.columns.tolist()
assert actual == EXPECTED_COLS, \
    f"FAIL: columnas incorrectas.\nEsperadas: {EXPECTED_COLS}\nObtenidas: {actual}"
print(f"  {len(actual)} columnas  → PASS")

# ── TEST 2: Physical ranges ───────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 2 — Rangos físicos\n{CELL}")
tx_pwr = scene.transmitters[0].tx_power_dbm
assert (df["tau_s"] > 0).all(),              "FAIL: tau_s <= 0"
assert (df["power_dbm"] <= tx_pwr + 0.5).all(), \
    f"FAIL: power_dbm supera TX ({tx_pwr:.1f}dBm)"
assert (df["theta_rad"].abs() <= math.pi/2 + 0.01).all(), "FAIL: theta_rad fuera de ±π/2"
assert (df["phi_rad"].abs() <= math.pi + 0.01).all(),     "FAIL: phi_rad fuera de ±π"
print(f"  tau_s range: [{df['tau_s'].min():.3e}, {df['tau_s'].max():.3e}] s")
print(f"  power range: [{df['power_dbm'].min():.1f}, {df['power_dbm'].max():.1f}] dBm")
print("  → PASS")

# ── TEST 3: Boolean flags ─────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 3 — Flags booleanos (0 o 1)\n{CELL}")
for col in ["is_uav_bounce", "is_los", "visible", "enable_dr", "agc_active"]:
    vals = set(df[col].unique())
    assert vals <= {0, 1, True, False}, f"FAIL: {col} tiene valores {vals}"
print(f"  is_uav_bounce: {df['is_uav_bounce'].sum()} True")
print(f"  is_los       : {df['is_los'].sum()} True")
print(f"  visible      : {df['visible'].sum()} True")
print("  → PASS")

# ── TEST 4: UAV metadata non-NaN ─────────────────────────────────────────────
print(f"\n{CELL}\nTEST 4 — UAV metadata no-NaN\n{CELL}")
for col in ["uav_pos_x","uav_pos_y","uav_pos_z","uav_vel_x","uav_vel_y","uav_vel_z"]:
    assert df[col].notna().all(), f"FAIL: {col} tiene NaN"
    assert (df[col] == df[col].iloc[0]).all(), f"FAIL: {col} no constante por frame"
print(f"  uav_pos=({df['uav_pos_x'].iloc[0]:.1f},{df['uav_pos_y'].iloc[0]:.1f},"
      f"{df['uav_pos_z'].iloc[0]:.1f})  → PASS")

# ── TEST 5: UAV bounces have non-zero Doppler ─────────────────────────────────
print(f"\n{CELL}\nTEST 5 — UAV-bounce rows tienen f_D ≠ 0\n{CELL}")
uav_rows = df[df["is_uav_bounce"]==True]
if len(uav_rows) > 0:
    assert (uav_rows["f_D"].abs() > 0).any(), \
        "FAIL: todos los f_D de UAV-bounce son 0"
    print(f"  {len(uav_rows)} UAV rows  f_D range: "
          f"[{uav_rows['f_D'].min():+.3f}, {uav_rows['f_D'].max():+.3f}] Hz")
    print("  → PASS")
else:
    print("  SKIP — sin UAV bounces en esta escena")

# ── TEST 6: LOS rows have n_bounces=0 ─────────────────────────────────────────
print(f"\n{CELL}\nTEST 6 — LOS rows corresponden a n_bounces=0\n{CELL}")
from src.core.scene.ray import Ray
los_rays    = [r for r in all_rays if r.n_bounces == 0]
los_in_df   = df[df["is_los"]==True]
assert len(los_in_df) == len(los_rays), \
    f"FAIL: {len(los_in_df)} LOS rows en CSV pero {len(los_rays)} LOS rayos"
print(f"  {len(los_rays)} LOS rayos  → PASS")

# ── TEST 7: CSV roundtrip ──────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 7 — Roundtrip CSV → reload\n{CELL}")
CSV_PATH = str(_ROOT / "cache" / "_schema_test_tmp.csv")
df.to_csv(CSV_PATH, index=False)
df2 = pd.read_csv(CSV_PATH)
assert list(df2.columns) == EXPECTED_COLS, "FAIL: columnas perdidas al recargar"
assert len(df2) == len(df), "FAIL: filas perdidas al recargar"
# tau_s preserved to 12 significant figures
max_err = (df2["tau_s"] - df["tau_s"]).abs().max()
assert max_err < 1e-20, f"FAIL: error tau_s en roundtrip: {max_err:.2e}"
os.remove(CSV_PATH)
print(f"  {len(df)} filas  error_tau={max_err:.2e}  → PASS")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
