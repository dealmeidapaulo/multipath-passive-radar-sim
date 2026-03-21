import sys, pathlib, time
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain          import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav
from src.outputs.observables        import to_dataframe

CELL = "=" * 62
C    = 3e8


def make_two_tx_scene(n_rays: int = 30_000) -> Scene:
    """
    Domain 120×60×60 m.
    TX-A at (5, 30, 30)   — left side, 700 MHz
    TX-B at (115, 30, 30) — right side, 700 MHz (same freq for simplicity)
    RX  at (60, 30, 15)   — centre
    UAV at (60, 30, 40)   — above RX, on specular axis from both TXs
    No obstacles — clean geometry to guarantee anchors from both TXs.
    """
    box = Box(np.zeros(3), np.array([120., 60., 60.]))
    tx_a = Transmitter(np.array([5.,  30., 30.]), 700e6, tx_power_w=500., tx_id=0)
    tx_b = Transmitter(np.array([115.,30., 30.]), 700e6, tx_power_w=500., tx_id=1)
    rx   = Receiver(np.array([60., 30., 15.]), radius=6.)
    uav  = UAV(np.array([60., 30., 40.]), np.array([1., 0., 0.]), radius=1.0)

    scene = Scene(box=box, transmitters=[tx_a, tx_b], receiver=rx,
                  uav=uav, obstacles=[], n_rays=n_rays, n_max=4)
    scene.use_physics   = True
    scene.roughness     = 0.0
    scene.bandwidth_hz  = 0.5e6
    scene.temperature_c = 20.
    scene.uav_roughness = 0.4
    scene.n_samples_uav = 8
    return scene


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


# ── TEST 1: Both TXs produce anchors ─────────────────────────────────────────
print(f"\n{CELL}\nTEST 1 — Dois TXs: ambos contribuyen anchors\n{CELL}")
scene  = make_two_tx_scene()
t0     = time.time()
static = precompute(scene, seed=42, cell_size=5.)
t_pre  = time.time() - t0

ids = [r.transmitter_id for r in static.anchors]
tx0_anchors = ids.count(0)
tx1_anchors = ids.count(1)

print(f"  precompute    : {_fmt(t_pre)}")
print(f"  total anchors : {len(static.anchors)}")
print(f"  from TX-A (id=0): {tx0_anchors}")
print(f"  from TX-B (id=1): {tx1_anchors}")
print(f"  total_rays    : {static.n_pts_cpu.shape[0]:,}  "
      f"(2 × {scene.n_rays:,} = {2*scene.n_rays:,} expected)")

assert len(static.anchors) > 0,  "FAIL: no anchors"
assert tx0_anchors > 0,          "FAIL: TX-A (id=0) no contribuyó anchors"
assert tx1_anchors > 0,          "FAIL: TX-B (id=1) no contribuyó anchors"
assert static.n_pts_cpu.shape[0] == 2 * scene.n_rays, \
    f"FAIL: expected {2*scene.n_rays} rays, got {static.n_pts_cpu.shape[0]}"
print("  → PASS")

# ── TEST 2: Symmetry — TX-A and TX-B contribute similar anchor counts ─────────
print(f"\n{CELL}\nTEST 2 — Simetría: TX-A ≈ TX-B en número de anchors\n{CELL}")
ratio = tx0_anchors / max(tx1_anchors, 1)
print(f"  TX-A/TX-B ratio: {ratio:.2f}  (esperado ≈ 1.0 por simetría)")
assert 0.3 < ratio < 3.0, \
    f"FAIL: ratio {ratio:.2f} muy asimétrico — geometría no simétrica"
print("  → PASS")

# ── TEST 3: apply_uav with two TXs ────────────────────────────────────────────
print(f"\n{CELL}\nTEST 3 — apply_uav con dos TXs\n{CELL}")
t0 = time.time()
vis, occ, bounces = apply_uav(static, scene.uav, scene)
t_app = time.time() - t0

print(f"  apply_uav  : {_fmt(t_app)}")
print(f"  vis={len(vis)}  occ={len(occ)}  bounces={len(bounces)}")

assert len(vis) + len(occ) == len(static.anchors), \
    "FAIL: anchors perdidos en la partición vis/occ"

if bounces:
    dops = [r.doppler_shift for r in bounces]
    print(f"  Doppler: min={min(dops):+.3f}  max={max(dops):+.3f} Hz")

print("  → PASS")

# ── TEST 4: tx_ids_cpu covers both TXs ───────────────────────────────────────
print(f"\n{CELL}\nTEST 4 — tx_ids_cpu registra ambos emisores\n{CELL}")
unique_tx_ids = set(static.tx_ids_cpu.tolist())
print(f"  tx_ids únicos en StaticField: {sorted(unique_tx_ids)}")
assert 0 in unique_tx_ids, "FAIL: TX id=0 no encontrado en tx_ids_cpu"
assert 1 in unique_tx_ids, "FAIL: TX id=1 no encontrado en tx_ids_cpu"
print("  → PASS")

# ── TEST 5: CSV — observables por TX ─────────────────────────────────────────
print(f"\n{CELL}\nTEST 5 — Observables CSV con dos TXs\n{CELL}")
all_rays = vis + occ + bounces
if all_rays:
    df = to_dataframe(all_rays, instance_id="multi_tx_test", time_step=0, uav=scene.uav)
    print(f"  filas={len(df)}  UAV_bounces={df['is_uav_bounce'].sum()}")
    assert "tau_s" in df.columns, "FAIL: columna tau_s ausente"
    assert (df["tau_s"] > 0).all(), "FAIL: tau_s <= 0"
    print("  → PASS")
else:
    print("  SKIP — sin rayos")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
