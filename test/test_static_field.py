import sys, pathlib, time, shutil
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_CACHE_DIR = _ROOT / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

from src.core.scene.domain          import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.scene.ray             import Ray
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav
from src.core.cache                 import get_or_compute, hash_scene
from src.outputs.observables             import to_dataframe


def make_scene(n_rays: int = 50_000) -> Scene:
    """
    Un obstáculo en (30-50, 20-40) — no bloquea la diagonal TX→RX.
    UAV en (55, 40, 45): eje especular exacto TX→UAV→RX,
    Doppler analítico esperado: −0.665 Hz con vel=(2,1,0).
    """
    box = Box(np.zeros(3), np.array([100., 100., 60.]))
    tx  = Transmitter(np.array([5., 5., 30.]),  700e6, tx_power_w=500., tx_id=0)
    rx  = Receiver(np.array([95., 95., 20.]),   radius=8.)
    uav = UAV(np.array([55., 40., 45.]),         np.array([2., 1., 0.]), radius=1.0)
    obs = [Obstacle(np.array([30., 20., 0.]),    np.array([50., 40., 25.]))]
    scene = Scene(box=box, transmitters=[tx], receiver=rx,
                  uav=uav, obstacles=obs, n_rays=n_rays, n_max=6)
    scene.use_physics   = True
    scene.roughness     = 0.3
    scene.bandwidth_hz  = 0.5e6
    scene.temperature_c = 20.
    scene.uav_roughness = 0.4
    scene.n_samples_uav = 24
    return scene


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"

CELL = "=" * 62

# ── TEST 1 ────────────────────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 1 — Precompute: anchors y spatial hash\n{CELL}")
scene  = make_scene()
t0     = time.time()
static = precompute(scene, seed=42, cell_size=10.)
t_pre  = time.time() - t0

sh = static.spatial_hash
mean_s, max_s, frac = sh.coverage_stats()
print(f"  precompute  : {_fmt(t_pre)}")
print(f"  anchors     : {len(static.anchors)}")
print(f"  total_rays  : {static.n_pts_cpu.shape[0]:,}")
print(f"  hash_entries: {sh.total_entries:,}  "
      f"({sh.total_entries // max(static.n_pts_cpu.shape[0],1):.0f} per ray)")
print(f"  cells       : {sh.n_cells:,}  ({sh.NX}×{sh.NY}×{sh.NZ})")
print(f"  per-cell    : mean={mean_s:.1f}  max={max_s}  nonempty={100*frac:.1f}%")
print(f"  noise_floor : {scene.noise_floor_dbm:.1f} dBm  "
      f"TX={scene.transmitters[0].tx_power_dbm:.1f} dBm")
assert len(static.anchors) > 0, "FAIL: no anchors"
assert sh.total_entries     > 0, "FAIL: hash vacío"
print("  → PASS")

# ── TEST 2 ────────────────────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 2 — apply_uav (posición del UAV)\n{CELL}")
cands = sh.query(scene.uav.position, float(scene.uav.radius))
print(f"  UAV pos    : {scene.uav.position}  (eje especular TX→UAV→RX)")
print(f"  UAV radius : {scene.uav.radius}m")
print(f"  candidates : {len(cands)}")

t0 = time.time()
vis, occ, bounces = apply_uav(static, scene.uav, scene)
t_app = time.time() - t0
print(f"  apply_uav  : {_fmt(t_app)}")
print(f"  vis={len(vis)}  occ={len(occ)}  UAV_bounces={len(bounces)}")
if bounces:
    dops = [r.doppler_shift for r in bounces]
    pwrs = [r.power_dbm    for r in bounces]
    print(f"  Doppler    : min={min(dops):+.3f}  max={max(dops):+.3f} Hz  "
          f"(analítico esperado ≈ −0.665 Hz)")
    print(f"  power      : min={min(pwrs):.1f}  max={max(pwrs):.1f} dBm")
assert len(vis) + len(occ) == len(static.anchors), "FAIL: anchors perdidos"
print("  → PASS")

# ── TEST 3 ────────────────────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 3 — Cache: MISS → SAVE → HIT\n{CELL}")
print("  Primera llamada (MISS)...")
t0 = time.time()
s1 = get_or_compute(scene, seed=42, cell_size=10., cache_dir=_CACHE_DIR, verbose=True)
t1 = time.time() - t0

print("  Segunda llamada (HIT)...")
t0 = time.time()
s2 = get_or_compute(make_scene(), seed=42, cell_size=10., cache_dir=_CACHE_DIR, verbose=True)
t2 = time.time() - t0

h = hash_scene(scene, seed=42, cell_size=10.)
npz = list((_CACHE_DIR / "precomputed_static_fields").glob("*.npz"))
jsn = list(_CACHE_DIR.glob("*.json"))
print(f"\n  hash       : {h}")
print(f"  1ª llamada : {_fmt(t1)}   2ª llamada: {_fmt(t2)}   speedup: {t1/max(t2,1e-6):.0f}×")
print(f"  anchors    : 1ª={len(s1.anchors)}  2ª={len(s2.anchors)}")
print(f"  Archivos en cache/:")
for f in npz: print(f"    {f.name}  ({f.stat().st_size//1024} KB)")
for f in jsn: print(f"    {f.name}")

assert len(s1.anchors) == len(s2.anchors), "FAIL: anchors difieren"
assert hash_scene(make_scene(), seed=42, cell_size=10.) == h, "FAIL: hash no determinístico"
print("  → PASS")

# ── TEST 4 ────────────────────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 4 — apply_uav: in-memory == cache\n{CELL}")
vis_c, occ_c, bou_c = apply_uav(s2, scene.uav, scene)
print(f"  in-memory : vis={len(vis)}  occ={len(occ)}  bounces={len(bounces)}")
print(f"  cache     : vis={len(vis_c)}  occ={len(occ_c)}  bounces={len(bou_c)}")
assert len(vis) == len(vis_c) and len(occ) == len(occ_c), "FAIL: resultados difieren"
print("  → PASS")

# ── TEST 5 ────────────────────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 5 — Observables CSV\n{CELL}")
all_rays = vis + occ + bounces
df = to_dataframe(all_rays, instance_id="t1", time_step=0, uav=scene.uav)
print(f"  filas={len(df)}  columnas={len(df.columns)}  UAV_bounces={df['is_uav_bounce'].sum()}")
print(f"  tau_s     : mean={df['tau_s'].mean():.3e}  min={df['tau_s'].min():.3e}")
print(f"  power_dbm : mean={df['power_dbm'].mean():.1f} dBm")
for col in ["tau_s", "power_dbm", "f_D", "is_uav_bounce", "visible"]:
    assert col in df.columns,      f"FAIL: columna '{col}' ausente"
assert (df["tau_s"] > 0).all(),    "FAIL: tau_s ≤ 0"
assert len(df) == len(all_rays),   "FAIL: filas != rayos"
print("  → PASS")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
