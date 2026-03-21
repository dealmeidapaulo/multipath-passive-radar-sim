import sys, pathlib, json, shutil, time
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain          import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav
from src.core.cache                 import get_or_compute, hash_scene, KERNEL_VERSION

CELL      = "=" * 62
CACHE_DIR = _ROOT / "cache"


def make_scene_a(n_rays=10_000):
    box = Box(np.zeros(3), np.array([100., 60., 50.]))
    tx  = Transmitter(np.array([5., 30., 25.]), 700e6, tx_power_w=300., tx_id=0)
    rx  = Receiver(np.array([95., 30., 15.]), radius=6.)
    uav = UAV(np.array([50., 30., 30.]), np.array([1., 0., 0.]), radius=1.0)
    scene = Scene(box=box, transmitters=[tx], receiver=rx,
                  uav=uav, obstacles=[], n_rays=n_rays, n_max=4)
    scene.use_physics=True; scene.roughness=0.0
    scene.bandwidth_hz=0.5e6; scene.temperature_c=20.
    return scene


def make_scene_b(n_rays=10_000):
    """Different geometry from scene_a."""
    box = Box(np.zeros(3), np.array([100., 60., 50.]))
    tx  = Transmitter(np.array([5., 30., 25.]), 700e6, tx_power_w=300., tx_id=0)
    rx  = Receiver(np.array([95., 30., 15.]), radius=6.)
    obs = [Obstacle(np.array([40., 10., 0.]), np.array([60., 50., 20.]))]  # added obstacle
    uav = UAV(np.array([50., 30., 30.]), np.array([1., 0., 0.]), radius=1.0)
    scene = Scene(box=box, transmitters=[tx], receiver=rx,
                  uav=uav, obstacles=obs, n_rays=n_rays, n_max=4)
    scene.use_physics=True; scene.roughness=0.0
    scene.bandwidth_hz=0.5e6; scene.temperature_c=20.
    return scene


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"


# ── TEST 1: Different scenes → different hashes ───────────────────────────────
print(f"\n{CELL}\nTEST 1 — Escenas distintas → hashes distintos\n{CELL}")
s_a = make_scene_a(); s_b = make_scene_b()
h_a = hash_scene(s_a, seed=42, cell_size=5.)
h_b = hash_scene(s_b, seed=42, cell_size=5.)
print(f"  hash_a = {h_a}")
print(f"  hash_b = {h_b}")
assert h_a != h_b, "FAIL: escenas distintas tienen el mismo hash"
print("  → PASS")

# ── TEST 2: Same scene + different seed → different hash ──────────────────────
print(f"\n{CELL}\nTEST 2 — Mismo escena, seeds distintos → hashes distintos\n{CELL}")
h_42 = hash_scene(s_a, seed=42, cell_size=5.)
h_99 = hash_scene(s_a, seed=99, cell_size=5.)
print(f"  hash(seed=42) = {h_42}")
print(f"  hash(seed=99) = {h_99}")
assert h_42 != h_99, "FAIL: seeds distintos producen el mismo hash"
print("  → PASS")

# ── TEST 3: Determinism ───────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 3 — Determinismo del hash\n{CELL}")
h1 = hash_scene(make_scene_a(), seed=42, cell_size=5.)
h2 = hash_scene(make_scene_a(), seed=42, cell_size=5.)
assert h1 == h2, f"FAIL: hash no determinístico ({h1} vs {h2})"
print(f"  hash estable: {h1}")
print("  → PASS")

# ── TEST 4: n_rays change → different hash ────────────────────────────────────
print(f"\n{CELL}\nTEST 4 — n_rays distinto → hash distinto\n{CELL}")
h10k = hash_scene(make_scene_a(n_rays=10_000), seed=42, cell_size=5.)
h20k = hash_scene(make_scene_a(n_rays=20_000), seed=42, cell_size=5.)
assert h10k != h20k, "FAIL: n_rays no afecta al hash"
print(f"  hash(10k) = {h10k}")
print(f"  hash(20k) = {h20k}")
print("  → PASS")

# ── TEST 5: Cache MISS → SAVE → HIT, apply_uav consistent ────────────────────
print(f"\n{CELL}\nTEST 5 — MISS→SAVE→HIT y apply_uav consistente\n{CELL}")
scene_c = make_scene_a(n_rays=15_000)

t0 = time.time()
st_live  = precompute(scene_c, seed=42, cell_size=5.)
t_live   = time.time() - t0

t0 = time.time()
st_cache = get_or_compute(scene_c, seed=42, cell_size=5.,
                          cache_dir=CACHE_DIR, verbose=True)
t_cache  = time.time() - t0

# Second call should be faster
t0 = time.time()
st_hit   = get_or_compute(make_scene_a(n_rays=15_000), seed=42, cell_size=5.,
                          cache_dir=CACHE_DIR, verbose=True)
t_hit    = time.time() - t0

print(f"\n  live precompute  : {_fmt(t_live)}")
print(f"  first cache call : {_fmt(t_cache)}")
print(f"  second (HIT)     : {_fmt(t_hit)}  speedup≈{t_live/max(t_hit,1e-6):.0f}×")
assert len(st_live.anchors) == len(st_hit.anchors), "FAIL: anchor count differs"

v_l, o_l, b_l = apply_uav(st_live,  scene_c.uav, scene_c)
v_h, o_h, b_h = apply_uav(st_hit,   scene_c.uav, scene_c)
assert len(v_l)==len(v_h) and len(o_l)==len(o_h), "FAIL: apply_uav result differs"
print("  → PASS")

# ── TEST 6: field_registry.json structure ─────────────────────────────────────
print(f"\n{CELL}\nTEST 6 — field_registry.json estructura\n{CELL}")
reg_path = CACHE_DIR / "field_registry.json"
assert reg_path.exists(), "FAIL: field_registry.json no existe"
entries  = json.loads(reg_path.read_text())
assert isinstance(entries, list) and len(entries) > 0, "FAIL: registry vacío"
for e in entries:
    for key in ["hash", "filename", "kernel_version", "precompute_time_s", "params_summary"]:
        assert key in e, f"FAIL: clave '{key}' ausente en entrada"
    assert e["kernel_version"] == KERNEL_VERSION, "FAIL: kernel_version mismatch"
    npz = CACHE_DIR / "precomputed_static_fields" / e["filename"]
    assert npz.exists(), f"FAIL: archivo {e['filename']} no existe en disco"
print(f"  {len(entries)} entradas en registry, todas válidas")
print("  → PASS")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
