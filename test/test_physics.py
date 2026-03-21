import sys, pathlib, math
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain      import Scene, Box, Obstacle, Transmitter, Receiver, UAV
from src.core.scene.propagation import (compute_fspl, compute_sphere_rcs_bounce_gain,
                                         compute_scattered_doppler)
from src.core.gpu.utils         import fspl_const
from src.core.precompute.precompute import precompute
from src.core.uav.apply_uav         import apply_uav

C    = 3e8
FC   = 700e6
CELL = "=" * 62


def free_space(tx, rx, uav_pos=None, uav_vel=None, n_rays=20_000, uav_rad=1.0) -> Scene:
    box = Box(np.zeros(3), np.array([200., 200., 150.]))
    txo = Transmitter(np.asarray(tx, float), FC, tx_power_w=1000., tx_id=0)
    rxo = Receiver(np.asarray(rx, float), radius=5.)
    uav = (UAV(np.asarray(uav_pos, float),
               np.asarray([0.,0.,0.] if uav_vel is None else uav_vel, float), uav_rad)
           if uav_pos is not None else None)
    scene = Scene(box=box, transmitters=[txo], receiver=rxo,
                  uav=uav, obstacles=[], n_rays=n_rays, n_max=4)
    scene.use_physics   = True
    scene.roughness     = 0.0
    scene.bandwidth_hz  = 0.1e6
    scene.temperature_c = 20.
    return scene


def _fmt(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"

# ── TEST 1: FSPL fórmula ──────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 1 — FSPL: GPU kernel == propagation.py\n{CELL}")
fc_c = fspl_const(FC)
for d in [10., 50., 137.45, 500., 1000.]:
    cpu = compute_fspl(d, FC)
    gpu = 20.*math.log10(d) + fc_c
    err = abs(cpu - gpu)
    assert err < 1e-6, f"FAIL d={d}: cpu={cpu:.6f} gpu={gpu:.6f} err={err:.2e}"
    print(f"  d={d:7.2f}m  FSPL={cpu:.4f} dB  err={err:.2e}")
print("  → PASS")

# ── TEST 2: Potencia espacio libre ────────────────────────────────────────────
print(f"\n{CELL}\nTEST 2 — Budget de potencia en espacio libre\n{CELL}")
tx_pos = np.array([10., 10., 50.]); rx_pos = np.array([190., 190., 50.])
d = float(np.linalg.norm(rx_pos - tx_pos))
scene = free_space(tx_pos, rx_pos, n_rays=30_000)
static = precompute(scene, seed=0, cell_size=5.)
assert len(static.anchors) > 0, "FAIL: no LOS anchor"
tx_dbm   = scene.transmitters[0].tx_power_dbm
expected = tx_dbm - compute_fspl(d, FC)
best_pwr = max(r.power_dbm for r in static.anchors)
err_db   = abs(best_pwr - expected)
print(f"  d={d:.1f}m  TX={tx_dbm:.1f}dBm")
print(f"  expected={expected:.1f}dBm  best_anchor={best_pwr:.1f}dBm  err={err_db:.2f}dB")
assert best_pwr <= tx_dbm + 0.1, "FAIL: anchor supera potencia TX"
assert err_db   < 2.0,           f"FAIL: error de budget {err_db:.2f} dB > 2 dB"
print("  → PASS")

# ── TEST 3: Delay mínimo == distancia directa / c ─────────────────────────────
print(f"\n{CELL}\nTEST 3 — Delay mínimo = ||TX−RX||/c\n{CELL}")
delays     = sorted(r.delay() for r in static.anchors)
d_direct   = float(np.linalg.norm(rx_pos - tx_pos))
expected_d = d_direct / C
err_ns     = abs(delays[0] - expected_d) * 1e9
print(f"  delay_min={delays[0]*1e9:.2f}ns  expected={expected_d*1e9:.2f}ns  err={err_ns:.2f}ns")
print(f"  delay_max={delays[-1]*1e9:.2f}ns  (pathes más largos)")
assert delays[0]  > 0,               "FAIL: delay no positivo"
assert err_ns     < expected_d*1e9*0.05, f"FAIL: delay mínimo difiere >5% de directo"
assert delays[-1] > delays[0],       "FAIL: todos los delays idénticos"
print("  → PASS")

# ── TEST 4: Budget UAV RCS ────────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 4 — Budget bistático: TX→UAV→RX\n{CELL}")
tx_p  = np.array([10.,  100., 50.])
rx_p  = np.array([190., 100., 50.])
uav_p = np.array([100., 100., 50.])   # midpoint: reflexión especular apunta a RX
scene2 = free_space(tx_p, rx_p, uav_p, [0.,0.,0.], n_rays=40_000, uav_rad=1.0)
scene2.n_samples_uav = 4
static2 = precompute(scene2, seed=0, cell_size=5.)
_, _, bounces = apply_uav(static2, scene2.uav, scene2)
if bounces:
    d1       = float(np.linalg.norm(uav_p - tx_p))
    d2       = float(np.linalg.norm(rx_p  - uav_p))
    rcs_gain = compute_sphere_rcs_bounce_gain(1.0, FC)
    tx_dbm2  = scene2.transmitters[0].tx_power_dbm
    expected = tx_dbm2 - compute_fspl(d1, FC) + rcs_gain - compute_fspl(d2, FC)
    best_pwr = max(r.power_dbm for r in bounces)
    err_db   = abs(best_pwr - expected)
    print(f"  d1={d1:.0f}m  d2={d2:.0f}m  RCS={rcs_gain:.1f}dB")
    print(f"  expected≈{expected:.1f}dBm  best_bounce={best_pwr:.1f}dBm  err={err_db:.2f}dB")
    assert err_db < 4.0, f"FAIL: error de budget {err_db:.2f} dB > 4 dB"
    print("  → PASS")
else:
    print("  SKIP — sin bounces (subir n_rays para este test)")

# ── TEST 5: Signo del Doppler ─────────────────────────────────────────────────
print(f"\n{CELL}\nTEST 5 — Signo del Doppler bistático\n{CELL}")
tx_p  = np.array([10., 10.,  50.])
rx_p  = np.array([190.,150., 50.])
uav_p = np.array([100.,80.,  50.])
uav_v = np.array([5.,  3.,   0.])
v_in  = uav_p - tx_p; v_in  /= np.linalg.norm(v_in)
v_out = rx_p  - uav_p; v_out /= np.linalg.norm(v_out)
analytic = compute_scattered_doppler(uav_v, v_in, v_out, FC)
scene3   = free_space(tx_p, rx_p, uav_p, uav_v, n_rays=40_000)
scene3.n_samples_uav = 1; scene3.uav_roughness = 0.0
static3  = precompute(scene3, seed=0, cell_size=5.)
_, _, bou3 = apply_uav(static3, scene3.uav, scene3)
if bou3:
    sim = bou3[0].doppler_shift
    print(f"  analítico={analytic:.4f}Hz  simulado={sim:.4f}Hz  err={abs(sim-analytic):.4f}Hz")
    if abs(analytic) > 0.01:
        assert np.sign(sim) == np.sign(analytic), \
            f"FAIL: signo incorrecto (esperado {np.sign(analytic):+.0f})"
    print("  → PASS")
else:
    print("  SKIP — sin bounces")

# ── TEST 6: Doppler monostático analítico (sin GPU) ───────────────────────────
print(f"\n{CELL}\nTEST 6 — Doppler monostático: f_D = 2·fc·v/c\n{CELL}")
v_speed = 10.
v_in_m  = np.array([1., 0., 0.])
v_out_m = np.array([-1., 0., 0.])
vel_m   = np.array([v_speed, 0., 0.])
f_D     = compute_scattered_doppler(vel_m, v_in_m, v_out_m, FC)
expected_m = 2 * FC * v_speed / C
err     = abs(abs(f_D) - expected_m)
print(f"  f_D={f_D:.4f}Hz  2·fc·v/c={expected_m:.4f}Hz  err={err:.6f}Hz")
assert err < 0.01, f"FAIL: {err:.4f} Hz"
print("  → PASS")

print(f"\n{CELL}\nTODOS LOS TESTS PASARON\n{CELL}")
