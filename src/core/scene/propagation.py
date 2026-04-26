import numpy as np


def compute_sphere_rcs_bounce_gain(radius: float, frequency_hz: float) -> float:
    """
    Bistatic radar equation balancing.
    Since the tracer applies FSPL on BOTH legs (TX->Target and Target->RX), 
    the bounce must inject the RCS and correct the wavelength geometry: 
    Gain_bounce = RCS * 4pi / lambda^2
    """
    c = 3e8
    lam = c / frequency_hz
    rcs_m2 = np.pi * (radius ** 2)
    gain_linear = rcs_m2 * 4.0 * np.pi / (lam ** 2)
    return 10.0 * np.log10(gain_linear)

def compute_scattered_doppler(velocity: np.ndarray, v_in: np.ndarray, v_out: np.ndarray, freq: float) -> float:
    """
    Computes bistatic Doppler shift.
    v_in: Unit vector TO the target.
    v_out: Unit vector FROM the target.
    """
    c = 3e8
    # Notice the (v_out - v_in) correct formulation
    return (freq / c) * float(np.dot(velocity, v_out - v_in))