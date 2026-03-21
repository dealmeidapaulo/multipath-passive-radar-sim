import numpy as np

def compute_fspl(distance_m: float, frequency_hz: float) -> float:
    """Free Space Path Loss in dB."""
    c = 3e8
    if distance_m < 1e-9:
        return 0.0
    return 20.0 * np.log10(distance_m) + 20.0 * np.log10(frequency_hz) + 20.0 * np.log10(4 * np.pi / c)

def sample_reflection_attenuation(base_factor: float = 0.1, std_dev: float = 0.03) -> float:
    """
    Samples an amplitude reduction factor from a normal distribution.
    Returns the equivalent power change in dB (will be a negative value).
    """
    val = np.random.normal(loc=base_factor, scale=std_dev)
    val = float(np.clip(val, 1e-6, 1.0))
    return 10.0 * np.log10(val)

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

def sample_scattering() -> float:
    raise NotImplementedError("Probabilistic scattering (cosine lobe) not yet implemented!")

def compute_bistatic_rcs(theta_in: float, phi_in: float, theta_out: float, phi_out: float) -> float:
    raise NotImplementedError("4D Bistatic RCS calculation is not yet implemented.")
