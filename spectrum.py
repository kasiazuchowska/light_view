import numpy as np
import colour

WL_MIN, WL_MAX = 380, 780


def wavelength_to_rgb(wl: float) -> str:
    if not (380 <= wl <= 780):
        return "rgb(0,0,0)"

    wls    = [380, 440, 490, 510, 580, 645, 780]
    r_vals = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    g_vals = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    b_vals = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    r = np.interp(wl, wls, r_vals)
    g = np.interp(wl, wls, g_vals)
    b = np.interp(wl, wls, b_vals)
    factor = np.interp(wl, [380, 420, 700, 780], [0.3, 1.0, 1.0, 0.3])

    r = int(255 * (r * factor) ** 0.8)
    g = int(255 * (g * factor) ** 0.8)
    b = int(255 * (b * factor) ** 0.8)
    return f"rgb({r},{g},{b})"


def _make_sd(wl_nm: np.ndarray, intensity: np.ndarray) -> colour.SpectralDistribution:
    return colour.SpectralDistribution(dict(zip(wl_nm.tolist(), intensity.tolist())))


def spectrum_to_xy(wl_nm: np.ndarray, intensity: np.ndarray) -> tuple[float, float]:
    XYZ = colour.sd_to_XYZ(_make_sd(wl_nm, intensity))
    return colour.XYZ_to_xy(XYZ / 100)


def spectrum_to_cri(wl_nm: np.ndarray, intensity: np.ndarray) -> float | None:
    try:
        return colour.colour_rendering_index(_make_sd(wl_nm, intensity))
    except Exception:
        return None


def spectrum_to_tm30(wl_nm: np.ndarray, intensity: np.ndarray):
    try:
        return colour.colour_fidelity_index(
            _make_sd(wl_nm, intensity), additional_data=True, method="ANSI/IES TM-30-18"
        )
    except Exception:
        return None


def gaussian_spectrum(peaks: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    wl = np.arange(WL_MIN, WL_MAX + 1, 5, dtype=float)
    intensity = np.zeros_like(wl)
    for p in peaks:
        sigma2 = p["fwhm"] ** 2 / (8 * np.log(2))
        peak = p["height"] * np.exp(-((wl - p["center"]) ** 2) / (2 * sigma2))
        intensity = np.maximum(intensity, peak)
    return (wl, intensity) if intensity.max() > 0 else None


def xy_to_uv(xy: np.ndarray) -> np.ndarray:
    """CIE xy → CIE 1960 uv (used for Robertson isotherms)."""
    x, y = xy
    d = -2 * x + 12 * y + 3
    return np.array([4 * x / d, 6 * y / d])


def uv_to_xy(uv: np.ndarray) -> np.ndarray:
    """CIE 1960 uv → CIE xy."""
    u, v = uv
    d = 2 * u - 8 * v + 4
    return np.array([3 * u / d, 2 * v / d])


def spectrum_to_duv(wl_nm: np.ndarray, intensity: np.ndarray) -> float | None:
    try:
        uv = xy_to_uv(np.array(spectrum_to_xy(wl_nm, intensity)))
        cct = float(np.clip(colour.temperature.uv_to_CCT_Robertson1968(uv)[0], 1667, 20000))
        uv_planck = xy_to_uv(colour.temperature.CCT_to_xy_Kang2002(np.array([cct]))[0])
        return float(np.linalg.norm(uv - uv_planck))
    except Exception:
        return None
