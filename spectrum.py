import numpy as np
import colour


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


def spectrum_to_xy(wl_nm: np.ndarray, intensity: np.ndarray) -> tuple[float, float]:
    sd = colour.SpectralDistribution(dict(zip(wl_nm.tolist(), intensity.tolist())))
    XYZ = colour.sd_to_XYZ(sd)
    return colour.XYZ_to_xy(XYZ / 100)


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
