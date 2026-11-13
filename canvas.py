import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spectrum import WL_MIN, WL_MAX

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
MARGIN_LEFT   = 45
MARGIN_RIGHT  = 20   # room for the "780" label
MARGIN_TOP    = 10   # room for the "1,00" label
MARGIN_BOTTOM = 30
PLOT_W        = 580
PLOT_H        = 260
CANVAS_WIDTH  = MARGIN_LEFT + PLOT_W + MARGIN_RIGHT
CANVAS_HEIGHT = MARGIN_TOP + PLOT_H + MARGIN_BOTTOM

# 380 nm sits on the Y axis and its label would collide with 400
X_TICKS = [400, 450, 500, 550, 600, 650, 700, 750, 780]
Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_background() -> Image.Image:
    img  = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for wl in X_TICKS:
        x = MARGIN_LEFT + int((wl - WL_MIN) / (WL_MAX - WL_MIN) * PLOT_W)
        draw.line([(x, MARGIN_TOP), (x, MARGIN_TOP + PLOT_H)], fill="#dddddd", width=1)
        draw.line([(x, MARGIN_TOP + PLOT_H), (x, MARGIN_TOP + PLOT_H + 5)], fill="#888888", width=1)
        label = str(wl)
        bbox  = draw.textbbox((0, 0), label, font=font)
        tw    = bbox[2] - bbox[0]
        draw.text((x - tw // 2, MARGIN_TOP + PLOT_H + 7), label, fill="#444444", font=font)

    for val in Y_TICKS:
        y = MARGIN_TOP + int((1.0 - val) * PLOT_H)
        draw.line([(MARGIN_LEFT, y), (MARGIN_LEFT + PLOT_W, y)], fill="#dddddd", width=1)
        draw.line([(MARGIN_LEFT - 5, y), (MARGIN_LEFT, y)], fill="#888888", width=1)
        label = f"{val:.2f}".replace(".", ",")
        bbox  = draw.textbbox((0, 0), label, font=font)
        tw    = bbox[2] - bbox[0]
        draw.text((MARGIN_LEFT - tw - 7, y - 6), label, fill="#444444", font=font)

    bottom = MARGIN_TOP + PLOT_H
    draw.line([(MARGIN_LEFT, MARGIN_TOP), (MARGIN_LEFT, bottom)], fill="#888888", width=1)
    draw.line([(MARGIN_LEFT, bottom), (MARGIN_LEFT + PLOT_W, bottom)], fill="#888888", width=1)
    return img


def read_spectrum_from_canvas(image_data: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (wl, intensity) arrays from canvas image data, or None if empty."""
    plot_alpha = image_data[MARGIN_TOP:MARGIN_TOP + PLOT_H, MARGIN_LEFT:MARGIN_LEFT + PLOT_W, 3]
    plot_w     = plot_alpha.shape[1]

    wavelengths_px = np.linspace(WL_MIN, WL_MAX, plot_w)
    intensity_px   = np.zeros(plot_w)
    for x in range(plot_w):
        drawn_rows = np.where(plot_alpha[:, x] > 0)[0]
        if len(drawn_rows) > 0:
            intensity_px[x] = 1.0 - drawn_rows.min() / PLOT_H

    wl        = np.arange(WL_MIN, WL_MAX + 1, 1)
    intensity = np.interp(wl, wavelengths_px, intensity_px)
    return (wl, intensity) if intensity.max() > 0 else None
