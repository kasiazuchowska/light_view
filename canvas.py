import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
MARGIN_LEFT   = 45
MARGIN_BOTTOM = 30
PLOT_W        = 580
PLOT_H        = 260
CANVAS_WIDTH  = MARGIN_LEFT + PLOT_W
CANVAS_HEIGHT = PLOT_H + MARGIN_BOTTOM
WL_MIN, WL_MAX = 380, 780

X_TICKS = [380, 400, 450, 500, 550, 600, 650, 700, 750, 780]
Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_background() -> Image.Image:
    img  = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#f8f8f8")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for wl in X_TICKS:
        x = MARGIN_LEFT + int((wl - WL_MIN) / (WL_MAX - WL_MIN) * PLOT_W)
        draw.line([(x, 0), (x, PLOT_H)], fill="#dddddd", width=1)
        draw.line([(x, PLOT_H), (x, PLOT_H + 5)], fill="#888888", width=1)
        label = str(wl)
        bbox  = draw.textbbox((0, 0), label, font=font)
        tw    = bbox[2] - bbox[0]
        draw.text((x - tw // 2, PLOT_H + 7), label, fill="#444444", font=font)

    for val in Y_TICKS:
        y = int((1.0 - val) * PLOT_H)
        draw.line([(MARGIN_LEFT, y), (MARGIN_LEFT + PLOT_W, y)], fill="#dddddd", width=1)
        draw.line([(MARGIN_LEFT - 5, y), (MARGIN_LEFT, y)], fill="#888888", width=1)
        label = f"{val:.2f}"
        bbox  = draw.textbbox((0, 0), label, font=font)
        tw    = bbox[2] - bbox[0]
        draw.text((MARGIN_LEFT - tw - 7, y - 6), label, fill="#444444", font=font)

    draw.line([(MARGIN_LEFT, 0), (MARGIN_LEFT, PLOT_H)], fill="#888888", width=1)
    draw.line([(MARGIN_LEFT, PLOT_H), (MARGIN_LEFT + PLOT_W, PLOT_H)], fill="#888888", width=1)
    return img


def read_spectrum_from_canvas(image_data: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (wl, intensity) arrays from canvas image data, or None if empty."""
    plot_alpha = image_data[:PLOT_H, MARGIN_LEFT:, 3]
    plot_w     = plot_alpha.shape[1]

    wavelengths_px = np.linspace(WL_MIN, WL_MAX, plot_w)
    intensity_px   = np.zeros(plot_w)
    for x in range(plot_w):
        drawn_rows = np.where(plot_alpha[:, x] > 0)[0]
        if len(drawn_rows) > 0:
            intensity_px[x] = 1.0 - drawn_rows.min() / PLOT_H

    wl        = np.arange(WL_MIN, WL_MAX + 1, 5)
    intensity = np.interp(wl, wavelengths_px, intensity_px)
    return (wl, intensity) if intensity.max() > 0 else None
