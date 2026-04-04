import streamlit as st
import numpy as np
import plotly.graph_objects as go
import colour
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Canvas layout constants
# ---------------------------------------------------------------------------
MARGIN_LEFT = 45
MARGIN_BOTTOM = 30
PLOT_W = 580
PLOT_H = 260
CANVAS_WIDTH = MARGIN_LEFT + PLOT_W
CANVAS_HEIGHT = PLOT_H + MARGIN_BOTTOM
WL_MIN, WL_MAX = 380, 780

X_TICKS = [380, 400, 450, 500, 550, 600, 650, 700, 750, 780]
Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Spectrum locus from colour-science CMFs
# ---------------------------------------------------------------------------
_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
_LOCUS_WL = _CMFS.wavelengths                        # 360–830 nm at 1 nm
_LOCUS_XYZ = _CMFS.values                            # shape (471, 3)
_denom = _LOCUS_XYZ.sum(axis=1, keepdims=True)
_LOCUS_XY = _LOCUS_XYZ[:, :2] / np.where(_denom > 0, _denom, 1)


def spectrum_to_xy(wl_nm, intensity):
    sd = colour.SpectralDistribution(dict(zip(wl_nm.tolist(), intensity.tolist())))
    XYZ = colour.sd_to_XYZ(sd)
    return colour.XYZ_to_xy(XYZ / 100)


def make_cie_diagram(cx, cy):
    fig = go.Figure()

    # Spectrum locus — one colored segment per nm
    for i in range(len(_LOCUS_WL) - 1):
        wl = _LOCUS_WL[i]
        color = wavelength_to_rgb(wl) if WL_MIN <= wl <= WL_MAX else "#999999"
        fig.add_trace(go.Scatter(
            x=[_LOCUS_XY[i, 0], _LOCUS_XY[i + 1, 0]],
            y=[_LOCUS_XY[i, 1], _LOCUS_XY[i + 1, 1]],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Line closing the locus (780 nm → 360 nm)
    fig.add_trace(go.Scatter(
        x=[_LOCUS_XY[-1, 0], _LOCUS_XY[0, 0]],
        y=[_LOCUS_XY[-1, 1], _LOCUS_XY[0, 1]],
        mode="lines",
        line=dict(color="#aaaaaa", width=1, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Wavelength labels on the locus
    for wl in [400, 450, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700]:
        idx = np.argmin(np.abs(_LOCUS_WL - wl))
        lx, ly = _LOCUS_XY[idx]
        fig.add_annotation(
            x=lx, y=ly, text=str(wl),
            showarrow=False, font=dict(size=9, color="#444444"),
            xshift=8, yshift=4,
        )

    # Planckian locus (blackbody, 1667–20000 K)
    pl_temps = np.arange(1667, 20001, 50)
    pl_xy = colour.temperature.CCT_to_xy_Kang2002(pl_temps)
    fig.add_trace(go.Scatter(
        x=pl_xy[:, 0], y=pl_xy[:, 1],
        mode="lines",
        line=dict(color="#333333", width=1.5),
        name="Planckian locus",
        hoverinfo="skip",
    ))

    # Temperature labels on the Planckian locus
    for T in [1500, 2000, 2700, 3000, 4000, 5000, 6500, 10000]:
        if T < 1667:
            continue
        xy_T = colour.temperature.CCT_to_xy_Kang2002(np.array([T]))[0]
        fig.add_annotation(
            x=xy_T[0], y=xy_T[1], text=f"{T}K",
            showarrow=False, font=dict(size=8, color="#333333"),
            xshift=6, yshift=-10,
        )
        fig.add_trace(go.Scatter(
            x=[xy_T[0]], y=[xy_T[1]],
            mode="markers",
            marker=dict(size=4, color="#333333"),
            showlegend=False,
            hovertemplate=f"{T} K<extra></extra>",
        ))

    # The drawn spectrum point + CCT connection
    try:
        cct = colour.temperature.xy_to_CCT_Hernandez1999(np.array([cx, cy]))
        cct_label = f"{cct:.0f} K"
        # Closest point on the Planckian locus for that CCT
        cct_clamped = float(np.clip(cct, 1667, 20000))
        xy_cct = colour.temperature.CCT_to_xy_Kang2002(np.array([cct_clamped]))[0]
        # Dashed line from spectrum point to locus
        fig.add_trace(go.Scatter(
            x=[cx, xy_cct[0]], y=[cy, xy_cct[1]],
            mode="lines",
            line=dict(color="black", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Marker on the locus
        fig.add_trace(go.Scatter(
            x=[xy_cct[0]], y=[xy_cct[1]],
            mode="markers",
            marker=dict(size=8, color="black"),
            showlegend=False,
            hovertemplate=f"CCT = {cct_label}<extra></extra>",
        ))
    except Exception:
        cct_label = "n/a"

    fig.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode="markers",
        marker=dict(size=14, color="white", line=dict(color="black", width=2)),
        name=f"Your spectrum (~{cct_label})",
        hovertemplate=f"x={cx:.4f}<br>y={cy:.4f}<br>CCT ≈ {cct_label}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(title="x", range=[-0.05, 0.85], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="y", range=[-0.05, 0.90]),
        plot_bgcolor="white",
        margin=dict(t=20),
        showlegend=True,
        legend=dict(x=0.75, y=0.95),
    )
    return fig, cct_label


# ---------------------------------------------------------------------------
# Canvas background
# ---------------------------------------------------------------------------
def make_background():
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#f8f8f8")
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
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, PLOT_H + 7), label, fill="#444444", font=font)

    for val in Y_TICKS:
        y = int((1.0 - val) * PLOT_H)
        draw.line([(MARGIN_LEFT, y), (MARGIN_LEFT + PLOT_W, y)], fill="#dddddd", width=1)
        draw.line([(MARGIN_LEFT - 5, y), (MARGIN_LEFT, y)], fill="#888888", width=1)
        label = f"{val:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((MARGIN_LEFT - tw - 7, y - 6), label, fill="#444444", font=font)

    draw.line([(MARGIN_LEFT, 0), (MARGIN_LEFT, PLOT_H)], fill="#888888", width=1)
    draw.line([(MARGIN_LEFT, PLOT_H), (MARGIN_LEFT + PLOT_W, PLOT_H)], fill="#888888", width=1)
    return img


def wavelength_to_rgb(wl):
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title("Light Spectrum Drawer")

stroke_width = st.slider("Brush size", 2, 20, 8)

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=stroke_width,
    stroke_color="#222222",
    background_color="#f8f8f8",
    background_image=make_background(),
    width=CANVAS_WIDTH,
    height=CANVAS_HEIGHT,
    drawing_mode="freedraw",
    key="spectrum_canvas",
)

if canvas_result.image_data is not None:
    img_data = canvas_result.image_data
    plot_alpha = img_data[:PLOT_H, MARGIN_LEFT:, 3]
    plot_w = plot_alpha.shape[1]

    wavelengths_px = np.linspace(WL_MIN, WL_MAX, plot_w)
    intensity_px = np.zeros(plot_w)

    for x in range(plot_w):
        drawn_rows = np.where(plot_alpha[:, x] > 0)[0]
        if len(drawn_rows) > 0:
            top_y = drawn_rows.min()
            intensity_px[x] = 1.0 - top_y / PLOT_H

    wl = np.arange(WL_MIN, WL_MAX + 1, 5)
    intensity = np.interp(wl, wavelengths_px, intensity_px)

    if intensity.max() > 0:
        colors = [wavelength_to_rgb(w) for w in wl]
        fig_bar = go.Figure(
            go.Bar(x=wl, y=intensity, marker_color=colors, marker_line_width=0)
        )
        fig_bar.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Relative intensity",
            yaxis_range=[0, 1.1],
            bargap=0,
            plot_bgcolor="white",
            margin=dict(t=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        cx, cy = spectrum_to_xy(wl, intensity)
        st.subheader("CIE 1931 Chromaticity")
        fig_cie, cct_label = make_cie_diagram(cx, cy)
        st.write(f"**x** = {cx:.4f} &nbsp;&nbsp; **y** = {cy:.4f} &nbsp;&nbsp; **CCT** ≈ {cct_label}")
        st.plotly_chart(fig_cie, use_container_width=True)
    else:
        st.info("Draw something on the canvas to see the spectrum.")
else:
    st.info("Draw something on the canvas to see the spectrum.")
