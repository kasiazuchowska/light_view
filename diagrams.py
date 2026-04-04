import numpy as np
import plotly.graph_objects as go
import colour

from spectrum import wavelength_to_rgb, xy_to_uv, uv_to_xy

# ---------------------------------------------------------------------------
# Precomputed CIE spectrum locus (module-level, computed once)
# ---------------------------------------------------------------------------
_CMFS      = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
_LOCUS_WL  = _CMFS.wavelengths          # 360–830 nm at 1 nm
_LOCUS_XYZ = _CMFS.values               # shape (471, 3)
_denom     = _LOCUS_XYZ.sum(axis=1, keepdims=True)
_LOCUS_XY  = _LOCUS_XYZ[:, :2] / np.where(_denom > 0, _denom, 1)


# ---------------------------------------------------------------------------
# CIE diagram helpers
# ---------------------------------------------------------------------------

def _add_spectrum_locus(fig: go.Figure) -> None:
    WL_MIN, WL_MAX = 380, 780
    for i in range(len(_LOCUS_WL) - 1):
        wl    = _LOCUS_WL[i]
        color = wavelength_to_rgb(wl) if WL_MIN <= wl <= WL_MAX else "#999999"
        fig.add_trace(go.Scatter(
            x=[_LOCUS_XY[i, 0], _LOCUS_XY[i + 1, 0]],
            y=[_LOCUS_XY[i, 1], _LOCUS_XY[i + 1, 1]],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ))
    # Closing line 780 nm → 360 nm
    fig.add_trace(go.Scatter(
        x=[_LOCUS_XY[-1, 0], _LOCUS_XY[0, 0]],
        y=[_LOCUS_XY[-1, 1], _LOCUS_XY[0, 1]],
        mode="lines",
        line=dict(color="#aaaaaa", width=1, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))
    for wl in [400, 450, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700]:
        idx = np.argmin(np.abs(_LOCUS_WL - wl))
        lx, ly = _LOCUS_XY[idx]
        fig.add_annotation(
            x=lx, y=ly, text=str(wl),
            showarrow=False, font=dict(size=9, color="#444444"),
            xshift=8, yshift=4,
        )


def _add_planckian_locus(fig: go.Figure) -> None:
    pl_temps = np.arange(1667, 20001, 50)
    pl_xy    = colour.temperature.CCT_to_xy_Kang2002(pl_temps)
    fig.add_trace(go.Scatter(
        x=pl_xy[:, 0], y=pl_xy[:, 1],
        mode="lines",
        line=dict(color="#333333", width=1.5),
        name="Planckian locus",
        hoverinfo="skip",
    ))


def _add_isothermal_ticks(fig: go.Figure) -> None:
    tick_len_uv = 0.015
    dT          = 100
    for T in [2000, 2700, 3000, 4000, 5000, 6500, 10000]:
        xy_T = colour.temperature.CCT_to_xy_Kang2002(np.array([T]))[0]
        fig.add_annotation(
            x=xy_T[0], y=xy_T[1], text=f"{T}K",
            showarrow=False, font=dict(size=8, color="#333333"),
            xshift=6, yshift=-10,
        )
        uv_lo    = xy_to_uv(colour.temperature.CCT_to_xy_Kang2002(np.array([max(1667, T - dT)]))[0])
        uv_hi    = xy_to_uv(colour.temperature.CCT_to_xy_Kang2002(np.array([min(20000, T + dT)]))[0])
        tangent  = uv_hi - uv_lo
        tangent /= np.linalg.norm(tangent)
        normal   = np.array([-tangent[1], tangent[0]])
        uv_T     = xy_to_uv(xy_T)
        p0       = uv_to_xy(uv_T - normal * tick_len_uv)
        p1       = uv_to_xy(uv_T + normal * tick_len_uv)
        fig.add_trace(go.Scatter(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]],
            mode="lines",
            line=dict(color="#333333", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))


def _add_chromaticity_point(fig: go.Figure, cx: float, cy: float) -> str:
    """Add the user's chromaticity point and its CCT connection. Returns the CCT label."""
    try:
        cct       = colour.temperature.xy_to_CCT_Hernandez1999(np.array([cx, cy]))
        cct_label = f"{cct:.0f} K"
        cct_xy    = colour.temperature.CCT_to_xy_Kang2002(np.array([float(np.clip(cct, 1667, 20000))]))[0]
        fig.add_trace(go.Scatter(
            x=[cx, cct_xy[0]], y=[cy, cct_xy[1]],
            mode="lines",
            line=dict(color="black", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[cct_xy[0]], y=[cct_xy[1]],
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
        marker=dict(size=8, color="white", line=dict(color="black", width=2)),
        name=f"Your spectrum (~{cct_label})",
        hovertemplate=f"x={cx:.4f}<br>y={cy:.4f}<br>CCT ≈ {cct_label}<extra></extra>",
    ))
    return cct_label


# ---------------------------------------------------------------------------
# Public figure builders
# ---------------------------------------------------------------------------

def make_cie_diagram(cx: float, cy: float) -> tuple[go.Figure, str]:
    fig = go.Figure()
    _add_spectrum_locus(fig)
    _add_planckian_locus(fig)
    _add_isothermal_ticks(fig)
    cct_label = _add_chromaticity_point(fig, cx, cy)
    fig.update_layout(
        xaxis=dict(title="x", range=[-0.05, 0.85], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="y", range=[-0.05, 0.90]),
        plot_bgcolor="white",
        margin=dict(t=20),
        showlegend=True,
        legend=dict(x=0.75, y=0.95),
    )
    return fig, cct_label


def make_spectrum_bar_chart(wl: np.ndarray, intensity: np.ndarray) -> go.Figure:
    colors = [wavelength_to_rgb(w) for w in wl]
    fig    = go.Figure(go.Bar(x=wl, y=intensity, marker_color=colors, marker_line_width=0))
    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title="Relative intensity",
        yaxis_range=[0, 1.1],
        bargap=0,
        plot_bgcolor="white",
        margin=dict(t=20),
    )
    return fig
