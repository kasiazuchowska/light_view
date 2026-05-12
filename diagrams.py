import numpy as np
import plotly.graph_objects as go
import colour

from spectrum import wavelength_to_rgb, xy_to_uv, uv_to_xy, _make_sd

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


def _duv(xy: np.ndarray) -> float:
    uv = xy_to_uv(xy)
    cct = float(np.clip(colour.temperature.uv_to_CCT_Robertson1968(uv)[0], 1667, 20000))
    uv_planck = xy_to_uv(colour.temperature.CCT_to_xy_Kang2002(np.array([cct]))[0])
    return float(np.linalg.norm(uv - uv_planck))


def _add_chromaticity_point(fig: go.Figure, cx: float, cy: float) -> tuple[str, str]:
    """Add the user's chromaticity point and CCT connections. Returns (hernandez_label, mccamy_label)."""
    xy = np.array([cx, cy])
    hernandez_label = "n/a"
    mccamy_label = "n/a"

    near_locus = _duv(xy) <= 0.05

    if not near_locus:
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers",
            marker=dict(size=8, color="white", line=dict(color="black", width=2)),
            name="Your spectrum",
            hovertemplate=f"x={cx:.4f}<br>y={cy:.4f}<extra></extra>",
        ))
        return hernandez_label, mccamy_label

    try:
        cct_h = colour.temperature.xy_to_CCT_Hernandez1999(xy)
        hernandez_label = f"{cct_h:.0f} K"
        cct_xy = colour.temperature.CCT_to_xy_Kang2002(np.array([float(np.clip(cct_h, 1667, 20000))]))[0]
        fig.add_trace(go.Scatter(
            x=[cx, cct_xy[0]], y=[cy, cct_xy[1]],
            mode="lines",
            line=dict(color="#1f77b4", width=1, dash="dot"),
            name=f"Hernandez 1999 ({hernandez_label})",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[cct_xy[0]], y=[cct_xy[1]],
            mode="markers",
            marker=dict(size=8, color="#1f77b4"),
            showlegend=False,
            hovertemplate=f"Hernandez 1999<br>CCT = {hernandez_label}<extra></extra>",
        ))
    except Exception:
        pass

    try:
        cct_m = colour.temperature.xy_to_CCT_McCamy1992(xy)
        mccamy_label = f"{cct_m:.0f} K"
        cct_xy = colour.temperature.CCT_to_xy_Kang2002(np.array([float(np.clip(cct_m, 1667, 20000))]))[0]
        fig.add_trace(go.Scatter(
            x=[cx, cct_xy[0]], y=[cy, cct_xy[1]],
            mode="lines",
            line=dict(color="#d62728", width=1, dash="dash"),
            name=f"McCamy 1992 ({mccamy_label})",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[cct_xy[0]], y=[cct_xy[1]],
            mode="markers",
            marker=dict(size=8, color="#d62728"),
            showlegend=False,
            hovertemplate=f"McCamy 1992<br>CCT = {mccamy_label}<extra></extra>",
        ))
    except Exception:
        pass

    fig.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode="markers",
        marker=dict(size=8, color="white", line=dict(color="black", width=2)),
        name="Your spectrum",
        hovertemplate=f"x={cx:.4f}<br>y={cy:.4f}<extra></extra>",
    ))
    return hernandez_label, mccamy_label


# ---------------------------------------------------------------------------
# Public figure builders
# ---------------------------------------------------------------------------

def make_cie_diagram(cx: float, cy: float) -> tuple[go.Figure, str]:
    fig = go.Figure()
    _add_spectrum_locus(fig)
    _add_planckian_locus(fig)
    _add_isothermal_ticks(fig)
    hernandez_label, mccamy_label = _add_chromaticity_point(fig, cx, cy)
    fig.update_layout(
        xaxis=dict(title="x", range=[-0.05, 0.85], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="y", range=[-0.05, 0.90]),
        plot_bgcolor="white",
        margin=dict(t=20),
        showlegend=True,
        legend=dict(x=0.75, y=0.95),
    )
    return fig, hernandez_label, mccamy_label


def make_color_vector_diagram(tm30_spec) -> go.Figure | None:
    if tm30_spec is None:
        return None

    ref_avg  = np.array(tm30_spec.averages_reference)   # (16, 2)  U*, V*
    test_avg = np.array(tm30_spec.averages_test)         # (16, 2)

    ref_r = float(np.mean(np.sqrt(ref_avg[:, 0] ** 2 + ref_avg[:, 1] ** 2)))
    theta = np.linspace(0, 2 * np.pi, 200)

    hue_colors = [
        f"hsl({int(h)},80%,45%)"
        for h in np.linspace(0, 360, 16, endpoint=False)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ref_r * np.cos(theta), y=ref_r * np.sin(theta),
        mode="lines",
        line=dict(color="#aaaaaa", width=1, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    for i in range(16):
        u_ref,  v_ref  = ref_avg[i, 0],  ref_avg[i, 1]
        u_test, v_test = test_avg[i, 0], test_avg[i, 1]
        color = hue_colors[i]

        fig.add_annotation(
            x=u_test, y=v_test, ax=u_ref, ay=v_ref,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=2, arrowcolor=color,
        )
        fig.add_trace(go.Scatter(
            x=[u_ref], y=[v_ref],
            mode="markers+text",
            marker=dict(size=6, color=color, line=dict(color="white", width=1)),
            text=[str(i + 1)], textposition="top center",
            textfont=dict(size=8),
            showlegend=False,
            hovertemplate=f"Bin {i + 1}<br>ref ({u_ref:.2f}, {v_ref:.2f})<br>test ({u_test:.2f}, {v_test:.2f})<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(title="U*", scaleanchor="y", scaleratio=1, zeroline=True, zerolinewidth=1, zerolinecolor="#cccccc"),
        yaxis=dict(title="V*", zeroline=True, zerolinewidth=1, zerolinecolor="#cccccc"),
        plot_bgcolor="white",
        margin=dict(t=20),
    )
    return fig


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
