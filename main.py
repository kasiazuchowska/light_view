import numpy as np
import streamlit as st

from canvas import make_background, read_spectrum_from_canvas, CANVAS_WIDTH, CANVAS_HEIGHT
from diagrams import make_cie_diagram, make_spectrum_bar_chart
from spectrum import WL_MIN, WL_MAX, spectrum_to_xy, gaussian_spectrum
from streamlit_drawable_canvas import st_canvas


def spectrum_to_text(wl: np.ndarray, intensity: np.ndarray) -> str:
    wl_1nm = np.arange(WL_MIN, WL_MAX + 1, 1, dtype=float)
    intensity_1nm = np.interp(wl_1nm, wl, intensity)
    max_val = intensity_1nm.max()
    if max_val > 0:
        intensity_1nm /= max_val
    lines = ["wavelength_nm\tintensity"]
    for w, i in zip(wl_1nm, intensity_1nm):
        lines.append(f"{int(w)}\t{i:.6f}")
    return "\n".join(lines)


def render_results(wl, intensity):
    st.plotly_chart(make_spectrum_bar_chart(wl, intensity), width='stretch', height=400)
    st.download_button(
        label="Export spectrum (TXT)",
        data=spectrum_to_text(wl, intensity),
        file_name="spectrum.txt",
        mime="text/plain",
    )
    cx, cy = spectrum_to_xy(wl, intensity)
    fig_cie, cct_label = make_cie_diagram(cx, cy)
    st.subheader("CIE 1931 Chromaticity")
    st.write(f"**x** = {cx:.4f} &nbsp;&nbsp; **y** = {cy:.4f} &nbsp;&nbsp; **CCT** ≈ {cct_label}")
    st.plotly_chart(fig_cie, width='stretch', height=550)


def draw_mode():
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    col1, col2 = st.columns([3, 1])
    with col1:
        stroke_width = st.slider("Brush size", 2, 20, 8)
    with col2:
        st.write("")
        if st.button("Clear canvas", width='stretch'):
            st.session_state.canvas_key += 1

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color="#222222",
        background_color="#f8f8f8",
        background_image=make_background(),
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        drawing_mode="freedraw",
        key=f"spectrum_canvas_{st.session_state.canvas_key}",
    )

    if canvas_result.image_data is not None:
        spectrum = read_spectrum_from_canvas(canvas_result.image_data)
        if spectrum:
            render_results(*spectrum)
        else:
            st.info("Draw something on the canvas to see the spectrum.")
    else:
        st.info("Draw something on the canvas to see the spectrum.")


def peaks_mode():
    if "peaks" not in st.session_state:
        st.session_state.peaks = []

    st.write("Define each peak by its center wavelength, peak height, and FWHM (full width at half maximum).")

    with st.form("add_peak", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            center = st.number_input("Center (nm)", min_value=float(WL_MIN), max_value=float(WL_MAX),
                                     value=450.0, step=1.0)
        with c2:
            height = st.number_input("Height (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        with c3:
            fwhm = st.number_input("FWHM (nm)", min_value=1.0, max_value=400.0, value=25.0, step=1.0)
        with c4:
            st.write("")
            submitted = st.form_submit_button("Add peak", width='stretch')

    if submitted:
        st.session_state.peaks.append({"center": center, "height": height, "fwhm": fwhm})

    if st.session_state.peaks:
        st.write("**Peaks:**")
        for i, p in enumerate(st.session_state.peaks):
            col_desc, col_del = st.columns([5, 1])
            with col_desc:
                st.write(f"Peak {i+1}: center={p['center']:.0f} nm, height={p['height']:.2f}, FWHM={p['fwhm']:.0f} nm")
            with col_del:
                if st.button("Remove", key=f"del_{i}"):
                    st.session_state.peaks.pop(i)
                    st.rerun()

        if st.button("Clear all peaks"):
            st.session_state.peaks = []
            st.rerun()

        spectrum = gaussian_spectrum(st.session_state.peaks)
        if spectrum:
            render_results(*spectrum)
    else:
        st.info("Add at least one peak to see the spectrum.")


def import_mode():
    uploaded = st.file_uploader("Upload a spectrum TXT file", type=["txt"])
    if uploaded is None:
        st.info("Upload a TXT file with two columns: wavelength_nm and intensity.")
        return

    try:
        content = uploaded.read().decode("utf-8")
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("wavelength"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))

        if not rows:
            st.error("No data rows found in file.")
            return

        wl_raw = np.array([r[0] for r in rows])
        intensity_raw = np.array([r[1] for r in rows])

        wl_clipped = np.clip(wl_raw, WL_MIN, WL_MAX)
        order = np.argsort(wl_clipped)
        wl_sorted = wl_clipped[order]
        intensity_sorted = intensity_raw[order]

        wl = np.arange(WL_MIN, WL_MAX + 1, 1, dtype=float)
        intensity = np.interp(wl, wl_sorted, intensity_sorted)

        max_val = intensity.max()
        if max_val <= 0:
            st.error("All intensity values are zero.")
            return
        intensity /= max_val

        render_results(wl, intensity)

    except Exception as e:
        st.error(f"Could not parse file: {e}")


def main():
    st.title("Light Spectrum Analyzer")
    mode = st.radio("Input mode", ["Draw", "Peaks", "Import"], horizontal=True)
    if mode == "Draw":
        draw_mode()
    elif mode == "Peaks":
        peaks_mode()
    else:
        import_mode()


main()
