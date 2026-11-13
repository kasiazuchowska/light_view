from pathlib import Path

import colour
import numpy as np
import streamlit as st
import torch

from canvas import make_background, read_spectrum_from_canvas, CANVAS_WIDTH, CANVAS_HEIGHT
from diagrams import make_cie_diagram, make_spectrum_bar_chart, make_color_vector_diagram, make_cri_ri_chart, make_rf_individual_chart
from model import ConditionalVAE, LATENT_DIM, MODEL_PATH
from spectrum import WL_MIN, WL_MAX, spectrum_to_xy, spectrum_to_cri, spectrum_to_cri_ri, spectrum_to_duv, spectrum_to_tm30, gaussian_spectrum
from streamlit_drawable_canvas import st_canvas


def spectrum_to_text(wl: np.ndarray, intensity: np.ndarray) -> str:
    wl_1nm = np.arange(WL_MIN, WL_MAX + 1, 1, dtype=float)
    intensity_1nm = np.interp(wl_1nm, wl, intensity)
    lines = ["wavelength_nm\tintensity"]
    for w, i in zip(wl_1nm, intensity_1nm):
        lines.append(f"{int(w)}\t{i:.6f}")
    return "\n".join(lines)


def render_results(wl, intensity):
    st.plotly_chart(make_spectrum_bar_chart(wl, intensity), width='stretch', height=400)
    st.download_button(
        label="Eksportuj widmo (TXT)",
        data=spectrum_to_text(wl, intensity),
        file_name="spectrum.txt",
        mime="text/plain",
    )
    cx, cy = spectrum_to_xy(wl, intensity)
    duv = spectrum_to_duv(wl, intensity)
    duv_label = f"{duv:.4f}" if duv is not None else "b.d."
    fig_cie, hernandez_label, mccamy_label = make_cie_diagram(cx, cy)

    st.subheader("Chromatyczność CIE 1931")
    st.write(
        f"**x** = {cx:.4f} &nbsp;&nbsp; **y** = {cy:.4f}"
        f"&nbsp;&nbsp;&nbsp; **CCT (Hernandez 1999)** ≈ {hernandez_label}"
        f"&nbsp;&nbsp; **CCT (McCamy 1992)** ≈ {mccamy_label}"
        f"&nbsp;&nbsp;&nbsp; **Duv** = {duv_label}"
    )
    st.plotly_chart(fig_cie, width='stretch', height=550)

    cri = spectrum_to_cri(wl, intensity)
    cri_ri = spectrum_to_cri_ri(wl, intensity)
    tm30 = spectrum_to_tm30(wl, intensity)
    cri_label = f"{cri:.0f}" if cri is not None else "b.d."
    rf_label  = f"{tm30.R_f:.0f}" if tm30 is not None else "b.d."
    rg_label  = f"{tm30.R_g:.0f}" if tm30 is not None else "b.d."
    fig_cvg = make_color_vector_diagram(tm30)
    st.subheader("Oddawanie barw")
    st.write(
        f"**CRI (Ra)** = {cri_label}"
        f"&nbsp;&nbsp;&nbsp; **Rf (TM-30)** = {rf_label}"
        f"&nbsp;&nbsp; **Rg (TM-30)** = {rg_label}"
    )
    if cri is not None and cri_ri is not None:
        st.markdown("**CRI – wartości szczegółowe R1–R14**")
        st.plotly_chart(make_cri_ri_chart(cri_ri, cri), width='stretch', height=350)
    if tm30 is not None and tm30.R_s is not None and len(tm30.R_s) > 0:
        st.markdown("**TM-30 – wierność szczegółowa Rf1–Rf99**")
        st.plotly_chart(make_rf_individual_chart(tm30.R_s), width='stretch', height=350)
    if fig_cvg is not None:
        st.plotly_chart(fig_cvg, width='stretch', height=500)


def draw_mode():
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    col1, col2 = st.columns([3, 1])
    with col1:
        stroke_width = st.slider("Rozmiar pędzla", 2, 20, 8)
    with col2:
        st.write("")
        if st.button("Wyczyść płótno", width='stretch'):
            st.session_state.canvas_key += 1

    st.markdown(
        "<style>.stCustomComponentV1 > iframe { border: 1px solid #cccccc !important; }</style>",
        unsafe_allow_html=True,
    )

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color="#222222",
        background_color="#ffffff",
        background_image=make_background(),
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        drawing_mode="freedraw",
        return_image_data=True,
        key=f"spectrum_canvas_{st.session_state.canvas_key}",
    )

    if canvas_result.image_data is not None:
        spectrum = read_spectrum_from_canvas(canvas_result.image_data)
        if spectrum:
            render_results(*spectrum)
        else:
            st.info("Narysuj coś na płótnie, aby zobaczyć widmo.")
    else:
        st.info("Narysuj coś na płótnie, aby zobaczyć widmo.")


def peaks_mode():
    if "peaks" not in st.session_state:
        st.session_state.peaks = []

    st.write("Zdefiniuj każdy pik przez długość fali środka, wysokość i FWHM (szerokość połówkową).")

    with st.form("add_peak", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            center = st.number_input("Środek (nm)", min_value=float(WL_MIN), max_value=float(WL_MAX),
                                     value=450.0, step=1.0)
        with c2:
            height = st.number_input("Wysokość (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        with c3:
            fwhm = st.number_input("FWHM (nm)", min_value=1.0, max_value=400.0, value=25.0, step=1.0)
        with c4:
            st.write("")
            submitted = st.form_submit_button("Dodaj pik", width='stretch')

    if submitted:
        st.session_state.peaks.append({"center": center, "height": height, "fwhm": fwhm})

    if st.session_state.peaks:
        st.write("**Piki:**")
        for i, p in enumerate(st.session_state.peaks):
            col_desc, col_del = st.columns([5, 1])
            with col_desc:
                st.write(f"Pik {i+1}: środek={p['center']:.0f} nm, wysokość={p['height']:.2f}, FWHM={p['fwhm']:.0f} nm")
            with col_del:
                if st.button("Usuń", key=f"del_{i}"):
                    st.session_state.peaks.pop(i)
                    st.rerun()

        if st.button("Usuń wszystkie piki"):
            st.session_state.peaks = []
            st.rerun()

        spectrum = gaussian_spectrum(st.session_state.peaks)
        if spectrum:
            render_results(*spectrum)
    else:
        st.info("Dodaj co najmniej jeden pik, aby zobaczyć widmo.")


def import_mode():
    uploaded = st.file_uploader("Wczytaj plik TXT z widmem", type=["txt"])
    if uploaded is None:
        st.info("Wczytaj plik TXT z dwiema kolumnami: wavelength_nm i intensity.")
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
            st.error("Nie znaleziono danych w pliku.")
            return

        wl_raw = np.array([r[0] for r in rows])
        intensity_raw = np.array([r[1] for r in rows])

        mask = (wl_raw >= WL_MIN) & (wl_raw <= WL_MAX)
        if not mask.any():
            st.error(f"Brak danych w zakresie {WL_MIN}–{WL_MAX} nm.")
            return
        wl_raw, intensity_raw = wl_raw[mask], intensity_raw[mask]
        order = np.argsort(wl_raw)
        wl_sorted = wl_raw[order]
        intensity_sorted = intensity_raw[order]

        wl = np.arange(WL_MIN, WL_MAX + 1, 1, dtype=float)
        intensity = np.interp(wl, wl_sorted, intensity_sorted)

        if intensity.max() <= 0:
            st.error("Wszystkie wartości natężenia są równe zero.")
            return

        render_results(wl, intensity)

    except Exception as e:
        st.error(f"Nie udało się odczytać pliku: {e}")


@st.cache_resource
def _load_model() -> ConditionalVAE | None:
    if not Path(MODEL_PATH).exists():
        return None
    model = ConditionalVAE()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def generate_mode():
    model = _load_model()
    if model is None:
        st.warning("Nie znaleziono wytrenowanego modelu. Najpierw uruchom `python train.py` w katalogu projektu.")
        return

    cct = st.number_input("Docelowa CCT (K)", min_value=100, max_value=100000, value=4000, step=100)
    if not (1630 <= cct <= 18273):
        st.warning("CCT poza zakresem treningowym (1630–18273 K). Wyniki mogą być niewiarygodne.")

    if st.button("Nowa próbka"):
        st.session_state.gen_z = ConditionalVAE.sample_z()

    if "gen_z" not in st.session_state:
        st.session_state.gen_z = ConditionalVAE.sample_z()

    temperature = st.slider(
        "Różnorodność", min_value=0.0, max_value=2.0, value=1.0, step=0.05,
        help="0 = zawsze to samo uśrednione widmo dla tej CCT. "
             "Wyżej = bardziej zróżnicowane / nietypowe kształty.",
    )

    wl = np.arange(WL_MIN, WL_MAX + 1, 1, dtype=float)
    z_scaled = st.session_state.gen_z * temperature
    intensity = model.decode_z(z_scaled, float(cct))
    intensity /= intensity.max()

    cx, cy = spectrum_to_xy(wl, intensity)
    try:
        actual_cct = colour.temperature.xy_to_CCT_Hernandez1999(np.array([cx, cy]))
        deviation = abs(actual_cct - cct) / cct
        if deviation > 0.2:
            st.warning(
                f"Wygenerowane widmo ma CCT ≈ {actual_cct:.0f} K (żądano {cct:.0f} K). "
                "Spróbuj zmniejszyć Różnorodność lub kliknij Nowa próbka."
            )
    except Exception:
        pass

    render_results(wl, intensity)


def main():
    st.title("Analiza widmowa źródeł światła")
    mode = st.radio("Tryb wprowadzania", ["Rysuj", "Piki", "Import", "Generuj"], horizontal=True)
    if mode == "Rysuj":
        draw_mode()
    elif mode == "Piki":
        peaks_mode()
    elif mode == "Import":
        import_mode()
    else:
        generate_mode()


main()
