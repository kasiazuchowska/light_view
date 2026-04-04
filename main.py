import streamlit as st

from canvas import make_background, read_spectrum_from_canvas, CANVAS_WIDTH, CANVAS_HEIGHT
from diagrams import make_cie_diagram, make_spectrum_bar_chart
from spectrum import spectrum_to_xy
from streamlit_drawable_canvas import st_canvas


def render_results(wl, intensity):
    st.plotly_chart(make_spectrum_bar_chart(wl, intensity), use_container_width=True, height=400)

    cx, cy = spectrum_to_xy(wl, intensity)
    fig_cie, cct_label = make_cie_diagram(cx, cy)
    st.subheader("CIE 1931 Chromaticity")
    st.write(f"**x** = {cx:.4f} &nbsp;&nbsp; **y** = {cy:.4f} &nbsp;&nbsp; **CCT** ≈ {cct_label}")
    st.plotly_chart(fig_cie, use_container_width=True, height=550)


def main():
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
        spectrum = read_spectrum_from_canvas(canvas_result.image_data)
        if spectrum:
            render_results(*spectrum)
        else:
            st.info("Draw something on the canvas to see the spectrum.")
    else:
        st.info("Draw something on the canvas to see the spectrum.")


main()
