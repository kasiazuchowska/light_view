import streamlit as st
import numpy as np
import plotly.graph_objects as go

wavelengths = np.arange(380, 781, 5)

# Planck blackbody
def blackbody(wl_nm, T):
    h, c, k = 6.626e-34, 3e8, 1.381e-23
    wl = wl_nm * 1e-9
    return (2*h*c**2) / (wl**5 * (np.exp((h*c)/(wl*k*T)) - 1))

T = st.slider("Temperature (K)", 1000, 10000, 5778)
intensity = blackbody(wavelengths, T)
intensity /= intensity.max()

fig = go.Figure(go.Bar(x=wavelengths, y=intensity))
st.plotly_chart(fig)
