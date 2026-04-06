# Light Spectrum Analyzer

A Streamlit app for visualizing light spectra and their color properties.

## Features

- **Draw mode** — freehand-draw a spectrum curve on a canvas
- **Peaks mode** — define Gaussian peaks by center wavelength, height, and FWHM
- Displays the spectrum as a bar chart and plots it on the CIE 1931 chromaticity diagram
- Calculates CIE xy coordinates and correlated color temperature (CCT)

## Running

```bash
uv sync
uv run streamlit run main.py
```
