# Light Spectrum Analyzer

A Streamlit app for visualizing light spectra and their color properties.

## Features

- **Draw mode** — freehand-draw a spectrum curve on a canvas
- **Peaks mode** — define Gaussian peaks by center wavelength, height, and FWHM
- Displays the spectrum as a bar chart and plots it on the CIE 1931 chromaticity diagram
- Calculates CIE xy coordinates and correlated color temperature (CCT)

## Installation

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install dependencies:

```bash
uv sync
```

## Running

```bash
uv run streamlit run main.py
```

## Training the generative model

The **Generate** mode uses a conditional VAE (`model.py`) trained on real light source SPDs.
The trained weights are committed as `model.pt`, so the app works without retraining.

- `data/real_light_source_spds.xlsx` — training data: *Real Light Source SPDs For Use in Research* (figshare, item 12947240)
- `data_loader.py` — reads the SPDs and CCTs from the Excel file, resamples to 380–780 nm @ 1 nm
- `train.py` — trains the model and saves `model.pt`

To retrain:

```bash
uv run python train.py
```
