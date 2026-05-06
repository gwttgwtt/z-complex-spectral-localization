# Z-Complex Spectral Localization

This project provides a Python-based signal analysis pipeline for structural recovery and localization in high-entropy spectral waterfall data. The framework combines Marchenko–Pastur filtering, Singular Spectrum Analysis (SSA), Z-complex reconstruction, Fisher-inspired localization, and Modified Shannon–Fisher Ratio (MSFR) mapping.

## What the project does

The pipeline processes spectral waterfall data stored as HDF5 files and generates a compact 2×3 reconstruction figure containing:

1. the original spectral waterfall,
2. the MP-filtered spectral backbone,
3. the SSA-recovered residual structure,
4. the reconstructed Z-complex amplitude,
5. the Fisher-inspired localization map,
6. the MSFR response map.

The goal is to reveal weak coherent spectral structures that may remain hidden inside high-entropy or noisy spectral observations.

## Input file format

The project expects spectral data in HDF5 format (`.h5`). Each HDF5 file contains one spectral measurement stored at the dataset path:

```text
entry/instrument/spectrometer/data/signal
