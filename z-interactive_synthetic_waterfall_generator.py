#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Synthetic Spectral Waterfall Series Generator
---------------------------------------------------------

Generates synthetic high-entropy spectral waterfall data for testing the
Z-Complex / MP / SSA / Fisher / MSFR reconstruction pipeline.

This version supports two selectable simulation modes:

1. Generic Waterfall
   - moderate stochastic background
   - narrow to medium carrier modes
   - controlled drift and wobble

2. Laser--Matter Interaction
   - initially high noise / plasma-like background
   - noise decays over time
   - narrow spectral modes strengthen over time
   - modes slightly broaden during evolution

Exports a SERIES of individual HDF5 files, one spectrum per file, compatible
with the original analysis script folder workflow:

    sim_data/synthetic_000000.h5
    sim_data/synthetic_000001.h5
    ...

Default HDF5 dataset path inside every file:

    entry/instrument/spectrometer/data/signal

Example:

    python interactive_synthetic_waterfall_series_generator_modes.py

Then analyze with:

    python z_complex_publication_figure.py --data-dir sim_data
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass

import h5py
import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons


# ============================================================
# CONFIG
# ============================================================
DEFAULT_H5_PATH = "entry/instrument/spectrometer/data/signal"
DEFAULT_OUT_DIR = "sim_data"
DEFAULT_OUT_PNG = "synthetic_waterfall_preview.png"

CMAP = "magma"
EPS = 1e-8

MODE_GENERIC = "Generic Waterfall"
MODE_LASER = "Laser Interaction"


@dataclass
class GeneratorState:
    time_steps: int = 360
    spectral_bins: int = 768
    noise_level: float = 0.10
    carriers: int = 4
    carrier_strength: float = 0.45
    drift: float = 0.025
    wobble: float = 0.004
    burst_strength: float = 0.08
    mode_width: float = 0.008
    seed: int = 12
    mode: str = MODE_GENERIC


# ============================================================
# SYNTHETIC MODEL HELPERS
# ============================================================
def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / max(float(sigma), EPS)) ** 2)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    lo = float(np.min(x))
    hi = float(np.max(x))
    return ((x - lo) / (hi - lo + EPS)).astype(np.float32)


# ============================================================
# GENERIC WATERFALL MODE
# ============================================================
def generate_generic_waterfall(state: GeneratorState) -> np.ndarray:
    T = int(max(16, state.time_steps))
    K = int(max(64, state.spectral_bins))
    carriers = int(max(0, state.carriers))

    rng = np.random.default_rng(int(state.seed))
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    f = np.linspace(0.0, 1.0, K, dtype=np.float32)

    W = state.noise_level * rng.standard_normal((T, K)).astype(np.float32)
    W += 0.35 * state.noise_level * rng.random((T, K)).astype(np.float32)

    background_envelope = 0.035 * (1.0 + np.sin(2 * np.pi * (0.35 * t + 0.1)))
    W += background_envelope[:, None] * (0.35 + 0.65 * rng.random((T, K)).astype(np.float32))

    # Dominant, mostly straight backbone mode.
    backbone_center = 0.50 + 0.012 * np.sin(2 * np.pi * (0.45 * t + 0.15))
    backbone_amp = 0.70 + 0.20 * np.sin(2 * np.pi * (0.35 * t + 0.3))
    for i in range(T):
        W[i] += backbone_amp[i] * gaussian(f, float(backbone_center[i]), state.mode_width * 1.7)

    if carriers > 0:
        base_centers = np.linspace(0.16, 0.84, carriers, dtype=np.float32)
        rng.shuffle(base_centers)

        for m in range(carriers):
            phase = rng.uniform(0, 2 * np.pi)
            local_drift = state.drift * rng.uniform(-1.0, 1.0)
            strength = state.carrier_strength * rng.uniform(0.35, 1.0)
            width = state.mode_width * rng.uniform(0.65, 1.20)

            center = base_centers[m] + local_drift * (t - 0.5)
            center += state.wobble * np.sin(
                2 * np.pi * (rng.uniform(0.3, 1.0) * t) + phase
            )
            center = np.clip(center, 0.03, 0.97)

            envelope_mu = rng.uniform(0.25, 0.75)
            envelope_sigma = rng.uniform(0.16, 0.32)
            envelope = np.exp(-0.5 * ((t - envelope_mu) / envelope_sigma) ** 2)

            for i in range(T):
                W[i] += strength * envelope[i] * gaussian(f, float(center[i]), width)

    if state.burst_strength > 0:
        burst_t_center = 0.63 + 0.03 * np.sin(0.3 * state.seed)
        burst_f_center = 0.70 + 0.03 * np.cos(0.2 * state.seed)
        burst_t = np.exp(-0.5 * ((t - burst_t_center) / 0.065) ** 2)
        burst_f = gaussian(f, burst_f_center, max(0.012, state.mode_width * 1.8))
        W += state.burst_strength * burst_t[:, None] * burst_f[None, :]

    ripple = 0.015 * np.sin(2 * np.pi * (6.0 * f[None, :] + 0.6 * t[:, None]))
    ripple *= np.exp(-0.5 * ((t[:, None] - 0.55) / 0.25) ** 2)
    W += ripple.astype(np.float32)

    return normalize01(W)


# ============================================================
# LASER--MATTER INTERACTION MODE
# ============================================================
def generate_laser_interaction_waterfall(state: GeneratorState) -> np.ndarray:
    """Simulate a laser--matter interaction spectral waterfall.

    Design:
    - high initial stochastic background, decreasing over time;
    - narrow spectral modes at early times;
    - modes strengthen as the background decays;
    - mode width slowly increases during evolution;
    - small drift/wobble only, giving mostly straight spectral traces.
    """
    T = int(max(16, state.time_steps))
    K = int(max(64, state.spectral_bins))
    carriers = int(max(1, state.carriers))

    rng = np.random.default_rng(int(state.seed))
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    f = np.linspace(0.0, 1.0, K, dtype=np.float32)

    # Noise is strongest at the beginning and is consumed/decays over time.
    noise_decay = 1.35 * np.exp(-3.0 * t) + 0.20
    W = (state.noise_level * noise_decay[:, None]) * rng.standard_normal((T, K)).astype(np.float32)
    W += (0.25 * state.noise_level * noise_decay[:, None]) * rng.random((T, K)).astype(np.float32)

    # Weak broadband plasma-like continuum that fades.
    continuum = 0.18 * state.noise_level * np.exp(-2.2 * t)
    spectral_tilt = 0.55 + 0.45 * np.exp(-2.5 * f)
    W += continuum[:, None] * spectral_tilt[None, :]

    # Signal grows as noise decays. This mimics emergence of structured emission.
    signal_growth = 1.0 - np.exp(-4.0 * t)
    late_gate = 0.55 + 0.45 / (1.0 + np.exp(-18.0 * (t - 0.35)))
    growth = signal_growth * late_gate

    # Dominant ablation/emission channel: narrow at first, broadens slightly.
    backbone_center = 0.52 + 0.006 * np.sin(2 * np.pi * (0.28 * t + 0.11))
    backbone_center += 0.45 * state.drift * (t - 0.5)
    backbone_center = np.clip(backbone_center, 0.05, 0.95)
    backbone_width = state.mode_width * (0.65 + 1.45 * t)
    backbone_amp = 0.35 + 1.10 * state.carrier_strength * growth

    for i in range(T):
        W[i] += backbone_amp[i] * gaussian(f, float(backbone_center[i]), float(backbone_width[i]))

    # Additional narrow emission lines. They grow and broaden mildly.
    base_centers = np.linspace(0.18, 0.84, carriers, dtype=np.float32)
    base_centers += rng.normal(0.0, 0.015, size=carriers).astype(np.float32)
    base_centers = np.clip(base_centers, 0.04, 0.96)

    for m in range(carriers):
        phase = rng.uniform(0, 2 * np.pi)
        local_slope = 0.45 * state.drift * rng.uniform(-1.0, 1.0)
        strength = state.carrier_strength * rng.uniform(0.25, 0.85)

        center = base_centers[m] + local_slope * (t - 0.5)
        center += state.wobble * np.sin(
            2 * np.pi * (rng.uniform(0.15, 0.55) * t) + phase
        )
        center = np.clip(center, 0.03, 0.97)

        # Starts narrow, broadens lightly with time.
        width_t = state.mode_width * rng.uniform(0.55, 0.95) * (0.75 + 1.10 * t)

        # Some lines emerge earlier/later.
        onset = rng.uniform(0.12, 0.45)
        line_gate = 1.0 / (1.0 + np.exp(-20.0 * (t - onset)))
        envelope = growth * line_gate

        for i in range(T):
            W[i] += strength * envelope[i] * gaussian(f, float(center[i]), float(width_t[i]))

    # Local emission burst, but cleaner and less chaotic than generic mode.
    if state.burst_strength > 0:
        burst_t_center = 0.38 + 0.10 * rng.random()
        burst_f_center = 0.60 + 0.10 * rng.uniform(-1.0, 1.0)
        burst_t = np.exp(-0.5 * ((t - burst_t_center) / 0.040) ** 2)
        burst_f = gaussian(f, burst_f_center, max(0.010, state.mode_width * 1.5))
        W += 0.65 * state.burst_strength * burst_t[:, None] * burst_f[None, :]

    # Faint instrumental ripple, also fading.
    ripple = 0.008 * np.sin(2 * np.pi * (5.5 * f[None, :] + 0.22 * t[:, None]))
    ripple *= np.exp(-1.8 * t[:, None])
    W += ripple.astype(np.float32)

    return normalize01(W)


# ============================================================
# MODE ROUTER
# ============================================================
def generate_synthetic_waterfall(state: GeneratorState) -> np.ndarray:
    if state.mode == MODE_LASER:
        return generate_laser_interaction_waterfall(state)
    return generate_generic_waterfall(state)


# ============================================================
# EXPORT
# ============================================================
def save_h5_series(out_dir: str, signal: np.ndarray, h5_path: str, state: GeneratorState) -> None:
    os.makedirs(out_dir, exist_ok=True)

    T, K = signal.shape

    for i in range(T):
        fn = os.path.join(out_dir, f"synthetic_{i:06d}.h5")
        with h5py.File(fn, "w") as h5:
            dset = h5.create_dataset(
                h5_path,
                data=np.asarray(signal[i], dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )
            dset.attrs["description"] = "Synthetic single-spectrum row"
            dset.attrs["generator"] = "Interactive Synthetic Spectral Waterfall Series Generator"
            dset.attrs["mode"] = state.mode
            dset.attrs["note"] = "Generated for algorithm testing; not experimental data."

            h5.attrs["spectrum_index"] = int(i)
            h5.attrs["total_spectra"] = int(T)
            h5.attrs["spectral_bins"] = int(K)
            h5.attrs["noise_level"] = float(state.noise_level)
            h5.attrs["carriers"] = int(state.carriers)
            h5.attrs["carrier_strength"] = float(state.carrier_strength)
            h5.attrs["drift"] = float(state.drift)
            h5.attrs["wobble"] = float(state.wobble)
            h5.attrs["burst_strength"] = float(state.burst_strength)
            h5.attrs["mode_width"] = float(state.mode_width)
            h5.attrs["seed"] = int(state.seed)
            h5.attrs["simulation_mode"] = state.mode
            h5.attrs["dataset_path"] = h5_path

        if (i + 1) % 50 == 0 or i == T - 1:
            print(f"Exported {i + 1}/{T}")

    print(f"Saved HDF5 series in: {os.path.abspath(out_dir)}")
    print(f"Files: {T}")
    print(f"Dataset path: {h5_path}")
    print(f"Each spectrum shape: ({K},)")


def save_h5_merged(path: str, signal: np.ndarray, h5_path: str, state: GeneratorState) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    with h5py.File(path, "w") as h5:
        dset = h5.create_dataset(
            h5_path,
            data=np.asarray(signal, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
            chunks=(1, signal.shape[1]),
        )
        dset.attrs["description"] = "Synthetic high-entropy spectral waterfall"
        dset.attrs["generator"] = "Interactive Synthetic Spectral Waterfall Series Generator"
        dset.attrs["mode"] = state.mode
        dset.attrs["note"] = "Generated for algorithm testing; not experimental data."

        h5.attrs["time_steps"] = int(state.time_steps)
        h5.attrs["spectral_bins"] = int(state.spectral_bins)
        h5.attrs["noise_level"] = float(state.noise_level)
        h5.attrs["carriers"] = int(state.carriers)
        h5.attrs["carrier_strength"] = float(state.carrier_strength)
        h5.attrs["drift"] = float(state.drift)
        h5.attrs["wobble"] = float(state.wobble)
        h5.attrs["burst_strength"] = float(state.burst_strength)
        h5.attrs["mode_width"] = float(state.mode_width)
        h5.attrs["seed"] = int(state.seed)
        h5.attrs["simulation_mode"] = state.mode
        h5.attrs["dataset_path"] = h5_path

    print(f"Saved merged HDF5: {os.path.abspath(path)}")
    print(f"Dataset path: {h5_path}")
    print(f"Shape: {signal.shape}")


def save_png(path: str, signal: np.ndarray, state: GeneratorState) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(signal, aspect="auto", origin="upper", cmap=CMAP, vmin=0, vmax=1)
    ax.set_title(f"Synthetic Spectral Waterfall - {state.mode}")
    ax.set_xlabel("Spectral sample / frequency bin")
    ax.set_ylabel("Spectrum index / time step")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG: {os.path.abspath(path)}")


# ============================================================
# GUI
# ============================================================
class SyntheticWaterfallGUI:
    def __init__(self, state: GeneratorState, h5_path: str):
        self.state = state
        self.h5_path = h5_path
        self.signal = generate_synthetic_waterfall(self.state)

        self.fig = plt.figure(figsize=(14.5, 8.2))
        try:
            self.fig.canvas.manager.set_window_title("Synthetic Spectral Waterfall Series Generator")
        except Exception:
            pass

        self.ax_img = self.fig.add_axes([0.06, 0.29, 0.62, 0.65])
        self.img = self.ax_img.imshow(self.signal, aspect="auto", origin="upper", cmap=CMAP, vmin=0, vmax=1)
        self.ax_img.set_title(f"Synthetic Spectral Waterfall - {self.state.mode}")
        self.ax_img.set_xlabel("Spectral sample / frequency bin")
        self.ax_img.set_ylabel("Spectrum index / time step")
        self.cbar = self.fig.colorbar(self.img, ax=self.ax_img, fraction=0.046, pad=0.02)

        x0, w = 0.74, 0.20
        y = 0.88
        dy = 0.057

        self.sliders = {}
        self.sliders["time_steps"] = self._add_slider(x0, y, w, "Spectra T", 32, 1000, self.state.time_steps, valstep=1)
        y -= dy
        self.sliders["spectral_bins"] = self._add_slider(x0, y, w, "Samples K", 64, 4096, self.state.spectral_bins, valstep=1)
        y -= dy
        self.sliders["noise_level"] = self._add_slider(x0, y, w, "Noise", 0.0, 0.8, self.state.noise_level)
        y -= dy
        self.sliders["carriers"] = self._add_slider(x0, y, w, "Carriers", 0, 12, self.state.carriers, valstep=1)
        y -= dy
        self.sliders["carrier_strength"] = self._add_slider(x0, y, w, "Carrier amp", 0.0, 1.5, self.state.carrier_strength)
        y -= dy
        self.sliders["drift"] = self._add_slider(x0, y, w, "Drift/slope", 0.0, 0.12, self.state.drift)
        y -= dy
        self.sliders["wobble"] = self._add_slider(x0, y, w, "Wobble", 0.0, 0.030, self.state.wobble)
        y -= dy
        self.sliders["burst_strength"] = self._add_slider(x0, y, w, "Burst", 0.0, 1.2, self.state.burst_strength)
        y -= dy
        self.sliders["mode_width"] = self._add_slider(x0, y, w, "Mode width", 0.003, 0.050, self.state.mode_width)
        y -= dy
        self.sliders["seed"] = self._add_slider(x0, y, w, "Seed", 0, 999, self.state.seed, valstep=1)

        for slider in self.sliders.values():
            slider.on_changed(self._on_slider_changed)

        # Mode selector.
        self.ax_mode = self.fig.add_axes([0.74, 0.015, 0.20, 0.055])
        self.mode_buttons = CheckButtons(
            self.ax_mode,
            [MODE_GENERIC, MODE_LASER],
            [self.state.mode == MODE_GENERIC, self.state.mode == MODE_LASER],
        )
        self.mode_buttons.on_clicked(self._on_mode_changed)

        # Output controls.
        self.ax_dir_box = self.fig.add_axes([0.06, 0.18, 0.36, 0.04])
        self.tb_dir = TextBox(self.ax_dir_box, "HDF5 dir", initial=DEFAULT_OUT_DIR)

        self.ax_png_box = self.fig.add_axes([0.06, 0.11, 0.36, 0.04])
        self.tb_png = TextBox(self.ax_png_box, "PNG out", initial=DEFAULT_OUT_PNG)

        self.ax_path_box = self.fig.add_axes([0.06, 0.04, 0.36, 0.04])
        self.tb_path = TextBox(self.ax_path_box, "H5 path", initial=self.h5_path)

        self.ax_regen = self.fig.add_axes([0.48, 0.18, 0.16, 0.045])
        self.btn_regen = Button(self.ax_regen, "Regenerate")
        self.btn_regen.on_clicked(self._regenerate)

        self.ax_export_series = self.fig.add_axes([0.48, 0.11, 0.16, 0.045])
        self.btn_export_series = Button(self.ax_export_series, "Export HDF5 Series")
        self.btn_export_series.on_clicked(self._export_h5_series)

        self.ax_export_png = self.fig.add_axes([0.48, 0.04, 0.16, 0.045])
        self.btn_export_png = Button(self.ax_export_png, "Export PNG")
        self.btn_export_png.on_clicked(self._export_png)

        self.ax_export_merged = self.fig.add_axes([0.66, 0.11, 0.12, 0.045])
        self.btn_export_merged = Button(self.ax_export_merged, "Merged H5")
        self.btn_export_merged.on_clicked(self._export_h5_merged)

        self.status = self.fig.text(0.74, 0.085, "Ready", fontsize=10)
        self._update_status()

    def _add_slider(self, x: float, y: float, w: float, label: str, lo: float, hi: float, val: float, valstep=None):
        ax = self.fig.add_axes([x, y, w, 0.028])
        return Slider(ax, label, lo, hi, valinit=val, valstep=valstep)

    def _read_state(self) -> None:
        self.state.time_steps = int(self.sliders["time_steps"].val)
        self.state.spectral_bins = int(self.sliders["spectral_bins"].val)
        self.state.noise_level = float(self.sliders["noise_level"].val)
        self.state.carriers = int(self.sliders["carriers"].val)
        self.state.carrier_strength = float(self.sliders["carrier_strength"].val)
        self.state.drift = float(self.sliders["drift"].val)
        self.state.wobble = float(self.sliders["wobble"].val)
        self.state.burst_strength = float(self.sliders["burst_strength"].val)
        self.state.mode_width = float(self.sliders["mode_width"].val)
        self.state.seed = int(self.sliders["seed"].val)

    def _update_status(self) -> None:
        self.status.set_text(
            f"Mode: {self.state.mode}\n"
            f"Shape: {self.signal.shape}\n"
            f"Noise={self.state.noise_level:.3f}, carriers={self.state.carriers}\n"
            f"Drift={self.state.drift:.3f}, wobble={self.state.wobble:.3f}\n"
            f"seed={self.state.seed}"
        )

    def _apply_signal_to_image(self) -> None:
        self.img.set_data(self.signal)
        self.img.set_clim(0, 1)
        self.ax_img.set_xlim(-0.5, self.signal.shape[1] - 0.5)
        self.ax_img.set_ylim(self.signal.shape[0] - 0.5, -0.5)
        self.ax_img.set_title(f"Synthetic Spectral Waterfall - {self.state.mode}")
        self._update_status()
        self.fig.canvas.draw_idle()

    def _regenerate(self, _event=None) -> None:
        self._read_state()
        self.signal = generate_synthetic_waterfall(self.state)
        self._apply_signal_to_image()

    def _on_slider_changed(self, _value) -> None:
        self._regenerate()

    def _on_mode_changed(self, label: str) -> None:
        # Make CheckButtons behave like radio buttons.
        statuses = self.mode_buttons.get_status()
        labels = [txt.get_text() for txt in self.mode_buttons.labels]
        clicked_index = labels.index(label)

        for i, active in enumerate(statuses):
            if i != clicked_index and active:
                self.mode_buttons.set_active(i)

        if label == MODE_LASER:
            self.state.mode = MODE_LASER
        else:
            self.state.mode = MODE_GENERIC

        # Keep current sliders, only regenerate model family.
        self._regenerate()

    def _export_h5_series(self, _event=None) -> None:
        self._read_state()
        h5_path = self.tb_path.text.strip() or DEFAULT_H5_PATH
        out_dir = self.tb_dir.text.strip() or DEFAULT_OUT_DIR
        save_h5_series(out_dir, self.signal, h5_path, self.state)
        self.status.set_text(f"Exported HDF5 series:\n{out_dir}")
        self.fig.canvas.draw_idle()

    def _export_h5_merged(self, _event=None) -> None:
        self._read_state()
        h5_path = self.tb_path.text.strip() or DEFAULT_H5_PATH
        out_dir = self.tb_dir.text.strip() or DEFAULT_OUT_DIR
        merged_path = os.path.join(out_dir, "synthetic_merged_waterfall.h5")
        save_h5_merged(merged_path, self.signal, h5_path, self.state)
        self.status.set_text(f"Exported merged HDF5:\n{merged_path}")
        self.fig.canvas.draw_idle()

    def _export_png(self, _event=None) -> None:
        out_path = self.tb_png.text.strip() or DEFAULT_OUT_PNG
        save_png(out_path, self.signal, self.state)
        self.status.set_text(f"Exported PNG:\n{out_path}")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive synthetic spectral waterfall series generator")
    parser.add_argument("--h5-path", default=DEFAULT_H5_PATH, help="Dataset path for exported HDF5 files")
    parser.add_argument("--T", type=int, default=360, help="Initial number of spectra / time steps")
    parser.add_argument("--K", type=int, default=768, help="Initial number of spectral samples / bins")
    parser.add_argument("--seed", type=int, default=12, help="Initial random seed")
    parser.add_argument(
        "--mode",
        choices=["generic", "laser"],
        default="generic",
        help="Initial simulation mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = MODE_LASER if args.mode == "laser" else MODE_GENERIC
    state = GeneratorState(time_steps=args.T, spectral_bins=args.K, seed=args.seed, mode=mode)
    gui = SyntheticWaterfallGUI(state, h5_path=args.h5_path)
    gui.show()


if __name__ == "__main__":
    main()
