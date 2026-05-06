#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Complex Publication Figure Generator

Creates a clean 2x3 experimental figure for the paper:
  (a) Original spectral waterfall
  (b) MP-filtered backbone
  (c) SSA residual recovery
  (d) Z-complex amplitude |Z|
  (e) Fisher-inspired localization map
  (f) MSFR response map

Removed from the exploratory app:
  - GF(16) symbolic maps
  - Skorokhod / median squeezing
  - transition matrices
  - large dashboard GUI

Usage:
  python z_complex_publication_figure_font_controls.py --data-dir ../data/ --out z_complex_pipeline.png
  python z_complex_publication_figure_font_controls.py --data-dir ../data/ --cpu
"""
from __future__ import annotations

import os
import glob
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy.ndimage import uniform_filter as scipy_uniform_filter
except Exception:
    scipy_uniform_filter = None


# ============================================================
# FONT / FIGURE STYLE CONTROLS
# ============================================================
# Edit only this block when the text in the exported figure is too small/large.
FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 14
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
COLORBAR_TICK_FONT_SIZE = 10
PANEL_LABEL_FONT_SIZE = 14
SUPTITLE_FONT_SIZE = 16
SUPTITLE_WEIGHT = "bold"
PANEL_TITLE_PAD = 12

FIGSIZE = (18, 10)          # Wider/larger figure for readable 2x3 publication layout
DPI = 300
THEME = "default"          # use "default" for publication; "dark_background" for screen
USE_COLORBARS = True        # set False for cleaner journal figure if colorbars are not needed
HIDE_INDIVIDUAL_AXIS_LABELS = True  # use shared figure-level x/y labels instead

plt.style.use(THEME)
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "figure.titlesize": SUPTITLE_FONT_SIZE,
})


# ============================================================
# CONFIG
# ============================================================
DATA_DIRECTORY = "sim_data" #"../data/"
H5_SIGNAL_PATH = "entry/instrument/spectrometer/data/signal"
USE_CUPY = True

DEFAULT_SSA_WINDOW = 30
DEFAULT_MSFR_WIN_T = 9
DEFAULT_MSFR_WIN_K = 9
EPS = 1e-8

CMAP_RAW = "magma"
CMAP_DETAIL = "inferno"
CMAP_Z = "magma"
CMAP_FISHER = "viridis"
CMAP_MSFR = LinearSegmentedColormap.from_list(
    "msfr_publication",
    [
        "#0b1026",
        "#123c69",
        "#1ca3ec",
        "#19d3da",
        "#38ef7d",
        "#f9f871",
        "#fdae61",
        "#f46d43",
        "#d73027",
    ],
    N=1024,
)


# ============================================================
# BACKEND
# ============================================================
@dataclass
class Backend:
    xp: object
    name: str
    cp: Optional[object]
    gpu_uniform_filter: Optional[object] = None


def setup_backend(use_cupy: bool = True) -> Backend:
    if not use_cupy:
        return Backend(np, "numpy/cpu", None, None)

    try:
        import cupy as cp

        try:
            cp.cuda.Device(0).use()
            _ = cp.zeros((1,), dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()

            gpu_uniform = None
            try:
                from cupyx.scipy.ndimage import uniform_filter as cupy_uniform_filter
                gpu_uniform = cupy_uniform_filter
                print("CuPy/CUDA backend enabled; cupyx uniform_filter available.")
            except Exception as nd_error:
                print(f"CuPy enabled, but cupyx uniform_filter unavailable: {nd_error}")

            return Backend(cp, "cupy/cuda", cp, gpu_uniform)
        except Exception as cuda_error:
            print(f"CuPy CUDA allocation failed; falling back to CPU: {cuda_error}")
    except Exception as import_error:
        print(f"CuPy import unavailable; falling back to CPU: {import_error}")

    return Backend(np, "numpy/cpu", None, None)


BACKEND = setup_backend(USE_CUPY)
xp = BACKEND.xp
cp_module = BACKEND.cp


def to_cpu(a):
    if BACKEND.name == "cupy/cuda" and cp_module is not None:
        return cp_module.asnumpy(a)
    return np.asarray(a)


def sync_gpu():
    if BACKEND.name == "cupy/cuda" and cp_module is not None:
        cp_module.cuda.Stream.null.synchronize()


# ============================================================
# NUMERIC HELPERS
# ============================================================
def clean_numeric_np(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def percentile_clim(x: np.ndarray, p: Tuple[float, float] = (2, 98), eps: float = EPS) -> Tuple[float, float]:
    a = np.asarray(x, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0

    lo, hi = np.percentile(finite, p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + eps:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if hi <= lo + eps:
            hi = lo + 1.0
    return float(lo), float(hi)


def normalize_signed_np(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    a = clean_numeric_np(x)
    scale = float(np.nanpercentile(np.abs(a), 98)) + eps
    return np.clip(a / scale, -1.0, 1.0).astype(np.float32)


def normalize_xp(a, eps: float = EPS):
    arr = xp.asarray(a, dtype=xp.float32)
    arr = xp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = xp.nanmin(arr), xp.nanmax(arr)
    return ((arr - lo) / (hi - lo + eps)).astype(xp.float32)


def uniform_filter_xp(a, size):
    if BACKEND.name == "cupy/cuda" and BACKEND.gpu_uniform_filter is not None:
        return BACKEND.gpu_uniform_filter(a, size=size, mode="reflect")

    if scipy_uniform_filter is None:
        raise RuntimeError("scipy.ndimage.uniform_filter is required for local entropy.")

    out = scipy_uniform_filter(to_cpu(a).astype(np.float32), size=size, mode="reflect")
    if BACKEND.name == "cupy/cuda":
        return xp.asarray(out, dtype=xp.float32)
    return out


def odd_or_valid_window(L: int, T: int, minimum: int = 2) -> int:
    return max(minimum, min(int(round(L)), max(minimum, T - 1)))


# ============================================================
# DATA LOADING
# ============================================================
def load_h5_waterfall(data_dir: str, h5_path: str):
    files = sorted(glob.glob(os.path.join(os.path.abspath(data_dir), "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {os.path.abspath(data_dir)}")

    rows = []
    for fn in files:
        with h5py.File(fn, "r") as h5:
            if h5_path not in h5:
                raise KeyError(f"Dataset path '{h5_path}' not found in {fn}")
            rows.append(np.asarray(h5[h5_path][()], dtype=np.float32).reshape(-1))

    return np.asarray(rows, dtype=np.float32), files


# ============================================================
# MP FILTERING
# ============================================================
@dataclass
class MPResult:
    clean: np.ndarray
    residue: np.ndarray
    eigvals: np.ndarray
    lambda_plus: float
    lambda_minus: float
    kept_components: int
    mean: np.ndarray
    std: np.ndarray


def mp_denoise(data_matrix: np.ndarray, sigma2: float = 1.0, eps: float = EPS) -> MPResult:
    X0 = clean_numeric_np(data_matrix)
    T, K = X0.shape

    if T < 2 or K < 2:
        mean = X0.mean(axis=0, keepdims=True)
        std = np.maximum(X0.std(axis=0, keepdims=True), eps)
        return MPResult(X0.copy(), X0 * 0, np.zeros(max(1, K), np.float32), 0, 0, 0, mean, std)

    mean = X0.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(X0.std(axis=0, keepdims=True), eps).astype(np.float32)
    Xz_np = ((X0 - mean) / std).astype(np.float32)

    q = K / max(1, T)
    root_q = np.sqrt(q)
    lambda_minus = float(sigma2 * (1.0 - root_q) ** 2)
    lambda_plus = float(sigma2 * (1.0 + root_q) ** 2)

    Xz = xp.asarray(Xz_np, dtype=xp.float32)
    C = (Xz.T @ Xz) / max(1, T)
    eigvals, eigvecs = xp.linalg.eigh(C)

    keep = eigvals > lambda_plus
    kept = int(to_cpu(xp.count_nonzero(keep)))

    # For extremely noisy or small datasets, keep the dominant component as fallback.
    if kept == 0:
        keep = eigvals == xp.max(eigvals)
        kept = 1

    V = eigvecs[:, keep]
    Xclean_z = (Xz @ V) @ V.T
    sync_gpu()

    clean = clean_numeric_np(to_cpu(Xclean_z).astype(np.float32) * std + mean)
    residue = (X0 - clean).astype(np.float32)

    return MPResult(clean, residue, to_cpu(eigvals).astype(np.float32), lambda_plus, lambda_minus, kept, mean, std)


# ============================================================
# SSA RESIDUAL RECOVERY
# ============================================================
def diagonal_averaging_xp(Xelem, T: int, L: int):
    Kwin = T - L + 1
    y = xp.zeros(T, dtype=xp.float32)
    c = xp.zeros(T, dtype=xp.float32)

    for i in range(L):
        y[i : i + Kwin] += Xelem[i, :]
        c[i : i + Kwin] += 1.0

    return y / xp.maximum(c, 1.0)


def ssa_leading_component_1d(signal: np.ndarray, L: int) -> np.ndarray:
    x0 = clean_numeric_np(signal).reshape(-1)
    T = x0.size
    L = odd_or_valid_window(L, T, 2)
    Kwin = T - L + 1

    if Kwin < 2:
        return x0.copy()

    xdev = xp.asarray(x0, dtype=xp.float32)
    idx = xp.arange(L)[:, None] + xp.arange(Kwin)[None, :]
    Xtraj = xdev[idx]

    try:
        U, S, Vh = xp.linalg.svd(Xtraj, full_matrices=False)
        leading = S[0] * xp.outer(U[:, 0], Vh[0, :])
        y = diagonal_averaging_xp(leading, T, L)
        sync_gpu()
        return to_cpu(y).astype(np.float32)
    except Exception as e:
        print(f"SSA SVD failed for one column; returning zeros. Error: {e}")
        return np.zeros_like(x0, dtype=np.float32)


def ssa_residue_leading(residue: np.ndarray, L: int, progress: bool = True) -> np.ndarray:
    R = clean_numeric_np(residue)
    T, K = R.shape
    L = odd_or_valid_window(L, T, 2)
    out = np.zeros_like(R, dtype=np.float32)

    for k in range(K):
        out[:, k] = ssa_leading_component_1d(R[:, k], L)
        if progress and (k + 1) % max(1, K // 10) == 0:
            print(f"SSA progress: {k + 1}/{K} columns, L={L}")

    return clean_numeric_np(out)


# ============================================================
# Z-COMPLEX, FISHER-LIKE DENSITY, MSFR
# ============================================================
def compute_z_complex(mp_clean: np.ndarray, ssa_detail: np.ndarray):
    re = normalize_signed_np(mp_clean)
    im = normalize_signed_np(ssa_detail)
    Z = re.astype(np.complex64) + 1j * im.astype(np.complex64)
    Z_amp = np.abs(Z).astype(np.float32)
    return Z, Z_amp, re, im


def compute_fisher_like_density(field: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Local Fisher-inspired density: |grad field|^2 / (|field| + eps)."""
    a = clean_numeric_np(field)
    gt, gk = np.gradient(a.astype(np.float32))
    out = (gt * gt + gk * gk) / (np.abs(a) + eps)
    return clean_numeric_np(out)


def compute_msfr_xp(data_matrix: np.ndarray, win_t: int = DEFAULT_MSFR_WIN_T, win_k: int = DEFAULT_MSFR_WIN_K, eps: float = EPS) -> np.ndarray:
    """Modified Shannon-Fisher Ratio-like response.

    This implementation follows the practical entropy-gradient coupling used for
    the figure. It computes local entropy of the normalized intensity and local
    entropy of the gradient field, then combines them into a normalized response map.
    """
    win_t, win_k = max(2, int(win_t)), max(2, int(win_k))
    x = xp.asarray(clean_numeric_np(data_matrix), dtype=xp.float32)
    x_pos = xp.clip(normalize_xp(x, eps), eps, None)
    area = float(win_t * win_k)

    def local_entropy(field):
        f = xp.clip(normalize_xp(field, eps), eps, None)
        local_sum = uniform_filter_xp(f, size=(win_t, win_k)) * area + eps
        local_xlogx = uniform_filter_xp(f * xp.log(f + eps), size=(win_t, win_k)) * area
        H = xp.log(local_sum) - local_xlogx / local_sum
        return xp.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)

    H_i = local_entropy(x_pos)
    gt, gk = xp.gradient(x_pos)
    grad_field = 0.5 * (xp.abs(gt) + xp.abs(gk)) + eps
    H_g = local_entropy(grad_field)

    # Practical MSFR response. Use product for stable visualization.
    # For strict ratio visualization, replace with: H_i / (H_g + eps)
    out = normalize_xp(H_i * H_g, eps)
    sync_gpu()
    return to_cpu(out).astype(np.float32)


# ============================================================
# FIGURE GENERATION
# ============================================================
def add_panel_label(ax, label: str):
    ax.text(
        0.015,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2.5),
    )


def show_image(ax, data, title: str, cmap, label: str, p=(2, 98)):
    im = ax.imshow(data, aspect="auto", origin="upper", cmap=cmap)
    im.set_clim(*percentile_clim(data, p))
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=PANEL_TITLE_PAD)

    if HIDE_INDIVIDUAL_AXIS_LABELS:
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("Frequency bin", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Time / file index", fontsize=AXIS_LABEL_FONT_SIZE)

    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    add_panel_label(ax, label)
    return im


def create_publication_figure(raw, mp_clean, ssa_detail, z_amp, fisher_map, msfr_map, out_path: str, show: bool = True):
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, constrained_layout=True)

    panels = [
        (axes[0, 0], raw, "Original spectral waterfall", CMAP_RAW, "a"),
        (axes[0, 1], mp_clean, "MP-filtered backbone", CMAP_RAW, "b"),
        (axes[0, 2], ssa_detail, "SSA residual recovery", CMAP_DETAIL, "c"),
        (axes[1, 0], z_amp, "Z-complex amplitude |Z|", CMAP_Z, "d"),
        (axes[1, 1], fisher_map, "Fisher-inspired localization", CMAP_FISHER, "e"),
        (axes[1, 2], msfr_map, "MSFR response map", CMAP_MSFR, "f"),
    ]

    images = []
    for ax, data, title, cmap, label in panels:
        images.append(show_image(ax, data, title, cmap, label))

    if HIDE_INDIVIDUAL_AXIS_LABELS:
        fig.supxlabel("Frequency bin", fontsize=AXIS_LABEL_FONT_SIZE)
        fig.supylabel("Time / file index", fontsize=AXIS_LABEL_FONT_SIZE)

    fig.suptitle(
        "Sequential Z-Complex Reconstruction and Shannon-Fisher Localization Pipeline",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight=SUPTITLE_WEIGHT,
    )

    if USE_COLORBARS:
        for ax, im in zip(axes.ravel(), images):
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".pdf":
        pdf_path = base + ".pdf"
        fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved figure: {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Create a 2x3 Z-Complex/MSFR publication figure from spectral waterfall H5 files.")
    p.add_argument("--data-dir", default=DATA_DIRECTORY, help="Directory containing .h5 files")
    p.add_argument("--h5-path", default=H5_SIGNAL_PATH, help="Dataset path inside each .h5 file")
    p.add_argument("--ssa-window", type=int, default=DEFAULT_SSA_WINDOW, help="SSA embedding window length")
    p.add_argument("--msfr-win-t", type=int, default=DEFAULT_MSFR_WIN_T, help="MSFR local entropy window in time")
    p.add_argument("--msfr-win-k", type=int, default=DEFAULT_MSFR_WIN_K, help="MSFR local entropy window in frequency")
    p.add_argument("--out", default="z_complex_pipeline_figure.png", help="Output PNG/PDF path")
    p.add_argument("--cpu", action="store_true", help="Force CPU backend")
    p.add_argument("--no-show", action="store_true", help="Save without opening the figure window")
    return p.parse_args()


def main():
    global BACKEND, xp, cp_module
    args = parse_args()

    if args.cpu:
        BACKEND = setup_backend(False)
        xp = BACKEND.xp
        cp_module = BACKEND.cp

    print(f"Backend: {BACKEND.name}")
    raw, files = load_h5_waterfall(args.data_dir, args.h5_path)
    raw = clean_numeric_np(raw)
    print(f"Loaded waterfall: T={raw.shape[0]}, K={raw.shape[1]}, files={len(files)}")

    print("Computing MP filtering...")
    mp = mp_denoise(raw)
    print(f"MP: lambda-={mp.lambda_minus:.4f}, lambda+={mp.lambda_plus:.4f}, kept={mp.kept_components}/{raw.shape[1]}")

    print("Computing SSA residual recovery...")
    ssa_detail = ssa_residue_leading(mp.residue, args.ssa_window)

    print("Computing Z-complex amplitude...")
    _, z_amp, _, _ = compute_z_complex(mp.clean, ssa_detail)

    print("Computing Fisher-inspired localization map...")
    fisher_map = compute_fisher_like_density(z_amp)

    print("Computing MSFR response map on |Z|...")
    msfr_map = compute_msfr_xp(z_amp, args.msfr_win_t, args.msfr_win_k)

    create_publication_figure(
        raw=raw,
        mp_clean=mp.clean,
        ssa_detail=ssa_detail,
        z_amp=z_amp,
        fisher_map=fisher_map,
        msfr_map=msfr_map,
        out_path=args.out,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
