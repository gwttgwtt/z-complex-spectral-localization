#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Entropy Fisher Complex Explorer with Control Window and GUI Progress Bar

Windows:
  1) Main 2x5 figure:
     raw, Fisher(raw), MSFR(raw), |Z_EF|(raw), phase(Z_EF)(raw)
     MP,  Fisher(MP),  MSFR(MP),  |Z_EF|(MP),  phase(Z_EF)(MP)

  2) SSA 2x2 figure:
     SSA amplitude raw, SSA phase raw
     SSA amplitude MP,  SSA phase MP

  3) Control window:
     - Entropy mode RadioButtons
     - SSA backend RadioButtons
     - sliders for windows/parameters
     - buttons: Recompute, Save Figures, Close
     - status text + progress bar

Complex information field:
    Z_EF(t, f) = H_E(t, f) + i F(t, f)

where:
    H_E is selected entropy/storage map;
    F is Fisher structural density/reaction map.

Entropy modes:
    shannon
    kapur
    renyi
    tsallis
    permutation_rows
    variational_modes
    local_variance

Run:
    python multi_entropy_fisher_explorer_progress.py --data-dir ../data/
    python multi_entropy_fisher_explorer_progress.py --entropy-mode renyi --renyi-alpha 0.7
    python multi_entropy_fisher_explorer_progress.py --ssa-backend cuda --ssa-batch 128
    python multi_entropy_fisher_explorer_progress.py --no-ssa-window
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import os
import glob
import argparse
import itertools
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable

import h5py
import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import RadioButtons, Slider, Button, CheckButtons
from matplotlib.patches import Rectangle

try:
    from scipy.ndimage import uniform_filter as scipy_uniform_filter
except Exception:
    scipy_uniform_filter = None


# ============================================================
# STYLE
# ============================================================
FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 12
TITLE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 13
TICK_FONT_SIZE = 10
COLORBAR_TICK_FONT_SIZE = 8
PANEL_LABEL_FONT_SIZE = 12
SUPTITLE_FONT_SIZE = 16
SUPTITLE_WEIGHT = "bold"
PANEL_TITLE_PAD = 9

FIGSIZE_MAIN = (30, 10)
FIGSIZE_SSA = (15, 10)
FIGSIZE_CTRL = (7.8, 10.2)
DPI = 300
THEME = "default"
USE_COLORBARS = True
HIDE_INDIVIDUAL_AXIS_LABELS = True

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
DATA_DIRECTORY = None  # None => open folder chooser dialog
H5_SIGNAL_PATH = "entry/instrument/spectrometer/data/signal"
USE_CUPY = True

DEFAULT_MSFR_WIN_T = 9
DEFAULT_MSFR_WIN_K = 9
DEFAULT_SSA_WINDOW = 30
DEFAULT_SSA_BATCH = 128
DEFAULT_ENTROPY_MODE = "shannon"
DEFAULT_RENYI_ALPHA = 0.7
DEFAULT_TSALLIS_Q = 1.4
DEFAULT_PERM_ORDER = 5
DEFAULT_PERM_DELAY = 1
EPS = 1e-8

ENTROPY_MODES = [
    "shannon",
    "kapur",
    "renyi",
    "tsallis",
    "permutation_rows",
    "variational_modes",
    "local_variance",
]

ENTROPY_LABELS = [
    "Shannon",
    "Kapur",
    "Rényi",
    "Tsallis",
    "Permutation rows",
    "Variational modes",
    "Local variance",
]

LABEL_TO_MODE = dict(zip(ENTROPY_LABELS, ENTROPY_MODES))
MODE_TO_LABEL = dict(zip(ENTROPY_MODES, ENTROPY_LABELS))

CMAP_RAW = "magma"
CMAP_FISHER = "viridis"
CMAP_COMPLEX = "turbo"
CMAP_PHASE = "twilight"

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

ProgressCB = Optional[Callable[[float, str], None]]


# ============================================================
# HELPERS
# ============================================================
def to_cpu(a):
    if BACKEND.name == "cupy/cuda" and cp_module is not None:
        return cp_module.asnumpy(a)
    return np.asarray(a)


def sync_gpu():
    if BACKEND.name == "cupy/cuda" and cp_module is not None:
        cp_module.cuda.Stream.null.synchronize()


def clean_numeric_np(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def percentile_clim(x: np.ndarray, p: Tuple[float, float] = (2, 98), eps: float = EPS):
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


def normalize_xp(a, eps: float = EPS):
    arr = xp.asarray(a, dtype=xp.float32)
    arr = xp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = xp.nanmin(arr), xp.nanmax(arr)
    return ((arr - lo) / (hi - lo + eps)).astype(xp.float32)


def robust_unit_np(a: np.ndarray, percentile: float = 99.5, eps: float = EPS) -> np.ndarray:
    x = clean_numeric_np(a)
    scale = float(np.nanpercentile(x, percentile)) + eps
    x = x / scale
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def uniform_filter_xp(a, size):
    if BACKEND.name == "cupy/cuda" and BACKEND.gpu_uniform_filter is not None:
        return BACKEND.gpu_uniform_filter(a, size=size, mode="reflect")

    if scipy_uniform_filter is None:
        raise RuntimeError("scipy.ndimage.uniform_filter is required.")

    out = scipy_uniform_filter(to_cpu(a).astype(np.float32), size=size, mode="reflect")
    if BACKEND.name == "cupy/cuda":
        return xp.asarray(out, dtype=xp.float32)
    return out


def uniform_filter_np(a: np.ndarray, size):
    if scipy_uniform_filter is None:
        raise RuntimeError("scipy.ndimage.uniform_filter is required.")
    return scipy_uniform_filter(np.asarray(a, dtype=np.float32), size=size, mode="reflect")


def odd_or_valid_window(L: int, T: int, minimum: int = 2) -> int:
    L = int(round(L))
    return max(minimum, min(L, max(minimum, T - 1)))


def pretty_entropy_name(mode: str) -> str:
    return MODE_TO_LABEL.get(mode, mode).replace(" rows", " Rows").replace(" modes", " Modes")


def auto_export_paths(entropy_mode: str, ssa_window: int, mp_kept: int, prefix: str = "complex") -> Tuple[str, str]:
    tag = entropy_mode.lower().replace(" ", "_")
    main = f"{prefix}_pipeline_{tag}_ssa{ssa_window}_mp{mp_kept}.png"
    ssa = f"{prefix}_ssa_{tag}_ssa{ssa_window}_mp{mp_kept}.png"
    return main, ssa


def progress_noop(frac: float, msg: str):
    pass


# ============================================================
# DATA
# ============================================================
H5_EXTENSIONS = ("*.h5", "*.hdf5")


def select_data_folder() -> str:
    """Open a standard folder chooser dialog and return selected folder."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    folder = filedialog.askdirectory(
        title="Select folder containing .h5 / .hdf5 files"
    )

    root.destroy()

    if not folder:
        raise RuntimeError("No data folder selected.")

    return os.path.abspath(folder)


def scan_h5_files(data_dir: str):
    """Find .h5 and .hdf5 files in selected directory."""
    files = []
    for ext in H5_EXTENSIONS:
        files.extend(glob.glob(os.path.join(os.path.abspath(data_dir), ext)))
    return sorted(files)


def inspect_h5_datasets(fn: str):
    """Return dataset paths with shape and dtype for diagnostics."""
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj.shape, obj.dtype))

    with h5py.File(fn, "r") as h5:
        h5.visititems(visitor)

    return datasets


def find_first_numeric_dataset(h5: h5py.File) -> Optional[str]:
    """Find first non-empty numeric or complex dataset in an HDF5 file."""
    candidates = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        if obj.shape is None or len(obj.shape) == 0:
            return

        if int(np.prod(obj.shape)) == 0:
            return

        dt = obj.dtype
        if np.issubdtype(dt, np.number) or np.issubdtype(dt, np.complexfloating):
            candidates.append(name)

    h5.visititems(visitor)
    return candidates[0] if candidates else None


def read_h5_spectrum(fn: str, h5_path: str) -> Tuple[np.ndarray, str, bool]:
    """Read one HDF5 dataset as a 1D real-valued spectrum.

    Returns:
        spectrum      : float32 1D vector
        dataset_path  : actual dataset path used
        was_complex   : True if source dataset was complex and np.abs was applied
    """
    with h5py.File(fn, "r") as h5:
        dataset_path = h5_path

        if dataset_path not in h5:
            dataset_path = find_first_numeric_dataset(h5)

            if dataset_path is None:
                available = inspect_h5_datasets(fn)
                raise KeyError(
                    f"No numeric or complex dataset found in {fn}. "
                    f"Available datasets: {available}"
                )

            print(
                f"Preferred path '{h5_path}' not found in {os.path.basename(fn)}; "
                f"using '/{dataset_path}'"
            )

        data = np.asarray(h5[dataset_path][()])

    was_complex = bool(np.iscomplexobj(data) or np.issubdtype(data.dtype, np.complexfloating))

    if was_complex:
        print(f"Complex dataset detected in {os.path.basename(fn)} -> using np.abs()")
        data = np.abs(data)
    else:
        print(f"Real dataset detected in {os.path.basename(fn)}")

    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data.reshape(-1), dataset_path, was_complex


def load_h5_waterfall(data_dir: Optional[str], h5_path: str):
    """Load folder of HDF5 files as a 2D waterfall matrix.

    If data_dir is None or empty, opens a folder chooser dialog.
    Automatically handles:
      - real datasets: used directly;
      - complex datasets: converted to amplitude by np.abs();
      - missing preferred path: first numeric/complex dataset is used.
    """
    if data_dir is None or str(data_dir).strip() == "":
        data_dir = select_data_folder()

    data_dir = os.path.abspath(data_dir)
    files = scan_h5_files(data_dir)

    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found in {data_dir}")

    print("=" * 72)
    print(f"Selected data directory: {data_dir}")
    print(f"Found HDF5 files: {len(files)}")
    print("=" * 72)

    rows = []
    dataset_paths = []
    complex_count = 0
    real_count = 0
    expected_len = None

    for i, fn in enumerate(files):
        spectrum, used_path, was_complex = read_h5_spectrum(fn, h5_path)

        if expected_len is None:
            expected_len = spectrum.size
        elif spectrum.size != expected_len:
            raise ValueError(
                f"Spectrum length mismatch in {os.path.basename(fn)}: "
                f"expected {expected_len}, got {spectrum.size}. "
                "All files must have the same spectrum length."
            )

        rows.append(spectrum)
        dataset_paths.append(used_path)

        if was_complex:
            complex_count += 1
        else:
            real_count += 1

        if (i + 1) % max(1, len(files) // 10) == 0:
            print(f"Loaded {i + 1}/{len(files)} files...")

    waterfall = np.asarray(rows, dtype=np.float32)
    unique_paths = sorted(set(dataset_paths))

    print("=" * 72)
    print("HDF5 LOAD SUMMARY")
    print("=" * 72)
    print(f"Waterfall shape:  {waterfall.shape}")
    print(f"Real datasets:    {real_count}")
    print(f"Complex datasets: {complex_count}")
    print("Dataset paths used:")
    for p in unique_paths:
        print(f"  /{p}")
    print("=" * 72)

    return waterfall, files


# ============================================================
# MARCHENKO–PASTUR FILTER
# ============================================================
@dataclass
class MPResult:
    clean: np.ndarray
    residue: np.ndarray
    eigvals: np.ndarray
    lambda_plus: float
    lambda_minus: float
    kept_components: int


def mp_denoise(data_matrix: np.ndarray, sigma2: float = 1.0, eps: float = EPS, progress_cb: ProgressCB = None) -> MPResult:
    progress_cb = progress_cb or progress_noop
    progress_cb(0.02, "MP: preparing data")
    X0 = clean_numeric_np(data_matrix)
    T, K = X0.shape

    if T < 2 or K < 2:
        return MPResult(
            clean=X0.copy(),
            residue=np.zeros_like(X0, dtype=np.float32),
            eigvals=np.zeros(max(1, K), dtype=np.float32),
            lambda_plus=0.0,
            lambda_minus=0.0,
            kept_components=0,
        )

    mean = X0.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(X0.std(axis=0, keepdims=True), eps).astype(np.float32)
    Xz_np = ((X0 - mean) / std).astype(np.float32)

    q = K / max(1, T)
    root_q = np.sqrt(q)
    lambda_minus = float(sigma2 * (1.0 - root_q) ** 2)
    lambda_plus = float(sigma2 * (1.0 + root_q) ** 2)

    progress_cb(0.15, "MP: uploading / covariance")
    Xz = xp.asarray(Xz_np, dtype=xp.float32)

    if K <= T:
        C = (Xz.T @ Xz) / max(1, T)
        progress_cb(0.45, "MP: eigendecomposition")
        eigvals, eigvecs = xp.linalg.eigh(C)

        keep = eigvals > lambda_plus
        kept = int(to_cpu(xp.count_nonzero(keep)))
        if kept == 0:
            keep = eigvals == xp.max(eigvals)
            kept = 1

        progress_cb(0.75, "MP: projection")
        V = eigvecs[:, keep]
        Xclean_z = (Xz @ V) @ V.T
    else:
        G = (Xz @ Xz.T) / max(1, T)
        progress_cb(0.45, "MP: temporal eigendecomposition")
        eigvals, U = xp.linalg.eigh(G)

        keep = eigvals > lambda_plus
        kept = int(to_cpu(xp.count_nonzero(keep)))
        if kept == 0:
            keep = eigvals == xp.max(eigvals)
            kept = 1

        progress_cb(0.75, "MP: temporal projection")
        U_keep = U[:, keep]
        Xclean_z = U_keep @ (U_keep.T @ Xz)

    sync_gpu()
    progress_cb(0.9, "MP: downloading result")

    clean = clean_numeric_np(to_cpu(Xclean_z).astype(np.float32) * std + mean)
    residue = clean_numeric_np(X0 - clean)
    eigvals_np = clean_numeric_np(to_cpu(eigvals))

    progress_cb(1.0, "MP: done")
    return MPResult(
        clean=clean,
        residue=residue,
        eigvals=eigvals_np,
        lambda_plus=lambda_plus,
        lambda_minus=lambda_minus,
        kept_components=kept,
    )


# ============================================================
# SSA CPU + CUDA
# ============================================================
def diagonal_averaging_np(Xelem: np.ndarray, T: int, L: int) -> np.ndarray:
    Kwin = T - L + 1
    y = np.zeros(T, dtype=np.float32)
    c = np.zeros(T, dtype=np.float32)
    for i in range(L):
        y[i:i + Kwin] += Xelem[i, :]
        c[i:i + Kwin] += 1.0
    return y / np.maximum(c, 1.0)


def ssa_leading_component_1d_np(signal: np.ndarray, L: int) -> np.ndarray:
    x0 = clean_numeric_np(signal).reshape(-1)
    T = x0.size
    L = odd_or_valid_window(L, T, minimum=2)
    Kwin = T - L + 1

    if Kwin < 2:
        return x0.copy()

    idx = np.arange(L)[:, None] + np.arange(Kwin)[None, :]
    Xtraj = x0[idx].astype(np.float32)

    try:
        U, S, Vt = np.linalg.svd(Xtraj, full_matrices=False)
        leading = S[0] * np.outer(U[:, 0], Vt[0, :])
        y = diagonal_averaging_np(leading.astype(np.float32), T, L)
        return clean_numeric_np(y)
    except Exception as e:
        print(f"CPU SSA failed for one column; returning zeros. Error: {e}")
        return np.zeros_like(x0, dtype=np.float32)


def ssa_matrix_columns_cpu(
    data_matrix: np.ndarray,
    L: int = DEFAULT_SSA_WINDOW,
    progress: bool = True,
    progress_cb: ProgressCB = None,
    base_msg: str = "CPU SSA",
) -> np.ndarray:
    progress_cb = progress_cb or progress_noop
    X = clean_numeric_np(data_matrix)
    T, K = X.shape
    L = odd_or_valid_window(L, T, minimum=2)
    out = np.zeros_like(X, dtype=np.float32)

    for k in range(K):
        out[:, k] = ssa_leading_component_1d_np(X[:, k], L)
        if progress and (k + 1) % max(1, K // 100) == 0:
            progress_cb((k + 1) / K, f"{base_msg}: {k + 1}/{K}")
        if progress and K >= 20 and (k + 1) % max(1, K // 10) == 0:
            print(f"CPU SSA progress: {k + 1}/{K} columns, L={L}")
    progress_cb(1.0, f"{base_msg}: done")
    return clean_numeric_np(out)


def diagonal_averaging_batch_cp(leading, T: int, L: int):
    cp = cp_module
    B, L2, Kwin = leading.shape
    assert L2 == L

    y = cp.zeros((B, T), dtype=cp.float32)
    c = cp.zeros((B, T), dtype=cp.float32)

    for i in range(L):
        y[:, i:i + Kwin] += leading[:, i, :]
        c[:, i:i + Kwin] += 1.0

    return y / cp.maximum(c, 1.0)


def ssa_matrix_columns_cuda(
    data_matrix: np.ndarray,
    L: int = DEFAULT_SSA_WINDOW,
    batch_size: int = DEFAULT_SSA_BATCH,
    progress: bool = True,
    progress_cb: ProgressCB = None,
    base_msg: str = "CUDA SSA",
) -> np.ndarray:
    progress_cb = progress_cb or progress_noop
    if BACKEND.name != "cupy/cuda" or cp_module is None:
        raise RuntimeError("CUDA SSA requested, but CuPy backend is not available.")

    cp = cp_module
    X = clean_numeric_np(data_matrix)
    T, K = X.shape
    L = odd_or_valid_window(L, T, minimum=2)
    Kwin = T - L + 1

    if Kwin < 2:
        return X.copy()

    out = np.zeros_like(X, dtype=np.float32)
    batch_size = max(1, int(batch_size))
    idx = cp.arange(L, dtype=cp.int32)[:, None] + cp.arange(Kwin, dtype=cp.int32)[None, :]

    for start in range(0, K, batch_size):
        end = min(K, start + batch_size)
        Xb = cp.asarray(X[:, start:end], dtype=cp.float32)
        traj = Xb[idx, :].transpose(2, 0, 1).astype(cp.float32)

        try:
            U, S, Vh = cp.linalg.svd(traj, full_matrices=False)
            leading = (
                S[:, 0, None, None]
                * U[:, :, 0][:, :, None]
                * Vh[:, 0, :][:, None, :]
            ).astype(cp.float32)

            yb = diagonal_averaging_batch_cp(leading, T=T, L=L)
            cp.cuda.Stream.null.synchronize()
            out[:, start:end] = cp.asnumpy(yb.T).astype(np.float32)

        except Exception as e:
            print(f"CUDA SSA batch failed for columns {start}:{end}; CPU fallback. Error: {e}")
            cp.get_default_memory_pool().free_all_blocks()
            out[:, start:end] = ssa_matrix_columns_cpu(
                X[:, start:end], L=L, progress=False, progress_cb=None, base_msg="CPU fallback SSA"
            )

        del Xb
        try:
            del traj, U, S, Vh, leading, yb
        except Exception:
            pass
        cp.get_default_memory_pool().free_all_blocks()

        frac = end / K
        if progress:
            progress_cb(frac, f"{base_msg}: {end}/{K}")
            print(f"CUDA SSA progress: {end}/{K} columns, L={L}, batch={batch_size}")

    progress_cb(1.0, f"{base_msg}: done")
    return clean_numeric_np(out)


def ssa_matrix_columns(
    data_matrix: np.ndarray,
    L: int = DEFAULT_SSA_WINDOW,
    backend: str = "auto",
    batch_size: int = DEFAULT_SSA_BATCH,
    progress: bool = True,
    progress_cb: ProgressCB = None,
    base_msg: str = "SSA",
) -> np.ndarray:
    backend = backend.lower().strip()

    if backend not in ["auto", "cuda", "cpu"]:
        raise ValueError("ssa backend must be 'auto', 'cuda', or 'cpu'")

    if backend == "cpu":
        return ssa_matrix_columns_cpu(data_matrix, L=L, progress=progress, progress_cb=progress_cb, base_msg=f"{base_msg} CPU")

    if backend in ["auto", "cuda"]:
        if BACKEND.name == "cupy/cuda" and cp_module is not None:
            try:
                return ssa_matrix_columns_cuda(
                    data_matrix,
                    L=L,
                    batch_size=batch_size,
                    progress=progress,
                    progress_cb=progress_cb,
                    base_msg=f"{base_msg} CUDA",
                )
            except Exception as e:
                if backend == "cuda":
                    raise
                print(f"CUDA SSA unavailable/failed; falling back to CPU SSA. Error: {e}")
        elif backend == "cuda":
            raise RuntimeError("CUDA SSA requested, but CuPy backend is unavailable.")

    return ssa_matrix_columns_cpu(data_matrix, L=L, progress=progress, progress_cb=progress_cb, base_msg=f"{base_msg} CPU")


# ============================================================
# FISHER STRUCTURAL DENSITY
# ============================================================
def compute_fisher_density_2d(field: np.ndarray, eps: float = EPS, progress_cb=None, **kwargs) -> np.ndarray:
    a = clean_numeric_np(field)
    a = a - np.min(a)
    a = np.maximum(a, 0.0)

    x = np.log1p(a)
    p = x / (np.max(x) + eps)
    p = np.maximum(p, eps)

    gt, gk = np.gradient(p.astype(np.float32))
    F = (gt * gt + gk * gk) / (p + eps)
    F = np.sqrt(F)
    F = F / (np.percentile(F, 99.5) + eps)
    F = np.clip(F, 0.0, 1.0)
    return clean_numeric_np(F)


# ============================================================
# ENTROPY INPUT + MODES
# ============================================================
def prepare_entropy_input_np(field: np.ndarray, eps: float = EPS) -> np.ndarray:
    """CPU fallback: log-compressed normalized positive input."""
    a = clean_numeric_np(field)
    a = a - np.min(a)
    a = np.maximum(a, 0.0)
    x = np.log1p(a)
    x = x / (np.max(x) + eps)
    x = np.clip(x, eps, 1.0)
    return x.astype(np.float32)


def prepare_entropy_input_xp(field, eps: float = EPS):
    """GPU/CPU xp version: log-compressed normalized positive input."""
    x = xp.asarray(field, dtype=xp.float32)
    x = xp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - xp.min(x)
    x = xp.maximum(x, 0.0)
    x = xp.log1p(x)
    x = x / (xp.max(x) + eps)
    x = xp.clip(x, eps, 1.0)
    return x.astype(xp.float32)


# Keep old name for CPU-only routines such as permutation entropy.
def prepare_entropy_input(field: np.ndarray, eps: float = EPS) -> np.ndarray:
    return prepare_entropy_input_np(field, eps=eps)


def entropy_shannon_xp(x, win_t: int, win_k: int, eps: float = EPS):
    area = float(win_t * win_k)
    local_sum = uniform_filter_xp(x, size=(win_t, win_k)) * area + eps
    local_xlogx = uniform_filter_xp(x * xp.log(x + eps), size=(win_t, win_k)) * area
    H = xp.log(local_sum) - local_xlogx / local_sum
    return xp.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)


def entropy_kapur_xp(x, win_t: int, win_k: int, eps: float = EPS):
    area = float(win_t * win_k)
    local_sum = uniform_filter_xp(x, size=(win_t, win_k)) * area + eps
    p_local = x / local_sum
    H = -uniform_filter_xp(p_local * xp.log(p_local + eps), size=(win_t, win_k)) * area
    return xp.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)


def entropy_renyi_xp(x, win_t: int, win_k: int, alpha: float, eps: float = EPS):
    if abs(alpha - 1.0) < 1e-5:
        return entropy_kapur_xp(x, win_t, win_k, eps=eps)

    area = float(win_t * win_k)
    local_sum = uniform_filter_xp(x, size=(win_t, win_k)) * area + eps
    p_local = x / local_sum
    local_p_alpha = uniform_filter_xp(xp.power(p_local + eps, alpha), size=(win_t, win_k)) * area
    H = xp.log(local_p_alpha + eps) / (1.0 - alpha)
    return xp.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)


def entropy_tsallis_xp(x, win_t: int, win_k: int, q: float, eps: float = EPS):
    if abs(q - 1.0) < 1e-5:
        return entropy_kapur_xp(x, win_t, win_k, eps=eps)

    area = float(win_t * win_k)
    local_sum = uniform_filter_xp(x, size=(win_t, win_k)) * area + eps
    p_local = x / local_sum
    local_p_q = uniform_filter_xp(xp.power(p_local + eps, q), size=(win_t, win_k)) * area
    H = (1.0 - local_p_q) / (q - 1.0)
    return xp.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)


def entropy_variational_modes_xp(x, win_t: int, win_k: int, eps: float = EPS):
    mean = uniform_filter_xp(x, size=(win_t, win_k))
    mean2 = uniform_filter_xp(x * x, size=(win_t, win_k))
    var = xp.maximum(mean2 - mean * mean, 0.0)
    cv = xp.sqrt(var + eps) / (mean + eps)
    return xp.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)


def entropy_local_variance_xp(x, win_t: int, win_k: int, eps: float = EPS):
    mean = uniform_filter_xp(x, size=(win_t, win_k))
    mean2 = uniform_filter_xp(x * x, size=(win_t, win_k))
    var = xp.maximum(mean2 - mean * mean, 0.0)
    H = xp.sqrt(var + eps)
    return xp.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)


def robust_unit_xp(a, percentile: float = 99.5, eps: float = EPS):
    """Normalize xp array to [0,1] using robust percentile.

    Note: cupy supports percentile; if unavailable for any reason, falls back to CPU
    for the scalar percentile only.
    """
    arr = xp.asarray(a, dtype=xp.float32)
    arr = xp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        scale = xp.percentile(arr, percentile) + eps
        out = arr / scale
        out = xp.clip(out, 0.0, 1.0)
        return out.astype(xp.float32)
    except Exception:
        scale = float(np.nanpercentile(to_cpu(arr), percentile)) + eps
        out = arr / scale
        out = xp.clip(out, 0.0, 1.0)
        return out.astype(xp.float32)


# CPU versions retained for compatibility and permutation fallback.
def entropy_shannon_2d(x: np.ndarray, win_t: int, win_k: int, eps: float = EPS) -> np.ndarray:
    area = float(win_t * win_k)
    local_sum = uniform_filter_np(x, size=(win_t, win_k)) * area + eps
    local_xlogx = uniform_filter_np(x * np.log(x + eps), size=(win_t, win_k)) * area
    return np.log(local_sum) - local_xlogx / local_sum


def entropy_kapur_2d(x: np.ndarray, win_t: int, win_k: int, eps: float = EPS) -> np.ndarray:
    area = float(win_t * win_k)
    local_sum = uniform_filter_np(x, size=(win_t, win_k)) * area + eps
    p_local = x / local_sum
    return -uniform_filter_np(p_local * np.log(p_local + eps), size=(win_t, win_k)) * area


def entropy_renyi_2d(x: np.ndarray, win_t: int, win_k: int, alpha: float, eps: float = EPS) -> np.ndarray:
    if abs(alpha - 1.0) < 1e-5:
        return entropy_kapur_2d(x, win_t, win_k, eps=eps)
    area = float(win_t * win_k)
    local_sum = uniform_filter_np(x, size=(win_t, win_k)) * area + eps
    p_local = x / local_sum
    local_p_alpha = uniform_filter_np(np.power(p_local + eps, alpha), size=(win_t, win_k)) * area
    return np.log(local_p_alpha + eps) / (1.0 - alpha)


def entropy_tsallis_2d(x: np.ndarray, win_t: int, win_k: int, q: float, eps: float = EPS) -> np.ndarray:
    if abs(q - 1.0) < 1e-5:
        return entropy_kapur_2d(x, win_t, win_k, eps=eps)
    area = float(win_t * win_k)
    local_sum = uniform_filter_np(x, size=(win_t, win_k)) * area + eps
    p_local = x / local_sum
    local_p_q = uniform_filter_np(np.power(p_local + eps, q), size=(win_t, win_k)) * area
    return (1.0 - local_p_q) / (q - 1.0)


def entropy_variational_modes(x: np.ndarray, win_t: int, win_k: int, eps: float = EPS) -> np.ndarray:
    mean = uniform_filter_np(x, size=(win_t, win_k))
    mean2 = uniform_filter_np(x * x, size=(win_t, win_k))
    var = np.maximum(mean2 - mean * mean, 0.0)
    return np.sqrt(var + eps) / (mean + eps)


def entropy_local_variance(x: np.ndarray, win_t: int, win_k: int, eps: float = EPS) -> np.ndarray:
    mean = uniform_filter_np(x, size=(win_t, win_k))
    mean2 = uniform_filter_np(x * x, size=(win_t, win_k))
    var = np.maximum(mean2 - mean * mean, 0.0)
    return np.sqrt(var + eps)


def permutation_entropy_rows(
    x: np.ndarray,
    order: int = DEFAULT_PERM_ORDER,
    delay: int = DEFAULT_PERM_DELAY,
    win_t: int = DEFAULT_MSFR_WIN_T,
    eps: float = EPS,
) -> np.ndarray:
    """CPU ordinal permutation entropy.

    This mode remains CPU because it requires ordinal sorting and histogramming
    of local permutation patterns. The rest of the entropy family is GPU-enabled.
    """
    x = clean_numeric_np(x)
    T, K = x.shape
    order = max(3, int(order))
    delay = max(1, int(delay))
    win_t = max(order * delay + 2, int(win_t))

    perms = list(itertools.permutations(range(order)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    n_perm = len(perms)
    max_entropy = np.log(n_perm + eps)

    out = np.zeros((T, K), dtype=np.float32)
    half = win_t // 2
    offsets = np.arange(order, dtype=np.int32) * delay
    span = int(offsets[-1]) + 1

    for k in range(K):
        col = x[:, k]
        if T < span:
            continue

        pattern_ids = np.full(T, -1, dtype=np.int32)

        for t0 in range(0, T - span + 1):
            vec = col[t0 + offsets]
            ordinal = tuple(np.argsort(vec, kind="mergesort"))
            pattern_ids[t0 + offsets[-1] // 2] = perm_to_idx.get(ordinal, -1)

        for t in range(T):
            lo = max(0, t - half)
            hi = min(T, t + half + 1)
            ids = pattern_ids[lo:hi]
            ids = ids[ids >= 0]
            if ids.size == 0:
                out[t, k] = 0.0
                continue
            counts = np.bincount(ids, minlength=n_perm).astype(np.float32)
            probs = counts / (np.sum(counts) + eps)
            H = -np.sum(probs * np.log(probs + eps))
            out[t, k] = H / (max_entropy + eps)

        if K >= 20 and (k + 1) % max(1, K // 10) == 0:
            print(f"Permutation entropy progress: {k + 1}/{K} modes")

    return clean_numeric_np(out)


def compute_local_entropy_np(
    field: np.ndarray,
    win_t: int,
    win_k: int,
    eps: float,
    mode: str,
    renyi_alpha: float,
    tsallis_q: float,
    perm_order: int,
    perm_delay: int,
    progress_cb=None,
    **kwargs,
) -> np.ndarray:
    """Compute selected real entropy/storage component.

    GPU-enabled modes:
      shannon, kapur, renyi, tsallis, variational_modes, local_variance

    CPU fallback:
      permutation_rows
    """
    mode = mode.lower().strip()
    if mode not in ENTROPY_MODES:
        raise ValueError(f"Unsupported entropy mode: {mode}. Choose from {ENTROPY_MODES}")

    win_t = max(2, int(win_t))
    win_k = max(2, int(win_k))

    if mode == "permutation_rows":
        print("Entropy backend: CPU permutation_rows")
        x_cpu = prepare_entropy_input_np(field, eps=eps)
        H_cpu = permutation_entropy_rows(
            x_cpu,
            order=perm_order,
            delay=perm_delay,
            win_t=win_t,
            eps=eps,
        )
        return robust_unit_np(H_cpu, percentile=99.5, eps=eps)

    print(f"Entropy backend: {BACKEND.name} for {mode}")

    x_gpu = prepare_entropy_input_xp(field, eps=eps)

    if mode == "shannon":
        H = entropy_shannon_xp(x_gpu, win_t, win_k, eps=eps)
    elif mode == "kapur":
        H = entropy_kapur_xp(x_gpu, win_t, win_k, eps=eps)
    elif mode == "renyi":
        H = entropy_renyi_xp(x_gpu, win_t, win_k, alpha=float(renyi_alpha), eps=eps)
    elif mode == "tsallis":
        H = entropy_tsallis_xp(x_gpu, win_t, win_k, q=float(tsallis_q), eps=eps)
    elif mode == "variational_modes":
        H = entropy_variational_modes_xp(x_gpu, win_t, win_k, eps=eps)
    elif mode == "local_variance":
        H = entropy_local_variance_xp(x_gpu, win_t, win_k, eps=eps)
    else:
        raise ValueError(f"Unsupported entropy mode: {mode}")

    H = robust_unit_xp(H, percentile=99.5, eps=eps)
    sync_gpu()
    return clean_numeric_np(to_cpu(H))


# ============================================================
# COMPLEX FIELD + MSFR
# ============================================================
def compute_entropy_fisher_complex_map(
    field: np.ndarray,
    win_t: int,
    win_k: int,
    eps: float,
    entropy_mode: str,
    renyi_alpha: float,
    tsallis_q: float,
    perm_order: int,
    perm_delay: int,
    progress_cb: ProgressCB = None,
):
    progress_cb = progress_cb or progress_noop
    H = compute_local_entropy_np(
        field,
        win_t=win_t,
        win_k=win_k,
        eps=eps,
        mode=entropy_mode,
        renyi_alpha=renyi_alpha,
        tsallis_q=tsallis_q,
        perm_order=perm_order,
        perm_delay=perm_delay,
        progress_cb=progress_cb,
    )
    progress_cb(0.92, "Complex field: Fisher")
    F = compute_fisher_density_2d(field, eps=eps)

    progress_cb(0.96, "Complex field: amplitude/phase")
    Z = H.astype(np.complex64) + 1j * F.astype(np.complex64)
    Z_amp = np.abs(Z).astype(np.float32)
    Z_phase = np.angle(Z).astype(np.float32)
    Z_phase_norm = Z_phase / ((0.5 * np.pi) + eps)
    Z_phase_norm = np.clip(Z_phase_norm, 0.0, 1.0).astype(np.float32)
    Z_amp = robust_unit_np(Z_amp, percentile=99.5, eps=eps)

    progress_cb(1.0, "Complex field: done")
    return Z, clean_numeric_np(Z_amp), clean_numeric_np(Z_phase), clean_numeric_np(Z_phase_norm), H, F


def compute_msfr_xp(data_matrix: np.ndarray, win_t: int, win_k: int, eps: float = EPS) -> np.ndarray:
    win_t = max(2, int(win_t))
    win_k = max(2, int(win_k))

    x = xp.asarray(clean_numeric_np(data_matrix), dtype=xp.float32)
    x_pos = xp.clip(normalize_xp(x, eps), eps, None)
    area = float(win_t * win_k)

    def local_entropy_xp(field):
        f = xp.clip(normalize_xp(field, eps), eps, None)
        local_sum = uniform_filter_xp(f, size=(win_t, win_k)) * area + eps
        local_xlogx = uniform_filter_xp(f * xp.log(f + eps), size=(win_t, win_k)) * area
        H = xp.log(local_sum) - local_xlogx / local_sum
        return xp.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(xp.float32)

    H_i = local_entropy_xp(x_pos)
    gt, gk = xp.gradient(x_pos)
    grad_field = 0.5 * (xp.abs(gt) + xp.abs(gk)) + eps
    H_g = local_entropy_xp(grad_field)
    out = normalize_xp(H_i * H_g, eps)
    sync_gpu()
    return to_cpu(out).astype(np.float32)


# ============================================================
# FIGURE HELPERS
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


def show_image(ax, data, title, cmap, label, p=(2, 98)):
    im = ax.imshow(data, aspect="auto", origin="upper", cmap=cmap)
    im.set_clim(*percentile_clim(data, p))
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=PANEL_TITLE_PAD)
    if HIDE_INDIVIDUAL_AXIS_LABELS:
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Time / file index")
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    add_panel_label(ax, label)
    return im


def save_with_optional_pdf(fig, out_path: str):
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".pdf":
        pdf_path = base + ".pdf"
        fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved figure: {pdf_path}")


# ============================================================
# EXPLORER APP
# ============================================================
class MultiEntropyFisherExplorer:
    def __init__(self, args):
        self.args = args
        self.state: Dict[str, Any] = {
            "entropy_mode": args.entropy_mode,
            "win_t": args.msfr_win_t,
            "win_k": args.msfr_win_k,
            "ssa_window": args.ssa_window,
            "renyi_alpha": args.renyi_alpha,
            "tsallis_q": args.tsallis_q,
            "perm_order": args.perm_order,
            "perm_delay": args.perm_delay,
            "ssa_backend": args.ssa_backend,
            "ssa_batch": args.ssa_batch,
            "show_ssa": not args.no_ssa_window,
        }

        self.progress_start = time.perf_counter()
        self.progress_ax = None
        self.progress_rect = None
        self.progress_text = None
        self.status_text = None

        print(f"Backend: {BACKEND.name}")
        print("Loading data...")
        self.raw, self.files = load_h5_waterfall(args.data_dir, args.h5_path)
        self.raw = clean_numeric_np(self.raw)
        print(f"Loaded waterfall: T={self.raw.shape[0]}, K={self.raw.shape[1]}, files={len(self.files)}")

        self.fig_main = None
        self.axes_main = None
        self.cbars_main = []

        self.fig_ssa = None
        self.axes_ssa = None
        self.cbars_ssa = []

        self.fig_ctrl = None
        self.current: Dict[str, np.ndarray] = {}

        self.create_main_window()
        self.create_ssa_window()
        self.create_control_window()

        self.update_progress(0.01, "Computing Marchenko–Pastur filtering...")
        self.mp = mp_denoise(self.raw, progress_cb=lambda f, m: self.update_progress(f, m))
        print(
            f"MP: lambda-={self.mp.lambda_minus:.4f}, "
            f"lambda+={self.mp.lambda_plus:.4f}, "
            f"kept={self.mp.kept_components}/{self.raw.shape[1]}"
        )

        self.update_progress(0.05, "Computing Fisher maps...")
        self.fisher_raw = compute_fisher_density_2d(self.raw)
        self.update_progress(0.55, "Computing Fisher maps: MP...")
        self.fisher_mp = compute_fisher_density_2d(self.mp.clean)
        self.update_progress(1.0, "Initialization ready")

        self.recompute(update_ssa=self.state["show_ssa"])

    def update_progress(self, frac: float, msg: str):
        frac = float(np.clip(frac, 0.0, 1.0))
        if self.progress_rect is not None:
            self.progress_rect.set_width(frac)
        if self.progress_text is not None:
            self.progress_text.set_text(f"{frac * 100:5.1f}%")
        if self.status_text is not None:
            elapsed = time.perf_counter() - self.progress_start
            self.status_text.set_text(f"STATUS: {msg}\nElapsed: {elapsed:6.1f}s")
        if self.fig_ctrl is not None:
            try:
                self.fig_ctrl.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:
                pass

    def reset_progress(self, msg: str):
        self.progress_start = time.perf_counter()
        self.update_progress(0.0, msg)

    def entropy_label(self):
        return pretty_entropy_name(self.state["entropy_mode"])

    def compute_current_maps(self):
        s = self.state
        self.update_progress(0.05, "MSFR: raw")
        msfr_raw = compute_msfr_xp(self.raw, s["win_t"], s["win_k"])
        self.update_progress(0.15, "MSFR: MP")
        msfr_mp = compute_msfr_xp(self.mp.clean, s["win_t"], s["win_k"])

        self.update_progress(0.25, f"Complex field raw: {s['entropy_mode']}")
        _, ef_amp_raw, _, ef_phase_raw, H_raw, F_raw = compute_entropy_fisher_complex_map(
            self.raw,
            win_t=s["win_t"],
            win_k=s["win_k"],
            eps=EPS,
            entropy_mode=s["entropy_mode"],
            renyi_alpha=s["renyi_alpha"],
            tsallis_q=s["tsallis_q"],
            perm_order=s["perm_order"],
            perm_delay=s["perm_delay"],
            progress_cb=lambda f, m: self.update_progress(0.25 + 0.30 * f, f"raw {m}"),
        )

        self.update_progress(0.58, f"Complex field MP: {s['entropy_mode']}")
        _, ef_amp_mp, _, ef_phase_mp, H_mp, F_mp = compute_entropy_fisher_complex_map(
            self.mp.clean,
            win_t=s["win_t"],
            win_k=s["win_k"],
            eps=EPS,
            entropy_mode=s["entropy_mode"],
            renyi_alpha=s["renyi_alpha"],
            tsallis_q=s["tsallis_q"],
            perm_order=s["perm_order"],
            perm_delay=s["perm_delay"],
            progress_cb=lambda f, m: self.update_progress(0.58 + 0.30 * f, f"MP {m}"),
        )

        self.current.update({
            "msfr_raw": msfr_raw,
            "msfr_mp": msfr_mp,
            "ef_amp_raw": ef_amp_raw,
            "ef_phase_raw": ef_phase_raw,
            "ef_amp_mp": ef_amp_mp,
            "ef_phase_mp": ef_phase_mp,
            "H_raw": H_raw,
            "F_raw": F_raw,
            "H_mp": H_mp,
            "F_mp": F_mp,
        })
        self.update_progress(0.9, "Main maps computed")

    def compute_current_ssa(self):
        s = self.state
        self.current["ssa_amp_raw"] = ssa_matrix_columns(
            self.current["ef_amp_raw"],
            L=s["ssa_window"],
            backend=s["ssa_backend"],
            batch_size=s["ssa_batch"],
            progress_cb=lambda f, m: self.update_progress(0.00 + 0.25 * f, f"SSA amp raw: {m}"),
            base_msg="SSA amp raw",
        )
        self.current["ssa_phase_raw"] = ssa_matrix_columns(
            self.current["ef_phase_raw"],
            L=s["ssa_window"],
            backend=s["ssa_backend"],
            batch_size=s["ssa_batch"],
            progress_cb=lambda f, m: self.update_progress(0.25 + 0.25 * f, f"SSA phase raw: {m}"),
            base_msg="SSA phase raw",
        )
        self.current["ssa_amp_mp"] = ssa_matrix_columns(
            self.current["ef_amp_mp"],
            L=s["ssa_window"],
            backend=s["ssa_backend"],
            batch_size=s["ssa_batch"],
            progress_cb=lambda f, m: self.update_progress(0.50 + 0.25 * f, f"SSA amp MP: {m}"),
            base_msg="SSA amp MP",
        )
        self.current["ssa_phase_mp"] = ssa_matrix_columns(
            self.current["ef_phase_mp"],
            L=s["ssa_window"],
            backend=s["ssa_backend"],
            batch_size=s["ssa_batch"],
            progress_cb=lambda f, m: self.update_progress(0.75 + 0.25 * f, f"SSA phase MP: {m}"),
            base_msg="SSA phase MP",
        )

    def create_main_window(self):
        self.fig_main, self.axes_main = plt.subplots(2, 5, figsize=FIGSIZE_MAIN, constrained_layout=True)
        self.fig_main.canvas.manager.set_window_title("Multi-Entropy Fisher Complex Field")

    def create_ssa_window(self):
        self.fig_ssa, self.axes_ssa = plt.subplots(2, 2, figsize=FIGSIZE_SSA, constrained_layout=True)
        self.fig_ssa.canvas.manager.set_window_title("SSA of Complex Information Field")

    def clear_colorbars(self, which="main"):
        cbars = self.cbars_main if which == "main" else self.cbars_ssa
        for cb in cbars:
            try:
                cb.remove()
            except Exception:
                pass
        if which == "main":
            self.cbars_main = []
        else:
            self.cbars_ssa = []

    def draw_main(self):
        label = self.entropy_label()
        axes = self.axes_main
        for ax in axes.ravel():
            ax.clear()
        self.clear_colorbars("main")

        panels = [
            (axes[0, 0], self.raw, "Original spectral waterfall", CMAP_RAW, "a", (2, 98)),
            (axes[0, 1], self.fisher_raw, "Fisher structural density (raw)", CMAP_FISHER, "b", (0, 99.8)),
            (axes[0, 2], self.current["msfr_raw"], "MSFR response map (raw)", CMAP_MSFR, "c", (2, 98)),
            (axes[0, 3], self.current["ef_amp_raw"], f"{label}–Fisher amplitude (raw)", CMAP_COMPLEX, "d", (0, 99.8)),
            (axes[0, 4], self.current["ef_phase_raw"], f"{label}–Fisher phase (raw)", CMAP_PHASE, "e", (0, 100)),
            (axes[1, 0], self.mp.clean, "MP-filtered spectral waterfall", CMAP_RAW, "f", (2, 98)),
            (axes[1, 1], self.fisher_mp, "Fisher structural density (MP)", CMAP_FISHER, "g", (0, 99.8)),
            (axes[1, 2], self.current["msfr_mp"], "MSFR response map (MP)", CMAP_MSFR, "h", (2, 98)),
            (axes[1, 3], self.current["ef_amp_mp"], f"{label}–Fisher amplitude (MP)", CMAP_COMPLEX, "i", (0, 99.8)),
            (axes[1, 4], self.current["ef_phase_mp"], f"{label}–Fisher phase (MP)", CMAP_PHASE, "j", (0, 100)),
        ]

        images = []
        for ax, data, title, cmap, panel_label, clim_p in panels:
            images.append(show_image(ax, data, title, cmap, panel_label, p=clim_p))

        self.fig_main.supxlabel("Frequency bin", fontsize=AXIS_LABEL_FONT_SIZE)
        self.fig_main.supylabel("Time / file index", fontsize=AXIS_LABEL_FONT_SIZE)
        self.fig_main.suptitle(
            f"Marchenko–Pastur Filtering and {label}–Fisher Complex Information Localization",
            fontsize=SUPTITLE_FONT_SIZE,
            fontweight=SUPTITLE_WEIGHT,
        )

        if USE_COLORBARS:
            for ax, im in zip(axes.ravel(), images):
                cbar = self.fig_main.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
                self.cbars_main.append(cbar)

        self.fig_main.canvas.draw_idle()

    def draw_ssa(self):
        if not self.state["show_ssa"]:
            return
        required = ["ssa_amp_raw", "ssa_phase_raw", "ssa_amp_mp", "ssa_phase_mp"]
        if not all(k in self.current for k in required):
            return

        label = self.entropy_label()
        axes = self.axes_ssa
        for ax in axes.ravel():
            ax.clear()
        self.clear_colorbars("ssa")

        panels = [
            (axes[0, 0], self.current["ssa_amp_raw"], f"SSA of {label}–Fisher amplitude (raw)", CMAP_COMPLEX, "a", (0, 99.8)),
            (axes[0, 1], self.current["ssa_phase_raw"], f"SSA of {label}–Fisher phase (raw)", CMAP_PHASE, "b", (0, 100)),
            (axes[1, 0], self.current["ssa_amp_mp"], f"SSA of {label}–Fisher amplitude (MP)", CMAP_COMPLEX, "c", (0, 99.8)),
            (axes[1, 1], self.current["ssa_phase_mp"], f"SSA of {label}–Fisher phase (MP)", CMAP_PHASE, "d", (0, 100)),
        ]

        images = []
        for ax, data, title, cmap, panel_label, clim_p in panels:
            images.append(show_image(ax, data, title, cmap, panel_label, p=clim_p))

        self.fig_ssa.supxlabel("Frequency bin", fontsize=AXIS_LABEL_FONT_SIZE)
        self.fig_ssa.supylabel("Time / file index", fontsize=AXIS_LABEL_FONT_SIZE)
        self.fig_ssa.suptitle(
            f"SSA Decomposition of {label}–Fisher Complex Information Field",
            fontsize=SUPTITLE_FONT_SIZE,
            fontweight=SUPTITLE_WEIGHT,
        )

        if USE_COLORBARS:
            for ax, im in zip(axes.ravel(), images):
                cbar = self.fig_ssa.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
                self.cbars_ssa.append(cbar)

        self.fig_ssa.canvas.draw_idle()

    def recompute(self, update_ssa=True):
        print("=" * 72)
        print("RECOMPUTE")
        print(self.state)
        self.reset_progress("Recompute started")
        self.compute_current_maps()
        self.update_progress(0.92, "Drawing main figure")
        self.draw_main()
        self.update_progress(1.0, "Main figure done")

        if update_ssa and self.state["show_ssa"]:
            self.reset_progress("SSA started")
            self.compute_current_ssa()
            self.update_progress(0.97, "Drawing SSA figure")
            self.draw_ssa()
            self.update_progress(1.0, "SSA done")

        self.update_progress(1.0, "Ready")
        print("Done.")

    def save_figures(self):
        main_path, ssa_path = auto_export_paths(
            self.state["entropy_mode"],
            self.state["ssa_window"],
            self.mp.kept_components,
            prefix="complex",
        )
        self.reset_progress("Saving figures")
        save_with_optional_pdf(self.fig_main, main_path)
        self.update_progress(0.5, "Saved main figure")
        if self.state["show_ssa"]:
            save_with_optional_pdf(self.fig_ssa, ssa_path)
        self.update_progress(1.0, "Figures saved")

    def create_control_window(self):
        self.fig_ctrl = plt.figure(figsize=FIGSIZE_CTRL)
        self.fig_ctrl.canvas.manager.set_window_title("Controls: Multi-Entropy Fisher Explorer")
        self.fig_ctrl.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.05)

        ax_title = self.fig_ctrl.add_axes([0.05, 0.935, 0.9, 0.045])
        ax_title.axis("off")
        ax_title.text(0.0, 0.5, "Multi-Entropy Fisher Controls", fontsize=14, fontweight="bold", va="center")

        ax_status = self.fig_ctrl.add_axes([0.07, 0.855, 0.86, 0.065])
        ax_status.axis("off")
        self.status_text = ax_status.text(0.0, 0.55, "STATUS: initializing", fontsize=10, va="center")

        self.progress_ax = self.fig_ctrl.add_axes([0.07, 0.825, 0.76, 0.025])
        self.progress_ax.set_xlim(0, 1)
        self.progress_ax.set_ylim(0, 1)
        self.progress_ax.set_xticks([])
        self.progress_ax.set_yticks([])
        self.progress_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#e6e6e6", edgecolor="#777777"))
        self.progress_rect = Rectangle((0, 0), 0.0, 1, facecolor="#2ca25f", edgecolor="none")
        self.progress_ax.add_patch(self.progress_rect)

        ax_percent = self.fig_ctrl.add_axes([0.84, 0.818, 0.12, 0.035])
        ax_percent.axis("off")
        self.progress_text = ax_percent.text(0.0, 0.5, "0.0%", fontsize=10, va="center")

        ax_radio = self.fig_ctrl.add_axes([0.08, 0.525, 0.40, 0.285])
        ax_radio.set_title("Entropy mode")
        active_idx = ENTROPY_MODES.index(self.state["entropy_mode"])
        radio = RadioButtons(ax_radio, ENTROPY_LABELS, active=active_idx)

        ax_check = self.fig_ctrl.add_axes([0.55, 0.735, 0.35, 0.07])
        check = CheckButtons(ax_check, ["SSA window"], [self.state["show_ssa"]])

        ax_backend = self.fig_ctrl.add_axes([0.55, 0.555, 0.35, 0.15])
        ax_backend.set_title("SSA backend")
        backend_options = ["auto", "cuda", "cpu"]
        backend_active = backend_options.index(self.state["ssa_backend"])
        radio_backend = RadioButtons(ax_backend, backend_options, active=backend_active)

        slider_specs = [
            ("win_t", "Win T", 3, 51, self.state["win_t"], 1),
            ("win_k", "Win K", 3, 51, self.state["win_k"], 1),
            ("ssa_window", "SSA L", 5, 100, self.state["ssa_window"], 1),
            ("ssa_batch", "SSA batch", 16, 512, self.state["ssa_batch"], 16),
            ("renyi_alpha", "Rényi α", 0.1, 3.0, self.state["renyi_alpha"], None),
            ("tsallis_q", "Tsallis q", 0.1, 3.0, self.state["tsallis_q"], None),
            ("perm_order", "Perm m", 3, 7, self.state["perm_order"], 1),
            ("perm_delay", "Perm τ", 1, 5, self.state["perm_delay"], 1),
        ]

        sliders = {}
        y0 = 0.485
        dy = 0.052
        for i, (key, label, vmin, vmax, val, step) in enumerate(slider_specs):
            ax = self.fig_ctrl.add_axes([0.13, y0 - i * dy, 0.74, 0.03])
            if step is None:
                sliders[key] = Slider(ax, label, vmin, vmax, valinit=val)
            else:
                sliders[key] = Slider(ax, label, vmin, vmax, valinit=val, valstep=step)

        ax_open = self.fig_ctrl.add_axes([0.05, 0.045, 0.18, 0.06])
        btn_open = Button(ax_open, "Open Folder")

        ax_recompute = self.fig_ctrl.add_axes([0.27, 0.045, 0.18, 0.06])
        btn_recompute = Button(ax_recompute, "Recompute")

        ax_save = self.fig_ctrl.add_axes([0.49, 0.045, 0.18, 0.06])
        btn_save = Button(ax_save, "Save Figures")

        ax_close = self.fig_ctrl.add_axes([0.71, 0.045, 0.18, 0.06])
        btn_close = Button(ax_close, "Close")

        def on_radio(label):
            self.state["entropy_mode"] = LABEL_TO_MODE[label]
            self.update_progress(0.0, f"Selected entropy: {self.state['entropy_mode']} — press Recompute")

        def on_backend(label):
            self.state["ssa_backend"] = label
            self.update_progress(0.0, f"Selected SSA backend: {label} — press Recompute")

        def on_check(label):
            self.state["show_ssa"] = not self.state["show_ssa"]
            self.update_progress(0.0, f"SSA window enabled: {self.state['show_ssa']} — press Recompute")

        def read_sliders():
            self.state["win_t"] = int(sliders["win_t"].val)
            self.state["win_k"] = int(sliders["win_k"].val)
            self.state["ssa_window"] = int(sliders["ssa_window"].val)
            self.state["ssa_batch"] = int(sliders["ssa_batch"].val)
            self.state["renyi_alpha"] = float(sliders["renyi_alpha"].val)
            self.state["tsallis_q"] = float(sliders["tsallis_q"].val)
            self.state["perm_order"] = int(sliders["perm_order"].val)
            self.state["perm_delay"] = int(sliders["perm_delay"].val)

        def on_open(event):
            try:
                new_dir = select_data_folder()
                print("=" * 72)
                print(f"Loading new folder: {new_dir}")

                self.raw, self.files = load_h5_waterfall(new_dir, self.args.h5_path)
                self.raw = clean_numeric_np(self.raw)

                print("Recomputing Marchenko–Pastur filtering...")
                self.mp = mp_denoise(self.raw)

                print("Recomputing Fisher maps...")
                self.fisher_raw = compute_fisher_density_2d(self.raw)
                self.fisher_mp = compute_fisher_density_2d(self.mp.clean)

                self.recompute(update_ssa=self.state["show_ssa"])

            except Exception as e:
                print(f"Open folder failed: {e}")

        def on_recompute(event):
            read_sliders()
            self.recompute(update_ssa=True)

        def on_save(event):
            self.save_figures()

        def on_close(event):
            plt.close(self.fig_ctrl)

        radio.on_clicked(on_radio)
        radio_backend.on_clicked(on_backend)
        check.on_clicked(on_check)
        btn_open.on_clicked(on_open)
        btn_recompute.on_clicked(on_recompute)
        btn_save.on_clicked(on_save)
        btn_close.on_clicked(on_close)

        self.control_widgets = {
            "radio": radio,
            "radio_backend": radio_backend,
            "check": check,
            "sliders": sliders,
            "btn_open": btn_open,
            "btn_recompute": btn_recompute,
            "btn_save": btn_save,
            "btn_close": btn_close,
        }

        self.update_progress(0.0, "Control window ready")
        self.fig_ctrl.canvas.draw_idle()


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive multi-entropy Fisher complex explorer with control window and progress bar."
    )

    p.add_argument("--data-dir", default=DATA_DIRECTORY,
                   help="Directory containing .h5/.hdf5 files. If omitted, opens a folder chooser dialog.")
    p.add_argument("--h5-path", default=H5_SIGNAL_PATH, help="Preferred dataset path inside each .h5 file")
    p.add_argument("--ask-dir", action="store_true", help="Force opening folder chooser dialog even if --data-dir is set")

    p.add_argument("--msfr-win-t", type=int, default=DEFAULT_MSFR_WIN_T)
    p.add_argument("--msfr-win-k", type=int, default=DEFAULT_MSFR_WIN_K)
    p.add_argument("--ssa-window", type=int, default=DEFAULT_SSA_WINDOW)
    p.add_argument("--ssa-batch", type=int, default=DEFAULT_SSA_BATCH)
    p.add_argument("--ssa-backend", choices=["auto", "cuda", "cpu"], default="auto")

    p.add_argument("--entropy-mode", choices=ENTROPY_MODES, default=DEFAULT_ENTROPY_MODE)
    p.add_argument("--renyi-alpha", type=float, default=DEFAULT_RENYI_ALPHA)
    p.add_argument("--tsallis-q", type=float, default=DEFAULT_TSALLIS_Q)
    p.add_argument("--perm-order", type=int, default=DEFAULT_PERM_ORDER)
    p.add_argument("--perm-delay", type=int, default=DEFAULT_PERM_DELAY)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-ssa-window", action="store_true")

    return p.parse_args()


def main():
    global BACKEND, xp, cp_module

    args = parse_args()

    if args.cpu:
        BACKEND = setup_backend(False)
        xp = BACKEND.xp
        cp_module = BACKEND.cp

    if getattr(args, 'ask_dir', False):
        args.data_dir = None

    if getattr(args, "ask_dir", False):
        args.data_dir = None

    explorer = MultiEntropyFisherExplorer(args)
    plt.show()


if __name__ == "__main__":
    main()
