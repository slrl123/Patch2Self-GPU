"""Patch2Self denoising for diffusion MRI, solved on the GPU.

This is a PyTorch re-implementation of the Patch2Self denoiser from DIPY
(https://dipy.org/documentation/latest/examples_built/preprocessing/denoise_patch2self/).
It follows DIPY's ``patch2self(..., model="ols", version=3)`` formulation and
solves the same leave-one-out ordinary least squares problem, but on the GPU
in float64 and over *all* voxels instead of a count-sketch subsample.

See ``README.md`` for the mathematical correspondence and the measured
agreement against the reference implementation.
"""

from __future__ import annotations

import time

import numpy as np
import torch

__all__ = ["patch2self_gpu"]

__version__ = "1.0.0"


def _resolve_device(device) -> torch.device:
    if device is None:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda:0")
    return torch.device(device)


def _lstsq_driver(tensor: torch.Tensor) -> str:
    """CUDA's LAPACK only ships 'gels'; the CPU path needs gelsd/gelsy."""
    return "gels" if tensor.is_cuda else "gelsd"


def _denoise_group(data: np.ndarray, indices: np.ndarray,
                   device: torch.device, verbose: bool = False) -> np.ndarray:
    """Leave-one-out OLS for one volume group (b0 or DWI).

    ``indices`` are the positions of the group's volumes in the full 4D array.
    Mirrors DIPY: a volume is predicted from the *other volumes of its own
    group* only; the intercept is carried by temporarily overwriting the
    held-out column with ones, which is algebraically the same problem as
    ``sklearn.linear_model.LinearRegression(fit_intercept=True)`` on the
    columns that remain.
    """
    n = indices.size
    shape3d = data.shape[:3]
    # train[v] == flattened v-th volume of the group
    train = np.ascontiguousarray(
        data[..., indices].transpose(3, 0, 1, 2)
    ).reshape(n, -1)

    X = torch.from_numpy(train.T).double().to(device)  # (n_voxels, n)
    driver = _lstsq_driver(X)
    den = np.empty((n,) + shape3d, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        for vi in range(n):
            y = X[:, vi].clone()
            X[:, vi] = 1.0
            sol = torch.linalg.lstsq(X, y, driver=driver).solution
            pred = (X @ sol).reshape(list(shape3d)).cpu().numpy()
            # restore the original column (float32 -> float64 is lossless)
            X[:, vi] = torch.from_numpy(train[vi]).to(device)
            den[vi] = pred
    finally:
        del X
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if verbose:
        print(f"  group of {n} volumes: {time.perf_counter() - t0:.1f} s",
              flush=True)
    return den


def patch2self_gpu(
    data,
    bvals,
    *,
    b0_threshold: float = 50,
    device=None,
    b0_denoising: bool = False,
    shift_intensity: bool = True,
    clip_negative_vals: bool = False,
    out_dtype=None,
    verbose: bool = True,
):
    """Patch2Self denoising on the GPU.

    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data, shape (X, Y, Z, N).
    bvals : array of shape (N,)
        The b-values of the acquisition. Volumes with ``bval <= b0_threshold``
        are treated as b0.
    b0_threshold : float, optional
        Threshold for considering a volume as b0. Default: 50.
    device : torch.device or str, optional
        Device to solve on. Default: ``cuda:0`` if CUDA is available, else
        ``cpu``.
    b0_denoising : bool, optional
        If False (default), b0 volumes are copied through untouched and the
        DWI volumes are regressed against the other DWI volumes only. If True,
        b0 volumes are additionally regressed against the other b0 volumes.

        Note that this default differs from DIPY, which denoises b0 volumes by
        default. Leaving b0 untouched is the recommended setting for
        downstream modelling.
    shift_intensity : bool, optional
        Shift each output volume so its minimum matches the corresponding
        noisy volume's minimum, i.e. keep the original intensity floor.
        Default: True.

        DIPY 1.10 documents this option as giving non-negative values, but its
        implementation computes ``min(denoised) - min(denoised)``, which is a
        no-op. This repository implements the intended behaviour, which is also
        what the original Patch2Self reference code does.
    clip_negative_vals : bool, optional
        Clamp negative values to 0 after denoising. Default: False.
    out_dtype : dtype, optional
        dtype of the returned array. Default: same as ``data``.
    verbose : bool, optional
        Print per-group timing. Default: True.

    Returns
    -------
    denoised : ndarray
        Denoised data, same shape as ``data``.

    Notes
    -----
    Memory: the design matrix is materialised on the device in float64, so the
    GPU needs roughly ``2.5 * 8 * n_voxels * n_group_volumes`` bytes. A full
    HCP-YA 3T case (145x174x145 voxels, 270 DWI volumes) needs about 8 GB for
    the matrix and fits comfortably on a 24 GB GPU.

    References
    ----------
    .. [1] Fadnavis, S., et al. "Patch2Self: Denoising Diffusion MRI with
       Self-Supervised Learning", NeurIPS 2020.
    .. [2] Fadnavis, S., et al. "Patch2Self: A Generalizable Deep Learning
       Framework for Denoising Diffusion MRI Data", NeuroImage 2024.
    """
    data = np.asanyarray(data)
    if data.ndim != 4:
        raise ValueError(f"data must be 4D (X, Y, Z, N), got shape {data.shape}")
    n_vols = data.shape[3]

    bvals = np.asarray(bvals, dtype=np.float64).ravel()
    if bvals.size != n_vols:
        raise ValueError(
            f"bvals must have {n_vols} entries, got {bvals.size}"
        )

    if out_dtype is None:
        out_dtype = data.dtype

    device = _resolve_device(device)
    if verbose:
        print(f"patch2self_gpu: solving on {device}", flush=True)

    is_b0 = bvals <= b0_threshold

    groups: list = []
    if (~is_b0).any():
        groups.append(np.flatnonzero(~is_b0))
    if b0_denoising and is_b0.any():
        groups.append(np.flatnonzero(is_b0))

    out = data.astype(np.float64, copy=True)

    for idx in groups:
        if idx.size < 2:
            if verbose:
                print(f"  group of {idx.size} volume(s): needs >= 2 to fit, "
                      f"left untouched", flush=True)
            continue
        if verbose:
            print(f"  denoising {idx.size} volumes "
                  f"({int((bvals[idx] <= b0_threshold).sum())} b0, "
                  f"{int((bvals[idx] > b0_threshold).sum())} DWI)", flush=True)
        den = _denoise_group(data, idx, device, verbose=verbose)
        out[..., idx] = den.transpose(1, 2, 3, 0)

    if shift_intensity:
        for i in range(n_vols):
            out[..., i] += data[..., i].min() - out[..., i].min()
    if clip_negative_vals:
        np.clip(out, 0.0, None, out=out)

    return out.astype(out_dtype)
