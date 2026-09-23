"""Numerical equivalence checks against the reference solver.

Runs entirely on CPU (``CUDA_VISIBLE_DEVICES`` is emptied on import) so the
suite can be executed on a machine without a GPU, and never competes with
running GPU work.

The reference is a direct float64 ``sklearn.linear_model.LinearRegression``
leave-one-out fit, which is the same solver DIPY's Patch2Self uses internally.
Note that comparing against DIPY's own ``patch2self`` output is *not* a useful
equality test: DIPY solves in the input dtype (float32 for most real data),
and on near-collinear shells its float32 path can differ from the float64
truth by ~15% relative (see README, "The float32 trap"). The tests below
therefore compare against the exact float64 solution.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

import torch  # noqa: E402  (import after the env var is set)

from patch2self_gpu import patch2self_gpu  # noqa: E402

B0_THRESHOLD = 50.0


def make_synth(shape=(8, 8, 8, 30), n_basis=25, noise=0.1, seed=0):
    """Synthetic DWI in general position: clean signal + Gaussian noise.

    ``n_basis`` is chosen below the volume count so the leave-one-out design
    matrices are well conditioned and the float64 reference is meaningful.
    """
    rng = np.random.default_rng(seed)
    n_vol = shape[3]
    n_vox = int(np.prod(shape[:3]))
    basis = rng.normal(size=(n_vox, n_basis))
    clean = basis @ rng.normal(size=(n_basis, n_vol))
    clean -= clean.min()
    noisy = clean + rng.normal(scale=noise, size=clean.shape)
    bvals = np.zeros(n_vol)
    bvals[n_vol // 3:] = 1000.0 + 1000.0 * (np.arange(n_vol - n_vol // 3) % 3)
    return noisy.reshape(shape).astype(np.float32), bvals


def reference_sklearn(data, bvals, b0_threshold=B0_THRESHOLD):
    """Leave-one-out OLS with intercept in float64."""
    from sklearn.linear_model import LinearRegression

    idx = np.flatnonzero(bvals > b0_threshold)
    out = data.astype(np.float64).copy()
    for pos, vi in enumerate(idx):
        others = np.delete(idx, pos)
        X = data[..., others].reshape(-1, others.size).astype(np.float64)
        y = data[..., vi].ravel().astype(np.float64)
        out[..., vi] = LinearRegression(fit_intercept=True).fit(
            X, y).predict(X).reshape(data.shape[:3])
    return out


def _rel(got, ref, mask):
    return (np.abs(got[..., mask] - ref[..., mask]).max()
            / max(np.abs(ref[..., mask]).max(), 1e-12))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_float64_reference(seed):
    """The GPU solve must reproduce the exact float64 OLS solution."""
    data, bvals = make_synth(seed=seed)
    got = patch2self_gpu(data, bvals, device="cpu", shift_intensity=False,
                         verbose=False)
    ref = reference_sklearn(data, bvals)
    dwi = bvals > B0_THRESHOLD
    assert _rel(got, ref, dwi) < 1e-6


def test_matches_on_rank_deficient_data():
    """Collinear shells (the realistic case): fitted values must still match.

    Least-squares fitted values are unique even when the design matrix is
    rank deficient, so the two solvers must agree here too.
    """
    data, bvals = make_synth(n_basis=4, noise=1.0, seed=3)
    got = patch2self_gpu(data, bvals, device="cpu", shift_intensity=False,
                         verbose=False)
    ref = reference_sklearn(data, bvals)
    dwi = bvals > B0_THRESHOLD
    assert _rel(got, ref, dwi) < 1e-6


def test_b0_untouched_by_default():
    data, bvals = make_synth(seed=4)
    got = patch2self_gpu(data, bvals, device="cpu", shift_intensity=False,
                         verbose=False)
    is_b0 = bvals <= B0_THRESHOLD
    assert is_b0.any()
    assert np.array_equal(got[..., is_b0], data[..., is_b0].astype(np.float64))


def test_denoising_actually_changed_the_dwi():
    """Sanity: the DWI volumes are not passed through untouched."""
    data, bvals = make_synth(noise=0.5, seed=5)
    got = patch2self_gpu(data, bvals, device="cpu", shift_intensity=False,
                         verbose=False)
    dwi = bvals > B0_THRESHOLD
    assert not np.array_equal(got[..., dwi], data[..., dwi].astype(np.float64))
    assert np.isfinite(got).all()


def test_shift_intensity_matches_the_noisy_floor():
    data, bvals = make_synth(seed=6)
    got = patch2self_gpu(data, bvals, device="cpu", verbose=False)
    for i in range(data.shape[3]):
        assert got[..., i].min() == pytest.approx(data[..., i].min())


def test_shape_and_dtype_preserved():
    data, bvals = make_synth(seed=7)
    got = patch2self_gpu(data, bvals, device="cpu", verbose=False)
    assert got.shape == data.shape
    assert got.dtype == data.dtype
    got32 = patch2self_gpu(data, bvals, device="cpu", out_dtype=np.float32,
                           verbose=False)
    assert got32.dtype == np.float32


def test_rejects_bad_shapes():
    data, bvals = make_synth(seed=8)
    with pytest.raises(ValueError):
        patch2self_gpu(data[..., 0], bvals, device="cpu", verbose=False)
    with pytest.raises(ValueError):
        patch2self_gpu(data, bvals[:-1], device="cpu", verbose=False)


def test_all_dwi_shell_layout():
    """No b0 volumes at all: every volume is denoised, none crashed."""
    data, bvals = make_synth(shape=(8, 8, 8, 16), seed=9)
    bvals[:] = 1000.0 + 1000.0 * (np.arange(16) % 3)
    got = patch2self_gpu(data, bvals, device="cpu", shift_intensity=False,
                         verbose=False)
    assert np.isfinite(got).all()
    assert not np.array_equal(got, data.astype(np.float64))
