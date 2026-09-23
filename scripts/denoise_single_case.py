"""Denoise a single 4D DWI volume with Patch2Self on the GPU.

Examples
--------
Denoise one HCP-YA 3T case and write it next to the input::

    python scripts/denoise_single_case.py \\
        /HCP-YA/3T/Diffusion_Preprocessed/121921/T1w/Diffusion/data.nii.gz \\
        --bvals .../Diffusion/bvals \\
        --visualize

The ``--bvals`` argument may be omitted if a ``bvals`` file sits next to the
DWI image, which is the convention for most processed dMRI datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

import setproctitle

setproctitle.setproctitle("dayong-denoise")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patch2self_gpu import patch2self_gpu  # noqa: E402


def default_bvals(dwi_path: Path) -> Path | None:
    """Look for a sibling ``bvals`` file, as dipy/HCP conventions lay out."""
    for cand in (dwi_path.parent / "bvals", dwi_path.parent / f"{dwi_path.stem}.bvals"):
        if cand.is_file():
            return cand
    return None


def visualize(dwi_path, orig, den, bvals, out_png):
    """3-view centre-slice comparison: original / denoised / residual."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_b0 = bvals <= 50.0
    dwi_idx = np.flatnonzero(~is_b0)
    v = int(dwi_idx[len(dwi_idx) // 2]) if dwi_idx.size else data_mid(orig)
    noisy, clean = orig[..., v], den[..., v]
    resid = noisy - clean

    fig, axes = plt.subplots(3, 3, figsize=(13, 13),
                             subplot_kw={"xticks": [], "yticks": []})
    views = [
        ("Axial", noisy[:, :, noisy.shape[2] // 2],
         clean[:, :, clean.shape[2] // 2], resid[:, :, resid.shape[2] // 2]),
        ("Coronal", noisy[:, noisy.shape[1] // 2, :],
         clean[:, clean.shape[1] // 2, :], resid[:, resid.shape[1] // 2, :]),
        ("Sagittal", noisy[noisy.shape[0] // 2, :, :],
         clean[clean.shape[0] // 2, :, :], resid[resid.shape[0] // 2, :, :]),
    ]
    for row, (name, o, d, r) in enumerate(views):
        vmax = max(np.percentile(o, 99.5), 1e-6)
        rmax = max(np.percentile(np.abs(r), 99.5), 1e-6)
        axes[row, 0].imshow(np.rot90(o), cmap="gray", vmin=0, vmax=vmax)
        axes[row, 1].imshow(np.rot90(d), cmap="gray", vmin=0, vmax=vmax)
        axes[row, 2].imshow(np.rot90(r), cmap="seismic", vmin=-rmax, vmax=rmax)
        axes[row, 0].set_ylabel(name, fontsize=12)
    axes[0, 0].set_title("Original", fontsize=12)
    axes[0, 1].set_title("Patch2Self denoised (GPU)", fontsize=12)
    axes[0, 2].set_title("Residual", fontsize=12)
    b = int(round(bvals[v] / 1000.0) * 1000)
    fig.suptitle(f"{dwi_path.name} - volume {v} (b~{b}) - "
                 f"{int(is_b0.sum())} b0 + {int((~is_b0).sum())} DWI volumes",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}", flush=True)


def data_mid(vol):
    return vol.shape[3] // 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dwi", help="4D DWI NIfTI image")
    ap.add_argument("--bvals", default=None,
                    help="b-values file (default: sibling 'bvals')")
    ap.add_argument("-o", "--out", default=None,
                    help="output image (default: <dwi>_denoised.nii.gz)")
    ap.add_argument("--device", default=None,
                    help="torch device, e.g. cuda:1 (default: cuda:0 if "
                         "available)")
    ap.add_argument("--b0-threshold", type=float, default=50.0)
    ap.add_argument("--denoise-b0", action="store_true",
                    help="also denoise the b0 volumes (default: leave them "
                         "untouched)")
    ap.add_argument("--no-shift-intensity", action="store_true",
                    help="disable the per-volume intensity shift")
    ap.add_argument("--clip-negative", action="store_true",
                    help="clamp negative output values to 0")
    ap.add_argument("--visualize", action="store_true",
                    help="also save a 3-view comparison PNG")
    ap.add_argument("--out-dtype", default=None,
                    help="output dtype, e.g. float32 (default: same as input)")
    args = ap.parse_args()

    dwi_path = Path(args.dwi)
    if not dwi_path.is_file():
        raise SystemExit(f"no such file: {dwi_path}")

    bvals_path = Path(args.bvals) if args.bvals else default_bvals(dwi_path)
    if bvals_path is None or not bvals_path.is_file():
        raise SystemExit(
            "no bvals file found: pass --bvals explicitly (needed to tell b0 "
            "from DWI volumes)"
        )

    out_path = Path(args.out) if args.out else \
        dwi_path.parent / f"{dwi_path.stem}_denoised.nii.gz"

    t0 = time.perf_counter()
    img = nib.load(str(dwi_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    bvals = np.loadtxt(str(bvals_path), ndmin=1).astype(np.float64)
    t_load = time.perf_counter() - t0
    print(f"loaded {dwi_path}", flush=True)
    print(f"  shape {data.shape}  dtype {data.dtype}  "
          f"({data.nbytes / 2**30:.2f} GB)  in {t_load:.1f} s", flush=True)

    is_b0 = bvals <= args.b0_threshold
    print(f"  b0 volumes: {int(is_b0.sum())}   DWI volumes: "
          f"{int((~is_b0).sum())}", flush=True)

    stats = {"dwi": str(dwi_path), "shape": list(data.shape)}

    t0 = time.perf_counter()
    den = patch2self_gpu(
        data,
        bvals,
        b0_threshold=args.b0_threshold,
        device=args.device,
        b0_denoising=args.denoise_b0,
        shift_intensity=not args.no_shift_intensity,
        clip_negative_vals=args.clip_negative,
        out_dtype=args.out_dtype,
    )
    stats["seconds"] = round(time.perf_counter() - t0, 1)

    nib.Nifti1Image(den, img.affine, header=img.header).to_filename(str(out_path))
    print(f"saved {out_path}  ({stats['seconds']} s)", flush=True)

    # quick sanity report: b0 untouched, no NaN/Inf
    if is_b0.any():
        b0_diff = float(np.abs(data[..., is_b0] - den[..., is_b0]).max())
        print(f"  b0 volumes max|diff|: {b0_diff:.3e} "
              f"({'untouched' if b0_diff == 0 else 'MODIFIED'})", flush=True)
        stats["b0_max_abs_diff"] = b0_diff
    finite = bool(np.isfinite(den).all())
    print(f"  output finite: {finite}", flush=True)
    stats["finite"] = finite

    if args.visualize:
        png = out_path.with_suffix(".png")
        visualize(dwi_path, data, den, bvals, png)

    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
