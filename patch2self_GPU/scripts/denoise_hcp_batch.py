"""Batch Patch2Self-GPU denoising for the HCP-YA dataset.

Denoises the DWI volumes (b1000/b2000/b3000) of a deterministic selection of
HCP-YA subjects and writes a self-contained copy of the dataset tree under a
new root, so existing pipelines keep working with only the data root changed.
b0 volumes are left untouched.

The output preserves the original HCP layout::

    <out_root>/<3T|7T>/Diffusion_Preprocessed/<subject>/T1w/
        Diffusion[_7T]/{data.nii.gz, bvals, bvecs, nodif_brain_mask.nii.gz}
        T1w_acpc_dc_restore_1.25.nii.gz   (3T)  /  1.05  (7T)

Multi-GPU support: the parent launches one worker subprocess per GPU with
``CUDA_VISIBLE_DEVICES`` set before any CUDA driver initialisation, so work
cannot silently land on GPU 0.

Usage::

    python scripts/denoise_hcp_batch.py --src-root /path/to/HCP-YA \\
        --out-root /data/HCP-denoise --gpus 2,3 --cases-3t 250 --cases-7t 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import setproctitle

setproctitle.setproctitle("dayong-denoise")

import nibabel as nib  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patch2self_gpu import patch2self_gpu  # noqa: E402

DIFFUSION_DIRS = {"3T": "Diffusion", "7T": "Diffusion_7T"}
T1_FILENAMES = {
    "3T": "T1w_acpc_dc_restore_1.25.nii.gz",
    "7T": "T1w_acpc_dc_restore_1.05.nii.gz",
}
SIDE_FILES = ("bvals", "bvecs", "nodif_brain_mask.nii.gz")

WORKER_MODE = "--_worker"


@dataclass(frozen=True)
class HCPSubject:
    field_strength: str
    subject: str
    dwi: Path
    bvals: Path
    t1: Path

    @property
    def key(self) -> str:
        return f"{self.field_strength}/{self.subject}"


def discover_subjects(src_root: str, field_strengths=("3T", "7T")) -> list:
    """Find every preprocessed HCP-YA case that has DWI data + bvals."""
    out: list = []
    for fs in field_strengths:
        base = Path(src_root) / fs / "Diffusion_Preprocessed"
        if not base.is_dir():
            print(f"WARNING: no such directory: {base}", flush=True)
            continue
        for subj_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            ddir = subj_dir / "T1w" / DIFFUSION_DIRS[fs]
            dwi = ddir / "data.nii.gz"
            bvals = ddir / "bvals"
            if not (dwi.is_file() and bvals.is_file()):
                continue
            out.append(HCPSubject(fs, subj_dir.name, dwi, bvals,
                                  subj_dir / "T1w" / T1_FILENAMES[fs]))
    return out


def _split_seed(base_seed: int, field_strength: str) -> int:
    """Stable per-field-strength seed so case selection is reproducible."""
    digest = hashlib.md5(f"{base_seed}:{field_strength}:split".encode()).hexdigest()
    return int(digest[:8], 16)


def select_cases(subjects: list, cases: dict, split_seed: int) -> list:
    """Deterministic shuffle-and-take per field strength."""
    by_field: dict = {}
    for s in subjects:
        by_field.setdefault(s.field_strength, []).append(s)

    selected = []
    for fs in ("3T", "7T"):
        items = by_field.get(fs, [])
        if not items:
            continue
        rng = np.random.default_rng(_split_seed(split_seed, fs))
        order = rng.permutation(len(items))
        n = min(int(cases.get(fs, 0)), len(items))
        if n < int(cases.get(fs, 0)):
            print(f"WARNING: {fs} requested {cases[fs]} but only {len(items)} "
                  f"available; using {n}", flush=True)
        selected.extend(items[i] for i in order[:n])
    return selected


def output_dir(subject: HCPSubject, out_root: str) -> Path:
    return (Path(out_root) / subject.field_strength / "Diffusion_Preprocessed"
            / subject.subject / "T1w" / DIFFUSION_DIRS[subject.field_strength])


def denoise_case(subject: HCPSubject, out_root: str,
                 b0_threshold: float = 50.0) -> dict:
    """Denoise one subject's DWI shells and mirror the HCP tree."""
    t_start = time.perf_counter()
    img = nib.load(str(subject.dwi))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine, header = img.affine, img.header
    bvals = np.loadtxt(subject.bvals, ndmin=1).astype(np.float64)

    n_b0 = int((bvals <= b0_threshold).sum())
    n_dwi = int((bvals > b0_threshold).sum())
    if n_dwi == 0:
        raise ValueError(f"{subject.key}: no DWI volumes above b0 threshold")

    den = patch2self_gpu(data, bvals, b0_threshold=b0_threshold, verbose=False)

    ddir = output_dir(subject, out_root)
    ddir.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(den, affine, header=header).to_filename(
        str(ddir / "data.nii.gz"))

    base = ddir.parent
    for name in SIDE_FILES:
        src = subject.dwi.parent / name
        if src.is_file() and not (ddir / name).is_file():
            shutil.copy2(src, ddir / name)
    if subject.t1.is_file() and not (base / T1_FILENAMES[subject.field_strength]).is_file():
        shutil.copy2(subject.t1, base / T1_FILENAMES[subject.field_strength])

    return {
        "subject": subject.subject,
        "field": subject.field_strength,
        "n_dwi": n_dwi,
        "n_b0": n_b0,
        "n_vox": int(np.prod(data.shape[:3])),
        "seconds": round(time.perf_counter() - t_start, 1),
    }


def run_worker(gpu: int, keys: list, src_root: str, out_root: str,
               b0_threshold: float, stats_path: Path) -> int:
    """Worker main: denoise its shard of subjects on one GPU."""
    setproctitle.setproctitle(f"dayong-denoise-gpu{gpu}")
    n_fail = 0
    for field_strength, subject_id in keys:
        try:
            subject = next(
                s for s in discover_subjects(src_root, (field_strength,))
                if s.subject == subject_id
            )
            stats = denoise_case(subject, out_root, b0_threshold)
            with open(stats_path, "a") as f:
                f.write(json.dumps(stats) + "\n")
            print(f"[gpu{gpu}] {stats['field']}/{stats['subject']}: "
                  f"{stats['n_dwi']} DWI vol, {stats['n_vox']:,} vox, "
                  f"{stats['seconds']} s", flush=True)
        except Exception as exc:
            n_fail += 1
            with open(stats_path, "a") as f:
                f.write(json.dumps({"subject": subject_id,
                                    "field": field_strength,
                                    "error": repr(exc)}) + "\n")
            print(f"[gpu{gpu}] FAILED {field_strength}/{subject_id}: {exc!r}",
                  flush=True)
    return n_fail


def shard(items: list, n: int) -> list:
    """Round-robin split into n shards."""
    out = [[] for _ in range(n)]
    for i, k in enumerate(items):
        out[i % n].append(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="HCP-YA root containing <3T|7T>/Diffusion_Preprocessed")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--cases-3t", type=int, default=250)
    ap.add_argument("--cases-7t", type=int, default=50)
    ap.add_argument("--split-seed", type=int, default=20260906)
    ap.add_argument("--b0-threshold", type=float, default=50.0)
    ap.add_argument("--force", action="store_true",
                    help="re-denoise even if output already exists")
    ap.add_argument(WORKER_MODE, type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--_keys-file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker is not None:
        keys = json.loads(Path(args._keys_file).read_text())
        sys.exit(run_worker(args._worker, keys, args.src_root, args.out_root,
                            args.b0_threshold,
                            Path(args.out_root) / "denoise_stats.jsonl"))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stats_path = out_root / "denoise_stats.jsonl"

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    cases = {"3T": args.cases_3t, "7T": args.cases_7t}

    subjects = discover_subjects(args.src_root)
    print(f"discovered {len(subjects)} cases with DWI data "
          f"({sum(1 for s in subjects if s.field_strength == '3T')} 3T, "
          f"{sum(1 for s in subjects if s.field_strength == '7T')} 7T)",
          flush=True)

    selected = select_cases(subjects, cases, args.split_seed)
    print(f"selected {len(selected)} cases: "
          f"{sum(1 for s in selected if s.field_strength == '3T')} 3T, "
          f"{sum(1 for s in selected if s.field_strength == '7T')} 7T",
          flush=True)

    todo, skipped = [], 0
    for s in selected:
        if (output_dir(s, str(out_root)) / "data.nii.gz").is_file() and \
                not args.force:
            skipped += 1
            continue
        todo.append(s)
    if skipped:
        print(f"skipped {skipped} already-denoised cases "
              f"(use --force to redo)", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    keys = [(s.field_strength, s.subject) for s in todo]
    shards = shard(keys, len(gpus))
    key_files = []
    for i, sh in enumerate(shards):
        p = out_root / f".shard-{os.getpid()}-{i}.json"
        p.write_text(json.dumps(sh))
        key_files.append(p)

    procs = []
    for gpu, kf in zip(gpus, key_files):
        w_env = dict(os.environ)
        w_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               WORKER_MODE, str(gpu),
               "--src-root", args.src_root,
               "--out-root", str(out_root),
               "--b0-threshold", str(args.b0_threshold),
               "--_keys-file", str(kf)]
        procs.append(subprocess.Popen(cmd, env=w_env))
        print(f"launched worker on GPU {gpu} with {len(shards[gpus.index(gpu)])} "
              f"cases", flush=True)

    t0 = time.perf_counter()
    rc = [p.wait() for p in procs]
    total = time.perf_counter() - t0

    for kf in key_files:
        kf.unlink(missing_ok=True)

    stats = [json.loads(line) for line in stats_path.read_text().splitlines()
             if line.strip()]
    ok = [s for s in stats if "error" not in s]
    bad = [s for s in stats if "error" in s]

    print("\n" + "=" * 60, flush=True)
    print(f"done      : {len(ok)} / {len(todo)}", flush=True)
    print(f"failed    : {len(bad)}", flush=True)
    for s in bad:
        print(f"  {s['field']}/{s['subject']}: {s['error']}", flush=True)
    if ok:
        print(f"mean/case : {np.mean([s['seconds'] for s in ok]):.1f} s",
              flush=True)
    print(f"total wall: {total / 60:.1f} min ({len(gpus)} GPUs)", flush=True)
    print(f"stats     : {stats_path}", flush=True)
    if any(r != 0 for r in rc) or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
