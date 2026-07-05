import argparse
import sys
from pathlib import Path
from typing import IO, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from config import VOLUME_SHAPE, HU_MIN, HU_MAX, NORM_BIAS, NORM_SCALE, DATA_PREP

# ── stdout tee ───────────────────────────────────────────────────────────────

class _Tee:
    """Write every print() to both the terminal and a log file."""
    def __init__(self, log_path: Path, original: IO):
        self._log  = log_path.open("w", encoding="utf-8")
        self._orig = original

    def write(self, s: str) -> int:
        self._orig.write(s)
        self._log.write(s)
        return len(s)

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def close(self) -> None:
        self._log.close()


# CT2Rep paper target spacing in (D, H, W) order (mm)
RESAMPLE_SPACING_DHW: Tuple[float, float, float] = (1.5, 0.75, 0.75)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_nifti_as_dhw(
    path: Path,
) -> Tuple[np.ndarray, Tuple[float, float, float], tuple]:
    """
    Load NIfTI, reorient to RAS+, return (arr_DHW, spacing_DHW_mm, orig_shape).

    Reorientation uses nibabel.as_closest_canonical() so that axis interpretation
    is consistent regardless of how the file was written:
        RAS+ dim 0 = x (L→R)   → width   (W)
        RAS+ dim 1 = y (P→A)   → height  (H)
        RAS+ dim 2 = z (I→S)   → depth   (D)  ← slice axis for axial CT
    We then transpose (x, y, z) → (z, y, x) = (D, H, W).

    nibabel.get_fdata() applies scl_slope / scl_inter automatically, so the
    returned array is in Hounsfield Units for standard CT NIfTI files.
    """
    img      = nib.load(str(path))
    orig_shape = img.shape[:3]

    img_ras  = nib.as_closest_canonical(img)
    zooms    = img_ras.header.get_zooms()[:3]          # (dx, dy, dz) in mm, RAS+ order
    data     = img_ras.get_fdata(dtype=np.float32)    # slope/inter already applied

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {data.shape} in {path}")

    # (x, y, z) → (D=z, H=y, W=x)
    data    = data.transpose(2, 1, 0).astype(np.float32)
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))  # (dz, dy, dx)

    return data, spacing, orig_shape


# ── transforms ────────────────────────────────────────────────────────────────

def normalize_hu(arr: np.ndarray) -> np.ndarray:
    """Clip to [HU_MIN, HU_MAX] then map to [-1, +1] with a safety clip."""
    arr = np.clip(arr, HU_MIN, HU_MAX)
    arr = (arr + NORM_BIAS) / NORM_SCALE
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def center_crop_or_pad(
    arr: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: float = -1.0,
) -> np.ndarray:
    """
    Central crop (if arr dim > target) or central pad (if arr dim < target).
    Accepts (D, H, W). Padding fills with pad_value.
    """
    out = np.full(target_shape, pad_value, dtype=arr.dtype)
    src_slices: list = []
    dst_slices: list = []

    for s, t in zip(arr.shape, target_shape):
        if s >= t:
            start = (s - t) // 2
            src_slices.append(slice(start, start + t))
            dst_slices.append(slice(0, t))
        else:
            pad_before = (t - s) // 2
            src_slices.append(slice(0, s))
            dst_slices.append(slice(pad_before, pad_before + s))

    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def resample_to_spacing(
    arr: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Resample to target voxel spacing via trilinear zoom."""
    zoom_factors = tuple(c / t for c, t in zip(current_spacing, target_spacing))
    return ndi.zoom(arr, zoom_factors, order=1).astype(np.float32)


# ── save / assert ─────────────────────────────────────────────────────────────

def _save(arr: np.ndarray, dst: Path) -> None:
    assert arr.shape == VOLUME_SHAPE, \
        f"shape mismatch: {arr.shape} != {VOLUME_SHAPE}"
    assert arr.dtype == np.float32, \
        f"dtype mismatch: {arr.dtype}"
    assert arr.min() >= -1.0 - 1e-5, \
        f"min out of range: {arr.min():.6f}"
    assert arr.max() <=  1.0 + 1e-5, \
        f"max out of range: {arr.max():.6f}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(dst), arr_0=arr)
    print(f"  saved → {dst}")


# ── mode: ct2rep_contract ─────────────────────────────────────────────────────

def process_ct2rep_contract(src: Path, dst: Path) -> None:
    # Matches the final CTViT input contract. Does NOT resample spacing.
    print(f"\n[ct2rep_contract]  {src.name}")

    arr, spacing, orig_shape = load_nifti_as_dhw(src)
    print(f"  original shape      : {orig_shape}")
    print(f"  voxel spacing (mm)  : D={spacing[0]:.3f}  H={spacing[1]:.3f}  W={spacing[2]:.3f}")
    print(f"  reoriented (D,H,W)  : {arr.shape}")
    print(f"  HU range            : [{arr.min():.1f}, {arr.max():.1f}]")

    arr = normalize_hu(arr)
    print(f"  normalized range    : [{arr.min():.4f}, {arr.max():.4f}]")

    arr = center_crop_or_pad(arr, VOLUME_SHAPE, pad_value=-1.0)
    print(f"  final shape         : {arr.shape}  (D, H, W)")
    print(f"  final range         : [{arr.min():.4f}, {arr.max():.4f}]")

    _save(arr, dst)


# ── mode: resample_crop_pad ───────────────────────────────────────────────────

def process_resample_crop_pad(src: Path, dst: Path) -> None:
    # Closer to the CT2Rep paper preprocessing from raw scans.
    # NOTE: the released CT2Rep GitHub loader assumes files are already at
    # target spacing and does NOT resample — this mode fills that gap.
    print(f"\n[resample_crop_pad]  {src.name}")

    arr, spacing, orig_shape = load_nifti_as_dhw(src)
    print(f"  original shape      : {orig_shape}")
    print(f"  voxel spacing (mm)  : D={spacing[0]:.3f}  H={spacing[1]:.3f}  W={spacing[2]:.3f}")
    print(f"  reoriented (D,H,W)  : {arr.shape}")
    print(f"  HU range            : [{arr.min():.1f}, {arr.max():.1f}]")

    tgt = RESAMPLE_SPACING_DHW
    print(f"  target spacing (mm) : D={tgt[0]}  H={tgt[1]}  W={tgt[2]}")
    arr = resample_to_spacing(arr, spacing, tgt)
    print(f"  resampled (D,H,W)   : {arr.shape}")

    arr = normalize_hu(arr)
    print(f"  normalized range    : [{arr.min():.4f}, {arr.max():.4f}]")

    arr = center_crop_or_pad(arr, VOLUME_SHAPE, pad_value=-1.0)
    print(f"  final shape         : {arr.shape}  (D, H, W)")
    print(f"  final range         : [{arr.min():.4f}, {arr.max():.4f}]")

    _save(arr, dst)


# ── check-only ────────────────────────────────────────────────────────────────

def check_npz(path: Path) -> bool:
    """Verify one NPZ meets the output contract. Returns True if all checks pass."""
    print(f"\n[check]  {path}")
    ok = True

    if not path.exists():
        print(f"  FAIL  file does not exist")
        return False

    try:
        npz = np.load(str(path))
    except Exception as e:
        print(f"  FAIL  cannot load: {e}")
        return False

    if "arr_0" not in npz:
        print(f"  FAIL  key 'arr_0' missing  (found: {list(npz.keys())})")
        return False

    arr  = npz["arr_0"]
    vmin = float(arr.min())
    vmax = float(arr.max())

    def _check(label: str, cond: bool, detail: str) -> bool:
        status = "OK  " if cond else "FAIL"
        print(f"  {status}  {label}: {detail}")
        return cond

    ok &= _check("shape", arr.shape == VOLUME_SHAPE,      f"{arr.shape}")
    ok &= _check("dtype", arr.dtype == np.float32,        f"{arr.dtype}")
    ok &= _check("range", vmin >= -1.0 - 1e-5 and vmax <= 1.0 + 1e-5,
                 f"[{vmin:.4f}, {vmax:.4f}]")

    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────

def _stem(p: Path) -> str:
    """Handle both .nii.gz and .nii extensions."""
    name = p.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return p.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess NIfTI → CTViT-ready NPZ  (arr_0: 240×480×480, float32, [-1,+1])",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("src", nargs="?",
                        help="Input .nii/.nii.gz file or directory")
    parser.add_argument("dst", nargs="?", default=str(DATA_PREP),
                        help="Output .npz file or directory  (default: data/preprocessed/)")
    parser.add_argument(
        "--mode",
        choices=["ct2rep_contract", "resample_crop_pad"],
        default="ct2rep_contract",
        help="Processing mode (default: ct2rep_contract)",
    )
    parser.add_argument(
        "--check-only", metavar="PATH",
        help="Verify an existing NPZ file or directory of NPZ files and exit.",
    )
    parser.add_argument(
        "--log", metavar="FILE",
        help="Write all diagnostic output to FILE in addition to the terminal.",
    )
    args = parser.parse_args()

    tee = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tee = _Tee(log_path, sys.stdout)
        sys.stdout = tee  # type: ignore[assignment]

    # ── check-only ────────────────────────────────────────────────────────────
    if args.check_only:
        target = Path(args.check_only)
        paths  = sorted(target.glob("*.npz")) if target.is_dir() else [target]
        if not paths:
            sys.exit(f"No .npz files found at {target}")

        results = [check_npz(p) for p in paths]
        n_ok    = sum(results)
        print(f"\n  {n_ok}/{len(results)} passed.")
        sys.exit(0 if all(results) else 1)

    # ── preprocess ────────────────────────────────────────────────────────────
    if not args.src:
        parser.print_help()
        sys.exit(1)

    process_fn = (process_ct2rep_contract
                  if args.mode == "ct2rep_contract"
                  else process_resample_crop_pad)

    src, dst = Path(args.src), Path(args.dst)

    try:
        if src.is_dir():
            files = sorted(src.rglob("*.nii.gz")) + sorted(src.rglob("*.nii"))
            if not files:
                sys.exit(f"No .nii/.nii.gz files found under {src}")
            for f in files:
                process_fn(f, dst / (_stem(f) + ".npz"))
        else:
            out = dst if dst.suffix == ".npz" else dst / (_stem(src) + ".npz")
            process_fn(src, out)
    finally:
        if tee:
            sys.stdout = tee._orig
            tee.close()
            print(f"  log saved → {args.log}")


if __name__ == "__main__":
    main()
