import argparse
from pathlib import Path

import numpy as np
import torch

import config

try:
    from ctvit import CTViT
    _CTVIT_AVAILABLE = True
except ModuleNotFoundError:
    _CTVIT_AVAILABLE = False


# ── helpers ───────────────────────────────────────────────────────────────────

def load_volume(npz_path: Path) -> torch.Tensor:
    """Load preprocessed NPZ → (1, 1, D, H, W) float32 tensor."""
    arr = np.load(str(npz_path))["arr_0"].astype(np.float32)
    assert arr.shape == config.VOLUME_SHAPE, f"{npz_path}: bad shape {arr.shape}"
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 240, 480, 480)


def volume_to_embedding(model: "CTViT", vol: torch.Tensor) -> torch.Tensor:
    """Run CTViT encoder; return att_feats (1, N_tokens, D)."""
    vol = vol.to(config.device)
    with torch.no_grad():
        tokens = model(vol, return_encoded_tokens=True)
    # tokens: (1, T_p, H_p, W_p, D) → flatten spatial/temporal → (1, N_tokens, D)
    B, T, H, W, D = tokens.shape
    return tokens.reshape(B, T * H * W, D).cpu()


def dummy_embedding() -> torch.Tensor:
    """Random embedding with the correct contract shape — no GPU required."""
    return torch.randn(1, config.N_TOKENS, config.EMBED_DIM)


def save_embedding(emb: torch.Tensor, dst: Path) -> None:
    assert emb.shape == (1, config.N_TOKENS, config.EMBED_DIM), \
        f"bad embedding shape: {emb.shape}"
    assert emb.dtype == torch.float32
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, str(dst))


def verify_round_trip(path: Path) -> None:
    emb = torch.load(str(path), map_location="cpu", weights_only=True)
    assert emb.shape == (1, config.N_TOKENS, config.EMBED_DIM), \
        f"round-trip shape mismatch: {emb.shape}"
    assert emb.dtype == torch.float32
    print(f"  round-trip OK  {path}  shape={tuple(emb.shape)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",   default=None, help="Single NPZ file to encode")
    parser.add_argument("--ckpt",  default=None, help="CTViT checkpoint .pt")
    parser.add_argument("--dummy", action="store_true",
                        help="Skip encoder; save random embeddings (shape contract only)")
    parser.add_argument("--out",   default=str(config.EMBED_DIR),
                        help="Output directory (default: embeddings/pilot/)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.dummy:
        print("  [dummy mode] saving random embeddings — shape contract only")
        npz_files = [Path(args.npz)] if args.npz else sorted(config.DATA_PREP.glob("*.npz"))
        if not npz_files:
            # No NPZ yet — save one dummy under a synthetic name
            npz_files = [Path("scan_000.npz")]
        for npz in npz_files:
            emb = dummy_embedding()
            dst = out_dir / (npz.stem + ".pt")
            save_embedding(emb, dst)
            print(f"  saved dummy {dst}")
            verify_round_trip(dst)
        return

    if not _CTVIT_AVAILABLE:
        import sys
        sys.exit("CTViT not found. Run setup first, or use --dummy.")

    model = CTViT(**config.CTVIT_CFG)
    if args.ckpt:
        state = config.extract_ctvit_state(args.ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        print(f"  loaded weights from {args.ckpt}  ({len(missing)} decoder keys absent — OK for encode-only)")
    model.eval().to(config.device)

    npz_files = [Path(args.npz)] if args.npz else sorted(config.DATA_PREP.glob("*.npz"))
    if not npz_files:
        import sys
        sys.exit(f"No NPZ files found under {config.DATA_PREP}. Run preprocess.py first.")

    for npz in npz_files:
        vol = load_volume(npz)
        emb = volume_to_embedding(model, vol)
        dst = out_dir / (npz.stem + ".pt")
        save_embedding(emb, dst)
        print(f"  saved {dst}  shape={tuple(emb.shape)}")
        verify_round_trip(dst)


if __name__ == "__main__":
    main()
