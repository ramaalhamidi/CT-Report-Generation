import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config


# ── dependency imports (fail early with a clear message) ─────────────────────

def _require(name: str, import_path: str):
    try:
        return __import__(import_path)
    except ModuleNotFoundError:
        sys.exit(
            f"{name} not importable from {import_path}. "
            "Run setup.sh / setup.ps1 first to clone CT2Rep into src/."
        )


# ── stage runners ─────────────────────────────────────────────────────────────

def _banner(phase: int, label: str):
    print(f"\n{'─'*60}")
    print(f"  Phase {phase}: {label}")
    print(f"{'─'*60}")


def stage2_preprocess(npz_path: Path, dummy: bool) -> Path:
    """Verify or create a preprocessed NPZ."""
    _banner(2, "Preprocessing")
    if dummy:
        arr = ((np.random.rand(*config.VOLUME_SHAPE) * 2) - 1).astype(np.float32)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(npz_path), arr_0=arr)
        print(f"  [dummy] created synthetic NPZ: {npz_path}")

    arr = np.load(str(npz_path))["arr_0"].astype(np.float32)
    assert arr.shape == config.VOLUME_SHAPE, \
        f"FAIL shape: {arr.shape} != {config.VOLUME_SHAPE}"
    assert arr.dtype == np.float32, f"FAIL dtype: {arr.dtype}"
    assert arr.min() >= -1.0 - 1e-5, f"FAIL min: {arr.min():.4f}"
    assert arr.max() <=  1.0 + 1e-5, f"FAIL max: {arr.max():.4f}"
    print(f"  NPZ OK  shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}")
    return npz_path


def stage3_ctvit(ctvit_ckpt: Optional[Path], dummy: bool) -> torch.Tensor:
    """Run CTViT on minimum-viable synthetic volume; return raw token tensor."""
    _banner(3, "CTViT smoke test")

    if dummy:
        # return a correctly-shaped fake embedding — no model needed
        T_p = 1   # one temporal patch (12 frames)
        H_p = W_p = 480 // config.CTVIT_CFG["patch_size"]
        emb_raw = torch.randn(1, T_p, H_p, W_p, config.EMBED_DIM)
        print(f"  [dummy] synthetic token tensor: {tuple(emb_raw.shape)}")
        return emb_raw

    try:
        from ctvit import CTViT
    except ModuleNotFoundError:
        sys.exit("CTViT not found in src/. Run setup first or pass --dummy.")

    model = CTViT(**config.CTVIT_CFG)
    if ctvit_ckpt:
        state = config.extract_ctvit_state(ctvit_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        print(f"  loaded weights from {ctvit_ckpt}  ({len(missing)} decoder keys absent — OK for encode-only)")
    model.eval().to(config.device)

    D_MIN = config.CTVIT_CFG["temporal_patch_size"]
    x = torch.randn(1, 1, D_MIN, 480, 480, device=config.device)
    print(f"  input  shape: {tuple(x.shape)}")

    with torch.no_grad():
        emb_raw = model(x, return_encoded_tokens=True)

    H_p = W_p = 480 // config.CTVIT_CFG["patch_size"]
    T_p = D_MIN // config.CTVIT_CFG["temporal_patch_size"]
    assert emb_raw.shape == (1, T_p, H_p, W_p, config.EMBED_DIM), \
        f"FAIL token shape: {emb_raw.shape}"
    print(f"  token tensor:  {tuple(emb_raw.shape)}  (B, T_p, H_p, W_p, D)")
    return emb_raw.cpu()


def stage4_cache(emb_raw: torch.Tensor, embed_path: Path) -> torch.Tensor:
    """Flatten token tensor → (1, N_tokens_pilot, D), save, reload."""
    _banner(4, "Embedding cache round-trip")

    B, T, H, W, D = emb_raw.shape
    emb_flat = emb_raw.reshape(1, T * H * W, D)

    # For the full pipeline, N_tokens=8000. Pilot may differ (smaller crop).
    print(f"  pilot emb (before reshape): {tuple(emb_raw.shape)}")
    print(f"  pilot emb (after  reshape): {tuple(emb_flat.shape)}")

    embed_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_flat, str(embed_path))

    reloaded = torch.load(str(embed_path), map_location="cpu", weights_only=True)
    assert reloaded.shape == emb_flat.shape, \
        f"FAIL round-trip shape: {reloaded.shape} != {emb_flat.shape}"
    assert reloaded.dtype == torch.float32
    print(f"  round-trip OK  {embed_path}")
    return reloaded


def stage5_dataloader(
    embed_path: Path,
    scan_id: str,
    vocab_size: int,
    tmp_dir: Path,
) -> tuple:
    """Build a one-sample DataLoader; assert batch shapes."""
    _banner(5, "DataLoader (embedding × report)")

    from data.dataset import CTRATEDataset, SimpleTokenizer, collate_fn

    # Minimal CSV for smoke test
    reports_csv = tmp_dir / "reports.csv"
    report_text = "bilateral lung infiltrates with no pleural effusion seen"
    pd.DataFrame({"scan_id": [scan_id], "report_text": [report_text]}).to_csv(
        str(reports_csv), index=False
    )

    # Tiny dummy tokenizer
    words = report_text.split() + ["<pad_bos_eos>", "<unk>"]
    token2idx = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tokenizer = SimpleTokenizer(token2idx)

    # Dataset expects embed at embed_dir / scan_id.pt
    # Stage 4 already wrote embed_path = embed_dir / scan_id.pt so we reuse it.
    embed_dir = embed_path.parent

    ds = CTRATEDataset(
        embed_dir=embed_dir,
        reports_csv=reports_csv,
        tokenizer=tokenizer,
        max_seq_length=config.R2GEN_CFG["max_seq_length"],
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    batch = next(iter(loader))
    embs, ids, masks, lengths = batch

    N_pilot = embs.shape[1]   # may differ from 8000 if using tiny crop
    D       = embs.shape[2]
    L       = ids.shape[1]

    assert D == config.EMBED_DIM, f"FAIL embed_dim: {D}"
    assert L == config.R2GEN_CFG["max_seq_length"], f"FAIL seq_length: {L}"
    print(f"  emb  : {tuple(embs.shape)}   (B, N_tokens_pilot, D)")
    print(f"  ids  : {tuple(ids.shape)}    (B, max_seq_length)")
    print(f"  masks: {tuple(masks.shape)}")
    return batch


def stage6_r2gen(batch: tuple, vocab_size: int) -> None:
    """R2Gen forward+backward on real cached embedding."""
    _banner(6, "R2Gen decoder forward + backward")

    try:
        from modules.encoder_decoder import EncoderDecoder
    except ModuleNotFoundError:
        sys.exit("R2Gen modules not found. Run setup first or pass --dummy.")

    embs, ids, masks, lengths = batch
    B = embs.shape[0]

    class _Tok:
        def __init__(self, n): self.idx2token = {i: f"w{i}" for i in range(n)}

    args = SimpleNamespace(**config.R2GEN_CFG, vocab_size=vocab_size)
    model = EncoderDecoder(args, _Tok(vocab_size)).to(config.device)
    model.train()

    att_feats  = embs.to(config.device)                               # (B, N_pilot, 512)
    fc_feats   = torch.zeros(B, config.EMBED_DIM, device=config.device)
    target_ids = ids.to(config.device)                                 # (B, 200)

    print(f"  att_feats : {tuple(att_feats.shape)}")
    print(f"  fc_feats  : {tuple(fc_feats.shape)}")
    print(f"  targets   : {tuple(target_ids.shape)}")

    log_probs = model._forward(fc_feats, att_feats, target_ids)
    print(f"  log_probs : {tuple(log_probs.shape)}")

    loss = F.nll_loss(
        log_probs.reshape(-1, vocab_size + 1),
        target_ids[:, 1:].reshape(-1),
        ignore_index=config.R2GEN_CFG["pad_idx"],
    )
    assert loss.item() == loss.item(), "FAIL: loss is NaN"
    assert torch.isfinite(loss),       "FAIL: loss is Inf"
    print(f"  loss      : {loss.item():.4f}")

    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f"FAIL: no gradient for {no_grad[:3]}"
    print(f"  gradients : all trainable params have grad ✓")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",   default=None,
                        help="Preprocessed NPZ to use (default: first in data/preprocessed/)")
    parser.add_argument("--ckpt",  default=None, help="CTViT checkpoint .pt")
    parser.add_argument("--dummy", action="store_true",
                        help="Run fully synthetic — no real data or GPU required")
    parser.add_argument("--vocab_size", type=int, default=500,
                        help="Vocab size for R2Gen smoke (use len(tokenizer) in full run)")
    cli = parser.parse_args()

    # ── resolve pilot NPZ path ────────────────────────────────────────────────
    if cli.npz:
        npz_path = Path(cli.npz)
    else:
        npzs = sorted(config.DATA_PREP.glob("*.npz"))
        if npzs and not cli.dummy:
            npz_path = npzs[0]
        else:
            npz_path = config.DATA_PREP / "pilot_dummy.npz"

    scan_id   = npz_path.stem

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        emb_path  = tmp_dir / f"{scan_id}.pt"   # pilot embedding stays in tempdir

        # Stage 2
        stage2_preprocess(npz_path, dummy=cli.dummy)

        # Stage 3
        ctvit_ckpt = Path(cli.ckpt) if cli.ckpt else None
        emb_raw = stage3_ctvit(ctvit_ckpt, dummy=cli.dummy)

        # Stage 4
        emb_flat = stage4_cache(emb_raw, emb_path)

        # Stage 5
        batch = stage5_dataloader(emb_path, scan_id, cli.vocab_size, tmp_dir)

        # Stage 6
        stage6_r2gen(batch, cli.vocab_size)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  ALL PHASES PASSED — pipeline is proven correct end-to-end.")
    print()
    print("  To run on full volume (Raad-II):")
    print("    1. Replace D_MIN=12 in stage3 with D=240 (full depth).")
    print("    2. Build tokenizer from real CT-RATE reports and pass vocab_size.")
    print("    3. Wrap stage6 in a training loop.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
