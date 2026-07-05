# Shape Contract

Source of truth for every tensor boundary in the pipeline.
All shapes confirmed against actual repo source code before writing any test.

## Repos confirmed against

| Repo | File | Commit date |
|------|------|-------------|
| CT2Rep | `CT2Rep/models/ct2rep.py` | ibrahimethemhamamci/CT2Rep |
| CT2Rep | `CT2Rep/modules/visual_extractor.py` | ibrahimethemhamamci/CT2Rep |
| CT2Rep | `CT2Rep/modules/data_ct.py` | ibrahimethemhamamci/CT2Rep |
| CT2Rep | `CT2Rep/main.py` | ibrahimethemhamamci/CT2Rep |
| CT-CLIP | `scripts/run_train.py` | ibrahimethemhamamci/CT-CLIP |
| CT-CLIP | `transformer_maskgit/transformer_maskgit/ctvit.py` | ibrahimethemhamamci/CT-CLIP |

---

## Stage-by-stage contract

### Stage 0 — Raw NIfTI on disk

- Format: `.nii` or `.nii.gz`
- Values: Hounsfield Units (HU), typically −1024 to +3000
- Shape: variable per scan

### Stage 1 — Preprocessed NPZ (output of `preprocess.py`)

| Property | Value |
|----------|-------|
| File key | `arr_0` |
| Shape | `(240, 480, 480)` — **(D, H, W)** |
| Dtype | `float32` |
| Value range | **[−1.0, +1.0]** |
| Normalization | `clip(HU, −1000, 200)` then `(HU + 400) / 600` |

Assertion at save time:
```python
assert arr.shape == (240, 480, 480)
assert arr.dtype == np.float32
assert arr.min() >= -1.0 and arr.max() <= 1.0
```

### Stage 2 — DataLoader batch tensor (input to CTViT)

| Property | Value |
|----------|-------|
| Shape | `(B, 1, 240, 480, 480)` |
| Axes | batch, channel=1, D, H, W |
| Dtype | `float32` |
| Source | `arr_0` + `unsqueeze(0)` for channel dim |

### Stage 3 — CTViT encoder (confirmed from `ct2rep.py`)

**Config** (must match pretrained weights exactly):
```python
CTViT(
    dim                 = 512,
    codebook_size       = 8192,
    image_size          = 480,   # H = W
    patch_size          = 24,    # ph = pw = 24
    temporal_patch_size = 12,
    spatial_depth       = 4,
    temporal_depth      = 4,
    dim_head            = 32,
    heads               = 8,
)
```

**Patch arithmetic:**
- H patches = W patches = 480 / 24 = **20**
- T patches = 240 / 12 = **20**
- Total tokens = 20 × 20 × 20 = **8 000**

**Call signature for feature extraction:**
```python
tokens = ctvit(video, return_encoded_tokens=True)
# tokens.shape == (B, 20, 20, 20, 512)  — (B, T, H, W, D)
```

### Stage 4 — VisualExtractor output (confirmed from `visual_extractor.py`)

```python
# AvgPool3d(kernel_size=20) over (B, 512, 20, 20, 20)
att_feats.shape  == (B, 8000, 512)   # N_tokens=8000, D=512  → passed to R2Gen cross-attn
fc_feats.shape   == (B, 512)         # global avg-pool summary
```

### Stage 5 — Cached embedding file (output of `cache_embeddings.py`)

```python
# Saved with torch.save, loaded with torch.load
emb = torch.load("embeddings/pilot/scan_001.pt")
assert emb.shape == (1, 8000, 512)   # B=1 preserved for DataLoader collation
assert emb.dtype == torch.float32
```

### Stage 6 — Tokenized report (output of `dataset.py` / R2Gen Tokenizer)

| Property | Value |
|----------|-------|
| `token_ids` shape | `(B, max_seq_length)` — `max_seq_length = 200` |
| Vocab | built from CT-RATE training reports, `threshold = 3` |
| `<bos>` / `<eos>` / `<pad>` idx | all `0` (R2Gen default) |

### Stage 7 — R2Gen decoder (confirmed from `CT2Rep/main.py`)

**Config:**
```python
d_model        = 512
d_ff           = 512
d_vf           = 512   # must match EMBED_DIM=512
num_heads      = 8
num_layers     = 3
rm_num_slots   = 3
rm_d_model     = 512
max_seq_length = 200
```

**Call (train mode):**
```python
# att_feats: (B, 8000, 512)  fc_feats: (B, 512)  targets: (B, 200)
log_probs = model._forward(fc_feats, att_feats, targets)
# log_probs.shape == (B, 199, vocab_size+1)  ← seq cropped by 1 inside _forward
```

**Loss:**
```python
loss = F.nll_loss(
    log_probs.reshape(-1, vocab_size + 1),
    targets[:, 1:].reshape(-1),
    ignore_index=0,
)
assert loss.item() == loss.item()  # not NaN
```

### Stage 8 — Generated report (inference output)

```
string: "The lungs are clear bilaterally. No pleural effusion..."
```

---

## Key constants (single source of truth: `config.py`)

```python
VOLUME_SHAPE = (240, 480, 480)
N_TOKENS     = 8_000
EMBED_DIM    = 512
```
