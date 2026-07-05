"""Central config — import this everywhere, never hardcode paths or device."""
import sys
from pathlib import Path

import torch

# ── device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── project paths ─────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_RAW      = ROOT / "data" / "raw"
DATA_PREP     = ROOT / "data" / "preprocessed"
EMBED_DIR     = ROOT / "embeddings" / "pilot"
CHECKPOINT_DIR= ROOT / "checkpoints"
LOG_DIR       = ROOT / "logs"

# ── source paths (cloned repos) ───────────────────────────────────────────────
CT2REP_ROOT   = ROOT / "src" / "CT2Rep"
CTVIT_SRC     = CT2REP_ROOT / "ctvit"        # contains ctvit/ package
CT2REP_SRC    = CT2REP_ROOT / "CT2Rep"       # contains modules/ and models/

for _p in (CTVIT_SRC, CT2REP_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── CTViT hyperparameters (matched to CT-CLIP_v2.pt checkpoint) ───────────────
# Input volume : (B, 1, 240, 480, 480) — (batch, channel, D, H, W)
# patch_size=20 → H_patches = W_patches = 480/20 = 24
# temporal_patch_size=10 → T_patches = 240/10 = 24
# N_tokens = 24^3 = 13,824   embed_dim = 512
# (CT2Rep's own config used patch_size=24/temporal_patch_size=12/dim_head=32
#  giving 8,000 tokens — incompatible with CT-CLIP_v2.pt weights)
CTVIT_CFG = dict(
    dim                = 512,
    codebook_size      = 8192,
    image_size         = 480,
    patch_size         = 20,
    temporal_patch_size= 10,
    spatial_depth      = 4,
    temporal_depth     = 4,
    dim_head           = 32,
    heads              = 8,
)
VOLUME_SHAPE  = (240, 480, 480)   # (D, H, W) stored in NPZ arr_0
_T = 240 // CTVIT_CFG["temporal_patch_size"]   # 24
_S = 480 // CTVIT_CFG["patch_size"]            # 24
N_TOKENS      = _T * _S * _S                   # 13 824
EMBED_DIM     = 512


def extract_ctvit_state(ckpt_path) -> dict:
    """Return a CTViT state_dict from either CT2Rep or CT-CLIP checkpoint format.

    CT2Rep format : raw CTViT state_dict, or {"model": state_dict}
    CT-CLIP format: full CLIP state_dict with keys prefixed "visual_transformer."
    """
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        state = state.state_dict()
    if "model" in state:
        return state["model"]
    if any(k.startswith("visual_transformer.") for k in state):
        return {k[len("visual_transformer."):]: v
                for k, v in state.items()
                if k.startswith("visual_transformer.")}
    return state

# ── R2Gen / CT2Rep decoder hyperparameters ────────────────────────────────────
# (confirmed from CT2Rep/CT2Rep/main.py)
R2GEN_CFG = dict(
    d_model        = 512,
    d_ff           = 512,
    d_vf           = 512,    # must match EMBED_DIM
    num_heads      = 8,
    num_layers     = 3,
    dropout        = 0.1,
    logit_layers   = 1,
    bos_idx        = 0,
    eos_idx        = 0,
    pad_idx        = 0,
    use_bn         = 0,
    drop_prob_lm   = 0.5,
    rm_num_slots   = 3,
    rm_num_heads   = 8,
    rm_d_model     = 512,
    max_seq_length = 200,
    sample_method  = "beam_search",
    beam_size      = 3,
    temperature    = 1.0,
    sample_n       = 1,
    group_size     = 1,
    output_logsoftmax = 1,
    decoding_constraint = 0,
    block_trigrams = 1,
)

# ── preprocessing constants ───────────────────────────────────────────────────
HU_MIN    = -1000.0
HU_MAX    =  200.0
NORM_BIAS =  400.0   # (HU + 400) / 600 → [-1, +1]
NORM_SCALE=  600.0
