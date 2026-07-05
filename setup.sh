#!/usr/bin/env bash
# CT-Pipeline setup script — Linux/HPC (Raad-II)
# Run once after cloning the project: bash setup.sh
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Conda environment ──────────────────────────────────────────────────────
echo "[1/4] Creating conda environment from environment.yml..."
conda env create -f "$PROJECT_DIR/environment.yml" || conda env update -f "$PROJECT_DIR/environment.yml"
echo "      Done. Activate with: conda activate ctvit-val"

# ── 2. Model repos ───────────────────────────────────────────────────────────
echo "[2/4] Cloning model repos into src/..."
mkdir -p "$PROJECT_DIR/src"

# CT2Rep contains both the CTViT encoder and adapted R2Gen decoder
CT2REP_REPO="https://github.com/ibrahimethemhamamci/CT2Rep"

if [ ! -d "$PROJECT_DIR/src/CT2Rep" ]; then
    git clone "$CT2REP_REPO" "$PROJECT_DIR/src/CT2Rep"
else
    echo "      src/CT2Rep already exists, skipping."
fi

# ── 3. Project folder structure ───────────────────────────────────────────────
echo "[3/4] Creating project directories..."
mkdir -p "$PROJECT_DIR/data/raw"
mkdir -p "$PROJECT_DIR/data/preprocessed"
mkdir -p "$PROJECT_DIR/embeddings/pilot"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/logs"

# ── 4. Verify install ─────────────────────────────────────────────────────────
echo "[4/4] Verifying PyTorch + CUDA..."
conda run -n ctvit-val python - <<'EOF'
import torch
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda}")
print(f"  GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
EOF

echo ""
echo "Setup complete. Next steps:"
echo "  conda activate ctvit-val"
echo "  python validate_pipeline.py"
