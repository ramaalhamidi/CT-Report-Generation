# CT-Pipeline setup script — Windows (PowerShell)
# Run once: .\setup.ps1
# Requires: Python 3.10 on PATH, git on PATH

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

# ── 1. Virtual environment ────────────────────────────────────────────────────
Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Cyan
$EnvPath = "$ProjectDir\venv"
if (-not (Test-Path $EnvPath)) {
    python -m venv $EnvPath
} else {
    Write-Host "      venv already exists, skipping."
}

$pip = "$EnvPath\Scripts\pip.exe"
$python = "$EnvPath\Scripts\python.exe"

Write-Host "[1/4] Installing dependencies..." -ForegroundColor Cyan
& $pip install --upgrade pip
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
& $pip install numpy nibabel pydicom SimpleITK einops transformers timm scipy scikit-image pandas tqdm h5py tensorboard matplotlib vector-quantize-pytorch accelerate openpyxl

# ── 2. Model repos ───────────────────────────────────────────────────────────
Write-Host "[2/4] Cloning model repos into src\..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force "$ProjectDir\src" | Out-Null

# CT2Rep contains both the CTViT encoder and adapted R2Gen decoder
$CT2RepRepo = "https://github.com/ibrahimethemhamamci/CT2Rep"

if (-not (Test-Path "$ProjectDir\src\CT2Rep")) {
    git clone $CT2RepRepo "$ProjectDir\src\CT2Rep"
} else {
    Write-Host "      src\CT2Rep already exists, skipping."
}

# ── 3. Project folder structure ───────────────────────────────────────────────
Write-Host "[3/4] Creating project directories..." -ForegroundColor Cyan
@("data\raw", "data\preprocessed", "embeddings\pilot", "checkpoints", "logs") | ForEach-Object {
    New-Item -ItemType Directory -Force "$ProjectDir\$_" | Out-Null
}

# ── 4. Verify install ─────────────────────────────────────────────────────────
Write-Host "[4/4] Verifying PyTorch + CUDA..." -ForegroundColor Cyan
& $python -c @'
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available"
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda}")
print(f"  GPU     : {gpu}")
'@

Write-Host ""
Write-Host "Setup complete. Next steps:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python validate_pipeline.py"
