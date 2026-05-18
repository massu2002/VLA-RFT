#!/bin/bash
# Setup .venv_pro6000 for NVIDIA RTX PRO 6000 Blackwell (sm_120)
# Based on README.md installation steps, with the following changes:
#   - PyTorch 2.7.0+cu128  (sm_120 support)
#   - vLLM 0.9.2           (officially supports torch 2.7.0 + sm_120; verl handles >=0.7.0 natively)
#   - flash-attn 2.8.3     (torch 2.7 pre-built wheel)
#   - verl[vllm] skipped   (pins vllm==0.6.3 for torch 2.4 — replaced by manual vllm+tensordict install)
set -e

UV="${HOME}/.local/bin/uv"
VENV=".venv_pro6000"

echo "=== Step 1: Create virtual environment ==="
"$UV" venv --seed -p 3.10 "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel

echo "=== Step 2: Install PyTorch 2.7.0 with CUDA 12.8 (sm_120 support) ==="
# Specify +cu128 explicitly: uv resolves torch==2.7.0 to the cu126 wheel from PyPI otherwise.
"$UV" pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128"

echo "=== Step 3: Install shared base packages ==="
"$UV" pip install \
  numpy==1.26.4 \
  "fsspec>=2023.1.0,<=2026.2.0" \
  transformers==4.45.0 \
  timm==0.9.10 \
  sentencepiece==0.1.99 \
  accelerate>=0.25.0 \
  peft==0.11.1 \
  protobuf \
  einops \
  rich \
  matplotlib \
  jsonlines \
  huggingface_hub \
  draccus==0.8.0 \
  imageio \
  uvicorn \
  fastapi \
  wandb

echo "=== Step 4: Install verl (gpu extras only; vllm extra skipped — handled in Step 5) ==="
"$UV" pip install -e train/verl/".[gpu]"
"$UV" pip install -r train/verl/requirements.txt

echo "=== Step 5: Install vLLM 0.9.2 + tensordict (replaces verl[vllm] which pins vllm==0.6.3) ==="
# vLLM 0.9.2 officially requires torch==2.7.0 and includes sm_120 kernel support.
# verl's third_party/vllm/__init__.py handles vllm>=0.7.0 via direct import (no custom patches needed).
# tensordict is required by verl core; install without the <=0.6.2 cap since 0.6.x also works with torch 2.7.
"$UV" pip install vllm==0.9.2
"$UV" pip install "tensordict>=0.6.2"

echo "=== Step 6: Re-pin torch/transformers/numpy after verl/vllm installs ==="
# Must specify +cu128 explicitly so uv does not fall back to the cu126 wheel from PyPI.
"$UV" pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128"
"$UV" pip install \
  numpy==1.26.4 \
  "fsspec>=2023.1.0,<=2026.2.0" \
  transformers==4.45.0

echo "=== Step 7: Install flash-attn 2.8.3 (torch 2.7 + CUDA 12.x pre-built wheel) ==="
"$UV" pip install packaging ninja
"$UV" pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'

echo "=== Step 8: Install TensorFlow side (openvla-oft / dlimp) ==="
"$UV" pip install \
  tensorflow==2.15.0 \
  tensorflow-datasets==4.9.3 \
  tensorflow-graphics==2021.12.3

echo "=== Step 9: Install dlimp ==="
"$UV" pip install git+https://github.com/moojink/dlimp_openvla.git --no-deps

echo "=== Step 10: Install openvla-oft ==="
"$UV" pip install -e train/verl/vla-adapter/openvla-oft --no-deps

echo "=== Step 11: Install LIBERO ==="
"$UV" pip install -e third_party/LIBERO

echo "=== Step 12: Final packages and verification ==="
"$UV" pip install json-numpy

python - <<'PY'
import torch, torchvision, torchaudio, numpy, transformers
print("torch       =", torch.__version__)
print("torchvision =", torchvision.__version__)
print("torchaudio  =", torchaudio.__version__)
print("numpy       =", numpy.__version__)
print("transformers=", transformers.__version__)
print("CUDA available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         =", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("SM capability =", f"sm_{cap[0]}{cap[1]}")
try:
    import vllm
    print("vllm        =", vllm.__version__)
except Exception as e:
    print("vllm import error:", e)
PY

"$UV" pip check || echo "[WARN] Some dependency conflicts detected — review above"

echo "=== Done! Activate with: source .venv_pro6000/bin/activate ==="
