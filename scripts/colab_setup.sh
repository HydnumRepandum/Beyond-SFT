#!/usr/bin/env bash
set -euo pipefail

CONSTRAINTS_FILE="/tmp/dpo_colab_constraints.txt"

cat > "${CONSTRAINTS_FILE}" <<'EOF'
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
sentence-transformers==3.4.1
trl==0.9.6
EOF

python -m pip install --upgrade pip

# Keep Colab on a CUDA 12.4 PyTorch stack. Floating upgrades currently pull
# CUDA 13 wheels on some runtimes, which breaks bitsandbytes and torchcodec.
python -m pip install --upgrade --force-reinstall --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install --upgrade --no-cache-dir -c "${CONSTRAINTS_FILE}" unsloth
python -m pip install --upgrade --no-cache-dir -c "${CONSTRAINTS_FILE}" -r requirements.txt

# Colab images can carry a partially upgraded PyArrow stack. Force reinstall
# PyArrow after the other ML packages so datasets/parquet imports see one ABI.
python -m pip install --upgrade --force-reinstall --no-cache-dir "pyarrow>=16.1.0" "datasets>=2.20.0"

PIP_CONSTRAINT="${CONSTRAINTS_FILE}" python -m pip install -e .

# This project uses bitsandbytes 4-bit loading, not TorchAO quantization or
# audio/video decoding. These optional packages are fragile across Colab's
# Torch/CUDA image changes and can break imports before any project code runs.
python -m pip uninstall -y torchao torchcodec || true

python - <<'PY'
import datasets
import pyarrow
import pyarrow.parquet
import torch
import torchvision
import torchaudio
import unsloth

print(f"Verified datasets={datasets.__version__}, pyarrow={pyarrow.__version__}")
print(f"Verified torch={torch.__version__}, torchvision={torchvision.__version__}, torchaudio={torchaudio.__version__}")
print(f"Verified unsloth={getattr(unsloth, '__version__', 'unknown')}")
PY

echo "Colab environment setup complete."
echo "If this runtime already hit a PyArrow or Torch/Torchvision binary-compatibility error, restart the runtime before continuing."
