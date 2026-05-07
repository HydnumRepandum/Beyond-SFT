#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install --upgrade --no-cache-dir unsloth
python -m pip install --upgrade --no-cache-dir -r requirements.txt
# Colab images can carry a partially upgraded PyArrow stack. Force reinstall
# PyArrow after the other ML packages so datasets/parquet imports see one ABI.
python -m pip install --upgrade --force-reinstall --no-cache-dir "pyarrow>=16.1.0" "datasets>=2.20.0"
python -m pip install -e .

python - <<'PY'
import datasets
import pyarrow
import pyarrow.parquet

print(f"Verified datasets={datasets.__version__}, pyarrow={pyarrow.__version__}")
PY

echo "Colab environment setup complete."
echo "If this runtime already hit a PyArrow binary-incompatibility error, restart the runtime before continuing."
