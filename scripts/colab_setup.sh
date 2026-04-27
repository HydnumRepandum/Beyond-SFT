#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install unsloth
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Colab environment setup complete."
