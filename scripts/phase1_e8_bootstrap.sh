#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/boltzmann-attention}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/seka}"
SEKA_DIR="$ROOT_DIR/external/SEKA"
DATA_DIR="$SEKA_DIR/data"
DATA_TMP_DIR="${DATA_TMP_DIR:-/scratch/boltzmann/tmp/SEKA-datasets}"
HF_DATASET_REPO="${HF_DATASET_REPO:-https://huggingface.co/datasets/waylonli/SEKA-datasets}"
SEKA_REPO="${SEKA_REPO:-https://github.com/waylonli/SEKA.git}"
if [ -d /home/azureuser/.cache/huggingface ]; then
  export HF_HOME="${HF_HOME:-/home/azureuser/.cache/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/azureuser/.cache/huggingface}"
fi

echo "[phase1_e8_bootstrap] root=$ROOT_DIR"
echo "[phase1_e8_bootstrap] venv=$VENV_DIR"

mkdir -p "$(dirname "$ROOT_DIR")" "$(dirname "$VENV_DIR")"

if ! command -v git-lfs >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git-lfs
fi

if [ ! -d "$ROOT_DIR/.git" ] && [ ! -f "$ROOT_DIR/README.md" ]; then
  echo "[phase1_e8_bootstrap] missing synced workspace at $ROOT_DIR" >&2
  exit 1
fi

if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  if ! python3 -m venv "$VENV_DIR"; then
    apt-get update
    apt-get install -y python3-venv python3.10-venv
    python3 -m venv "$VENV_DIR"
  fi
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

python -m pip install \
  torch==2.7.0 \
  transformers==4.51.3 \
  datasets==3.5.1 \
  accelerate==1.11.0 \
  dataclasses_json==0.6.7 \
  nltk==3.9.4 \
  spacy==3.8.14 \
  anchoring==0.1.0 \
  evaluation==0.0.2 \
  ipdb==0.13.13 \
  matplotlib==3.8 \
  numpy==1.26.4 \
  scikit_learn==1.5.2 \
  scipy==1.13.1 \
  tqdm==4.67.1 \
  wget==3.2 \
  huggingface_hub

python -m spacy download en_core_web_sm
python - <<'PY'
import nltk
nltk.download("punkt_tab")
PY

mkdir -p "$ROOT_DIR/external"
if [ ! -d "$SEKA_DIR/.git" ]; then
  git clone "$SEKA_REPO" "$SEKA_DIR"
else
  git -C "$SEKA_DIR" fetch --all --tags
  git -C "$SEKA_DIR" pull --ff-only
fi

mkdir -p "$(dirname "$DATA_TMP_DIR")" "$DATA_DIR"
if [ ! -d "$DATA_TMP_DIR/.git" ]; then
  rm -rf "$DATA_TMP_DIR"
  git clone "$HF_DATASET_REPO" "$DATA_TMP_DIR"
else
  git -C "$DATA_TMP_DIR" fetch --all --tags
  git -C "$DATA_TMP_DIR" pull --ff-only
fi
git -C "$DATA_TMP_DIR" lfs install
git -C "$DATA_TMP_DIR" lfs pull

cp -a "$DATA_TMP_DIR"/. "$DATA_DIR"/

cd "$ROOT_DIR"
python scripts/phase1_seka_env_audit.py
echo "[phase1_e8_bootstrap] done"
