#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${1:-/home/v-seungplee/boltzmann-attention}"
REMOTE_ROOT="${2:-/scratch/boltzmann/boltzmann-attention}"

AZ_PYTHON="${AZ_PYTHON:-/opt/az/bin/python3}"
CONNECTOR="${CONNECTOR:-$HOME/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
URL="${URL:-wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms}"
SSH_BASE=(
  ssh -T
  -o ConnectTimeout=20
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL"
  -i "$SSH_KEY"
  azureuser@placeholder
)

echo "[phase1_sync_to_e8] syncing $LOCAL_ROOT -> $REMOTE_ROOT"
"${SSH_BASE[@]}" "mkdir -p '$REMOTE_ROOT'"

tar \
  --exclude='.git' \
  --exclude='results' \
  --exclude='tmp_remote_results' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -C "$LOCAL_ROOT" -cf - . \
| "${SSH_BASE[@]}" "tar -C '$REMOTE_ROOT' -xf -"

echo "[phase1_sync_to_e8] done"
