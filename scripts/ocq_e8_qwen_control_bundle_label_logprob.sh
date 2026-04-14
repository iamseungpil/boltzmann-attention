#!/usr/bin/env bash
set -euo pipefail

SCORING_MODE=label_logprob LOGPROB_NORMALIZE="${LOGPROB_NORMALIZE:-mean}" \
  bash "$(dirname "$0")/ocq_e8_qwen_control_bundle.sh" "$@"
