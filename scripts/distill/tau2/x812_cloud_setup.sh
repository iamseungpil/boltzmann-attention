#!/usr/bin/env bash
# x812_cloud_setup — 클라우드 인스턴스에 우리 vLLM 스택을 세운다 (계획서 x812 §3).
# 사용:  bash x812_cloud_setup.sh <REPO_GIT_URL>
# ⛔키는 절대 인자로 받지 않는다 — 파일로만 주입한다([[30]] 2026-06-16 유출 사고).
set -eu
REPO_URL="${1:?repo git url}"
BASE="$HOME"
TB="$BASE/tau2-bench"
REPO="$BASE/boltzmann-attention-pi"
T2="$REPO/scripts/distill/tau2"
TAU2_SHA="fc0055d"                 # ★로컬 iso_tau3 와 동일(clean clone·수정 0)
VLLM_VER="0.27.1"                  # ★로컬 실측
MODEL="Qwen/Qwen3.8-27B-FP8"

echo "=== [1/7] 시스템 확인"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python3 --version; df -h "$BASE" | tail -1

echo "=== [2/7] 엔진 venv (vllm==$VLLM_VER)"
python3 -m venv "$BASE/venv_vllm"
"$BASE/venv_vllm/bin/pip" -q install --upgrade pip
"$BASE/venv_vllm/bin/pip" -q install "vllm==$VLLM_VER" huggingface_hub
"$BASE/venv_vllm/bin/python" -c "import vllm,torch;print(' vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"

echo "=== [3/7] tau2-bench (upstream @ $TAU2_SHA · 로컬과 동일)"
[ -d "$TB" ] || git clone -q https://github.com/sierra-research/tau2-bench.git "$TB"
cd "$TB" && git checkout -q "$TAU2_SHA"
grep -q '"alltools"' src/tau2/domains/banking_knowledge/retrieval.py \
  && echo "  alltools 변형 등재 확인 ✅" || { echo "  ⛔alltools 없음 — 중단"; exit 1; }

echo "=== [4/7] 러너 venv"
python3 -m venv "$TB/venv"
"$TB/venv/bin/pip" -q install --upgrade pip
"$TB/venv/bin/pip" -q install -e "$TB" litellm

echo "=== [5/7] repo"
[ -d "$REPO" ] || git clone -q -b facet-rft-2026 "$REPO_URL" "$REPO"
cd "$REPO" && echo "  $(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD)"

echo "=== [6/7] 모델 (29GB · 10~20분)"
"$BASE/venv_vllm/bin/huggingface-cli" download "$MODEL" --quiet
echo "  받음"

echo "=== [7/7] 디렉터리"
mkdir -p "$BASE/logs" "$BASE/q"
echo
echo "다음: ⛔키를 손으로 넣는다 (인자·커밋 금지)"
echo "   printf 'export OPENROUTER_API_KEY=sk-...\n' > ~/.openrouter_key && chmod 600 ~/.openrouter_key"
echo "그다음: bash $T2/x812_cloud_engine.sh   → bash $T2/x812_cloud_gates.sh"
