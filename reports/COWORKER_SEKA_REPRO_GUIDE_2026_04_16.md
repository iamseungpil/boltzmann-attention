# SEKA Reproduction on A100 — Coworker Guide

**대상**: A100 보유 coworker
**작성**: 2026-04-16 03:35 KST
**목적**: 우리 RTX A6000 환경에서 SEKA paper 수치 재현 실패 → **A100 에서 정확 재현 여부 검증**

---

## TL;DR (3-line summary)

1. 우리 A6000 (cc 8.6) 환경에서 SEKA canonical ES=0.952 재현 **실패** (0.374 나옴)
2. 환경 (torch 2.3.0+cu121, transformers 4.51.3, Qwen3-4B-Base) 다 맞춰도 실패
3. **A100 (cc 8.0) 에서 돌려보고 0.952 나오는지 확인 요청** — 나오면 hardware 원인 확정, 안 나오면 다른 원인 isolated

---

## 1. Phase 1 — SEKA Canonical CounterFact 재현 (1 시간 이내)

### 1.1 Environment setup

```bash
# 1. venv (paper-exact)
python3.12 -m venv ~/venvs/seka_paper_exact
~/venvs/seka_paper_exact/bin/pip install --upgrade pip setuptools wheel

# 2. Paper-exact torch (다른 버전 쓰면 안 됨)
~/venvs/seka_paper_exact/bin/pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Remaining deps (numpy/scikit pin 은 python 3.12 호환성 이슈로 최신화)
~/venvs/seka_paper_exact/bin/pip install \
  transformers==4.51.3 accelerate==1.11.0 \
  dataclasses_json==0.6.7 datasets==3.5.1 \
  'numpy<2' scikit-learn scipy matplotlib \
  nltk==3.9.1 pastalib==0.1.3 anchoring==0.1.0 \
  tqdm evaluation tokenizers
```

**Verify**:
```bash
~/venvs/seka_paper_exact/bin/python -c "
import torch, transformers
print('torch:', torch.__version__)          # 2.3.0+cu121
print('transformers:', transformers.__version__)  # 4.51.3
print('cuda:', torch.cuda.is_available(), torch.version.cuda)  # 12.1
import pynvml; pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print('GPU:', pynvml.nvmlDeviceGetName(h))
# compute_cap via torch
print('compute_cap:', torch.cuda.get_device_capability(0))
"
```

### 1.2 Data download (SEKA-datasets on HuggingFace)

```bash
cd ~/workspace_common/boltzmann-attention
~/venvs/seka_paper_exact/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='waylonli/SEKA-datasets', repo_type='dataset',
                       allow_patterns=['pasta_bench/*', 'synthetic/*'])
print('downloaded to:', p)
"
```

이미 repo 에 `external/SEKA/data/pasta_bench/` 는 symlink/copy 로 세팅되어 있음 (git pull 하면 따라옴). 만약 없으면:
```bash
DS=$(ls -d ~/.cache/huggingface/hub/datasets--waylonli--SEKA-datasets/snapshots/*/ | head -1)
PB=external/SEKA/data/pasta_bench
for f in counterfact.jsonl counterfact.json attribute_snippets.json tfidf_vocab.json; do
  [ -f "$PB/$f" ] || ln -sfn "$DS/pasta_bench/$f" "$PB/$f"
done
```

`idf.npy` 는 이미 repo 에 있음 (`external/SEKA/data/pasta_bench/idf.npy`).

### 1.3 Model download

```bash
~/venvs/seka_paper_exact/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Base')
"
```

**확인 대상 model snapshot**: `906bfd4b4dc7f14ee4320094d8b41684abff8539`
- 이것이 우리가 테스트한 유일한 snapshot. coworker 환경에서도 같은 hash 나와야 함.
- 다른 hash 나오면 Qwen 모델 weights 가 변경된 것이므로 그 자체가 재현 실패 원인 가능.

### 1.4 **THE critical command**

```bash
cd ~/workspace_common/boltzmann-attention/external/SEKA
export PYTHONPATH=$PWD:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 ~/venvs/seka_paper_exact/bin/python benchmarks/eval_fact_gen.py \
  --model Qwen/Qwen3-4B-Base \
  --data_path data/pasta_bench \
  --output_dir ~/workspace_common/boltzmann-attention/reports/seka_repro_A100_2026_04_16/counterfact_qwen3_4b \
  --overwrite_output_dir \
  --example_subset=:500 \
  --benchmarks efficacy paraphrase \
  --add_unmediated_fact True \
  --batch_size 8 \
  --max_new_tokens 64 \
  --seka \
  --pos seka_projections/counterfact/Qwen3-4B-Base/Qwen3-4B-Base_pos_proj.pt \
  --neg seka_projections/counterfact/Qwen3-4B-Base/Qwen3-4B-Base_neg_proj.pt \
  --amplify_pos 1.0 \
  --amplify_neg 0.8 \
  --layers last10 \
  --add_marker
```

**Expected runtime**: ~3 분 (N=500 efficacy + N=5000 paraphrase).

---

## 2. Success Criterion (A100 재현 검증)

### 2.1 핵심 체크

| Metric | Expected (paper ref) | Our A6000 | A100 expected |
|---|---|---|---|
| **efficacy ES** | **0.952** | **0.374** ❌ | ??? (이게 핵심) |
| paraphrase PS | 0.962 | 0.428 ❌ | ??? |
| baseline (no-SEKA) ES | 0.402 | 0.406 ✓ | should be 0.402 ✓ |

**Primary verification**:
```bash
cat ~/workspace_common/boltzmann-attention/reports/seka_repro_A100_2026_04_16/counterfact_qwen3_4b/efficacy_metrics.json
# score.mean should be ~0.95 (±0.02)
```

### 2.2 결과 3 가지 시나리오

**A. A100 에서 ES=0.952 나옴** → **hardware 가 원인 확정**
- A6000 (cc 8.6) 의 bf16/TF32 Tensor Core 가 SEKA steering 에서 numerical 편차 유발
- Paper claim: "We note reproducibility depends on A100-class hardware; RTX A6000 fails to reproduce under identical software stack."
- MetaTool 비교는 A100 에서 계속 진행해야 canonical

**B. A100 에서도 ES ≠ 0.952 (예: 0.4–0.7)** → hardware 원인 아님, 더 숨은 변수 있음
- HF model snapshot 확인 (906bfd4b 맞는지)
- Pre-built P_pos 파일 content 확인
- SEKA author 에게 issue 로 재현 재현 환경 세부사항 요청

**C. A100 에서 원인 불명 오류 발생** → environment 문제, 재셋업

---

## 3. Phase 2 — Canonical SEKA on MetaTool (Phase 1 성공 조건)

### 3.1 Purpose
Phase 1 에서 canonical SEKA 재현 성공하면, 같은 A100 환경에서 **MetaTool Subtask4 (multi-tool) 에 canonical SEKA 적용** → 우리 Q-coverage 와 head-to-head 비교.

### 3.2 문제: MetaTool 에 대한 canonical SEKA direction 부재

SEKA 는 원래 CounterFact/BiasBios (single-concept editing) benchmark 만 지원. MetaTool multi-tool task 용 contrastive pair 가 prior work 에 없음.

### 3.3 Option 3A — MetaTool 용 canonical SEKA direction build

SEKA 의 `synthetic_qa_builder.py` 를 MetaTool 데이터에 적용:

```bash
cd external/SEKA

# MetaTool contrastive pair 생성 (우리가 작성)
# scripts/diagnostics_2026_04_16/build_metatool_contrastive_pairs.py 참조

~/venvs/seka_paper_exact/bin/python src/custom_builders/synthetic_qa_builder.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data ~/workspace_common/boltzmann-attention/reports/metatool_contrastive_pairs.jsonl \
  --output_dir seka_projections/metatool-qwen25-7b \
  --max_samples 500 \
  --min_diff 0.20 \
  --top_pct 0.90
```

이후:
```bash
CUDA_VISIBLE_DEVICES=0 ~/venvs/seka_paper_exact/bin/python \
  ~/workspace_common/boltzmann-attention/scripts/ocq/eval_subtask4_with_real_seka.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda:0 \
  --b-ont external/SEKA/seka_projections/metatool-qwen25-7b/Qwen2.5-7B-Instruct_pos_proj.pt \
  --max-samples 497 --amplify 1.0 3.0 5.0 --max-new-tokens 256 \
  --attn-impl eager \
  --out ~/workspace_common/boltzmann-attention/reports/canonical_seka_metatool_A100/qwen_st4.json
```

**Expected runtime**: Phase 2A 전체 ~4-6 시간 (pair gen ~1h, SVD build ~1h, eval 3 amp × 3h = 6h 총).

### 3.4 Option 3B — AdaSEKA canonical (less work)

우리가 이미 `external/SEKA/seka_projections/adaseka-qwen25-7b-metatool/` 에 per-facet B_f 를 expert list 형태로 빌드해 둠. Canonical AdaSEKA operator 바로 사용 가능.

```bash
cd ~/workspace_common/boltzmann-attention
mkdir -p reports/adaseka_canonical_A100/
CUDA_VISIBLE_DEVICES=0 ~/venvs/seka_paper_exact/bin/python scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda:0 \
  --expert-path external/SEKA/seka_projections/adaseka-qwen25-7b-metatool/expert_paths.json \
  --amplify 1.0 3.0 5.0 \
  --layers last10 --max-new-tokens 256 \
  --out reports/adaseka_canonical_A100/qwen_st4.json
```

**Expected runtime**: ~2-3 시간 (Qwen + Llama 두 모델 × 3 amp).

**권장**: 3B 먼저 (Phase 1 success 가정), 3A 는 여유 시 추가.

---

## 4. 우리가 이미 돌린 것 (A6000 environment, 참조용)

### 4.1 Env verified identical to paper
- torch 2.3.0+cu121 ✓
- transformers 4.51.3 ✓
- SEKA commit `679149e` (release) ✓

### 4.2 Our reproduction failures
| Condition | ES |
|---|---|
| baseline (no SEKA) | 0.406 ✓ matches ref 0.402 |
| SEKA amp=1.0 bf16 | 0.374 ❌ |
| SEKA amp=1.0 fp32 | 0.400 ❌ |
| SEKA amp=1.0 TF32 off | 0.390 ❌ |
| SEKA amp=5.0 | 0.416 ❌ |
| SEKA amp=50.0 | 0.398 ❌ |
| **Reference (paper)** | **0.952** |

### 4.3 Key diagnostic observations
- Generation byte-identical to reference (greedy deterministic)
- Baseline log-probs match reference within 0.14 nats
- SEKA steering log-probs drift 5+ nats from reference
- Hook attaches on 10 layers as expected
- Pre-built P_pos file has only layer 35 non-zero (as reference has)
- Amp scaling (1→5→50) doesn't increase effect proportionally

---

## 5. 필요한 reporting back

A100 에서 Phase 1 실행 후 다음 정보 주세요:

1. **`efficacy_metrics.json` score.mean** — 0.95 에 가까운지 (재현 성공) 또는 0.40 에 가까운지 (실패)
2. **GPU 정보**: `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv`
3. **`efficacy.json` 의 sample 0** — prompt/generation/target_score/comparator_score
   - Sample 0 target_score 가 −12.58 (ref) 또는 −17.7 (우리) 근처인지
4. **실제 GPU 사용된 MBs**: `torch.cuda.max_memory_allocated()` (optional)

5분이면 알 수 있음 (Phase 1 3분 실행 + 값 확인 2분).

---

## 6. File 포인터 (이 repo 에 모두 들어있음)

### Code
- `external/SEKA/` — canonical SEKA source (release 679149e)
- `external/SEKA/benchmarks/eval_fact_gen.py` — 주 eval entry
- `external/SEKA/src/model/seka_llm.py` — SEKA operator (k_norm hook)
- `scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py` — our AdaSEKA canonical eval (Phase 2B)
- `scripts/diagnostics_2026_04_16/build_adaseka_experts_from_bont.py` — per-facet B_f → AdaSEKA expert list

### Data (already in repo via symlinks)
- `external/SEKA/data/pasta_bench/{counterfact.jsonl, attribute_snippets.json, tfidf_vocab.json, idf.npy}`
- `external/SEKA/seka_projections/counterfact/Qwen3-4B-Base/{Qwen3-4B-Base_pos_proj.pt, Qwen3-4B-Base_neg_proj.pt}`
- `external/SEKA/seka_projections/adaseka-qwen25-7b-metatool/expert_paths.json` — ready for Phase 2B
- `external/SEKA/seka_projections/adaseka-llama31-8b-metatool/expert_paths.json` — ready for Phase 2B

### Reference results (SEKA authors, pre-computed)
- `external/SEKA/benchmarks/counterfact/results/seka-qwen3-4b-500/efficacy_metrics.json` — ES=0.9520
- `external/SEKA/benchmarks/counterfact/results/baseline-qwen3-4b-500/efficacy_metrics.json` — ES=0.4020

### Our failed reproduction logs
- `logs/seka_repro_2026_04_16/counterfact_qwen3_4b_torch230.log`
- `reports/seka_repro_2026_04_16/counterfact_qwen3_4b_torch230/` — full outputs (generations, scores)
- `reports/seka_repro_2026_04_16/counterfact_qwen3_4b_baseline/` — our baseline (matches ref)

---

## 7. 최악 시나리오 대비

만약 A100 에서도 SEKA 재현 실패 (Scenario B):
- SEKA author 에게 GitHub issue 또는 email
- "Same env, baseline matches, SEKA steering 미작동" 이라는 diagnostic 첨부
- 그 동안 우리 paper 는 "SEKA reference 수치 (ES=0.952) 를 published baseline 으로 인용" 으로 진행

만약 A100 에서 성공 (Scenario A):
- 그 환경에서 Phase 2 (AdaSEKA canonical on MetaTool) 진행
- 결과가 우리 Q-coverage 결과와 direct comparable
- Paper §5.5.3.1 완성

---

## 8. 우리 투자 시간 (기록용)

**2026-04-16 01:00–03:35 (약 2h 30m)**:
- SEKA install + data/model download: 20 min
- torch 2.3 venv 구축: 25 min
- 재현 시도 × 6 (torch 2.7/2.3, fp32, TF32 off, amp 1/5/50): 40 min
- Hook instrumentation + diagnostic: 45 min
- 분석 + 문서 작성: 20 min

A100 에서는 Phase 1 단독 5–10 분이면 해결. Request urgent.
