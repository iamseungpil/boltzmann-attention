# 🌙 Coworker 긴급 실험 요청 — 2026-04-15 Night Sprint

**발신**: develop side (mais)
**시간**: 2026-04-15 17:30 KST (last update 20:35 KST)
**대상**: A100×4 보유 coworker
**Deadline**: **NeurIPS 2026 abstract 2026-05-04 / full paper 2026-05-06 (D-19 ~ D-21)** ⚠️ 정정: 5/15 아님
**목적**: NeurIPS 2026 main-track 진입 — 현재 6.3/10 → 목표 6.8-7.0/10 (accept 60-70%)

---

## 🚨 [2026-04-15 21:00 update] CRITICAL FIX — SEKA hang/gibberish ROOT CAUSE

**우리 develop side 에서 SEKA 가 2 회 hang + 4 회 gibberish 를 낸 root cause 최종 확정** (T1-T8 isolation):

### Real root cause: `SEKALLM` 의 자동 multi-GPU sharding

`external/SEKA/src/model/seka_llm.py:32` 에 다음 코드:
```python
multi_gpu = torch.cuda.device_count() > 1 and str(device).startswith("cuda")

if multi_gpu:
    self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_or_path, device_map="auto", **hf_kwargs
    ).eval()
```

→ **`torch.cuda.device_count() > 1` 이면 자동으로 `device_map="auto"` 로 sharding**.
→ 우리 시스템 (2× A6000) + cuda:0 가 다른 wave 점유 중 → memory pressure → **silent decoder collapse** (gibberish: `"halves halves halves..."`, `"!!! hue modelBuilderkey..."` 등) + **eager attention path 가 100x slow** (sharding 시 cross-GPU communication overhead).

A100×4 환경에서도 같은 문제 발생 가능 — coworker GPU 다중 + 동일 노드에서 다른 wave 점유 중일 때.

### ✅ 검증된 Fix (T8): `CUDA_VISIBLE_DEVICES=<single_gpu>` 강제

```bash
# 단 하나의 GPU 만 보이게 하면 sharding 안 함
CUDA_VISIBLE_DEVICES=1 python scripts/ocq/eval_subtask4_with_real_seka.py \
    --device cuda:0 \   # ← CVD=1 후 visible GPU 는 cuda:0 으로 remapping
    --model NousResearch/Meta-Llama-3.1-8B-Instruct \
    --b-ont external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt \
    ...
```

T8 검증 결과 (SEKA + Llama-3.1-8B + CVD=1, eager+bf16):
- T8.1 vanilla: 1.6 초 + 정상 출력 `'I would choose the following two tools... <tool_call>{"name": "NewsTool", ...}'`
- T8.2 SEKA steered: 1.6 초 + 정상 출력 `'I will choose the following two tools... <tool_call>{"name": "NewsTool", ...}'`

→ Sharding 비활성화하면 SEKA 정상 동작. 어떤 transformers 버전 (4.51.3 / 5.4.0) 이든 상관 없음.

### Secondary issue: Qwen2.5 + eager + SWA

T2 결과 (SEKA + Qwen2.5-7B-Inst + CVD=2, eager+bf16): **여전히 gibberish**.
이유: Qwen2.5 가 Sliding Window Attention 사용. SEKA 가 eager attention 강제 → SWA 미구현 → broken outputs.
경고 메시지: `"Sliding Window Attention is enabled but not implemented for `eager`; unexpected results may be encountered."`

**Qwen2.5 SEKA 를 위한 추가 fix**: `attn_implementation="sdpa"` 또는 SEKA 의 hook 이 eager 만 지원하면 Qwen2.5 는 **out of scope** 표시 (Llama 만 SEKA 비교 가능).

### ✅ 우리가 사용한 venv 설정 (transformers pin 도 권장 — secondary, primary 는 위 fix)

```bash
mkdir -p /home/<user>/venvs
python3.12 -m venv /home/<user>/venvs/seka_env
/home/<user>/venvs/seka_env/bin/python -m pip install \
    torch==2.7.0 \
    transformers==4.51.3 \
    accelerate==1.11.0 \
    tokenizers==0.21.1 \
    safetensors==0.5.3 \
    huggingface-hub==0.30.2 \
    numpy==1.26.4 sentencepiece protobuf scikit-learn matplotlib nltk scipy \
    spacy wget dataclasses_json ipdb pastalib regex tqdm
```

### ⚠️ Anti-patterns (동일 mistake 회피)

❌ Default `python` 이 transformers 5.x 일 수 있음 → 그래도 **위 root cause 는 transformers 무관**
❌ Multi-GPU 환경에서 `--device cuda:0` 만 지정 → SEKA 가 무시하고 sharding 함 → **반드시 `CUDA_VISIBLE_DEVICES`**
❌ 첫 generate output 의 quality check 안 함 (gibberish 면 즉시 GPU sharding 의심)
❌ 점유된 GPU 위에 SEKA load → memory pressure 로 silent gibberish

### Quick sanity check before full eval

```bash
# 1 sample 1 gen 으로 정상성 확인 (10 초 안에 완료되어야 함)
CUDA_VISIBLE_DEVICES=1 python -c "
import sys; sys.path.insert(0, 'external/SEKA'); sys.path.insert(0, 'scripts/ocq')
from src.model.seka_llm import SEKALLM
from eval_metatool_subtask4 import build_fc_prompt
from eval_metatool_subtask1 import parse_candidates
import torch, json, time
seka = SEKALLM('NousResearch/Meta-Llama-3.1-8B-Instruct', device='cuda:0',
               pos_pt='/tmp/seka_p_pos_llama_debug.pt', layers='last10',
               torch_dtype=torch.bfloat16, attn_implementation='eager')
data = json.load(open('/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json'))
e = data[0]; cands = parse_candidates(e['action_prompt'])
prompt = build_fc_prompt(seka.tok, e['action_prompt'], cands)
ids = seka.tok(prompt, return_tensors='pt').to(seka.device)
t0 = time.time()
out = seka.model.generate(**ids, max_new_tokens=32, do_sample=False, pad_token_id=seka.tok.eos_token_id)
print(f'time={time.time()-t0:.1f}s')
print('out:', seka.tok.decode(out[0, ids[\"input_ids\"].shape[1]:], skip_special_tokens=True)[:150])
# Expected: time<3s, output starts with 'I would choose the following...' or '<tool_call>{...'
# If: time>30s OR repetitive gibberish ('halves halves' / '!!! ...') → CUDA_VISIBLE_DEVICES not set
"
```

## ⚡ 현재 상황 — 왜 긴급한가

Develop side (A6000×2) 는 주간 내내 16 시간 run 중. 밤샘 sprint 로 2-3 cell 추가 가능하지만, **4 개 high-leverage 실험** 이 필요. A100×4 병렬 실행 시 coworker 단독으로 3-4 일 내 완료 가능 — 이것이 NeurIPS submission 시점 최대 변수.

## 🎯 4 개 P0 실험 (ROI 순서)

### 🥇 P0-A — SEKA & AdaSEKA head-to-head (기여 +0.4, 1-2 day)

**문제**: 우리 논문 §2.1 에서 SEKA (Li 2026 ICLR) 를 closest prior art 로 인정. 수식 $k' = k + gPk$ 동일. 단 *head-to-head 비교 표 부재* → reviewer 가 §5 에서 즉시 감지 → reject 리스크.

**우리 자체 SEKA 실행 시도 (2 회)**:
- 첫 번째 (20 분): hang, GPU0 1% 활용
- 두 번째 (15 분, micro N=5): 같은 hang
- 원인 의심: FlashAttention vs SEKA hook 비호환, 또는 eager mode 에서도 encode_with_markers slow path

**coworker 요청**:
1. A100×1 에서 `external/SEKA/src/model/seka_llm.py` 와 `adaptive_seka_llm.py` 를 **original 그대로** 사용 (프록시 작성 금지 — 우리 4월 14일 사고 reference: `memory/feedback_external_baseline_use_original_source.md`).
2. B_ont → P_pos 변환 wrapper: `scripts/ocq/eval_subtask4_with_real_seka.py` 참조 (convert_bont_to_ppos 함수).
3. 평가 target:
   - **MetaTool Subtask1** (N=995, Qwen2.5-7B-Instruct): no_steer + SEKA amplify_pos ∈ {1, 2, 5} + AdaSEKA M=2/3 × T=0.1
   - **MetaTool Subtask4** (N=497, Qwen2.5-7B-Instruct): 동일 sweep
   - **Cross-model**: 시간 여유 시 Llama-3.1-8B-Instruct 도 같은 sweep
4. Steer_mask: user query span 을 `**...**` markers 로 감싸서 자동 추출 (encode_with_markers)
5. Output: `reports/baselines/seka/{subtask}_{model}_amp{N}.json` 형식

**예상 GPU-hr**: 6-8h (all cells × 2 models 가정, A100 속도로 3-4h wall-clock).

### 🥈 P0-B — Thm 6.18 attention-weighted bit allocation full WT2 PPL (기여 +0.4, 1-2 day)

**문제**: §3.6.2 Thm 6.18 는 예측 $-2.5$ PPL only, empirical 측정 zero. Unified Pareto 의 4 leg 중 마지막 leg — 닫으면 reviewer 의 "3/4 full" 비판 해소.

**Develop side 가 작성한 코드**: `scripts/ocq/measure_thm618_attn_weighted_bits.py` (오늘 17:30 KST commit). 현재 allocation-only (b*(t,f) 계산). **PPL eval 파트는 미구현** — coworker 가 다음을 추가해야:

1. Calibration forward (우리 코드로 가능) → b*(t,f) 추출
2. **Variable-bit quantizer hook** on k_proj output:
   - Per-(t, f, d) bit count 로 asymmetric min-max quantization
   - Existing `scripts/ocq/quantizer.py` 의 `ocq_kivi_quantize_head` 패턴 재사용
3. WT2 PPL: `math/experiment/exp4_2_v3_full_quant_ppl.py` 패턴 (ctx=2048 non-overlap, 전체 test set)
4. Target sweep: avg_bits ∈ {1.81, 2.0, 2.5, 3.0, 4.0}
5. Baselines (동일 protocol): KIVI 2.00, OCQ 1.81 (이미 측정)

**목표 숫자**: OCQ 1.81 현재 PPL 15.60. Thm 6.18 예측: **12.5-13.5 at 1.81 bits** (−2-3 PPL). 만약 달성되면 Thm 6.18 empirically verified.

**⭐ 성공 기준 (명시)**:
- **Full credit (+0.35)**: WT2 full PPL ≤ **13.5** at avg 1.81 bits (예측 상한 달성).
- **Partial credit (+0.15)**: PPL 13.5–15.0 (예측 구간 근처, "improved but not to spec").
- **Null (0)**: PPL ≥ 15.5 (예측 미달, allocation 이 uniform 대비 개선 없음).

Iteration 여부는 coworker 자체 판단 — 초기 결과가 13.5–15.0 구간이면 lambda* 재조정 또는 facet 경계 재배치로 1 회 iteration 시도 권장. 최악 2 iterations 이내 goal 미달 시 중단 + partial 결과 그대로 제출.

**예상 GPU-hr**: 8-10h (calib + full WT2 PPL × 5 sweep points).

### 🥉 P0-C — 6 baselines degrade-gracefully (CAA/ITI/PASTA/Focus/LoRA-FT/RAG)

SEKA/AdaSEKA 는 P0-A 에서 커버됨. 나머지 6 baselines 중:

**⭐ 완료 정의 (degrade gracefully)**:
- **Full credit (+0.30)**: **모든 6 개** Subtask1+4 complete
- **Partial credit (+0.15)**: **우선 3개 (CAA + ITI + LoRA-FT)** Subtask1+4 complete — SEKA (P0-A) 외 가장 많이 인용되는 prior, LoRA-FT 는 우리 Cor 6.16 와 직접 비교
- **Null**: 3 개 미만

18-24 GPU-hr 에 6 개 완료는 타이트하므로 priority 순서 (CAA → ITI → LoRA-FT → PASTA → Focus → RAG). CAA/ITI/LoRA-FT 는 source 이미 public (clone + wrapper 만), PASTA/Focus 는 구현 복잡도 높을 수 있음.

**Source-first 정책 (필수)**:
- CAA: clone https://github.com/nrimsky/CAA
- ITI: clone https://github.com/likenneth/honest_llama
- PASTA: clone https://github.com/QingruZhang/PASTA
- Focus Directions: GitHub 검색 후 clone
- LoRA-tool-FT: develop side v4 recipe (`scripts/ocq/lora_train_metatool_v3.py` + Subtask1+2+3 mixed)
- RAG: LangChain 표준

자세한 hyperparameter: memory `baseline_recipes_attention_steering.md` 참조.

**예상 GPU-hr**: 18-24h (6 method × 2 dataset × ~2h each).

### 🥇 P0-D — Thm 6.20 τ²-bench retail multi-turn (기여 +0.3, 1 day)

**배경**: 오늘 N=100 Subtask4 single-turn proxy 에서 AUROC 0.976 확보 (`reports/thm620_smoke/eps_q_predictor_N100.json`). 그러나 *real multi-turn plan* 에서 AUROC 는 아직 unverified. τ²-bench retail 이 실제 multi-turn agent benchmark.

**Task**: τ²-bench retail 50-200 conversations 에서:
- 각 conversation 의 per-step ε_q(q_t) trajectory 측정 (Qwen2.5-7B-Instruct)
- 최종 task success (binary)
- AUROC(min_t ε_q, success) 계산

**B_ont**: `external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt` (이미 build 됨)

**코드 시작점**: `scripts/ocq/measure_epsilon_q_plan_predictor.py` 의 ε_q hook 로직 + τ²-bench 의 agent loop 통합.

**예상 GPU-hr**: 6-8h (τ²-bench 50-200 conversation agent run).

## 📊 점수 영향 — 4 개 P0 모두 성공 시

| 항목 | 현재 locked | P0 성공 시 |
|---|---|---|
| Soundness | 3.15 | 3.5-3.6 |
| Contribution | 3.0 | 3.4 |
| Presentation | 3.0 | 3.0 |
| **Overall Expected** | **6.30** | **6.8-7.0** |
| **Acceptance 확률** | **~52%** | **65-70%** |

## ⏱ Timeline

| 일자 | Task |
|---|---|
| D+0 (오늘 밤) | P0-A SEKA debug 시작 (develop 도 시도 중) |
| D+1 (4/16 목) | P0-A SEKA 결과 + P0-B Thm 6.18 calibration |
| D+2 (4/17 금) | P0-B Thm 6.18 WT2 PPL + P0-D τ²-bench 시작 |
| D+3 (4/18 토) | P0-C 6 baselines 병렬 시작 |
| D+4 (4/19 일) | P0-D 완료 |
| D+5 (4/20 월) | P0-C 6 baselines 완료 |
| D+10 (4/25 금) | 모든 결과 통합 + paper writing |
| D+15 (4/30 수) | 최종 review + submit polish |
| D+30 (5/15 목) | **NeurIPS 2026 submit** |

## 📌 중요 Reminder

1. **Source-first 정책** (memory `feedback_external_baseline_use_original_source.md`): proxy 절대 금지. 발견 시 즉시 retraction 필요 (develop side 4/15 AdaSEKA proxy 사고 + paper retraction commit `c5e4f2f` reference).
2. **결과 JSON 형식**: `{"model": ..., "method": ..., "n_queries": ..., "results": [{"macro": {...}, "per_sample": [...]}]}` — develop side 의 eval_metatool_subtask{1,4}.py 와 호환.
3. **매 주요 결과 commit + push 후 Slack 통보** — paper 통합 시 delay 방지.

## Contact
- develop side git: iamseungpil/boltzmann-attention branch develop
- develop side working dir: `/home/woori/workspace_common/boltzmann-attention/`
- Canonical paper: `math/paper/benchmark_design/PAPER_DRAFT_v1_2026_04_14.md` (+ `_ko.md`)

---

**이 4 개 실험 중 3 개 이상 성공이 NeurIPS 2026 main-track 진입의 단일 결정 요인**. Develop side 는 병행하여 paper writing + 작은 Mistral null-control / α-sweep / Llama L=13 Var_s V 측정에 집중.
