# ▶▶ Coworker 요청서 — TaskBench scale 실험 (Qwen3 32B/235B-A22B, 4×A100 80GB) — 2026-06-10 **v2**

> **★v2 (2026-06-10 PM, 사용자 결정·정정 반영): 이원화 — Qwen2.5는 비교-통제용 유지 + Qwen3는 곡선-연장용 신규.**
> - **fact**: 72B는 Qwen2.5에만 존재(Qwen3 dense는 32B 천장, 그 위는 MoE 30B-A3B/235B-A22B). SOPBench 리더보드(Track-B 기준선)도 Qwen2.5 계열 → **SOPBench 작업은 Qwen2.5 유지(전환 아님)**.
> - **Q2(전이 NULL=용량? — P1)는 동일-family 통제 필수**: 오늘 7B NULL이 Qwen2.5-7B 측정 → **P1 SFT = Qwen2.5-32B**(Qwen3-32B면 family+scale 동시 변경 = confound).
> - **Q1/Q3(곡선 연장·72B-초과)는 Qwen3**: 단일 family가 0.6B→235B 커버. Track A가 0.6–14B 재측정(전환), coworker가 32B/235B-A22B. 오늘까지의 Qwen2.5 0.5–14B 곡선은 **P0에서 32B/72B만 보태면 완성** = 2-family 증거(곡선 모양의 family-불변성 체크)로 승격.
> **⚠️전 실험 공통 통제 (신규, 필수)**: Qwen3는 hybrid thinking 모드 → **non-thinking 고정**. 방법 = 요청 body에 `"chat_template_kwargs": {"enable_thinking": false}` (vLLM OpenAI 서버 지원; TaskBench inference.py payload에 1줄 추가) 또는 user 프롬프트에 `/no_think`. **두 트랙 같은 방법 사용**(차이나면 비교 무효). thinking-on은 별도 arm으로만(선택, 비교축 오염 금지).
> **⚠️MoE 표기**: 235B-A22B는 total 235B/activated 22B — 곡선에 올리되 "MoE(A22B)" 병기(dense 곡선과 시각 구분).

> 발신: Track A (7B/14B, 리모트 2×49GB). 채널 = branch `facet-rft-2026`. 마스터 정합: `scripts/distill/FIELD_GAP_LLM_VALUE_DESIGN.md` §18.1 Exp-A/C + `scripts/distill/HANDOFF_2026_06_10_taskbench_learning.md`. 결과 권위본 = `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` (오늘 수치 전부 여기).
> **보고 규율 (사전등록, 위반 금지)**: ① TaskBench LODO = **supporting 전이만**(moat-(3) 주장 금지, FIELD_GAP §17.9 리뷰7-1) ② 지표 = **edge-F1 중심**(node ~포화) + type(single/chain/dag) 층화 ③ GT=GPT-4 back-instruct → **GPT-4를 teacher/증강에 쓰지 말 것**(순환, 리뷰7-2) ④ exact-match 채점이라 천장<100(valid 대안 penalize; A-0 감사: miss의 ~27%=관례).

## 0. 왜 요청하나 (Track A 오늘 실측 — 비교 타깃)

**(a) base prompted 곡선 (Qwen2.5-Instruct, 500-sub/도메인; 7B만 full)** — edge-F1은 1.5B→3B서 emerge, **14B까지 비포화**:

| 크기 | HF n/e-F1 | MM n/e-F1 | daily n/e-F1 |
|---|---|---|---|
| 0.5B | 10.5 / 0.0 | 17.0 / 0.0 | 5.1 / 0.2 |
| 1.5B | 50.3 / 2.3 | 55.9 / 3.0 | 54.5 / 18.6 |
| 3B | 64.4 / 19.1 | 72.6 / 27.6 | 70.5 / 33.9 |
| 7B (full) | 73.6 / 32.2 | 84.4 / 50.0 | 90.8 / 68.1 |
| 14B | 77.5 / 39.6 | 89.5 / 52.8 | 93.5 / 77.4 |
| (published gpt-4) | — | 90.9 / 69.3 | — |

**(b) gold-SFT LODO 7B 1차 = held-out 전이 NULL/소폭 회귀**: HF+daily 학습(3869ex, LoRA r16 2ep)→held-out MM full: node 84.4→82.5·edge 50.0→**48.3**. 해석(잠정): base가 형식은 이미 보유, SFT가 배운 것=도메인-특정 grounding→전이 없음. **미해결 질문 = 이 NULL이 7B 용량 한계인가(스케일로 풀리나), 방식 한계인가(alias/RFT 필요).**

## 1. 질문 3개 (이 실험이 답할 것)
- **Q1 (곡선 연장)**: edge-F1이 32B/72B(Qwen2.5)·32B/235B-A22B(Qwen3)에서 계속 오르나, 어디서 published gpt-4(69.3 MM)에 접근/포화하나? + **곡선 모양이 family-불변인가**(Qwen2.5 vs Qwen3 두 곡선 비교).
- **Q2 (★핵심, transfer-vs-scale, 동일-family 통제 = Qwen2.5)**: 7B서 NULL인 gold-SFT held-out 전이가 **Qwen2.5-32B**(→+면 72B)서 양(+)으로 뒤집히나? = "전이할 공통 edge-스킬의 학습이 용량-바운드인가" — Exp-A 다음 수(alias/RFT) 설계를 결정.
- **Q3 (선택, frontier-adjacent 오픈 앵커)**: 235B-초과/타-family 오픈모델은 prompted로 어디까지 가나? (별도 표 — family/양자화 confound로 곡선에 못 올림.)

## 2. 실험 arms (우선순위순; Qwen3 run은 **전부 non-thinking 고정**)

**P0 — prompted baseline, 두 family (전부 500-sub ×3도메인, 각 도메인 `user_requests.json` 앞 500줄 — §4 스크립트가 자동 생성)**
- **P0a (Qwen2.5 곡선 완성)**: Qwen2.5-32B(TP1-2, bf16) + **Qwen2.5-72B(TP4, bf16 ~145GB)** — Track A 기측정 0.5–14B에 보태 Qwen2.5 곡선 완결. 견적 합 ~2-4h.
- **P0b (Qwen3 곡선 대형단)**: Qwen3-32B(dense, bf16) + Qwen3-235B-A22B(공식 AWQ/GPTQ-INT4 ~120-130GB, TP4). (선택: Qwen3-30B-A3B = dense-32B 동구간 MoE 대조점.) 견적: 32B ~30-60분, 235B-INT4 ~2-4h. Track A가 같은 라인 0.6–14B 측정 → **0.6B→235B 단일-family 곡선 완성.**
- 산출: 두 family 곡선 분리 표기(같은 그림에 겹쳐도 좋으나 family 라벨 필수).

**P1 — ★gold-SFT LODO_mm @ **Qwen2.5-32B** (Q2 본명 — 동일-family 통제)**
- 학습 데이터 = Track A와 **동일 생성**(아래 §4 `tb_build_sft.py`, seed 42 → byte-동일 jsonl): HF(single400/chain1000/dag전부795=2194) + daily(400/1000/275=1675) = 3869.
- 레시피 = 7B와 동일 통제: **LoRA r16/alpha32, 2ep, seqlen 6144, lr 1e-4, val 2%** (`scripts/distill/lora_train_chat_toolcall.py` 그대로 사용 가능; 32B는 grad-ckpt+TP/FSDP로 2~4GPU. 자체 트레이너 쓸 경우 위 하이퍼 고정).
- 평가 = held-out **MM full**(시간 빠듯하면 sub500 먼저→full 보충) + in-domain HF/daily sub500. 비교 4열: `base-32B / 32B+gold-SFT / base-7B / 7B+gold-SFT`(7B 수치는 §0).
- 견적: 32B LoRA 7584 step seqlen6144 ≈ **12-24h**(설정 따라). 판정: held-out edge-F1 Δ가 7B의 −1.7을 넘어 **+로 뒤집히면 용량-바운드 확정**(→ Track A는 14B 재시도+alias), 32B서도 NULL이면 **방식-바운드**(→ alias-마스킹/RFT로 피벗, 스케일 추가투자 중단).
- ⚠️ in-domain이 두 스케일 다 평평하면 "gold-SFT 자체 무가치(base 포화)" — 그것도 1급 결과.

**P2 — (P1이 + 신호 시) Qwen2.5-72B gold-SFT LODO_mm**: 동일 레시피·동일 family로 용량 축 한 단 더(7B→32B→72B 3점 = 전이-vs-용량 곡선). 견적 2-3일 → P1이 +면만.

**P3 — (선택) 타-family 오픈 앵커, prompted-only**
- 후보: **gpt-oss-120b**(MoE, MXFP4 native ~63GB — 1-2 GPU, 최신 vllm 필요) / (한계 도전) Llama-3.1-405B-INT4(~205GB+KV, TP4 빠듯 — 실패해도 무방). (Qwen3-235B는 P0b로 승격됨.)
- 500-sub ×3도메인 prompted만(SFT 불가/불요). **별도 표** "frontier-adjacent open anchors"(family·양자화 병기, 곡선과 분리). 목적 = "오픈 최상위가 published gpt-4(69.3)를 넘나" 단일 질문.

## 3. 지표·산출물 (고정)
- metrics: `node_micro_f1_no_matching` / `link_binary_f1` (+가능하면 `-m argument`의 t/v-F1). **type 층화**(metrics json에 single/chain/dag split 포함됨).
- 산출물: ① 위 표 형식으로 `TASKBENCH_EXPERIMENT_RESULTS.md`에 추가(§1 표에 32B/72B행, §3에 P1 4열표, P3는 별도표) ② pred/metrics json은 자체 보관+경로 공유(대용량이라 커밋 불요, metrics json만 커밋 권장).

## 4. 재현 절차 (전부 박제됨 — 재발견 금지)
1. **클론**: `git clone https://github.com/microsoft/JARVIS && cd JARVIS/taskbench` (Apache-2.0). 데이터 3도메인 동봉(`data_huggingface` 7458 / `data_multimedia` 5555 / `data_dailylifeapis` 4318).
2. **venv**: python3.10+; `pip install numpy scikit-learn networkx python-Levenshtein "datasets==2.14.5" "pyarrow==12.0.0" rouge_score aiohttp emoji click` + vllm(서빙용; P3 모델은 최신 vllm).
3. **inference.py 버그 패치 필수**: `loop = asyncio.get_event_loop()` 다음 줄에 `results = []` 추가 (미초기화 크래시).
4. **dependency_type**: HF/MM = `resource`, **daily = `temporal`**(tool_desc에 input-type 없음 — 틀리면 assert).
5. **500-sub 생성·serve·infer·eval 전부** = 우리 repo `scripts/distill/taskbench/` 재사용:
   - `tb_scale_curve.sh` — sub500 생성+모델 루프(serve→infer→eval)의 견본(경로 변수만 자체 클러스터로 수정).
   - `tb_build_sft.py` — P1 학습 jsonl 생성: `--tb_dir <클론> --domain data_huggingface --out sft_hf.jsonl --n_single 400 --n_chain 1000 --n_dag 0` + daily 동일(`--n_dag 0`=전부) → concat. seed 기본 42 고정.
   - `tb_build_eval.py` — **id-정렬·기형 gold skip·기형 pred sanitize 전부 내장**(직접 evaluate.py 돌리면 KeyError 함정 3종). `--tb_dir <클론> --domain <dom> --llm <tag> [--pred_file ... --dst ...]`.
   - `tb_eval_adapter.sh` — 어댑터 서빙+held-out/in-domain 평가 견본(GPU/UTIL/PORT 인자화 돼 있음).
6. **함정 요약**: pred id 순서≠gold 순서(→`tb_build_eval.py`가 처리) / `-m node`은 무효(`-m f1`이 node-F1) / metrics = `{dst}/metrics/{tag}.json`의 `overall_overall`.

## 5. 일정 제안
P0(반나절) → P1(1-2일) → 판정 공유(채널) → P2/P3(조건부, 1-3일). **P1 결과가 Track A의 다음 수(alias vs RFT vs 스케일)를 게이트하므로 P1 우선 완주 요청.** 기존 Track-B(32B SOPBench SFT, v1.42 #0/#1)와 GPU 경합 시 — Track-B #0 sanity가 먼저, 그 다음 본 요청 P0/P1 권장(어차피 32B 서빙/학습 인프라 공유).
