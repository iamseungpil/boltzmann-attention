# Coworker 요청 (2026-06-16) — 대형모델 floor/scale sweep (Int8-cap vs reasoning-floor 확정)

> 자기완결 요청서. 이전 요청서(reports/COWORKER_*·4월)는 **무관**(다른 라인). 이것만 보면 됨.
> 권위 컨텍스트: `scripts/distill/ma/M_A_RESULTS.md`(§8 floor 결과)·`MIN_CONTEXT_FORMALIZER_DESIGN.md`·`DECOMPOSITION_OPTIMALITY.md`.

## 0. TL;DR — 무엇을 / 왜
**우리 committed eval 파이프(`scripts/distill/ma/`)를 32B-bf16 + 72B에서 돌려달라.** 로컬은 A6000 49GB라 32B는 **Int8**까지만 가능했고, 그 결과 핵심 질문이 *quant 교란*에 묶여있다:
- 로컬: concrete-emit 정확도 **7B 0.438 → 14B 0.719 → 32B-Int8 0.719 (14B→32B *평탄*)**.
- **이 평탄이 (a)reasoning floor(scale로 안 넘는 한계)인지 (b)Int8 양자화 cap인지 모른다.** → **coworker 32B-bf16이 가른다.** + 72B로 천장 위치 확인.

## 1. 설계 한 장 (맥락)
에이전트를 **기능별로 분담**: 정확-명세가능→**결정론**(resolver/gate/verify) / 도메인-불변 추론→**LLM 학습-일반** / 도메인 사실→**retrieval/ABox**. 주장 = 이 협업이 monolith를 **비용·성능 Pareto-지배**(LLM에게 *못하는* 일[날조·환각·과다호출]을 안 시킴). on-prem 작은 모델로 큰 모델 성능을 내는 게 목표.

**테스트 태스크 = τ² retail exchange**(29 케이스·"키보드를 clicky로, 없으면 무백라이트" 류). 모델은 **새 variant를 고른다**. 입력 정보수준(L0–L3)·출력방식(concrete/selector)·모델크기를 교차.

## 2. 지금까지 핵심 발견 (로컬·`M_A_RESULTS.md §8`)
| arm | 7B | 14B | 32B-Int8 | tok/case |
|---|---|---|---|---|
| L0 (availability 없음) | 0.375 | 0.531 | 0.594 | ~855 |
| L1 (full+avail) | 0.531 | 0.688 | 0.750 | ~900 |
| L2a (가용필터) | 0.406 | 0.656 | 0.812 | ~625 |
| **L2b (가용·표 formalized)** | 0.531 | 0.625 | **0.844** | **~508** |
| L3 (diff 주석) | 0.406 | 0.656 | 0.750 | ~628 |
| A (concrete) | 0.438 | 0.719 | 0.719 | ~918 |
| Bfair (selector·공정정보) | 0.375 | 0.500 | 0.656 | ~810 |
- 정보 floor 실재(L0→L1 +16pp 전 scale)·**MSC(입력formalize)는 scale 대체 못 함**(7B ~0.53 천장)·**formalize(L2b)=비용-Pareto 우월**(토큰 절반·32B 최고)·**selector(Bfair)는 공정정보로도 concrete에 짐**.
- 별개로 step-reasoning(2-call CoT)은 7B를 0.656(≈14B 0.719)까지 끌어올림 = 유망 레버(다음 라인).

## 3. coworker가 답할 결정 질문
1. **14B→32B A 평탄(0.719=0.719)이 reasoning-floor냐 Int8-cap이냐** — 32B-bf16 A가 14B보다 오르면 Int8-cap·같으면 floor.
2. **reasoning 천장이 scale로 계속 오르나** — 72B서 A·L2b가 어디까지.
3. **formalize(L2b) 이득이 bf16/72B서도 유지/확대되나** (32B-Int8서 +9pp).
4. **selector(Bfair)는 대형서도 concrete에 지나** (전 scale 음성 지속?).

## 4. 정확한 실험 스펙 (committed 파이프 그대로)
```bash
cd <REPO> && git pull --ff-only
# (scratch에 tau2-bench db.json/tasks.json + seka/vllm env 필요 — 우리와 동일 경로 가정;
#  경로 다르면 ma_eval_scale.sh 상단 PY/VLLM/S 변수만 조정)
# 32B-bf16 (TP=2 or A100-80G/H100 1장):
bash scripts/distill/ma/ma_eval_scale.sh "Qwen/Qwen2.5-32B-Instruct" \
  "A,Bfair,L0,L1,L2a,L2b,L3" "_bf16" <GPU> <PORT>     # 필요시 vllm에 --tensor-parallel-size 2 추가
# 72B (bf16면 2×80G/4×48G TP·아니면 AWQ-Int4; 어느쪽이든 tag에 명시):
bash scripts/distill/ma/ma_eval_scale.sh "Qwen/Qwen2.5-72B-Instruct" \
  "A,Bfair,L0,L1,L2a,L2b,L3" "_bf16" <GPU> <PORT>
```
- arms 정확히 **`A,Bfair,L0,L1,L2a,L2b,L3`** (우리 7B/14B/32B-Int8과 동일 비교축).
- 72B가 bf16 불가면 **AWQ/GPTQ로 돌리되 suffix/tag에 양자화 명시**(예 `_awq`) — Int8/AWQ도 데이터로 유용(quant 곡선).
- ⚠ `ma_eval_scale.sh`는 **`$4=GPU·$5=PORT` 인자** 받음·로그=`ma_eval_scale_p<PORT>.log`·출력=`ma_eval_<tag><suffix>.jsonl`. GPU/포트 충돌 피해 빈 GPU 지정.
- ⚠ 32B+ TP가 필요하면 `ma_eval_scale.sh`의 vllm serve 줄에 `--tensor-parallel-size N` 추가(현재 단일GPU 가정).

## 5. 산출물 (git으로 회수)
- `/scratch/ma_eval_Qwen2_5_32B_Instruct_bf16.jsonl`·`..._72B...jsonl` (per-case 레코드·**비용계측 포함**: prompt_tokens/completion_tokens/n_calls).
- `ma_eval_scale_p<PORT>.log`의 `=== SUMMARY ===` 블록(arm별 acc + tok).
- **이 파일들을 repo에 commit**하거나 경로 알려주면 우리가 집계. (집계 스크립트: `M_A_RESULTS.md §8`의 python 스니펫·arm별 sum(item_correct)/len.)

## 6. 하지 말 것 / 규율
- **도메인-fit 금지**(SFT/finetune 아님·base 모델 inference만). [[feedback-thesis-tbox-transfer-direction]].
- arms·태스크·프롬프트 **변경 금지**(committed `ma_eval.py` 그대로) — 7B/14B/32B-Int8과 직접 비교해야 함.
- 양자화 쓰면 **반드시 tag 명시**(bf16 vs Int8 vs AWQ 구분이 핵심 질문).
- 토큰 비용 꼭 기록(이미 계측됨) — 정확도-당-비용 곡선용.

## 7. (예고·다음 라운드) step-decomposition arm
곧 "결정론 scaffold가 typed 증분스텝 강제+per-step 검증" arm을 추가할 예정(작은모델 reasoning↑ 강한형). 그건 별도 요청서로 — 이번엔 **floor/scale(위)만**.
