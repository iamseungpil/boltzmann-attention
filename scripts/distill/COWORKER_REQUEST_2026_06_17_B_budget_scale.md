# Coworker 요청 (2026-06-17) — B(절차예산) 스케일 실험: 대형 32B/72B/235B in-head 매핑 임계 측정

> 자기완결. 권위 = `scripts/distill/B_BUDGET_SCALE_DESIGN_2026_06_17.md`·`NL_PROCEDURE_OFFLOAD_THEORY_2026_06_17.md §7d-bis`. 직전 요청서(floor·factorial)는 완료·무관.
> 불변: 추론-only(학습 0)·결정론 분담([[feedback-selector-verifier-deterministic]])·**τ² 무관**(이건 합성 통제실험).

## 0. TL;DR — 무엇을/왜
**합성 통제-깊이 selection 과제를 *in-head(CoT 없음)*로 돌려, "superlative/comparative(the most/the better)를 *매핑만으로* 푸는 *임계 모델크기*"를 측정.** 명제 = *충분히 크면 얕은 모델도 절차를 forward-pass 매핑으로 푼다*(유계예산 B가 깊이 d 초과). **woori = 0.5–14B / coworker = 32B·72B·235B**(H100×4). 합쳐 1.5B→235B 스케일링 곡선 S*(d,N).

## 1. 실행 (synth_depth+depth_eval 커밋 후 — 의존)
> ⚠️ **선행**: 내가 `synth_depth.py`·`depth_eval.py`·`depth_scale_batch.sh`를 단일소스 커밋(곧·핑). coworker는 `git pull` 후 `ls scripts/distill/ma/depth_scale_batch.sh` 확인 후 시작.
```bash
cd <REPO> && git pull --ff-only
# 대형 3개·각 bf16(가능하면)·H100×4 TP. depth_scale_batch가 serve→eval→집계 캡슐화.
bash scripts/distill/ma/depth_scale_batch.sh "Qwen/Qwen2.5-32B-Instruct"  32B  <GPUs>
bash scripts/distill/ma/depth_scale_batch.sh "Qwen/Qwen2.5-72B-Instruct"  72B  <GPUs>
bash scripts/distill/ma/depth_scale_batch.sh "Qwen/Qwen3-235B-A22B"       235B <GPUs>   # MoE·확장점
```
- `depth_scale_batch.sh <model> <tag> <gpus>` = serve(TP) → `depth_eval.py`(in-head + CoT 조건·합성 N∈{5,10,20,50}×d∈{1..4}·n≥200) → `depth_<tag>.json`(acc(d,N) per-condition).
- **양자화 불가피하면 tag에 명시**(_awq 등)·235B는 active≠total 별표.
- ⚠ TP 필요(32B+)·serve 전 GPU kill·port/log 분리.

### 1b. ★추가 (2026-06-18) — WIDTH 스윕 (B의 width 축·depth의 쌍둥이)
**왜**: `M_A_RESULTS §20-21`서 transfer 천장 = **multi-attr `set` 과소추출**(요청 변경 k개 중 일부만). base 7B set-추출 정확도가 width 1→4서 0.64→0.25 하락(width-budget 벽). **이게 scale로 풀리나(=offload 불요)·frontier도 못 맞추나(=decomposition-offload 필연) 판정**. depth와 같은 모델·serve로 width_eval 추가 실행:
```bash
bash scripts/distill/ma/width_scale_batch.sh "Qwen/Qwen2.5-32B-Instruct" 32B "0,1"     8065
bash scripts/distill/ma/width_scale_batch.sh "Qwen/Qwen2.5-72B-Instruct" 72B "0,1,2,3" 8066
bash scripts/distill/ma/width_scale_batch.sh "Qwen/Qwen3-235B-A22B"      235B "0,1,2,3" 8067
```
- `width_scale_batch.sh` = serve(TP) → `width_eval.py`(통제-width substitute·arm A in-head + arm B **set-추출**·width∈{1..5}·n=100·gloss=1) → `depth/c8/width/width_<tag>.json`(width별 arm_A_item·SET_EXACT·set_recall).
- **핵심 지표 = SET_EXACT(width)**: 모델이 요청된 k개 변경을 *전부* 추출하나. 높게 유지=scale이 width 맞춤(offload 불요)·하락=width 벽(offload 필연). woori 7B + frontier(gpt-4.1, 진행중) 곡선과 합쳐 S*(width).

## 2. coworker가 답할 것 (집계는 내가)
- 각 크기의 **acc(d, N, condition)** — in-head / +CoT.
- **모델 메타: L(layers)·d_model·총params·(MoE면 active)** 꼭 기록(S* ∝ L vs params 분리용).
- woori 0.5–14B + coworker 32–235B 합쳐 **S*(d,N) phase diagram** + 스케일링 곡선.

## 3. 핵심 질문 (이론 판정)
1. 각 (d,N)에 **유한 임계 S\* 존재**하나(충분히 크면 매핑으로 풀림)? = 사용자 명제.
2. **S\* 단조↑ in (d,N)**인가(깊을수록·N 클수록 더 큰 모델)?
3. in-head S\*가 **L에 묶이나 params에 묶이나**(메타 기록으로).
4. CoT가 S\*를 **낮추나**(작은 모델 + 외부직렬 = 큰 모델 매핑 대체)?
5. (대조) 결정론 offload는 전 d/N서 **1.0**(B=∞)·소형도 → *임계가 크면 offload 지배*.

## 4. 하지 말 것 / 규율
- **`synth_depth.py`·`depth_eval.py`·하이퍼·프롬프트 편집 금지**(크기간 동일해야 스케일링 유효).
- **CoT 조건은 *지정된 2-stage 프롬프트* 그대로**(자의적 prompt 금지).
- 학습 0·base inference만·temp0·n≥200/셀.
- 모델 메타(L·width·params) 반드시 회수.

## 5. 산출물 (git 회수) + 인프라
- `depth_<tag>.json`(acc(d,N,condition) + 모델 메타) → repo commit 또는 경로 통보.
- GPU 분리·`git pull --ff-only`([[reference-remote-server-environment]]).

## 6. 조율
- 선행(synth_depth+batch) 커밋되면 핑 → 시작. 235B는 확장점(불가하면 32/72B만으로도 명제 판정).
