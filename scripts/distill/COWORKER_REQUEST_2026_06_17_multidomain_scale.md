# Coworker 요청 (2026-06-17 PM) — 다도메인 content-routing 전이의 *스케일* 측정 (추론-only)

> 자기완결. 권위 = `scripts/distill/HANDOFF_2026_06_17_PM.md §2,§7`·`scripts/distill/GENERATOR_ALGEBRA_DESIGN_2026_06_17.md §3,§6`·`scripts/distill/ma/M_A_RESULTS.md §19`. 직전 요청서(`COWORKER_REQUEST_2026_06_17_B_budget_scale.md` = in-head 깊이 스케일)와 **상보**(같은 thesis·다른 축). 둘 다 진행 가능.
> 불변: **추론-only(학습 0)**·결정론 분담([[feedback-selector-verifier-deterministic]])·도메인-타깃 금지([[feedback-thesis-tbox-transfer-direction]]).

## 0. TL;DR — 무엇을/왜
오늘 woori가 **content 생성원 5→7**(substitute=keep-rest·create) 추가 → *동일* op-IR/결정론 resolver로 **retail exchange 32/32 + airline cabin 27/27** 표현적합성 증명(`M_A_RESULTS §19`). 그리고 7B에 **synth-only 라우팅 LoRA**를 학습해 retail+airline config-swap 전이를 측정 중(§20).

**coworker가 답할 질문**: *도메인-일반 op-라우팅(substitute vs comparative vs create를 두 도메인서 옳게 명명)이 **학습**(7B LoRA)을 요하나, 아니면 **스케일 + in-context 연산정의(gloss)**로 충분한가?* → 32B/72B(/235B) **base 모델**(학습 0)을 동봉 케이스에 돌려 floor(gloss0)·ceiling(gloss1) 라우팅 곡선을 측정. woori 7B-LoRA(§20)와 합쳐 **"offload 구조가 소형+학습 = 대형+gloss를 따라잡나"** 판정.

## 1. 실행 (스크립트·케이스 전부 커밋됨 — git pull만)
```bash
cd <REPO> && git pull --ff-only
ls scripts/distill/ma/multidomain_scale_eval.sh scripts/distill/ma/cases/   # 존재 확인
# <MODEL> <TAG> <GPUS(콤마=TP)> [PORT] [EXTRA_VLLM]. 추론-only·serve→eval→teardown 캡슐화.
bash scripts/distill/ma/multidomain_scale_eval.sh "Qwen/Qwen2.5-32B-Instruct" 32B "0,1"     8055
bash scripts/distill/ma/multidomain_scale_eval.sh "Qwen/Qwen2.5-72B-Instruct" 72B "0,1,2,3" 8056
bash scripts/distill/ma/multidomain_scale_eval.sh "Qwen/Qwen3-235B-A22B"      235B "0,1,2,3" 8057   # 확장점(MoE)
```
- 각 실행 = base 모델 serve(TP=GPU수) → `tau2_op_eval.py`로 **retail·airline × gloss∈{0,1}** 4-eval → `…/depth/c8/multidomain/results/<TAG>__<domain>_g<0|1>.json`.
- **케이스 = repo 동봉**(`scripts/distill/ma/cases/tau2_{retail,airline}_cases.jsonl`) → tau2-bench 불요·전 크기 동일 케이스(스케일 비교 유효).
- **환경 경로 다르면 env override**: `REPO=… PY=… VLLM=… SCRATCH=… bash …`. 기본=woori 경로.
- 양자화 쓰면 TAG에 명시(_awq)·235B는 active≠total 별표.

## 2. coworker가 답할 것 (집계·박제는 woori `M_A_RESULTS §20`)
각 (크기 × 도메인 × gloss)에서 `tau2_op_eval` 출력 회수:
- **overall new_item_id 정확도**, **by case_op**(substitute/create) 분해, **op-routing recognition**(emitted op == gold case_op), **emitted op 분포**.
- 모델 메타(L·d_model·총/active params) — B_budget 요청과 동일 규율(S* ∝ L vs params 분리).

## 3. 핵심 질문 (이론 판정)
1. **ceiling(gloss1)**: 큰 base가 두 도메인서 substitute/comparative/create를 **옳게 명명**하나(recognition↑·op분포 정상)? = 라우팅 인식이 스케일+in-context 능력인가.
2. **floor(gloss0)**: gloss 없이도 라우팅되나? **S1−S0 gap** = 명시적 연산어휘의 가치(7B서 gloss가 comparative 0→1.00 회복: `M_A_RESULTS §15-bis`).
3. **도메인-일반성**: retail·airline 라우팅 정확도/recognition가 **같이** 높나(한쪽만↑=특화)? — 스케일이 도메인-일반을 *학습 없이* 주나.
4. **소형+학습 vs 대형+gloss**: woori 7B-LoRA 전이(§20)가 큰-base-gloss1에 근접/초과하나? = offload 구조가 scale 대체(thesis 핵심).
5. (대조) **over-comparative 회귀** 여부: 7B-LoRA가 substitution을 comparative로 오라우팅했던 §17/§18 병리가 큰 base서도 나타나나, 아니면 스케일이 해소하나.

## 4. 하지 말 것 / 규율
- **학습 0**(base inference만)·`tau2_op_eval.py`·`tau2_op_resolver.py`·케이스·프롬프트/gloss **편집 금지**(크기간 동일해야 유효).
- 도메인 보고 op·휴리스틱 추가 금지(생성원 taxonomy는 동결·τ²-blind).
- temp0·GPU 분리·serve 전 GPU kill·port/log 분리([[reference-remote-server-environment]]).

## 5. 산출물 (git 회수) + 인프라
- `…/multidomain/results/<TAG>__{retail,airline}_g{0,1}.json` (8파일/크기) → repo commit 또는 경로 통보.
- 로그 `…/mdscale_<TAG>.log`. `git pull --ff-only` 선행.

## 6. 조율
- 스크립트·케이스 이미 커밋(이 요청서와 동시). `git pull` 후 바로 시작 가능.
- 32B/72B로 명제 판정 충분·235B는 확장점(MoE active-param 별).
- woori는 동시에 7B-LoRA §20(학습 전이)·K-sweep(다양성 곡선) 진행 중 — 결과 합쳐 phase diagram.
