# Rung1 v3-inductive — 단계별(bottom-up) reduction 트리평가 게이트: 설계서

> 2026-06-03. 진입점=`HANDOFF_2026_06_03_v3_treeval.md`. 마스터=`EXPERIMENT_DESIGN.md §3`.
> 결과권위본=`reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` (Exp-4-rung1-v3-AB).
> 문헌근거=`RUNG1_V3_TREE_EVAL_LITREVIEW.md` (deep-research 27소스→25주장 검증본).
> 이 문서 = **v3 단일-스텝 → inductive 다단계 derivation 변경의 설계·근거·검증·실행계획**.

## 0. 한 줄
v3 단일-스텝 grounded 트리평가는 (truncation 제거 후) **수렴하나 BOTH=control=5 = 무개선**.
원인 = **단일-스텝 whole-expression 평가의 globality 한계**. 처방 = **bottom-up reduction 체인**
(각 내부 AND/OR 노드를 supervised step으로 fold). 문헌이 *증명 수준*으로 지지하는 트리학습 경로.

## 1. 실증 트리거 (왜 이 변경인가 — Exp-4-rung1-v3-AB 전수조사)
- **헤드라인(maxtok=24)**: treeval BOTH 2 vs control 5 → "v3 회귀"로 보였음.
- **행동층 전수조사**: treeval should_T 48 중 **35개가 max_steps=10 루프**(avg 8→20스텝). 콜드붕괴 아님 = **전역 비수렴**.
- **메커니즘층(SOPBENCH_RLLOG)**: control terminal **27/27=100% 결정 도달**, treeval **0/29=0%** — 전부
  `ready=true; gate = AND(op_32=false, AND(op_39=true, op_25=true` 에서 **planner max_tokens=24 절단**(종결 토큰 전).
  → **모델은 grounded tree-eval emit을 정확히 학습**했으나 디코드 예산이 잘라 결정 미파싱 → 재게더 → 루프.
- **무재학습 재시험(`SOPBENCH_PLAN_MAXTOK=1024`)**: treeval **loop 35→0, 도달 0%→100%, BOTH 2→5**.
  → "회귀"는 100% truncation 아티팩트. **진짜 판정 = BOTH 5 = control 5 = 무개선**(단 거부↓·행동↑·dirgraph↓:
  콜드 보수성은 풀렸으나 결정을 *완전 게더* 위에 못 얹음 = acted 15 중 ~10이 premature).
- **해석**: 식을 *emit*하는 것(✅ 학습됨) ≠ 식을 한 스텝에 *정확 평가*하는 것. **단일-스텝 globality 한계.**

## 2. 변경 — 단일식 → 단계별 reduction 체인
**Before (단일-스텝, `--treeval`)**: terminal 1줄에 중첩식 전체 + 단일 값.
```
ready=true; gate = AND(op_32=false, AND(op_39=true, op_25=true)) = false; STOP
```
**After (inductive, `--treeval_inductive`)**: bottom-up post-order, 각 내부 노드를 **이미 해소된 자식 값**으로 fold, 중간결과(t1,t2…)를 명명·재사용, 마지막에 gate 재확인.
```
ready=true; t1=AND(op_39=true, op_25=true)=true; t2=AND(op_32=false, t1)=false; gate=false; STOP
```
- 잎 = **게더한 truth 룩업**(treeval_expr와 동일 의미; 재추론 없음).
- 각 `tK=OP(자식들)=v` = **국소 스텝**(2–3개 *기지(旣知)* 값 결합) = autoregressive serial compute.
- 중간값 `t1,t2`가 **SFT target에 포함**(supervised) = 효율분리의 핵심 기제.
- 단일-leaf 트리는 fold 없이 `ready=true; op_28=true; gate=true; ACT`.

## 3. 설계결정 ↔ 선행연구 매핑 (각 선택의 근거)
| 설계 결정 | 근거 논문 | 판정 |
|---|---|---|
| 중첩식 1샷 평가 → **노드별 reduction**(globality 분해) | **Abbe 2406.06467 (NeurIPS24)** — globality 장벽은 *inductive(구조적) scratchpad만* 깬다; free-form/agnostic 불가 | ✅ SUPPORTED |
| **중간 집계결과(t1,t2)를 target에 포함**(supervise) | **Kim & Suzuki 2410.08633 (ICLR25 Oral)** — 중간스텝 loss면 parity 1-step 학습, 없으면 *증명적으로* 불가(효율분리) | ✅ SUPPORTED(analogical) |
| reduction 체인 = **serial 토큰으로 serial 평가** | **Feng 2305.15408/2402.12875** — no-CoT const-depth=AC0/TC0(serial boolean 불가), CoT면 size-T boolean circuit 평가(CVP) | ✅ SUPPORTED(expressivity) |
| **pairwise/소(小)-AND로 fold**(고민감도 회피) | **Bhattamishra 2211.12316 (ACL23)·Wang 2412.02823** — flat-AND 쉬움·고민감도 붕괴, ICL정확도↔식복잡도 r=−0.88(Qwen2-7B) | ✅✅ 우리 실측 직접설명 |
| **잎=게더 truth 룩업**(재추론 금지) | 자체 T1T2 census(grounded AND(preconds) **0오류**) + litreview §4.3 | ✅ 자체실증 |
| 분해 자체의 *필요성* | **Dziri 2305.18654 (NeurIPS23)** — 트랜스포머=linearized subgraph matching, 복잡도↑서 error→1 | ✅(negative) |
| 재귀 locate-then-evaluate(잎부터 위로) | He 2512.02677 | ◐ (depth 일반화는 RISK) |

**과대주장 경계(litreview §3 그대로)**: 대부분 1-layer/from-scratch 합성(parity/CVP)·GPT-2급 →
**7B LoRA tool-use 전이는 미검증**. expressivity 증명 ≠ SFT가 그 해를 복원한다는 보장. 본 실험은 *탐색적*.

## 4. 가설·지표·판정
- **H(inductive)**: 단계별 reduction이 단일-스텝의 globality 한계를 풀어 **BOTH > 5**(단일-스텝/control 천장 초과).
  특히 premature(acted∧¬dirgraph) ~10건이 "게더 완성 후 fold 닫힘"으로 **dirgraph↑→BOTH 전환**.
- **A/B**: control = 기존 single-step `treeval` 어댑터(데이터 byte-identical, 재학습 불요) vs `treevalind`(신규).
  **둘 다 PLAN_MAXTOK=1024**(truncation 통제). LODO holdout=bank, alias_s3, source=3, ep3, r16, SOLO.
- **분리지표(헤드라인)**: should_T(48) **BOTH**(dirgraph∩goal) + dirgraph + acted + premature + over-refuse;
  should_F(86) **STOP-recall**. + RLLOG **terminal-도달%**(둘 다 ~100% 기대, 비수렴 통제 확인).
- **판정**:
  - **성공** = treevalind **BOTH > 5**(특히 ≥ single-step+3) & STOP 비회귀 → inductive가 단일-스텝 한계를 깸 → §3.10 북극성 경로 + 조건수 분해.
  - **부분** = BOTH 6~7(소폭) → 방향 맞음, depth/조건수 분해로 어디서 새는지 확인 후 강화(노드별 별도 스텝화 등).
  - **무효** = BOTH ≈ 5 → 단계별 supervise도 7B 단일-call CoT로는 globality 못 깸 → **planner-call 분할**(노드당 1 call) 또는 DPO/T1c/탐색증류로 분기.
- ⚠️ Mean Pass Rate 단독 판정 금지(거부 부풀림). 분모=/48 주, /40 보조.

## 5. 측정 부가(ablation·진단)
- **조건수별 BOTH 분해**(2/4/6/8): depth-decay(He #5) 회피 여부 — reduction이 깊이를 평탄화하는지.
- **OR-케이스 정확도**(litreview 21% OR): grounded OR-fold가 AND만큼 되는지.
- **RLLOG 체인 정확도**: 모델이 emit한 reduction의 중간 fold가 게더 truth와 일치하는지(허위 fold 검출).

## 6. 리스크·교란(정직)
1. **단일-call 내 CoT vs call분할**: 본 구현은 *한* planner call이 체인을 autoregressive 생성. Abbe/Feng은
   "serial 스텝"을 요구하나 call-내 토큰열도 serial. 다만 7B가 call-내 다단계 fold를 신뢰성있게 할지 미검증
   → 무효 시 **노드당 1 planner-call**(진짜 멀티스텝)이 다음 후보.
2. **체인 길이**: 최심 트리(set_safety_box 7조건)도 ~6 fold ~80토큰 « 1024. 단 train `--max-seq-len 2048`,
   `--skip-overlong` → 프롬프트+체인 초과분 드롭(희귀). 빌드 후 overlong 비율 확인.
3. **grounded/fallback 셋이 arm 간 미세상이**: inductive는 chain_val(=tv) 기반 일관성 가드, single-step은
   disp(est_failed 포함) 기반 → est_failed-flip 희귀 케이스서 grounded 셋 다름. 영향 미미(빌드 검증: ~90% grounded 동일).
4. **전이·NL→트리·깊이 일반화**(litreview §5)는 본 in-domain LODO로 부분만 커버 → 별도 후속.

## 7. 코드 맵 + 정합성 가드
- `build_tbox_planner_sft.py`:
  - `treeval_reduce(tree, observed, amap)` — post-order fold, (chain_str, value) 반환. 잎 의미=`treeval_expr`와 동일.
  - `--treeval_inductive`(→ `--treeval`·`--scratchpad`·`--gate-token` 함의). 파일태그 `_treevalind`.
  - emit: **inductive grounded는 `chain_val`가 GT결정과 일치할 때만**(ACT↔true/STOP↔false), 아니면 fallback
    → **체인 마지막 값 == 표기 gate 보장**(est_failed-flip 모순 제거; single-step보다 엄격).
- `rung1_v3ind_train_eval.sh` — inductive teacher 빌드 → LODO train(비-bank) → GPU0 solo 학습 →
  treeval(기존)+treevalind(신규) **maxtok=1024** eval → 분리지표 헤드라인.
- `two_stage_client.py` — `SOPBENCH_PLAN_MAXTOK`(기본 24; 본 실험 1024).

## 8. 실행 전 검증 게이트(빌드 후·학습 전)
1. SYNTAX OK + 빌드 성공(✅ 7도메인 4846 예제).
2. **체인 포맷 육안검증**(✅): `t1=AND(..)=v; t2=AND(..,t1)=v; gate=v; ACT|STOP`, grounded≈90%.
3. ⏳ **일관성 재검증**(버그픽스 후): chain 마지막 값 == gate, fallback에 chain 누출 없음. (이 문서 리뷰 직후 재빌드·확인)
4. ⏳ overlong(>2048) 드롭 비율 < ~5% 확인.
통과 시 학습 launch(~4h GPU0 solo) → eval(~25분).
