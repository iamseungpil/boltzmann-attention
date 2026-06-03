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
> ⚠️**load-bearing 근거 = Kim&Suzuki(중간-스텝 loss), Abbe globality 아님**(리뷰 D1).
> 우리 잎은 **이미 게더된 기지(旣知) boolean**(observed 룩업) → terminal은 *기지 비트의 AND/OR fold*이고
> 얕은 망도 가능 = **globality-hard 아님**. 단일-스텝이 막는 진짜 병목 = **깊은 중첩식 뒤 단일-토큰
> boolean 예측**(long-range). 처방(중간 sub-result t1,t2 supervise)은 정확히 **Kim&Suzuki의 기제**.

| 설계 결정 | 근거 논문 | 판정 |
|---|---|---|
| **중간 집계결과(t1,t2)를 target에 supervise** (★load-bearing) | **Kim & Suzuki 2410.08633 (ICLR25 Oral)** — 중간스텝 loss면 parity 1-step 학습, 없으면 *증명적으로* 불가(효율분리) | ✅ SUPPORTED(가장 직접) |
| **bottom-up reduction이 OOD depth-decay 완화**(우리 형식과 가장 근접한 실증) | **He 2025 2512.02677**(Looped Locate-and-Replace; depth12 66.7% vs 51.8%) · **Yehudai/Amsel/Bruna 2503.01544 (NeurIPS25)**(CoT가 depth↔n 순차 sub-result 토큰 교환; 2-layer가 Boolean-formula 평가) | ✅ SUPPORTED(★재탐색; 단 He=inference-루프, 우리=SFT 변형) |
| 깊은 중첩식 → **노드별 단일-토큰 예측으로 분해**(long-range 완화) | **Feng 2305.15408/2402.12875** · **RoPE bound 2411.07602(EMNLP25)** — no-CoT const-depth=TC0(BFVP=NC1-complete 평가 불가), CoT면 TC0 탈출·boolean circuit 평가 | ✅ SUPPORTED(expressivity) |
| **pairwise/소(小)-AND로 fold**(고민감도 회피) | **Bhattamishra 2211.12316 (ACL23)·Wang 2412.02823** — flat-AND 쉬움·고민감도 붕괴, ICL정확도↔식복잡도 r=−0.88(Qwen2-7B) | ✅✅ 우리 실측 직접설명 |
| 구조적(inductive) scratchpad(free-form 아님) | **Abbe 2406.06467 (NeurIPS24)** — globality 장벽은 inductive scratchpad만 깸 | ◐ **analogical**(우리 잎은 기지값→globality 직접 적용 아님; 형식 유추로만) |
| **잎=게더 truth 룩업**(재추론 금지) | 자체 T1T2 census(grounded AND(preconds) **0오류**) + litreview §4.3 | ✅ 자체실증 |
| 분해 자체의 *필요성* | **Dziri 2305.18654 (NeurIPS23)** — 트랜스포머=linearized subgraph matching, 복잡도↑서 error→1 | ✅(negative) |

**과대주장 경계(litreview §3 그대로)**: 대부분 1-layer/from-scratch 합성(parity/CVP)·GPT-2급 →
**7B LoRA tool-use 전이는 미검증**. expressivity 증명 ≠ SFT가 그 해를 복원한다는 보장. 본 실험은 *탐색적*.

## 4. 가설·지표·판정
- **H(inductive)**: 단계별 reduction이 단일-스텝의 globality 한계를 풀어 **BOTH > 5**(단일-스텝/control 천장 초과).
  특히 premature(acted∧¬dirgraph) ~10건이 "게더 완성 후 fold 닫힘"으로 **dirgraph↑→BOTH 전환**.
- **A/B**: control = 기존 single-step `treeval` 어댑터(데이터 byte-identical, 재학습 불요) vs `treevalind`(신규).
  **둘 다 PLAN_MAXTOK=1024**(truncation 통제). LODO holdout=bank, alias_s3, source=3, ep3, r16, SOLO.
- **분리지표(헤드라인)**: should_T(48) **BOTH**(dirgraph∩goal) + dirgraph + acted + premature + over-refuse;
  should_F(86) **STOP-recall**. + RLLOG **terminal-도달%**(둘 다 ~100% 기대, 비수렴 통제 확인).
- **판정** (★STOP 기준선 = **nt=42%**, single-step의 33% 아님 — 리뷰 D3. single-step grounding은 STOP을
  42→33%로 *9pp 악화*시키고 BOTH는 평탄 = net-negative였음. inductive STOP 35%를 "비회귀"로 오판 금지):
  - **성공** = treevalind **BOTH > 5**(특히 ≥ single-step+3) & **STOP-recall ≥ nt 42%** → inductive가 단일-스텝 한계를 깸 → §3.10 북극성 경로 + 조건수 분해.
  - **부분** = BOTH 6~7(소폭) → 방향 맞음, depth/조건수 분해로 어디서 새는지 확인 후 강화(노드당 별도 call 등).
  - **무효** = BOTH ≈ 5 → 단계별 supervise도 7B 단일-call로는 천장 못 깸. ★**무효의 의미(D2)**: terminal 형식
    변경이 천장 원인(=premature gather-termination, 상류)을 *못 건드림*을 뜻함 → **gather-side 레버(T1c)** 또는
    **노드당 1 planner-call**(§6.1)로 분기, *grounding 추가* 아님.
- ⚠️ Mean Pass Rate 단독 판정 금지(거부 부풀림). 분모=/48 주, /40 보조.

## 5. 측정 부가(ablation·진단)
- **조건수별(=fan-in) BOTH 분해**(2/4/6/8): ★재탐색 단서 — **Beam Tree 2305.19999**가 *인자개수(fan-in) 일반화*를
  길이/깊이와 별개 실패축으로 documented(≤5인자 학습→15인자 67.9%). **우리 "조건수↑서 BOTH↓"가 콜드붕괴가 아니라
  이 arg-count 일반화 실패일 수 있음** → 학습 조건수 분포 대비 held-out 조건수서 BOTH 곡선 확인(콜드붕괴 vs fan-in decay 분리).
- **OR-케이스 정확도**(litreview 21% OR): grounded OR-fold가 AND만큼 되는지.
- **RLLOG 체인 정확도**: 모델이 emit한 reduction의 중간 fold가 게더 truth와 일치하는지(허위 fold 검출). + 형식혼합(chain vs permitted)·chain끝값↔ACT/STOP 모순율(드라이버 census).

## 6. 리스크·교란(정직)
0. ★**#1 리스크 — 천장 원인이 terminal 형식의 *상류*다(리뷰 D2)**: §1 정정결과에서 단일-스텝 천장(BOTH=5)의
   직접 원인은 *틀린 gate boolean*이 아니라 **premature action**(acted 15 중 ~10이 게더 완성 전 행동).
   모델은 게이트를 *틀리게 계산*하는 게 아니라 **게더를 끝내기 전에 행동**한다. 그런데 inductive terminal은
   teacher-side에서 **게더 완료 후에만** 발화(코드 L323–325) → terminal emit을 늘려도 "더 게더하라"를 *직접*
   가르치지 않는다. §4의 "fold=checklist → 완전 게더 강제 → dirgraph↑"는 **emergent-behavior 가설**이며
   Abbe/Kim&Suzuki/Feng 어느 것도 *gather-완전성*을 다루지 않는다(전부 *함수 계산*에 관한 것).
   → **무효(BOTH≈5)면 결론 = "형식 변경은 gather 행동을 못 고친다"**, 처방은 grounding 추가가 아니라
   **gather-side(T1c)** 또는 **노드당 1 call**(아래 #1).
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
