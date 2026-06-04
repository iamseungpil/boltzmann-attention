# RUNG1 재설계 (2026-06-04) — credential-augment(⓪) → gather 스케일-임계 + 결정 offload

> **수정 이력**: 최초 §1은 "over-call=teacher dep 결함→prune(H1)"로 LOCK 수정을 시도했으나 **reliable leave-one-out(실제 evaluator)로 철회** — login은 dirgraph-REQUIRED(12/17), 실패원인은 credential 부재. 레버 = **credential-augment**(prune 아님). LOCK 유지.

> **상태**: 리뷰용 사전등록 설계. 진입점 = 이 문서. 권위본 결과 = `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`(★★Gate-A). 상위 = `EXPERIMENT_DESIGN.md §2`. LOCK = `RUNG1_SOURCE_LADDER_DESIGN.md`(§아래 수정).

---

## §0 읽기 전 — SETTLED (재유도 금지, 인용만)

이 결론들은 확정됐다. 다시 census/probe로 재유도하지 말 것 (반복 사고 방지 — 메모리 `feedback-check-authority-before-rederive`).

- **정직 should_T 분모 = 34/48** = 48 − Part A 8(credit_card 결함) − Part B 6(login-mismatch 결함). 권위 = `BUGREPORT_SOPBench_bank_impossible_tasks.md` Part A/B. "24"·"40"은 철회/중간값.
- **Part B 6**: `directed_action_graph`(dep_full 생성)가 login mandatory인데 `constraints`/`user_instruction`은 username-only·cred 미제공 → unwinnable. **augment 아닌 제외**(username-only가 태스크 의도). pay_loan 66/67은 no-login OR분기로 통과 = 제외.
- **auth 필요/불요 구분 가능**: 권위 = `task["constraints"]`(79% 즉시) + dirgraph mandatory-vs-OR + cred 가용성(21%, Part B 방법). `dep_innate`엔 auth 전무 = 어떤 goal도 login innate 요구 아님.

---

## §1 이번 세션이 바꾼 것 — over-call 근본원인 (코드 확정)

**0a 진단**(권위본 ★★Gate-A): 학습 모델 should_T 한계 = **over-call**(genuine ≈ 25/34) — 태스크 constraint가 요구하지 않는 `login_user`/`authenticate_admin_password`를 호출·실패 → 충족불가 auth leaf가 treeval AND-게이트를 붕괴 → `gathered_then_REFUSE`(42) → goal 미호출.

**⚠️ 폐기된 1차 가설 (H1, prune-login) — RETRACTED**: 최초 이 문서는 "teacher required-set이 `dep_full∪ops[precond]`(272행)로 login/admin 과대포함 = 실행 dep(constraints,219행)과 불일치 = 데이터 결함 → gleaves=dep_innate-only로 prune하면 over-call 소멸"이라 주장하고 LOCK을 수정했다. **이는 reliable test로 반증됨 — pin 철회.** (재유도 패턴 재발: 코드 주석 240-242 rationale·credential 진단과 모순되는데 검증 없이 박았다.)

**★근본원인 (reliable leave-one-out, 실제 evaluator, 재현 17/17 검증) — login은 dirgraph-REQUIRED, 과대포함 아님**:
- 테스트: should_T 17건(login∉constraints)의 *실제 T1c 궤적*에서 login/admin 호출만 제거 → `evaluator_function_directed_graph`(미변경 실제 evaluator)로 재채점. (hand-rolled graph-replay 아님 = `mre_bank_impossible.py` 비신뢰 회피.)
- 결과: **login 제거 시 `dirgraph_satisfied` True→False = 12 (MANDATORY) / 유지 = 3 (OR-bypass) / 이미 False = 2.** ⇒ **login은 12/17서 호출이 dirgraph에 load-bearing.** prune하면 dirgraph가 무너짐 = 주석 240-242의 under-login 경고 그대로. **bench evaluator가 dep_innate_full(innate+full dep, login 포함) 기반**이라 directed_action_graph가 login을 요구; 219행은 `constraint_not_violated`용 dep일 뿐 dirgraph dep 아님 — **1차 가설은 두 eval 축을 혼동했다.**
- **호출 vs 성공 구분 (66/67 화해)**: dirgraph는 login이 *호출*되면 충족(반환값 무관) / treeval grounded 게이트는 login이 *성공*해야 leaf=true. 66/67은 호출-mandatory지만 성공-불요(constraint OR분기, Part B). **credential 부재 → login=false → grounded 게이트 붕괴 → STOP.** = **dirgraph-required login의 credential-실패**이지 과대포함 아님.
- **admin (혼합)**: `ops[precond]`에 admin 있는 goal(set_safety_box·cancel_cc·transfer)=teacher 게더 / 없는 goal(apply_cc·get_loan·pay_bill·deposit)=**MODEL 회귀**. 어느 쪽도 prune 레버를 정당화 못함.

**★LOCK 유지 (수정 없음, auth축)**: login over-call은 dirgraph-required login의 **credential 실패**(teacher 결함 아님) → 레버 = **credential-augment**(⓪, 비번 surface→login 성공→leaf true). admin over-call은 (admin∉precond goal서) **MODEL 회귀** = LOCK대로 SFT-positive 불가. **LIGHTEN 선례(login−59%·should_T 불변)도 prune류 무효를 지지** — login 호출수 줄여도 binding 실패(credential 부재 + policy-leaf cold-bias)는 그대로. ⇒ augment 후 잔여 거부 = policy-leaf cold-bias(T1c census op_2=false on gathered-true) = LOCK'd emission 문제 → offload(메모장)/DPO.

---

## §2 가설 + 사전등록 (리뷰 대상)

| # | 가설 | 조작 | 사전등록 예측 | 합격 기준 | NULL이면 |
|---|---|---|---|---|---|
| **⓪** | login over-call 실패 = dirgraph-required login의 **credential 부재**(prune 아님; leave-one-out 12/17 mandatory) | **credential-augment**: 비번만 user_known surface(누출 금지, Part B 제외) | login 성공률↑ | — | — |
| **⚠️⓪ zero-cost 게이트 결과 (2026-06-04, alias-independent, 실제 eval) = ⓪-단독 NULL 예측** | login=True인 should_T 30건 중 **success 0·refused 28·acted 2** → login 이미 성공해도 전부 실패. credential-augment가 고치는 건 login=False 18건뿐인데, **login=True 30건이 grounded-gate 붕괴(non-login leaf cold-bias·`op_20=650` emission)로 이미 전멸**. ⇒ **⓪-단독은 should_T 못 올림.** credential-augment = H3의 *필요 입력*(deterministic 게이트가 login=true를 보게)이지 단독 레버 아님. | ⓪+H3 결합 | — | 이미 결정: H3로 |
| ~~H1~~ | ~~teacher dep 불일치 prune~~ | **RETRACTED** — leave-one-out 반증(login dirgraph-required 12/17). prune=under-login. | — | — | — |
| **H2** | robust gather는 어느 최소 스케일서 학습·전이되나 | 현 teacher(prune 아님; login 유지 = dirgraph-정합)로 0.5/1.5/3/7/14B gather SFT 곡선 | `dirgraph_satisfied`·LODO 전이 곡선이 임계 드러냄(≤7B 예상) | 임계 국소화(±noise 밖) | 임계 없음/평평 = "≤최소" 보고 |
| **H3** | 결정(permitted)은 SFT로 안 되면 결정론 offload | 메모장형 `check_permitted`(결정론 over 모델 게더결과, unknown→deny) | BOTH = gather품질의 함수 | BOTH ≈ gather-bound(offload 후 결정실패 0) | 결정에 잔여 = DPO |

**사전등록 임계 (⓪)**: should_T BOTH ≥ **12/34** ∧ should_F STOP ≥ nt-baseline의 42% (over-refuse 비회귀 가드, LOCK 게이트B 승계). seed 2(가능시), 분모는 **항상 honest-34**(Part A/B 제외). ⓪이 NULL이면(LIGHTEN형) binding=policy-leaf cold-bias → H3.

---

## §3 Phased plan (비용·의존 순서)

| Phase | 무엇 | 모델 | 비용 | 게이트 |
|---|---|---|---|---|
| **0** | 진단·정직분모 (DONE) | — | zero | 0a 완료(over-call 지배·teacher 원인). |
| **⓪** | ~~credential-augment 단독~~ → **zero-cost 게이트로 NULL 예측**(login=True 30건 success 0). ⇒ **⓪-단독 skip**, credential-augment는 H3의 입력으로 흡수. | — | zero(완료) | **H3 직행** |
| **H3'** | **메모장형 `check_permitted` offload** (결정론 게이트 over 모델 게더결과; credential-augment로 login=real-true 보장; unknown→deny) → 모델 emission gate 우회 | 7B; 구현 필요 | 중 | gate 붕괴 제거 → should_T↑면 = 진짜 레버 |
| **2** | H2 gather 스케일곡선 | **0.5/1.5/3/7/14B**(sub-7B=R1) + A1 스크린(base-gather+valid-call) + full-LODO(7B)+대표 holdout 2 | 중(작은쪽 쌈) | 임계 국소화. |
| **3** | H3 메모장형 offload(unknown→deny) + (잔여시) DPO | 7B; **대형 32B/72B = B축(decision-emission) 전용** | 저~비쌈(좁게) | BOTH=gather-bound. |

**메타규칙(승계)**: 각 4h launch 전 ①이 변종이 dead-end(emission 스캐폴드)인가 ②zero-cost 진단 끝났나. ⓪(credential-augment)는 무재학습 우선이라 launch 게이트 가벼움. **★이번 사고 교훈**: 강한 주장(LOCK 수정 등)을 박기 전 reliable test(실제 evaluator) 필수 — H1을 추론만으로 박았다가 leave-one-out으로 철회.

---

## §4 통제 (사전등록, R1–R9 + A1/A2/A4)

- **R1 (비용역전)**: gather 곡선에 sub-7B(0.5/1.5/3B) 필수(7B가 이미 gather→임계 ≤7B). 대형은 B축 전용·full 매트릭스 금지.
- **A1 (능력 바닥)**: 스케일별 base-gather + valid-tool-call rate 선측정 → "gather 못함" vs "도구조작 못함" 분리.
- **R3/A2 (offload)**: 메모장형(oracle 아님=upper bound). unknown→deny 사전등록. BOTH = dirgraph ∧ goal-call-correctness(slot/arg) ∧ 게이트.
- **R4 (LoRA)**: r 고정 + 1스케일 rank-sweep(r8/16/32) → rank-bound 아님. base 전부 Qwen2.5-Instruct.
- **R5 (전이 n>1)**: 7B full-LODO(≥4 holdout) + 타스케일 대표 2. "≥70%"는 in-domain 상대값 사전등록.
- **R6 (credential)**: **scoped** — augment=비번만 surface(누출 금지), **base 측정 통제로만**, realistic 병행. **Part B엔 부적용**(username-only 의도 → 제외). 학습 모델 천장 레버 아님(0a).
- **R7 (tool-change)**: rename/add/remove는 7B 파일럿 후 대형 투입.
- **R8 (검정력)**: honest-34, 도메인-mix 고정, 곡선판정 사전등록, seed 2.
- **R9 (positioning)**: offload-decide = LLM-Modulo(Kambhampati) 인용. novelty = ①gather 스케일-임계 ②LODO ABox-swap 무재학습 전이 ③도구변경 robust.

---

## §5 지표·분모

- **분모 = honest-34** (절대 /48, /24, /40 아님). Part A 8 + Part B 6 제외.
- **헤드라인 = BOTH**(should_T: dirgraph_satisfied ∩ goal-call-correctness) + **should_F STOP 비회귀**(거부축). gather 1차지표 = `dirgraph_satisfied`.
- 결과는 권위본 `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-4 행에 기록(scratch 방치 금지).

---

## §6 리뷰 훅 — 비판 요청 (reviewer가 박제 전 확인할 것)

1. **⓪ credential-augment vs LIGHTEN 선례**: LIGHTEN(login−59%·should_T 불변)은 login *호출수* 감소였고, augment는 login *성공*을 노린다(다른 레버). 단 LIGHTEN이 should_T 무개선이었던 진짜 이유가 (a) credential 부재 외에 (b) policy-leaf cold-bias도면, augment도 부분개선에 그칠 수 있음 → ⓪ NULL이면 즉시 H3. **augment는 prune 아님**(login은 dirgraph-required, leave-one-out 12/17).
2. **dirgraph-required 범위**: leave-one-out = login mandatory 12·OR-bypass 3·base-False 2 (17 should_T, login∉constraints). admin은 login과 함께 제거해 분리 안 됨 → admin-only leave-one-out으로 admin이 dirgraph-required인지 추가 확인 가능(set_safety_box류 admin∈precond).
3. **잔여 over-call/refuse**: augment 후 잔여 = policy-leaf cold-bias(emission, LOCK'd) + auth 외 over-call(constraint-violation 12). 재census(권위본 인용)로 H3 범위 확정.
4. **offload 메커니즘 정의**: 메모장형 게이트의 unknown-handling(deny vs abstain)이 BOTH를 움직이는 튜너블 — deny 사전등록이 배포-보수와 정합한가?
5. **분모 합의**: honest-34가 모든 비교의 고정 분모임을 coworker(32B/72B)와도 공유(레이스·셔플 가드).

---

## §7 구현 메모 (⓪ credential-augment)

- **prune 금지**: required-set의 login은 dirgraph-required(leave-one-out 12/17)라 *유지*. `build_tbox_planner_sft.py:272`의 gleaves union은 dirgraph와 정합하므로 손대지 않음.
- ⓪ augment: should_T 태스크의 `user_known`에 **그 태스크가 login을 요구할 때만** 실제 identification surface(`initial_database.accounts[user].identification`). **admin_password는 신중**(Part B 6은 username-only 의도 → 제외; admin∈precond goal은 별도 판단). 누출 통제: 비번 외 제약 truth·게이트 결정 미surface.
- 측정: realistic(현행) vs augmented 병행 보고. login 성공률·`gathered_then_REFUSE`·should_T(honest-34)·should_F 비회귀.
- 무재학습 우선(프롬프트/user_known만 변경) → 효과 보이면 재학습으로 확정. 권위본 freshness-guard.
- **검증 도구**: leave-one-out 재현 스크립트(본 세션, `evaluator_function_directed_graph` 사용) = dirgraph-required 판정의 reliable 근거. 재실행으로 admin-only·타도메인 확장 가능.
