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
| **H2** | robust gather는 어느 최소 스케일서 학습·전이되나 | 현 teacher로 0.5/1.5/3/7/14B gather SFT 곡선 | **goal-precond-complete / clean-H3-success** 곡선이 임계 드러냄(⚠️**natural `dirgraph_satisfied` 금지** = refuse 궤적서 over-report) | 임계 국소화(±noise 밖) | 임계 없음/평평 = "≤최소" 보고 |
| **H3** | 결정(permitted)은 SFT로 안 되면 결정론 offload | 메모장형 `check_permitted`(결정론 over 모델 게더결과, unknown→deny) | BOTH = gather품질의 함수 | BOTH ≈ gather-bound(offload 후 결정실패 0) | 결정에 잔여 = DPO |

**사전등록 임계 (H3, ⓪-단독 skip 후 이전)**: 실제 H3 rollout eval에서 should_T success ≥ **12/34** ∧ should_F STOP 비회귀(LOCK 게이트B 승계). seed 2(가능시), 분모 **항상 honest-34**(Part A/B 제외). ⚠️offline forced-ACT는 측정수단 아님(§8 RETRACT) — H3는 실제 rollout으로만 측정.

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

---

## §8 A-arm 헤드룸 시도 (2026-06-04) — ⚠️RETRACTED: offline full-success 신뢰불가, one-vs-two-axis 미해결

**측정**: 각 should_T 태스크의 *모델 실제 게더 궤적* + login/admin **augment**(실제 cred 주입) + **강제 ACT**(goal 호출 append) → 미변경 `evaluator_function_directed_graph` 재채점. = "결정을 offload하고 ACT시키면 현 게더로 몇 건 성공?"

**결과 (honest 분모 주의)**: FULL success = **11/48**. goal *실행 가능* = **40/48**(Part A 8만 실행불가). 비-성공 37의 게이트 분해: **`dirgraph_satisfied`=False 34** / action_called=False 8(Part A) / tool_call_error 3. **`constraint_not_violated`·`database_match`는 통과.**
- per-goal: 단순-선행 goal 성공(apply_credit_card 4/4·deposit 2/2), **복합-선행 goal 전멸**(set_safety_box 0/10·transfer_funds 0/8·pay_loan 0/4·cancel_cc 0/6). ⇒ **artifact 아님 = 실제 게더 결핍**: 모델이 goal의 *실제 dirgraph 선행조건*을 못 세움(auth 과다게더·goal 실제 dep 과소게더·slot 불완전 예 transfer 이중 username-check).

**⚠️⚠️ §8 RETRACT (2026-06-04, 리뷰어 push → clean 측정으로 철회)**: 아래 forced-ACT full-success(11) 및 §8.1 "DB-read 처방"은 **offline 신뢰불가로 철회**. 검증:
- **clean 측정 (honest-40, Part A 제외)**: forced-ACT full-success **WITHOUT fix 11/40 → WITH `internal_get_database` 6/40 (↓!)**. DB-read 추가가 **dirgraph는 올리나(14→42) full-success는 내림**(close_account 1→0·deposit 2→0·get_loan 1→0: DB 전체읽기가 `constraint_not_violated` 위반). ⇒ **§8.1 "DB-read = A축 처방"은 net 음성 = 철회.**
- **offline full-success 자체가 artifact-prone**: 결정론 clean gather(probe)도 full-success **9**, 모델 11, 모델+DB 6 — 구성 궤적은 무엇이든 constraint/database 게이트서 낮게 나옴. ⇒ **forced-ACT offline로 H3 헤드룸 측정 불가. "11→34 / 두 축 필요" sizing 철회.**
- **신뢰 잔존 사실**: 게더 incomplete는 reliable(모델 dirgraph 14/48 vs clean 48/48, 동일 append 구성; natural 43은 over-report=goal 미호출). 단 **dirgraph-incomplete가 곧 success-deficit은 아님**(DB-read가 dirgraph↑·success↓로 증명). ⇒ **one-vs-two-axis는 offline로 미해결 — 실제 H3 eval(check_permitted in-loop rollout) 필요.**

**~~★해석 (프레이밍 수정)~~ — 위 RETRACT로 무효. 정직한 현 위치**:
1. goal 실행가능 40/48(천장 존재)는 유효.
2. 게더는 dirgraph 기준 incomplete(14 vs 48) = reliable. **하지만 offload 단독이 몇 건 회복하는지는 offline 측정불가**(constraint/database artifact).
3. **다음 = 실제 H3 구현·rollout eval**(offline 재구성 아님)이 one-vs-two-axis를 가름.

### §8.1 ~~gather-타겟팅 처방~~ (RETRACTED — DB-read는 net 음성)
- 신뢰 게이트(clean 48 vs 모델 14)·근본원인 분석(`internal_get_database` ∈dirgraph 46·∉constraints 0·predicate 아님 → teacher가 구조적으로 못 담음)은 **사실로 유효**.
- **그러나 "DB-read를 게더에 추가" 처방은 철회**: full-success를 내림(constraint 위반). teacher가 constraints를 타깃하고 eval은 dirgraph를 채점하는 **mismatch는 실재**하나, dirgraph를 DB-read로 맞추면 constraint를 깬다 = **벤치 내부 tension**(dirgraph vs constraint). 단순 처방 없음.
- **★통합 근본원인 (login·DB-read 동일, 사실로 유효)**: teacher 게더타깃=`constraints` / eval 게더metric=`directed_action_graph`. 단 이 mismatch의 *해법*은 미정(DB-read 추가는 역효과). transfer 0/8 = slot-axis(teacher slot-fix됐으나 모델 미준수=별개)·cancel 0/6 = Part A defect(gather 무관).

## §9 결정-축 격상 (decision-axis = 소형모델 환각 제거 비교) — 사용자 제안 (2026-06-04)

**문제**: T1c는 도구를 실제로 돌려놓고도 **결정을 모델이 emit(재-생성)** 하게 해서 cold-bias 환각(login 실제 True 30건 → 게이트에 false 적음 → 0 성공·`op_20=650` emission). **핵심 = 생성 ≠ 복사**: 맥락에 정답이 있어도 작은 모델은 구조화 emit서 prior로 덮어씀. (사용자 §1 원칙 위반: "노트 도구는 raw 관찰만, ready 판정 담지 말 것".)

**논문 축 = 결정의 충실성을 어떻게 얻나, 3~4 arm 비교**:
| arm | 무엇 | 위치 |
|---|---|---|
| **C** (완료·NULL) | 자기-emit derivation (T1c) | 환각 기준선 (login 30→0) |
| **A** | 결정론 offload — 기록된 raw 결과로 시스템이 permitted | 충실성 상한 (= LLM-Modulo, 단독 novelty 약). **A-arm §8: offload→gather-bound 실증** |
| **B** | 모델 판단 + **결정론 verifier와의 차이를 DPO/RFT로 교정** 학습 | **핵심: 충실 판단을 weight 내재화 가능한가** (LOCK의 음성신호 escape, SFT-positive emission 아님 = 재-tread 아님) |
| **B'** (옵션) | derivation 없이 **기록값 grounded-copy** 학습(최소 emission 표면) | cold-bias 이기는 최소 개입 |

**novelty = A↔B**(offload는 룰엔진 의존, B 성공시 = 검증 weight-baking → 엔진 없이 전이). **양 결과 게재가능**(B 성공=내재화 가능 / B NULL=offload 필수, LLM-Modulo 강화). **메모장 = harness가 도구출력 verbatim 기록, 모델은 읽고 판단/교정만**(재-state 금지). 위험: B가 cold-bias(LIGHTEN·30→0 robust) 이길지 미지수=시험 / B는 A 대비 *전이*로 정당화 / positioning vs process-supervision·verifier-RFT.

**현 데이터의 함의**: A-arm(§8)이 offload→11(gather-bound)을 보였으므로, decision-axis 비교는 **gather를 먼저 정상화한(또는 동시) 위에서** 해야 결정 효과가 분리됨. 순서 = (A축 게더-타겟팅 개선) ∥ (offload A) → B/B' 학습.

## §10 H3 check_permitted 구현 + offline deny-by-unknown 프리뷰 (2026-06-05)

**구현(`two_stage_client.py` env SOPBENCH_OFFLOAD)** — 모델 emit 게이트 폐기, 게더된 실제 도구결과로 결정론 ACT/STOP. **5 정확성 잠금**:
- #1 게더-only: leaf truth를 history에서만, 미게더→unknown→deny. **DB는 getter가 게더된 leaf에만 읽음(getter==DB값이라 오라클 아님)**; arg-matching 보수적(불일치→deny, over-permit 불가).
- #2 벤치 `Dependency_Evaluator` subclass — combinators+condition 계산 재사용, `_single`만 evidence-gated override.
- #3 ACT시 모델이 goal 호출(arg-correctness 측정), harness auto-call 아님.
- #4 deny 분해: **ungathered(도구 미호출=게더축) / argmismatch(호출·args불일치=slot축) / false** — reviewer 지적(두 work-stream 분리).
- #5 credential-augment companion(미구현, 다음).

**offline 프리뷰 (T1c 궤적, augment login=True, honest n=40) — 검증·artifact-free**:
- 변환 충실도 확인(raw non-error tool == captured `called`); deny-by-unknown은 *unchanged gather 궤적의 terminal-time 속성*이라 offline=live의 check_permitted와 동일(§8 trap 아님 — append 없음). **단 BOTH/full-success는 절대 offline 보고 금지 = live rollout만.**
- **결과**: ACT permitted **10** / STOP ungathered **22** / STOP argmismatch **8**.
  - ungathered(게더축) 지배 = **get_account_balance 18·get_account_owed_balance 7** 미호출(balance 조건 getter); 모델은 credit getter 과다게더. = **게더-타겟팅(엉뚱한 getter)**.
  - argmismatch(slot축) = **internal_check_username_exist 8** = transfer 이중-username(별개 처방, slot-fix 잔여).
- **one-vs-two-axis(artifact-free) = TWO-AXIS**: 게더 진짜 incomplete(22 ungathered), strict-gather 14/48과 일치. 단 게더축은 specific(balance getter), slot축은 분리.
- ⚠️ v1 confound(computed condition을 callable 오취급→spurious unknown) 자가검출·수정(evidence-gated bench-compute v2). 검증 census로 사유 정정 확인.

**★active-H3 레버 (reviewer, no-retrain)**: 게이트가 *어느 leaf가 ungathered인지* 앎(info dict) → passive STOP 대신 **"not permitted — gather get_account_balance" 피드백**으로 게더 유도 = 결정론 게이트를 gather-driver로. 22 ungathered 중 상당수를 **무재학습**으로 ACT 전환 가능 → gather-retrain 전 1차 시도.

**다음**: (1) credential-augment task 전처리(lock #5) (2) **live rollout**(SOPBENCH_OFFLOAD + augment) → BOTH 헤드라인 + deny-by-unknown 라이브; 대조 T1c-emit(BOTH 1)/C-none(3) (3) active-H3 파일럿. push=57fe779.

### §10.1 active-H3 전 zero-cost 진단 2개 (2026-06-05, 리뷰어 요청) — ordering 결정
**① login 커플링 (deny 버킷 × 원래 login 결과)**: 22 ungathered = **login=False 9** (augment가 세션 unblock해 전환할 *상한*) + **login=True 13** (로그인 됐는데도 balance getter 스킵 = augment-독립 게더 결핍). ⇒ credential-augment는 ≤9 전환, 핵심 게더 결핍(13+)은 별개 게더-축.
**② teacher-trace census (coverage-gap vs model-skip)** — `build_tbox_planner_sft.py:282-288` branch 재현(올바른 available tool_names): teacher required-set이 balance 조건을 **branch B(getter_map→get_account_balance)로 GATHERS**(safety_box_eligible·sufficient_account_balance·pay_loan_*_restr 전부; argonly 미게더는 max_deposit/exchange 3건뿐). ⇒ **MODEL-SKIP, coverage gap 아님**: teacher 시범하는데 모델 스킵 = **게더도 SFT-positive 저항**(decision축과 동일 실패모드) = 리뷰어 통합가설 지지. **둘 다 RFT/음성신호 필요.** ⚠️caveat: alias 반-암기(train≠eval salt)로 eval서 의미매칭 실패 가능성 — 단 credit getter는 over-gather·balance만 스킵 = *선택적* → content-bias/SFT-limit 쪽; non-alias 1회로 완전분리.
- ⚠️신뢰성: 이 진단서 observed-called를 available-tool로 착각 2회 + string-search aliasing 아티팩트 자가검출, 올바른 dss 빌드로 정정.

**★ordering 정정 (active-H3 먼저 철회)**: (1) zero-cost 진단 2개=완료 → (2) **credential-augment + passive-H3 live rollout**(구현됨; 정직 베이스라인=모델 자율게더∧완벽결정; augment ≤9 전환; offline deny-by-unknown 라이브 확인; 대조 T1c-emit 1/C-none 3) → (3) **active-H3**(gate-DRIVEN gather=배포/LLM-Modulo 증명, axis-A 아님; passive 후라야 자율게더 vs gate-driver 분리). **active-H3 성공=모델이 자율 게더 못 함의 증거라 axis-A와 긴장 — 헤드라인으로 쓰지 말 것.**
