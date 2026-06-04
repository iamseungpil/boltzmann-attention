# RUNG1 재설계 (2026-06-04) — over-call 근본원인 fix → gather 스케일-임계 + 결정 offload

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

**근본원인 = teacher required-set의 dep 불일치 (버그)**:
- `build_tbox_planner_sft.py:272` — required-set = `cleaves`(task constraints) **∪ `gleaves`**, 여기서 `gleaves = dep_innate[goal] ∪ dep_full_raw[goal] ∪ ops[goal]["precondition"]` (250-253행).
- 모든 은행 operator의 `ops[precondition]`엔 `logged_in_user`+`authenticated_admin_password`가 있음 → **required-set이 모든 태스크에서 auth를 과대포함** → teacher가 항상 login/admin 게더 시범 → 모델이 universal over-call 학습.
- 반면 teacher의 **실행 dep = task constraints**(219행 `task_dep[goal]=task["constraints"]`). ⇒ **게더-dep(dep_full)과 실행-dep(constraints)이 불일치.** 게다가 teacher의 login은 should_T서 GT cred 신뢰로 성공(345행)하나, eval서 모델은 실제 비번 없어 실패.
- 도입 경위(주석 240-242): "98/830 goal innate login 필요 → constraints-only면 under-login" 우려로 추가. **그 전제가 틀림**: `dep_innate` auth=0 (§0). dep_full에 있음 ≠ innate 요구.

**★LOCK 수정 (RUNG1_SOURCE_LADDER_DESIGN LOCK 보완)**: LOCK은 *"over-call = MODEL 회귀 = SFT-positive 불가, teacher는 이미 parsimonious"*라 했으나 — **auth축에서 teacher는 비-parsimonious**(dep_full 과대포함). 따라서 **auth-over-call은 음성신호(DPO) 없이 teacher consistency fix(SFT-positive)로 시험 가능**. LOCK의 "결정-emission SFT 3-NULL 종결"은 유효(treeval/T1c emission 변종 금지)지만, **required-set dep 정합화는 emission 변종이 아니라 데이터 결함 수정** = LOCK 범위 밖.

**⚠️보수적 단서 (반증 가능성)**: zero-train `LIGHTEN` 선례 = login 렌더 −59%인데 should_T 4/48 불변(추론시 login 감소만으론 무개선). teacher-fix는 *학습시* 수정이라 메커니즘 다르나, **"login 줄이면 should_T 오른다"는 보장 없음** → H1은 *가정 아닌 시험*. genuine over-call 25 중 auth가 아닌 잔여(constraint-violation 12 등)·intrinsic over-refuse는 fix로 안 사라질 수 있음.

---

## §2 가설 + 사전등록 (리뷰 대상)

| # | 가설 | 조작 | 사전등록 예측 | 합격 기준 | NULL이면 |
|---|---|---|---|---|---|
| **H1** | auth-over-call은 teacher dep 불일치가 원인 = SFT-positive로 가능 | required-set `gleaves`를 **dep_innate만**으로(272행 dep_full_raw·ops[precond] 제거) → 7B 재학습 | over-call↓·`gathered_then_REFUSE`↓·should_T(honest-34)↑ | should_T BOTH ↑ **유의**(사전등록 임계 아래) ∧ should_F STOP 비회귀 | over-call에 model-intrinsic/게이트설계 성분 → H3(offload/DPO)로 |
| **H2** | robust gather는 어느 최소 스케일서 학습·전이되나 | H1 teacher로 0.5/1.5/3/7/14B gather SFT 곡선 | `dirgraph_satisfied`·LODO 전이 곡선이 임계 드러냄(≤7B 예상) | 임계 국소화(±noise 밖) | 임계 없음/평평 = "≤최소" 보고 |
| **H3** | 결정(permitted)은 SFT로 안 되면 결정론 offload | 메모장형 `check_permitted`(결정론 over 모델 게더결과, unknown→deny) | BOTH = gather품질의 함수 | BOTH ≈ gather-bound(offload 후 결정실패 0) | 결정에 잔여 = DPO |

**사전등록 임계 (H1)**: should_T BOTH ≥ **12/34** (현 genuine over-call 25가 일부라도 전환되면 달성) ∧ should_F STOP ≥ nt-baseline의 42% (over-refuse 비회귀 가드, LOCK 게이트B 승계). seed 2(가능시), 분모는 **항상 honest-34**(Part A/B 제외).

---

## §3 Phased plan (비용·의존 순서)

| Phase | 무엇 | 모델 | 비용 | 게이트 |
|---|---|---|---|---|
| **0** | 진단·정직분모 (DONE) | — | zero | 0a 완료(over-call 지배·teacher 원인). |
| **1** | **H1 teacher consistency fix** (272행 dep_innate-only) → 재학습 → honest-34 eval. **LOCK 수정의 시험대.** | 7B (싼 트랙) | 중(~4h) | over-call↓∧should_T↑면 H2 진행; NULL이면 H3 직행. |
| **2** | H2 gather 스케일곡선 | **0.5/1.5/3/7/14B**(sub-7B=R1) + A1 스크린(base-gather+valid-call) + full-LODO(7B)+대표 holdout 2 | 중(작은쪽 쌈) | 임계 국소화. |
| **3** | H3 메모장형 offload(unknown→deny) + (잔여시) DPO | 7B; **대형 32B/72B = B축(decision-emission) 전용** | 저~비쌈(좁게) | BOTH=gather-bound. |

**메타규칙(승계)**: 각 4h launch 전 ①이 변종이 dead-end(emission 스캐폴드)인가 — H1은 *데이터 fix*라 아님 ②zero-cost 진단 끝났나 — 0a 완료. 둘 통과.

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

1. **H1 vs LIGHTEN 선례**: 추론시 login−59%가 should_T 무개선이었는데, *학습시* teacher-fix가 성공할 근거가 충분한가? (우리 주장: 모델이 over-call을 *학습 안 함* vs 학습된 행동에 프롬프트만 바꿈 — 메커니즘 상이. 단 보수적, H1은 시험.)
2. **constraints 충분성**: H1이 dep_full을 버리는데, eval의 `dirgraph_satisfied`는 dep_full dirgraph를 강제. honest-34에선 OR-bypass/no-auth만 남아 안전(Part B 제외)이나 — **OR-bypass 태스크에서 모델이 login을 *안* 부르면 dirgraph 비-auth 분기로 통과함이 확실한가?** (66/67 실증=통과. 일반화 검증 필요.)
3. **잔여 over-call**: auth 외 over-call(constraint-violation 12)은 H1로 안 사라질 수 있음. H1 후 잔여를 재census(권위본 인용)해 H3 범위 확정.
4. **offload 메커니즘 정의**: 메모장형 게이트의 unknown-handling(deny vs abstain)이 BOTH를 움직이는 튜너블 — deny 사전등록이 배포-보수와 정합한가?
5. **분모 합의**: honest-34가 모든 비교의 고정 분모임을 coworker(32B/72B)와도 공유(레이스·셔플 가드).

---

## §7 구현 메모 (H1)

- fix 1줄: `build_tbox_planner_sft.py` — `gleaves`를 `dep_innate[goal]`만으로 구성(250-253행에서 `dep_full_raw`·`ops[goal]["precondition"]` 제거), 272행은 그대로. 효과: required-set = constraint leaves(cleaves) + innate 바닥(auth 0). constraints에 `logged_in_user` 있는 31건은 cleaves로 login 유지, 나머지는 auth 미게더.
- 검증(빌드 후, 학습 전): set_safety_box류(login∉constraints) teacher 트레이스에 login/admin 호출 0; apply_credit_card류(login∈constraints) 트레이스엔 login 유지. = zero-cost 사전검증.
- 재학습: 기존 `rung1_train_eval.sh` 레시피(LODO holdout=bank, ep3, alias_s3). eval `SOPBENCH_SOURCE`·maxtok=1024. 권위본 freshness-guard.
