# 통합 실험 계획 (2026-07-06) — 결과 반영 재정리

> 목적: (1) 현재까지 실측으로 확정된 것, (2) **전이 폭 확장**(retail 외 다도메인), (3) **실패-경로 전수 포렌식→성능 개선**, (4) **논문용 기-준비 실험 재검토**를 하나로 통합.
> 불변(행동 전 점검): [[05]] 고정={TBox weights + Scaffold 엔진}·변경={ABox}만 · [[11]] 전이=ABox-swap·도메인-타깃 학습 금지 · [[13]] 흡수 우선순위 scale→학습(무망각)→(최후)scaffold/A2 · [[08]] 집계→결론 금지·전수 궤적 포렌식·robust pass^k(pass^1 금지) · [[09]] 유료런=로컬 무료검증 선행·스모크·승인·최소 scope.

---

## 0. 한 장 요약 (지금 위치)

- **확정(claim 가능)**: ① compliance 규모-불변(g2 per-write flat)·게이트가 전 규모 위반 0 = **강한형 백본**. ② 조립 스택 robust pass^3: **14B 0.313 / 32B 0.457** — 조립 14B > bare 32B(plain 0.281·compliant 0.219). ③ frontier 리더보드서 조립 32B(0.457) = o4-mini 수준·하위 frontier 진입·compliant 기준 우리 낙폭 0(결정론 보장). ④ **SOPBench 도메인-간 전이(bank→held-out 6도메인) 평균 77.3% public success·재학습 0·ABox-swap** = 전이가 *한 벤치 안에서는* 실증됨.
- **미확정/취약(현 최대 공백)**: ⓐ **τ²-retail 밖 전이 미측정**(airline 스펙만·bank/telecom 미착수). ⓑ **cross-bench 통합-TBox→τ² 전이 = 현재 0 pass**(Synth v6, 2-hop binding P2b 미해결). 즉 "학습된 도메인-일반 스킬이 τ²로 ABox-swap 전이"라는 **헤드라인 전이 주장은 아직 미실증**(특허 P5 각주로 로드맵 처리 완료). ⓒ 비용 배수 = 토큰 계측 전 추정(정성만). ⓓ gpt-5.2-sim 32k 재측정 in-flight(40/64).
- **정리된 방향**: 결정론 레버는 τ²-retail서 **대체로 소진**(make-or-break NO-GO·operand 32B gap 없음). 다음 성능개선 여지 = **orchestration-under-load**(GPU-free plan-probe가 learn GO/NO-GO 판가름) + **전이 폭 확장**(진짜 whitespace).

---

## 1. 확정된 실측 (재유도 금지·`_WRITING_BRIEF §3` 정본)

| 축 | 결과 | 출처 |
|---|---|---|
| compliance 규모-불변 | g2 per-write flat(7B .103/14B .070/32B .075·CI중첩)·게이트 전 규모 위반 0 | §3.1 |
| operand 실행 | given-spec 32B 88/88(100%)·goal-only 70%·격차=기준해석 | §3.2 |
| 부하 은퇴 | 유효부하≈{길이·조건} 2D·L_state/L_interf 규모 은퇴 | §3.3 |
| 무학습 결정론 회복 | grounding 33→9·통과 0.14→0.264 | §3.4 |
| 조립 스택 | robust pass^3 14B 0.313 / 32B 0.457·floor compliant 7B .035/14B .152/32B .219 | §3.5 |
| 실패 원인 분포 | clean robust-fail 32B 21·14B 30·단일 지배 없음 | §3.5 |
| globality 라우팅 | max-select 77%→S 회복·intent 68%→T 후보·결합 50→85% | §3.7 |
| frontier 리더보드 | 조립 32B 0.457 = o4-mini·compliant 낙폭 우리 0 vs frontier 0~2.6pp | §3.9·3.9b |
| **SOPBench 전이** | bank→held-out 6도메인 avg **77.3%**·재학습0·arm-4a 9.5→26.1 | SOPBENCH_EXPERIMENT_RESULTS Exp-5 |
| TaskBench scale | edge-F1 72B 45.8·GPT-4 69.3 대비 −25.5pp·LODO 홀드아웃 전이 null(단일템플릿) | TASKBENCH_EXPERIMENT_RESULTS |

---

## 2. Track 1 — 전이 폭 확장 (retail 외 다도메인) ★최우선 whitespace

**문제**: 헤드라인은 "도메인-일반 TBox가 ABox-swap으로 전이"인데, τ²는 retail 한 도메인만 실측. SOPBench-내부 전이(77.3%)는 있으나 **벤치-교차 전이(학습=SOPBench/TaskBench/Synth → 타깃=τ²)는 현재 0 pass**. 이걸 닫는 게 논문 핵심.

### 2.1 근거리(설계 완료·미실행) — 대부분 GPU-free 또는 저비용
| # | 실험 | 상태 | 비용 | 산출 |
|---|---|---|---|---|
| T1a | **τ²-airline ABox-swap**(동일 TBox·게이트, ABox만 airline) — base ±gate robust pass/compliant | 스펙 컴파일 완료·게이트 replay-clean 0/108 over-deny·**agent 트라이얼 0** | 로컬 무료(base) + 확인용 유료 1회 | retail 밖 첫 전이점·게이트 도메인-일반성 실증 |
| T1b | **τ²-airline 게이트 낙폭 0 재현** — compliant 낙폭이 airline서도 0인가 | 미실행 | 무료(기존 궤적 재분석 우선) | 규모-불변 준수의 도메인-불변성 |
| T1c | **ODCV-Bench refusal-axis 스팟**(40 task) base vs +gate 위반율 | 설계됨·미실행 | 무료/저비용 | KPI-유혹 시나리오서 게이트 0% 실증(2차 준수 벤치) |
| T1d | **Amazon SOP-Bench 12-domain LODO base census** | step-0 착수 | 무료(census)→학습은 coworker | 전이 폭 12도메인으로 확대(SOPBench 밖) |

### 2.2 원거리(핵심·미해결) — cross-bench 통합 TBox → τ² 전이
- **현상**: 통합 7B TBox(SOPBench rollout + TaskBench graph) → τ² = **v6 0 pass**(vs base 0.17). 진단 = **P2b 2-hop entity binding(fetch→arg)** 미학습. v7 = ComplexFuncBench 병합으로 P2b/P4 primitive 타깃.
- **T2a (GPU)**: v7 데이터(2-hop binding·값 무작위화·D5 ask/fetch 게이트)로 통합 TBox 재학습 → τ²-retail A2-swap robust pass 측정. **성공 판정 = base 0.17 상회 + 전이(airline)도 동반 상승.**
- **T2b**: T2a 성공 시에만 → τ²-airline/bank로 A2-swap 확장(재학습 0). **실패 시 = 헤드라인을 "SOPBench-내부 전이 실증(77.3%) + τ² 전이는 로드맵"으로 정직 축소**(특허 현 상태와 일치).
- ⚠️ [[11]][[12]] 가드: τ² 자체로 학습 금지·다양성(표현/구조) 없는 단일템플릿 SFT는 표면매핑 역전이(TaskBench LODO null·C4·M-σ 전례). 전이는 반드시 **학습벤치서만 학습**.

**Track 1 우선순위**: T1a(airline base·무료) → T1b/T1c(무료 재분석·스팟) → T2a(GPU·v7) → 성공 시 T2b/T1d 확장.

---

## 3. Track 2 — 실패-경로 전수 포렌식 → 성능 개선

**원칙**([[08]]): 집계 metric에서 결론 직행 금지. 종료사유 분포(infra 배제)·단계별 분류·교차표·궤적 2-3개 정독. robust(fail-all-trial)만.

### 3.1 확정된 원인 분포(조립 스택·정본 ASSEMBLED_FAILURE_FORENSIC)
- clean robust-fail: **32B 21 / 14B 30**(infra 32B 6·14B 3 배제 후).
- 원인 %[32B/14B]: ⋈ORDER 14/23 · ORCHESTRATION 19/17 · OVER-ACTION 19/10 · CRITERION 14/13 · WRONG-OP 5/10 · PAYMENT 5/10 · NL/REPORT 19/10 · FORMAT 5/3.
- **규모로 은퇴**: ⋈ORDER·WRONG-OP·PAYMENT. **규모로 잔존/악화**: OVER-ACTION(32B↑·파괴적)·CRITERION.
- **벤치 아티팩트(모델오류 아님·분리 필수)**: REASON enum(~3)·tracking#(~2)·FORMAT/ADDRESS user-sim 변이(~3-4)·고정턴 infra(~1) = **~8-12/51(15-24%)**.

### 3.2 레버 판정(재론 금지)
| 레버 | 판정 | 근거 |
|---|---|---|
| Lever B ⋈ present-quality | **NO-GO**(이미 최대 발화·candidate_summary가 전 실패건에 이미 이름/가격 노출·모델이 보고도 오선택=LLM operand-formalize 잔여) | NEXT_DET_LEVERS §B |
| Lever A refund-target 게이트 | buildable·**저ROI**(⋈ 하류·right-order-wrong-card ~2-4건·gift-card over-block 위험) | NEXT_DET_LEVERS §A·DB 5/5 검증 |
| Lever C over-action | 대부분 **LLM scope 잔여**(게이트 금지·[[06]] lever-type≠해결)·중복쓰기만 idempotence gate=저ROI | NEXT_DET_LEVERS §C |
| calc 범위확장 | subset-refund·order-total ~2-3건·tracking#=범위밖(설계선택) | ASSEMBLED_FORENSIC |

### 3.3 남은 성능개선 실험 (핵심)
- **F1 (GPU-free) = orchestration plan-probe — ✅ 판정 완료(2026-07-06·정본 `PLAN_PROBE_PHASE0_VERDICT_2026_07_06.md`)**: Phase-0 `plan_probe`(2026-06-26 실행·미판정)를 전수 per-case 판정. **결과: (i) SELECT 지배·plan-GENERATE 학습 = NO-GO.** core_ok 6/10 + non-core 4건 전수 정독 = 전부 {단일샷 변이·status-decidable·⋈·배칭}으로 환원(genuine plan-generate 잔여 0). 실제 런 orchestration 실패(t20/t36/t99)가 **격리 계획선 CORE_OK** → 실패=실행부하(H1), planning 능력 아님. 한 줄 요약 "절반 planning miss→Paper3"= 집계 착시로 **기각**.
  - ⇒ **learn NO-GO 확정·make-or-break 강화.** 지배 잔여 = over-action·⋈(scale/LLM-resident) + 실행부하(결정론).
  - **robust 확증(무료·권장)**: `plan_probe` k=4 다-trial 재실행 — 단 in-flight 유료 gpt-5.2와 GPU 경합 회피로 **PERSISTED 후**.
  - **다음 결정론 개선 = F1b Phase-1 C1 하네스**(아래).
- **F1b (유일 headroom) = C1 plan/execute 분리 하네스 — ✅ 빌드+오프라인 검증 완료(2026-07-06)**: `plan_execute_orch.py`(plan 1회 + 결정론 controller: batch-merge·status-fix·provenance-drop; live k-trial + `--replay` 오프라인 모드). controller 순수로직 = **오프라인 단위테스트(`test_c1_controller.py`) ALL PASS**(db·32B·gpt-4.1 0): t71 배칭병합·t109 status remap·t111 날조drop+⋈-miss 회복불가(정직 분리). **[[05]] 가드: 도메인지식=ACTION_SPEC(ABox 6action)만·controller 로직 retail 리터럴 0(테스트 강제).**
  - **남은 = live k=4 robust 실행**(32B): 부하집합 pre(=plan_probe) vs post(controller) core_ok robust 회복 측정. **in-flight 유료 gpt-5.2와 GPU 경합 회피로 PERSISTED 후 실행**([[09]]). post>>pre면 H1 확정·C1 승리·learn NO-GO 종결. 잔여(⋈ wrong-valid·over-reach)=scale.
  - end-to-end(user-sim) 유료 확인 = 결론 후 승인·1회.
- **F2 (저ROI·선택)**: Lever A refund-target 게이트 빌드(유일 남은 clean 결정론 정책) — 빌드하거나 "결정론 소진" 선언. 권장 = 선언 후 F1로 피벗.
- **F3 (무료)**: 7B assembled(미실행) 채우기 → leaderboard 매트릭스 7B 셀 완성(현재 공백).

---

## 4. Track 3 — 학습(T) 날개: 게이트드, 재론 금지

**정본 판정(make-or-break 2026-06-26)**: τ²-retail서 faithful-formalize SFT = **NO-GO**(operand 32B gap 없음·실패=criterion해석[결정론 compute/present]+아티팩트). 학습은 **3회 실패**(C4 copy·M-σ derivation 전이음성·G5 eligibility≈0). ⇒ 재귀 SFT=함정.

### 4.1 GO 게이트 4조건(모두 충족 시만 SFT 실행)
(a) 잔여가 **실재**(present+nested+gate 후 남음) (b) **C4/M-σ 계열 아님**(operand copy/derivation 전이음성) (c) **게이트-redundant 아님**(eligibility vs sequencing 분리 선결) (d) **Probe-B처럼 격리 가능**(capability-bound 아님). → 하나라도 실패 시 SFT off·헤드라인=결정론+TCO.

### 4.2 in-scope 학습(도메인-일반만·[[11]])
- 타깃 = **faithful-formalize**(NL→predicate/operator/operand 선택 충실도) + **abstain 대칭 커리큘럼**(∅→ASK·σ>1→ASK·σ=1→act). 도메인-타깃 값 학습 금지.
- 벤치 = SOPBench(control-flow)·TaskBench(data-flow)·Synth(content-op). 전이 = τ² A2-swap 재학습0. 측정 = pass^k + 4축 scorecard(pass·seq_F1·over-ask·loop escape).
- **인프라 상태**: LoRA 트레이너·데이터빌더·grpo_reward·escape_det_census = **재사용 준비됨**. 신규 3(GPU-free 착수 가능): A2-σ-use 궤적빌더·abstain 커리큘럼·held-out formalize 평가셋. **SFT 실행 결정 = F1 plan-probe 결과에 종속.**

### 4.3 명시적 범위 밖(후속 논문)
- **A2-frontend(NL→A2 자동생성·NL2CA)** = 별도 논문. 현 논문 = A2 수작성·고정 전제. distill PoC는 live하나 real-domain 전이 stall → 현 논문 범위 제외([[41]] 귀속).

---

## 5. Track 4 — 논문용 기-준비 실험 재검토 (마스터 매트릭스)

**마스터 설계**(EXPERIMENT_DESIGN §0 / CAPABILITY_LEVER): C1–C12 능력 × 5레버(scaffold/A2/prompt/learn/scale) × scale(7/14/32/72B) × 양자화(int8/int4/native) × 도메인. decidability-first.

### 5.1 측정됨 vs 공백
- **측정**: 7B/14B/32B scale sweep·SOPBench LODO 전이·operand plateau·retail alias-transfer·조립 스택.
- **공백(우선순위순)**:
  1. **도메인-수평 전이 비용**(retail↔airline↔bank A2-swap labor + 성능유지) = TCO ⑤열 미측정 → **Track 1과 통합**.
  2. **72B same-scaffold headroom**(큰 모델이 *더* 하는 것·B축 decision-emission SFT lock) — coworker 유료.
  3. **235B sweep = 폐기 확정**(2026-06-21·"소형+scaffold가 frontier 도달하나"만 필요·frontier 증명 불요). 재론 금지.
  4. **양자화 능력손실 매트릭스**(int8/int4 operand/compliance floor) — 무료·다음 런 attach.
  5. **C8 error-recovery·C12 NL-comm 방법집합** — deepresearch 미결(scale-bound vs 회복가능).

### 5.2 비용(TCO) 정밀화 — 현 상태 정성만
- 현: 32B+게이트 ≈$0.0019/req(0.573) vs frontier ≈$0.044/req(0.82)·fleet 0.860@$0.021 — **전부 추정·litellm 토큰계측 OFF**.
- **필요 실험(무료)**: 토큰 계측 ON 재집계 → 정확 배수 확정(현재 특허·덱 전부 "미측정 추정"으로 정성화됨). **fleet 배수는 decidable 라우터 미구현이라 oracle 상한**.
- **make-or-break(fleet 헤드라인)**: 값싼 결정론 confidence 신호(토큰 불확실성·제약위반 카운트)로 "32B 충분" 예측·false-escalate <15% → fleet 준수-비용 우위 실증. 무료 baseline + ≤$100 확인.

---

## 6. 실행 순서 & 비용 규율

**즉시(GPU-free·무료)**: 
1. F1 orchestration plan-probe(learn GO/NO-GO 종결) 
2. T1a τ²-airline base(전이 첫 확장) + T1b/T1c 무료 재분석 
3. F3 7B assembled(매트릭스 채움) 
4. TCO 토큰계측 재집계 
5. (F1이 GENERATE 잔존 시) Track 4 학습 GPU-free prep(궤적빌더·커리큘럼·평가셋)

**GPU(무료 온프렘)**: T2a v7 통합 TBox 재학습(2-hop binding) → 성공 시 T2b 다도메인 A2-swap.

**유료(승인·스모크·최소 scope·[[09]])**: gpt-5.2-sim 재측정 완료 회수(in-flight) → 확인용 τ²-airline 1회 → 72B headroom(선택).

**게이트**: 각 단계는 앞 단계 결과가 조건. 특히 **F1 plan-probe 결과 없이 SFT(GPU) 착수 금지**. **T2a 실패 시 헤드라인 전이 주장 축소**(SOPBench-내부 전이로 한정).

---

## 7. 불변 가드 (매 실험 전 재확인)
- [[05]] ABox만 변경·scaffold 도메인분기 0·게이트 증식 금지.
- [[11]] τ² 학습 금지·전이=A2-swap. [[12]] 다양성 없는 단일템플릿 SFT=역전이.
- [[13]] 흡수 우선순위 scale→학습(무망각)→scaffold/A2.
- [[08]] robust pass^k·전수 포렌식·집계→결론 금지. [[09]] 유료=무료검증 후·승인·스모크.
- [[03]] 예측으로 갈아엎기 금지·진단=adjudicator. 결정론 레버 소진/make-or-break NO-GO 재론 금지.
