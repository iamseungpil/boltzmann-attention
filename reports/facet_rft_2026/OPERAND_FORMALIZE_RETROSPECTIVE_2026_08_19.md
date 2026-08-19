# OPERAND / FORMALIZE 실험 전면 회고 (2026-06-13 ~ 2026-08-19)

> **질문**: "operand formalize 를 위한 오랫동안의 실험이 모두 잘못된 건가?"
> **전제**: "LLM 이 calc 를 잘 못하므로 정책을 보고 calc 를 결정기로 해주면 된다. LLM 이 formalize 하고, formalize 된 값만 calc 로 계산한다."
>
> **한 줄 답**: 실험이 전부 잘못된 것은 아니다. **잘못된 것은 종점이다.** 174건 중 라이브 pass 를 종점으로 삼은 것은 **33건(19.0%)** 뿐이고 **81건(46.6%)이 gold 일치율**을 종점으로 삼았다. 그리고 전제가 말하는 아키텍처 — *LLM 이 formalize 하고 formalize 된 값만 결정기가 계산한다* — 는 **그 형태 그대로 대조와 함께 측정된 적이 한 번도 없다**(§5). 측정된 것은 *엔진이 gold 로 맞춘 규칙으로 채점되는 값을 직접 채우는* 다른 형태였고, 그것은 2026-08-19 커밋 `b220745d` 로 제거됐다.

작성 근거: 174건 전수 판정 + 본 회고의 로컬 원자료 독립 재현. 재현 스크립트는 `scratchpad/ep.py`(분류 데이터 인라인).

---

## 0. 요약 판정

| 축 | 수 | 비율 |
|---|---|---|
| 전체 판정 항목 | **174** | 100% |
| SURVIVES | 69 | 39.7% |
| RAW_MISSING (원자료 부재) | 32 | 18.4% |
| SUPERSEDED (후속이 뒤집음) | 31 | 17.8% |
| UNDERPOWERED (검출력 없음) | 19 | 10.9% |
| **GOLD_TAINTED ([[23]] 경유)** | **14** | **8.0%** |
| REFUTED | 9 | 5.2% |

- **부정통제([[57]])가 아예 없는 실험 = 48건 (27.6%)**.
- **라이브 pass 종점 33건 중 부정통제가 없는 것 = 16건 (48.5%)**.
- **39종 operand 레버 중 부정통제를 갖춘 대조로 pass 이득이 측정된 것 = 0건**(2026-08-19 감사·본 회고 재계수로 확인).

---

## 1. 연표

열 = 날짜 · 질문 · 설계 · **종점** · 부정통제 · 판정.
종점 코드: **PASS**(라이브 sim reward/pass) · **GOLD**(gold 일치율·gold-diff census) · **BEH**(행동 지표 — 발화·호출·날조율, gold 무관) · **OFF**(오프라인 replay·유닛·구현 게이트) · **INST**(계기·채점규약 감사) · **NONE**(미실행·측정 0).

### 1.1 2026-06 (n=44 · PASS 8 · GOLD 27 · BEH 4 · OFF 1 · INST 2 · NONE 2)

| 날짜 | 실험 | 질문 | 설계 | 종점 | 부정통제 | 판정 |
|---|---|---|---|---|---|---|
| 06-13 | M-A 3-arm (§1) | concrete-emit vs criteria+resolver | 3 arm × n=32 오프라인 | GOLD | arm A/C만 | UNDERPOWERED |
| 06-13 | M-A A-vs-B 짝 궤적 대조 (§2·3·4) | 무엇이 고쳐지고 무엇이 깨지나 | 짝-정렬 전수 32 | GOLD | 없음 | SUPERSEDED |
| 06-14 | dist 정밀도-벽 전수추적 (§5) | SFT 가 과다호출을 심나 | 샘플당 도구호출 계수 | BEH | base 7B 2.65 | **SURVIVES** |
| 06-16 | FLOOR SWEEP L0–L3 × 5 scale (§8) | 정보 사다리 vs 스케일 | 4 입력수준 × 5 규모 | GOLD | L0 박탈 하한 | **SURVIVES** |
| 06-16 | Sstep 증분 typed-스텝 (§9) | 검증이 binding 을 잡나 | typed 스텝 + per-step 검증 | GOLD | Snover **미실행** | UNDERPOWERED |
| 06-16 | M-σ in-dist derivation (§10) | `$ref` derivation 을 배우나 | LoRA + 채점기 | GOLD | base 0/280 | SUPERSEDED |
| 06-17 | C8 2차 τ² selection 전이 (§17) | 학습이 τ²로 전이되나 | 오프라인 op-eval n=32 | GOLD | S0/S1 | SUPERSEDED |
| 06-17 | §17b 형식 통제 진단 | 붕괴가 형식 탓인가 학습 탓인가 | 프롬프트 골격 고정 base↔trained | BEH | 형식-통제 arm | **SURVIVES** |
| 06-17 | 표현 다양화 재전이 (§18) | 등방화가 복사를 끊나 | DIV_ep3 vs 단일템플릿 | GOLD | 단일템플릿 S2 | SUPERSEDED |
| 06-17 | substitute op-IR 오라클 (§19) | IR 표현이 gold 를 담나 | 손으로 쓴 IR → resolver | GOLD | 없음 | **GOLD_TAINTED** |
| 06-17 | 인자(set) 실패 전수조사 (§20) | set 인자가 왜 틀리나 | 결정론 실패분류 | GOLD | 없음 | SUPERSEDED |
| 06-17 | 다도메인 라우팅 전이 (§21) | 학습이 도메인-일반인가 | retail/airline × g0/g1 | GOLD | 없음 | SUPERSEDED |
| 06-17 | width × scale SET_EXACT + decomp (§22·33) | 폭이 과소추출을 만드나 | width 1~5 × 규모 × 분해 | GOLD | frontier 상한 | **SURVIVES** |
| 06-17 | §17 closure 3-way (§23A) | 누락 생성원이 원인인가 | base / 5-op / 7-op | GOLD | base | SUPERSEDED |
| 06-18 | wide-substitute τ² 전이 (§23D) | 한 축 고치면 다른 축 깨지나 | 학습본 τ² op-eval | GOLD | 직전 MD_route | SUPERSEDED |
| 06-18 | op-IR 어댑터 native e2e (§23E) | 어댑터가 e2e 에서 사나 | 라이브 pass vs base | **PASS** | base 0.17 | **SURVIVES** |
| 06-18 | base τ² e2e GBW headroom (§24) | 벽의 본체가 무엇인가 | 단일 arm n=40 | **PASS** | 없음 | SUPERSEDED |
| 06-18 | collapse autopsy 정정 (§25) | collapse 안에 무엇이 있나 | 에러난 도구콜 인자 전수 | BEH | 3중 삼각측량 | **SURVIVES** |
| 06-18 | facet(3) native keystone (§28) | 형식 변경이 스킬을 깨나 | 생성기 held-out | GOLD | 없음 | SUPERSEDED |
| 06-18 | facet(3) gate① τ² 전이 (§29) | native 형식이 전이되나 | 오프라인 op-eval | GOLD | base | SUPERSEDED |
| 06-18 | operand 전수 궤적 3분해 (§30) | operand 실패 3원인 | base vs trained 분해 | GOLD | base vs trained | SUPERSEDED |
| 06-18 | enum-snap 시뮬 (§32) | 값 정규화가 레버인가 | 저장 rows 소급 시뮬 | GOLD | snap 전/후 | SUPERSEDED |
| 06-18 | airline 하네스 추출 버그 (§34) | 입력이 실 에이전트와 같나 | 계기 감사 | INST | 해당없음 | SUPERSEDED |
| 06-18 | **전면 철회 — 오프라인 op-eval 무효** | 이 계기가 τ²를 재현하나 | 구조 논증 | INST | 해당없음 | **SURVIVES** |
| 06-19 | 단독 LoRA e2e (solo/cfb 6종) | LoRA 단독이 pass 를 올리나 | 6 어댑터 × retail e2e | **PASS** | base 0.205 | **SURVIVES** |
| 06-20 | 스케일 분해 A vs B (§35) | operand 가 스케일-불변 잔여인가 | 7B/14B/32B census | GOLD | 없음 | SUPERSEDED |
| 06-20 | **S-min 3-arm autofetch (§35b)** | 막기인가 주기인가 | 3 arm within-batch | **PASS** | **arm1a(차단만)** | **SURVIVES** |
| 06-21 | C10 operand-formalize 최소 LoRA | 학습이 정당한가 | 사전등록 GO/NO-GO 3조건 | NONE | 사전등록만 | RAW_MISSING(**미실행**) |
| 06-21 | C8 recovery retry 3-arm (§35c) | 복구가 벽인가 | floor/gate/gate_retry | **PASS** | 레버마커 0/0/0 | **SURVIVES** |
| 06-22 | 32B vs frontier gap census | 갭이 능력인가 신뢰성인가 | best-of-3 vs 단일시행 | **PASS** | 없음 | UNDERPOWERED |
| 06-24 | escape-scope Stage-1 카탈로그 | 기권 표면이 있나 | 수동 분류 n=15 | GOLD | 없음 | UNDERPOWERED |
| 06-24 | Arm-II select-probe v1 | 격리하면 고르나 | 라벨된 후보 표 | GOLD | 없음 | SUPERSEDED(**떠먹이기**) |
| 06-25 | Arm-II Probe-B RAW | raw dict 에서 뽑나 | 중첩 dict 후보 n=7 | GOLD | 없음 | **REFUTED** |
| 06-25 | present(σ)+g15 결정론 census | 레버 단독 부작용은 | 4조건 × 342 sim | GOLD | floor + 단독 팔 | RAW_MISSING |
| 06-25 | present+nested+g15 실패 census | 스택 이후 잔여는 | 전수 실패 taxonomy | GOLD | floor/present | RAW_MISSING |
| 06-25 | 전체-궤적 pass-블로커 지도 | 단일 원인이 있나 | reward_info 7버킷 n=42 | GOLD | 없음 | UNDERPOWERED |
| 06-25 | compute_facts(calc) 도입 + 4게이트 | 도메인-일반 calc 를 지을 수 있나 | 구현 게이트 4종 | OFF | 구현게이트 | **SURVIVES** |
| 06-26 | **calc 미발화 버그 적발** (`fad66e26`) | 평탄함이 천장인가 死레버인가 | 발화 수 대 기회 수 | BEH | 해당없음 | **SURVIVES** |
| 06-26 | Claude-as-user-sim 배치 | 유저 천장을 0원으로 재나 | gold 아는 대본 10 태스크 | **PASS** | 없음 | **REFUTED** |
| 06-26 | 격리 operand-pick 프로브 | 격리에서 고르나 | 단일 arm·`reason_for_call[:700]` | GOLD | 없음 | **GOLD_TAINTED** |
| 06-26 | **통제 operand 실험 = make-or-break** | operand 능력 갭이 있나 | GIVEN-SPEC ↔ GOAL | GOLD | 없음 | **GOLD_TAINTED** |
| 06-26 | assembled 스택 실런 (asmscale) | 스택이 pass 를 사나 | 32B/14B nt3 | **PASS** | floor(별도 런) | **SURVIVES** |
| 06-27 | assembled 실패 per-case 포렌식 | 스택 이후 지배 원인은 | robust-fail 51건 정독 | GOLD | pure-infra 배제 | **SURVIVES** |
| 06-26 | 1차 present+nest+g15 데이터 소실 | — | 사고 기록 | NONE | 해당없음 | **SURVIVES** |

### 1.2 2026-07 (n=72 · PASS 11 · GOLD 32 · BEH 9 · OFF 8 · INST 6 · NONE 6)

| 날짜 | 실험 | 질문 | 설계 | 종점 | 부정통제 | 판정 |
|---|---|---|---|---|---|---|
| 07-06 | operand_controlled 7B/14B/32B | 실행은 scale 축인가 | C1 GIVEN ↔ C2 GOAL | GOLD | 없음 | SUPERSEDED |
| 07-06 | operand_probe 궤적 오선택 열거 | 어느 축이 지배인가 | 격리 census 88/219/307 | GOLD | 없음 | **SURVIVES** |
| 07-08 | E9 operand grounding gate Phase A | 술어 오탐 0 ∧ 근인성인가 | 무료 오프라인 사전등록 | OFF | passing 12건 | RAW_MISSING |
| 07-09 | **C43 날조 = 정박 치환** | 날조가 망각인가 조기-쓰기인가 | 4 arm 층화 456 sim | BEH | H-load/distractor/forget | **SURVIVES** |
| 07-09 | C45 날조 프롬프트 4-arm | 4지선다가 날조를 막나 | A~D 짝지음 n=30 | BEH | C팔(금지문) | RAW_MISSING |
| 07-10 | C47/C48 D′/D″ 검증기 | 출처 선언이 갈래를 가르나 | FAB/CLEAN 짝 n=30 | OFF | FAB/CLEAN 짝 | RAW_MISSING |
| 07-10 | C53 출처선언 레버 e2e (prov) | 출처선언이 pass 를 사나 | 456 sim × 114 태스크 | **PASS** | 없음 | UNDERPOWERED |
| 07-10 | **C58 T6h 디폴트 불변성 기각** | 디폴트가 정책인가 분포인가 | rand/통계/원리 3팔 | GOLD | **rand .065** | **SURVIVES** |
| 07-10 | **C59 T5-A 후보 열거가 ⋈를 연다** | 표면화가 여는가 | c51 격리 400행 A/B | GOLD | 불변 팔(order/addr) | **SURVIVES** |
| 07-11 | C60 T5-B DISAMB 라우터 e2e | 격리 이득이 전이되나 | routerv1 456 sim | **PASS** | 없음 | UNDERPOWERED |
| 07-11 | T5-C V0 화이트리스트 검증 | 어떤 인자를 치환할까 | fix/break 2×2 (gold 축) | GOLD | break 계수(gold) | **GOLD_TAINTED** |
| 07-11 | T5-C P-A/P-B silent repair 구현 | 배선이 도나 | UNIT 48 checks | OFF | UNIT만 | SUPERSEDED |
| 07-12 | T5-C V2.5b 6표적 궤적 포렌식 | 표적에서 무엇이 바뀌나 | nt=1 6표적 | **PASS** | 없음 | UNDERPOWERED |
| 07-11 | BANKING_FLOOR_LEVER_FIT (170 sim) | banking 날조 지분은 | per-case census + 정독 | GOLD | 없음 | **SURVIVES** |
| 07-13 | UNIFIED_OPERAND_A2 (설계) | 통일 스키마가 가능한가 | 설계·측정 0 | NONE | 해당없음 | **SURVIVES** |
| 07-13 | bankar_uni5 통일 스택 스모크 | 배선이 사나 | 단일 팔 5 sim | **PASS** | 없음 | SUPERSEDED |
| 07-13 | **T2_RESOLVE 표준 user-sim A/B (G/GR)** | resolve 레버가 pass 를 사나 | 동시·동일 5태스크·단일 플래그 | **PASS** | 없음 | UNDERPOWERED |
| 07-13 | STEP0 compute-사정권 (§8-0) | compute 사정권이 30% 넘나 | 사전등록 게이트 | GOLD | 사전 30% 게이트 | **SURVIVES** |
| 07-14 | C77 reference-filter 사정권 | ⋈가 지배 버킷인가 | 파싱신뢰 필터 853 | GOLD | 모델 간 일관성 | SUPERSEDED |
| 07-14 | C78 keystone 오프라인 REPLAY | filter 천장은 | gold 파생 criteria replay | GOLD | 없음 | **GOLD_TAINTED** |
| 07-14 | C79 formalize half 실측 | formalize 가 병목인가 | 32B 853 케이스 | GOLD | dispute 밖 1.5% | SUPERSEDED |
| 07-14 | **C80 ⋈ 오염 발각·재정량** | 진짜 ⋈ 몫은 | 제출집합 ↔ gold집합 3분할 | GOLD | 해당없음 | **SURVIVES** |
| 07-14 | C81 COMPUTE slice 실측 | compute 몫이 큰가 | gold action_checks 3904 | GOLD | 없음 | **GOLD_TAINTED** |
| 07-14 | §8-1 gold-blind 저작 게이트 | 상수를 정책에서만 뽑나 | 저작 규율 + 유닛 | OFF | 해당없음 | **REFUTED** |
| 07-14 | §8-2 재현율 측정 (liability) | 규칙이 gold 를 재현하나 | 임계 스윕 | GOLD | 없음 | **GOLD_TAINTED** |
| 07-14 | §8-3 Δspurious 측정 | 레버가 선택적인가 | agent-correct/wrong 2셀 | GOLD | agent-correct 셀(gold) | **GOLD_TAINTED** |
| 07-14 | **§8-4 liability flat config 스윕** | 어느 config 가 맞나 | {flat/min}×{cal/biz}×임계 | GOLD | business_days 동률 | **GOLD_TAINTED** |
| 07-14 | §8-5 오프라인 통합 replay | 배선이 실발화하나 | 240 sim replay | GOLD | 없음 | **GOLD_TAINTED** |
| 07-14 | C82 거래고정→파생 slot-fill | 논리곱 붕괴가 되나 | 파생정확도 × 기존오답률 | GOLD | 교차표 | **SURVIVES** |
| 07-14 | C83 H_min (gather-horizon) | 필요 정보량은 | joint 엔트로피 | GOLD | 없음 | **GOLD_TAINTED** |
| 07-14 | **C89 per-step regime partition** | 재샘플로 열리나 | k8 T0.7 · 853행 | GOLD | **gold-free 라우터 670/671 실패** | **SURVIVES** |
| 07-15 | C92 해소연산 오분류 자동 라벨 | 실패의 해소연산은 | 4범주 자동 라벨 | GOLD | 부분통제(자인) | SUPERSEDED |
| 07-15 | bank_operator_replay | COMPUTE 강제 시 천장은 | `would_pass_after_COMPUTE` | GOLD | 없음 | SUPERSEDED(**당일 철회**) |
| 07-15 | **C93 ceiling proxy 포렌식** | 내 종점이 채점되는 종점인가 | reward_basis 전수 6515 sim | INST | 2×2 tightness | **SURVIVES** |
| 07-16 | C96(a) compute 규칙 gold 적합 | 필드가 결정론인가 | 전수 적합 + base rate | GOLD | **base rate 65%** | **GOLD_TAINTED**(부분) |
| 07-16 | C96(c) 전-액션 DAG replay | 구조 결정론이 얼마나 닫나 | 상·하한 2 bound | GOLD | 2 bound | **SURVIVES** |
| 07-16 | **C102 A2 도구 `_f` 주입 사고** | 모델이 스키마를 본 적 있나 | 소스 직독 + 전/후 런 | INST | ctl/dreq | **SURVIVES** |
| 07-16 | §9.1c/§9.1d op vs operand 분리 | 어느 축이 실패인가 | 궤적 진단 | BEH | ctl arm(무의미) | SUPERSEDED(**6h 뒤 자기무효**) |
| 07-16 | bank_op_operand_probe | p(operand\|op) 는 | tool_choice 고정 조건화 | NONE | 설계상 조건화 | RAW_MISSING(**미기록**) |
| 07-18 | **C112/C108 env 거짓말** | 우리 문장이 죽이나 | 문장 1개 교체 (25 tok 차) | BEH | **unlock_truth** | **SURVIVES** |
| 07-17 | **C113 rate-formalize 분담선 격리** | 어디까지 LLM, 어디부터 엔진 | arm 사다리 3단 | GOLD | 없음 | RAW_MISSING |
| 07-18 | RATE §2c iso021 라이브 | 격리 서브가 e2e 로 도나 | n=1 존재증명 | **PASS** | ratefix arm | UNDERPOWERED |
| 07-18 | RATE §2d iso5 5태스크 페어 | 격리가 부하를 줄이나 | 5태스크 페어 | **PASS** | 해당없음 | **SURVIVES**(완주 0) |
| 07-18 | bank_overflag_probe | 온도가 분산을 만드나 | temp 0 vs 0.7 | GOLD | temp0 arm(**계기 고장**) | RAW_MISSING |
| 07-19 | **ACCOUNT_APY_OFFLOAD `group_reduce`** | 조항-결합을 결정기로 옮기나 | 도메인-일반 프리미티브 + UNIT | OFF | UNIT만 | RAW_MISSING(**효과 미측정**) |
| 07-19 | FUNCTION_AGENT_ISOLATION wrap | 자료-read 격리가 되나 | 기본 OFF | NONE | 라이브 arm 0 | RAW_MISSING |
| 07-23 | C114 052 approve-when-deny | 결정론 판정을 따르나 | 3 sim 텍스트 대조 | BEH | 없음 | UNDERPOWERED |
| 07-23 | C115/116/117 have-value→act | 값을 쥐고도 안 쓰나 | 격리 8/9 + 라이브 8+8 | **PASS** | 무효(**발화 0**) | **REFUTED** |
| 07-23 | C124 039 정보-맞춘 격리 | 경계인가 부하인가 | A_minimal vs B_fullctx | GOLD | 없음 | RAW_MISSING |
| 07-24 | C126 T2_REF_ISO + rall21 | 격리 서브가 교정하나 | per-firing 계수기 | BEH | 해당없음 | **SURVIVES**(switched 0) |
| 07-24 | C128 E-F3-ISO Phase1 + 매처 | 용량반응 + 결정론 매처 | S/A/B 사다리 + 8/8·25/25 | GOLD | gold 25/25 통과 | RAW_MISSING(**매처만 재현**) |
| 07-24 | **C129 rall22 gold→오답 switch** | 재선택이 해로운가 | per-switch 포렌식 | BEH | 없음 | **SURVIVES** |
| 07-24 | C130 T2_REF_VERIFY | 결정론 검증기가 잡나 | replay 짝 종점 | OFF | gold 통과율 | RAW_MISSING(**테스트 깨짐**) |
| 07-24 | C144/146/147 BRANCH-REGROUND | 문서 재환기가 인자를 사나 | matched pair 단일변수 | **PASS** | matched pair | UNDERPOWERED |
| 07-25 | C176/C179 card_type 오선택 | 문서 유입이 원인인가 | 3궤적 전수 + W2 | GOLD | W2(무유입) | RAW_MISSING |
| 07-25 | **C181/C185 `catalog_filter` 3버킷** | 미지를 주장으로 바꾸지 않나 | eligible/excluded/unverified | OFF | UNIT | **SURVIVES**(라이브 [?]) |
| 07-25 | **C186 검증 레버 43개 이탈** | 어떤 arm 이었나 | 코드×런처×런 3자 대조 | INST | 해당없음 | **SURVIVES** |
| 07-26 | **C197 입력-결함 침묵 통과** | 서브인가 우리 층인가 | as-is / 반사실 replay | OFF | **반사실 arm** | **SURVIVES** |
| 07-26 | C202/C203 D4 폐기 → D4′ | 값-접지가 날조를 막나 | 표적 포착 + 오탐 | BEH | 003 보존 | **SURVIVES** |
| 07-27 | D6 `spend_category` operand | 파생 operand 가 닫나 | 사전등록 지표 | NONE | 사전지표 **미측정** | SUPERSEDED |
| 07-29 | **C215 023 정책상수를 LLM 인자화** | 이 구간 A/B 가 성립하나 | arm 교집합 검사 | **PASS** | arm 교집합 **0** | **SURVIVES**(측정 무효 확정) |
| 07-29 | AXIS_DECISION_DETERMINISM_LEARN | 결정론이 어디서 성공하나 | 이론 판정 + 분할선 | NONE | 해당없음 | **SURVIVES** |
| 07-29 | C218/219/220 DECLARATION_FIRST | 전면 formalize 가 되나 | 설계 LOCK | NONE | 라이브 arm 0 | UNDERPOWERED |
| 07-30 | C240 X8 formalize 3분기 triage | 인터페이스가 구속조건인가 | 3 arm × 3 seed × 48 | GOLD | 볼드-베이스라인 | **SURVIVES** |
| 07-30 | C243 실행 잔여 census (OPERAND 57%) | operand 지분은 | 자동 분류 695 | GOLD | 없음 | SUPERSEDED |
| 07-30 | **C244+C245 채점 규약 소스 직독** | 채점이 실제로 무엇을 보나 | 재구현 자기검정 | INST | **재구현 1080/1080** | **SURVIVES** |
| 07-30 | C246 x9 REF_ISO 페르소나 프로브 | 도메인 명사가 바꾸나 | 81쌍 짝지음 | BEH | 짝지음 | **SURVIVES**(null) |
| 07-30 | C251/255/259 Y1 인자-수준 분류 | 실패의 인자 축은 | 전수 분류 + 안정성 열 | GOLD | 없음 | **SURVIVES** |
| 07-31 | C254 A2 gold-출처 감사(33키) | A2 가 gold 경유인가 | 문자열-출처 코퍼스 분할 | INST | gold/비-gold 분할 | **REFUTED** |
| 07-31 | TXNID_ISOLATION_PROBE(X22) | txn 오선택이 부하인가 경계인가 | A_minimal/B_fullctx/3-arm | GOLD | A/B 동일 후보 | RAW_MISSING(**부분 실행·소실**) |
| 07-31 | C264 Y2-B 중단 | 우리가 gold 를 막았나 | deny 계수 + replay | INST | 사전등록 계기 | UNDERPOWERED |
| 07-31 | C267/C268 V7 give 인자 제거 | 인자를 실을까 뺄까 | 자연실험 분할 | **PASS** | 자연실험 분할 | RAW_MISSING(**3회 번복**) |

### 1.3 2026-08 (n=58 · PASS 14 · GOLD 22 · BEH 8 · OFF 3 · INST 8 · NONE 3)

| 날짜 | 실험 | 질문 | 설계 | 종점 | 부정통제 | 판정 |
|---|---|---|---|---|---|---|
| 08-01 | **C274 채점 진단층 리터럴 아티팩트** | gold-miss 가 진짜 오류인가 | semantic 재매칭 400 | INST | reward 무결 대조 | **SURVIVES** |
| 08-01 | **C275/276 quote-ground false-abstain** | 가드가 옳은 행을 지우나 | gold-free 상인 census 138 | BEH | **gold 무참조** | **SURVIVES** |
| 08-01 | C278/279 `pin_kind` 재설계 | 열린 술어를 닫나 | LLM 선언 + 종류별 검사 | OFF | 리터럴 0 검정 | **SURVIVES**(라이브 −) |
| 08-01 | C282 quote-pin 라이브 스모크 | 022 가 통과하나 | n=1 before/after | **PASS** | 없음 | RAW_MISSING |
| 08-01 | C286 019 차단 ON/OFF replay | 식별표가 차단하나 | 무료 재생 ON/OFF | OFF | OFF arm | RAW_MISSING(효과 0) |
| 08-01 | C289 quote-pin 라이브 nt2 | 어느 층이 발화했나 | 로그 층 귀속 | **PASS** | 층 귀속 | **SURVIVES** |
| 08-14 | T2_WRITE_SUB 계보 확인 | 이 구간에 있었나 | 소스 계보 확인 | NONE | 해당없음 | **SURVIVES**(부정 존재) |
| 08-04 | T2_MATCH_COUNT 라이브 A/B (ax33n↔b4) | 회수 경계 표면화가 사나 | 시드-맞춤 64/64 | **PASS** | 없음 | **REFUTED**(처치 37% 거짓) |
| 08-04 | x75 MATCH_COUNT 문구 분포 | 무엇이 붙었나 | role=tool 정규식 3분류 | BEH | arm A 0줄 | **SURVIVES** |
| 08-04 | F2/F2b MATCH_COUNT 등재 + 코퍼스 수리 | 왜 0건이었나 | 소스 + 하류 발화 | OFF | 해당없음 | **SURVIVES** |
| 08-19 | **MATCH_COUNT 거짓 완결 인증 감사** | 우리 인증이 참인가 | 닫힌 검산 `matches:` ↔ `ID: doc` | INST | 닫힌 검산 | **SURVIVES** |
| 08-13 | x294 중복-write 격리 | DEDUP 을 지을까 | 사전 고정 A_ASIS ≥6/8 | GOLD | 사전 게이트 | RAW_MISSING(**미출시**) |
| 08-13 | x292/x292b BYREF deny 문면 격리 | deny 문면이 오유도하나 | n=8 → 16 → 원거리 | GOLD | 없음 | RAW_MISSING |
| 08-13 | t7274w FIX-5 fee 차액 operand | 차액을 내나 | nt=1 궤적 직독 | BEH | 없음 | **SURVIVES**(존재증명) |
| 08-12 | x275 축(계열) formalize 격리 | 축을 형식화하나 | 격리 8/8 | GOLD | 없음 | **REFUTED**(라이브 gold deny) |
| 08-14 | x301 fee 자기-참조 함정 | 라벨 정보량이 바꾸나 | 4셀 대비 | GOLD | 4셀 대비 | RAW_MISSING |
| 08-14 | x307~x310 write 착수 격리 사슬 | 착수가 부하인가 | 사슬 + D_NOBASIS | GOLD | D_NOBASIS 0/8 | SUPERSEDED |
| 08-15 | **C483 n=8 프로브 잡음 바닥** | 우리 눈금은 얼마인가 | 동일 본문 5블록 × 8 | INST | 동일 본문 | RAW_MISSING(방법 유효) |
| 08-14 | **C486 `action_match` 계기 정정** | gold 일치율이 종점이 되나 | 통과 sim 대조 | INST | **통과 sim** | **SURVIVES** |
| 08-14 | x322 OPERATOR_PINPOINT 격리 5팔 | 지목이 여는가 | 5팔 + E_NEG | GOLD | E_NEG 0/24 | RAW_MISSING(라이브 인과 [S]) |
| 08-12 | x330 terminal write emit | 이름과 실행이 갈리나 | 도구 바인딩 격리 | BEH | D_EARLY 0/24 | SUPERSEDED(라이브 6%) |
| 08-15 | t7296 전달 복구 라이브 A/B | 전달이 사나 | 16 sim/팔 | **PASS** | 없음 | UNDERPOWERED(1↔1) |
| 08-16 | t7297 행동 촉구 A/B (ACT_DEMAND) | 촉구가 사나 | 단일 처치 20/팔 | GOLD | 없음 | **GOLD_TAINTED** |
| 08-16 | x335/x335b 자격-제거 격리 | 전달이 결손인가 | 라이브 경로 정합 재료 | GOLD | 붕괴(자기 미판정) | **REFUTED** |
| 08-16 | **t7298 C494 라이브 예측 검정** | 격리 24/24 가 라이브로 오나 | 단일 팔 반증 설계 | **PASS** | 반증 설계 | **SURVIVES**(055 0/4) |
| 08-16 | t7299 T2_MATERIAL_RESERVE A/B | 예약이 사나 | 8 sim/팔 | **PASS** | 없음 | UNDERPOWERED |
| 08-17 | t7303 배달 **객체** 교체 A/B | 객체를 바꾸면 | 1차 종점이 처치 팔 마커 | **PASS** | 순환 마커 | **REFUTED** |
| 08-17 | t7304 `T2_DOCS_AT_WRITE` A/B | 배달 **자리**를 옮기면 | 단일-변수 | **PASS** | 없음 | SUPERSEDED(**양팔 0 발화**) |
| 08-19 | `T2_DOCS_AT_WRITE` dark 판정 | 자기 태그가 있나 | 소스 감사 | INST | 해당없음 | **SURVIVES** |
| 08-17 | x343/x343rep/x344/x344v2 (H_iso) | 요구를 넣으면 배제하나 | 격리 24/24 | GOLD | 같은 축 다른 요구 | SUPERSEDED(**x349 소급 무효**) |
| 08-17 | x345 formalize 충실도 격리 | 정규식→formalize 이설 가능한가 | 3값 복원 + D_NEG | GOLD | D_NEG 날조율 | RAW_MISSING |
| 08-17 | x347 `param_cap` formalize 이설 | 검증기를 지워도 되나 | 등급별 + D_NEG | GOLD | D_NEG | RAW_MISSING(**정반대 부호**) |
| 08-17 | t7305 `T2_SUB_REQUIREMENT` A/B | 요구 주입이 사나 | 8/팔 + aux | GOLD | 없음 | **GOLD_TAINTED** |
| 08-17 | **x351/352/352c/353 순서·재료·정박 격리** | 탐색인가 정박인가 | A_REF/B/C/D_NEG/E_MINUS | GOLD | **D_NEG 0/16** | **SURVIVES** |
| 08-18 | x354/x355 낱개 판정·제거형 질의 | 낱개면 되나 | B_FILTERED + D_NEG | GOLD | D_NEG 0/16 | **SURVIVES** |
| 08-18 | **x356b/x357(v2) 판정 줄만 올리기** | 답만 올리면 되나 | A_LIVEREF ↔ B_LIVEREQ 25축 | GOLD | D_NEG 2 | **SURVIVES**(라이브 미전이) |
| 08-18 | **C516 `formalize_groups` 필터가 삭제** | 파싱이 맞는 군을 지우나 | 코드 + 라이브 로그 125줄 | BEH | 해당없음 | **SURVIVES** |
| 08-18 | **C518 `business_*` 66%는 우리 필터** | 레버의 표적이 실재하나 | 전수 재계수 49건 | BEH | 해당없음 | **SURVIVES** |
| 08-18 | C519 선택축 없는 60태스크 | 태스크 성질인가 분류기인가 | `gold_axes()` 소스 직독 | INST | 해당없음 | **SURVIVES**(크기 [?]) |
| 08-18 | t7307/t7308 HANDOFF_PREDICATE A/B | 술어 도달이 사나 | 12/팔 + 지연 | **PASS** | 없음 | **SURVIVES**(완전 null·1.90×) |
| 08-18 | t7310 S2 배선 스모크 (VC·EL) | 발화·무오염인가 | 마커 + fb 사이드카 | BEH | ctl 오염 0 | **SURVIVES** |
| 08-18 | t7312/t7313 1단계 census A/B | 레버가 사나 | 16/팔 | **PASS** | 없음 | UNDERPOWERED(바닥 효과) |
| 08-18 | **x380/x381 CAP 래치 손해 + replay** | 래치가 답인가 | 56 sim + 줄-추적 replay | BEH | 외부 호출 대조 | **SURVIVES** |
| 08-18 | x382/x383 CAP latch 회귀 3건 | 손해 크기는 | 시드-맞춤 자연실험 | **PASS** | 시드-맞춤 | RAW_MISSING |
| 08-18 | **x384 20태스크 실패 해부** | A/B 를 읽을 수 있나 | 사전 고정 우선순위 분류 | GOLD | **양팔 11:11 대칭** | **SURVIVES** |
| 08-18 | t7314/15/16 VC 노브 분해 | 073 이 회귀인가 레버 음수인가 | nt=1 단발 3런 | **PASS** | 없음 | RAW_MISSING |
| 08-19 | x385/386/386b/387 VC 범위 술어 | 범위를 자를 수 있나 | 3갈래 술어 | GOLD | 사전문턱 미달 | RAW_MISSING(구조 명제만) |
| 08-19 | t7324 declfirst 가이드 A/B | 가이드가 사나 | 4태스크 × nt=2 | NONE | 배선 0(**무효**) | RAW_MISSING |
| 08-19 | **t7326 1단계 재베이스라인** | 수리 후 기준선은 | 단일 스택 40 sim | **PASS** | 단일 스택 | **SURVIVES**(7/40·DB35/A4) |
| 08-19 | **`compute_ops` gold-fit 제거 (`b220745d`)** | 엔진이 채점 인자를 썼나 | 커밋+소스+로그 100회 | INST | 해당없음 | **SURVIVES** |
| 08-19 | x398 조기 종결 교락 요인설계 | 기권 선택지가 만드나 | 창 4 × 시스템 2 × 288 | GOLD | 요인설계 | SUPERSEDED |
| 08-19 | **x399→x400/x402 실패 형태 재계수** | 실패의 최대 덩어리는 | 대칭 조인 재분류 144 | GOLD | 닫힌 술어 | **SURVIVES**(MISCALLED 38%) |
| 08-19 | x408 R3 격리 | 재제시가 계획을 여나 | A_slice/B_live/C_neg/D_both | GOLD | C_neg | UNDERPOWERED(**분기 비배타**) |
| 08-19 | x417/x419 operand 격리 2연속 무효 | — | 요약으로 원문 대체 | NONE | 해당없음 | **SURVIVES**(무효 판정) |
| 08-19 | x420 재생(replay) 격리 operand | 문서를 주면 맞추나 | R_asis/R_doc/R_neg 35표적 | GOLD | R_neg(고정) | RAW_MISSING(**절단 결함**) |
| 08-19 | x421 KB/정책 operand 스모크 | 적합한 정책을 주면 | tf-idf 결정론 검색 n=4 | GOLD | D_neg 무작위 | UNDERPOWERED |
| 08-19 | t7328 gold-fit 청소 후 재베이스라인 | 청소 후 기준선은 | 단일 스택 40 sim | **PASS** | 단일 스택 | RAW_MISSING(**미완결**) |
| 08-19 | OPERAND 레버 최종 판정표(실효 39) | 레버가 산 것이 있나 | 2신호 발화 + 전수 감사 | INST | 2신호 발화 | **SURVIVES** |

---

## 2. ★종점 분포 — 이 회고의 핵심

| 1차 종점 | 건수 | 비율 |
|---|---|---|
| **PASS** — 라이브 sim reward/pass (채점되는 종점) | **33** | **19.0%** |
| **GOLD** — gold 일치율 · gold-diff census ([[23]] 경유) | **81** | **46.6%** |
| BEH — 행동 지표(발화·호출·날조율·전환 계수, gold 무관) | 21 | 12.1% |
| OFF — 오프라인 replay · 유닛 · 구현 게이트 | 12 | 6.9% |
| INST — 계기·채점규약 감사 | 16 | 9.2% |
| NONE — 미실행 · 측정 0 · 설계만 | 11 | 6.3% |
| **합계** | **174** | 100% |

> **한 줄**: 3개월 동안 174건을 돌면서 **채점되는 종점(pass)을 본 것은 다섯 중 하나(19.0%)**이고, **절반 가까이(46.6%)는 gold 일치율을 성적처럼 읽었다.**

### 2.1 이 분포가 왜 치명적인가 — 세 겹의 증거

1. **gold 일치율은 reward 와 끊겨 있다.** C93 전수 측정: `reward_basis` **DB 80.9% / ACTION 9.0%**. 본 회고 재계수 t7326 40 sim = **DB 35 / ACTION 4 / 없음 1**(원자료 `bank_t7326_{halfA,halfB}_20260819q.results.json.gz` 직접 로드). 즉 실패의 대부분은 *성공한 변이 호출 집합*에서 갈리므로, dispute `action_checks` 를 고쳐도 `db_match` 는 안 바뀐다. C93 축자: *"dispute action_checks fix는 db_match를 안 바꾼다"*. 사정권은 완전 1.9% · 부분 17.9%.
2. **`action_match` 계기 자체가 무너진다.** C486(본 회고 재현): `bank_t7290_b_20260814m` task_073 은 **reward 1.0 · db_match True 인데 크레딧 3건의 `action_match` 가 전부 False** — gold 문자열 `9.50/9.00/1.50` ↔ 에이전트 `9.5/9.0/1.5`. 소수점 표기 차이 하나로 무너지는 축이다.
3. **채점 규약이 우리가 믿던 것과 달랐다.** C244/C245(본 회고 `x12_action_fail_exact.py` 재실행, 재구현 일치 **1080/1080 = 100%**): `compare_args` 는 **예측의 키에서 온다**(예측 인자가 빈 dict 면 무조건 매치) · 1:1 배정 없음 · requestor 미비교. 실패 750 중 지배는 operand 가 아니라 **미실행 NAME_ABSENT 341(45.5%)**.

### 2.2 PASS 종점 33건의 내부 사정

| 하위 판정 | 건수 |
|---|---|
| SURVIVES | 11 |
| UNDERPOWERED | 10 |
| RAW_MISSING | 5 |
| REFUTED | 4 |
| SUPERSEDED | 3 |

- 33건 중 **부정통제가 없는 것 16건(48.5%)**.
- 33건 중 **레버 이득이 부정통제와 함께 양성으로 나온 것 = 1건**: **2026-06-20 §35b S-min autofetch** (base 7B retail pass^1 0.140 → 0.264, arm1a(차단만)가 **이득 0**). 그리고 그 레버는 *operand formalize* 가 아니라 **producer 대리 호출로 실값을 공급**하는 것이다.
- 나머지 라이브 A/B 는 본 회고 재계수에서 전부 null 이거나 검출력 없음: t7296 1↔1 · t7297 8↔9 · t7299 3↔4 · t7303 5↔6 · t7304 0↔0 · t7305 0↔0 · t7308 2↔2 · t7310 1↔1 · t7313 2↔3 · ax33n↔b4 24↔24 · T2_RESOLVE G 3/5 ↔ GR 0/5(p≈0.167).

### 2.3 GOLD 종점 81건의 내부 사정

| 하위 판정 | 건수 |
|---|---|
| SUPERSEDED | 23 |
| SURVIVES | 19 |
| RAW_MISSING | 16 |
| **GOLD_TAINTED** | **14** |
| UNDERPOWERED | 6 |
| REFUTED | 3 |

**81건 중 37건(45.7%)이 폐기되거나 gold 오염 확정**이다.

---

## 3. 살아남는 것 (SURVIVES 69건)

무너뜨리기를 시도했으나 무너지지 않은 것들. **공통점: 종점이 gold 가 아니거나(BEH/INST), 결론 방향이 자기편 레버를 죽이는 쪽이거나, 부정통제가 실제로 무언가를 죽였다.**

### 3.1 결손의 위치와 성질 — 확립된 사실

| 실험 | 확립한 것 |
|---|---|
| §5 dist 과다호출 | 도구 부분집합 위 concrete-emit SFT 는 선택 추론을 안 가르치고 습관적 과다호출을 심는다(6.02 vs base 2.65 호출/샘플·과다호출 92% vs 1.4%). |
| §8 FLOOR SWEEP | 가용성 정보를 넣어 주면 오프라인 변형선택이 **다섯 스케일 전부에서 +16pp**. 정보 결손은 스케일이 아니라 정보 공급으로 닫힌다. |
| §22·§33 width | multi-attr set 과소추출은 폭에 따라 커지고 소형에서만 급하며, per-attr 분해는 **폭이 넓을 때만** 회복시킨다(7B w4 0.51→0.87). **폭이 좁으면 오히려 해친다**(7B w1 0.66→0.33) = 레버에 음의 구간이 있다. |
| §25 collapse autopsy | base e2e 붕괴의 본체는 generic flow 실패가 아니라 **가져올 수 있는 concrete id 를 발명**하는 것(order_id 79·item_id 15)과 같은 인자로 재시도(16/27). |
| **C43 정박 치환** | 날조는 긴 문맥에서 잊어서가 아니라 **조회하기 전에 쓰기 때문에** 난다(날조 시점 문맥 median 6,355자 vs 정상 write 9,421자 · read 0–3회 13.6% → 6–7회 1.0%). H-load/H-distractor/H-forget 3갈래 모두 자기 기각. |
| C80 ⋈ 재정량 | banking hard-core 에서 agent 는 transaction_id 를 **대체로 맞게 고른다**(2904/3904 = 74.4%). 진짜 ⋈ 오선택 159(4.1%). 지배는 **아예 제출하지 않는 것**(1000 = 25.6%). |
| **C89 regime partition** | 정답이 8회 재샘플 support 에 **사실상 없다**(5/825) ⇒ voting(0.1%)·field-verify(1.6%) 둘 다 레버가 아니다. maj 의 90%가 frontier 선택과 일치 = 모델 독립적·체계적. |
| C96(c) DAG replay | 순수 구조 결정론이 닫는 것은 banking DB-basis 실패의 **27.8%(상한 49.1%)**뿐이고, 잔여의 지배는 **F3-enum(NL→정규화) 의미매핑** = LLM 쪽. |
| C240 X8 triage | 선언-강제는 ~7pp 를 사고 날조 slot 51→32·스키마 밖 라벨을 닫는다. 그러나 48건 중 **32건(67%)이 세 arm 모두 실패** — 인터페이스는 formalize 실패의 구속조건이 아니다. |
| **x351/352/353 정박** | gold 문서를 강제로 쥐어줘도 3/8 인데 **오답 `Green Account` 하나를 빼면 6/8**. 결손은 탐색도 후보 수도 아니고 **특정 오답으로의 정박**이며, 닫히는 조작은 **제거뿐**([[63]]). |
| x354/x355 | 후보를 **낱개로 물으면** gold OK 6/8·위반 검출 6/8. 무너지는 것은 그 판정들을 **동시에 유지**할 때(4/8) = 능력이 아니라 유지·배치 결손. |
| **x356b/x357v2** | 문서 전문 대신 **후보별 판정 줄만** 결정점에 올리면 격리 표적 25축 gold 일치 8 → 15(D_NEG 2·UNCLEAR 0). [[65]] 의 유일한 정량 확증. **라이브로는 전이되지 않았다.** |
| x399→x400 | t7326 미매치 gold 144건의 최대 덩어리는 **MISCALLED — 호출은 했는데 인자가 어긋남**(38%, write 만 보면 49%). '허위 완료 신념 42%'·'knowing–doing 7%'는 계수 결함의 산물이었다. |
| x384 | t7313 실패 35 중 READ_MISS **22(63%)**이고 **양팔 11:11 정확 대칭** ⇒ 그 런에서 레버 pass 효과는 원리상 측정 불가(바닥 효과). |

### 3.2 결손이 **우리 층**이었던 것 — 이 회고에서 가장 큰 덩어리

| 실험 | 확립한 것 |
|---|---|
| **C102 `_f` 주입 사고** | 2026-07-16 이전 operand·도구선택 결손 증거 **전부가 우리 하네스 산물**. A2 도구가 `{name:'_f', description:'_f', parameters:{k:{}}}` 로 주입돼 모델에게 스키마가 제시된 적이 없었고, PROV 는 정당한 호출을 매번 반려했다(`'id'` 가 `'provided'` 의 부분문자열). |
| **C112/C108 env 거짓말** | 우리 env 응답 문장 **하나**(*"This tool is not available"*)가 결정점의 56%를 포기로, 22%를 배회로 보내고 producer 직접호출을 **88%→6%** 로 떨어뜨렸다. 두 arm 프롬프트 차 **25 토큰**. |
| **C186 레버 43개 이탈** | 커밋 `bf81ec86` 축자: go_stack 이 18을 실었는데 마지막 검증 런 스크립트는 56을 설정했다 — C167~07-25 사이 모든 operand 레버 효과가 **어떤 레버가 켜져 있었는지 모르는 상태**에서 측정됐다. |
| C197 입력-결함 침묵 | task_020 formalize 서브의 operand 는 전부 정확했고, 실패는 에이전트가 필수 leaf(`account_open`)를 26행 전부 누락시켜 **엔진이 그것을 침묵 중 False 로 강제**한 데서 왔다. |
| **C264 Y2-B** | 채널을 무시한 블랙리스트가 env 거부 문자열을 키로 삼아 **올바른 채널의 gold 액션을 18회 차단**하고 모델에게 그 이름을 쓰지 말라고 지시했다 — 양 arm 공통 천장. |
| **C516 `formalize_groups`** | 우리 군 파싱이 포함관계로 **맞는 군을 조용히 삭제**. 라이브 로그 축자: `raw='checking_accounts business_checking_accounts savings_accounts …'` → 출력은 `business_checking_accounts, business_savings_accounts` 뿐(16건). |
| **C518 `business_*`** | '모델이 개인 손님에게 business 군을 골랐다'는 관측 **49건 전량**이 우리 필터 산물. 자격축 레버 계열의 표적 자체가 존재하지 않았다. |
| **MATCH_COUNT 거짓 인증** | 우리 도구가 발급한 '전부 표시' 인증의 **18.6%(19/102)가 반증 가능하게 거짓**(t7326). 본 회고에서 2026-08-04 B4 로 소급 적용 → **46/126(37%) 추가 검출** ⇒ 그 A/B 는 '회수 경계 표면화'를 시험한 적이 없다. [[25]] 정면 위반. |
| **x380/x381 CAP latch** | 조기 종료 래치 뒤 53/53 이 새 도구를 실행하고 6이 pass 하는데 **927회 정지 동안 리셋 0회** — 관측용 `print` 가 상태 대입보다 앞에 있어 `NameError` 가 대입을 삼켰다. |
| **C519 `gold_axes()`** | '선택축이 없는 60 태스크'는 태스크 성질이 아니라 우리 분류기가 세 꼴만 인정하기 때문(소스 검증). |
| C129 rall22 | write 지점의 LLM 재선택이 **gold 인자를 손님이 언급한 적 없는 값으로 바꾸고 그 치환을 메모이즈**했다. 같은 데이터의 집계 판독은 *"5 교정·039 6/8"* 로 성공처럼 보였다 — [[08]] 집계 오도의 코퍼스 내 최고 사례. |

### 3.3 계기·규율 — 방법론 자산

- **오프라인 op-eval 전면 철회**(2026-06-18): *"§17-§32의 τ² 전이/operand 수치는 전부 신뢰불가 프록시"*. 6월 코퍼스에서 유일하게 무너지지 않은 메타-결과이며, **6월 말 make-or-break 이 위반한 바로 그 규칙**이다.
- **C93 reward_basis census** — 내가 재려는 종점이 채점되는 종점인가를 무료로 답한다.
- **C244/C245 채점 규약 오라클** — 재구현 자기검정 1080/1080.
- **C486** — `action_match` 는 종점이 될 수 없다.
- **C274** — gold-miss 의 24.5%(본 회고 재계산 33.3%)가 직렬화/공백 차이. reward 는 무결, **진단층만 오염**.
- **C483** — n=8 프로브 잡음 바닥 ±4/8, 온도 0에서 8/8 같은 오답(적중은 표집 꼬리).
- **C58 rand/통계/원리 3팔** — 어떤 A2 값이든 '정책 도출인가 분포 베끼기인가'를 판정한다. **§8-4 의 T1=30 에 걸었더라면 그날 걸렸다.**
- **C215 arm 교집합 검사** — 본 회고에서 재실행: `bank_day6frontA` ∩ `bank_day6frontB` = **0**.
- **`b220745d`** — 엔진이 채점 인자를 쓰는 배선의 제거. 본 회고 4중 확인(커밋·소스·A2 주석·로그 100회).

---

## 4. 무효 / 오염 — GOLD_TAINTED 14건과 인용 금지 수치

**정의**: 종점 또는 A2 내용의 출처가 gold(벤치 정답)인 것. [[23]] 위반이거나 그 경유.

| 날짜 | 실험 | **인용 금지 수치** | 오염 기전 |
|---|---|---|---|
| 06-17 | substitute op-IR 오라클 (§19) | **32/32 · 27/27 · 420/420 = 1.000** | 입력도 gold(손으로 쓴 IR), 종점도 gold. 모델이 루프에 없다. `substitute` 생성원 자체가 관찰 뒤에 도입된 순환. |
| 06-26 | 격리 operand-pick 프로브 | **모든 정답률** | 결정점 열거 = `evaluation_criteria.actions`(gold) · 실패 특징 `gold_argmax/argmin/avail` · 채점 = `gold_new in ans` 부분문자열 · 입력 `reason_for_call[:700]`(8일 전 무효 판정된 필드) · 비교 팔 없음. |
| 06-26 | **통제 operand 실험(make-or-break)** | **GIVEN-SPEC 88/88(100%)** · 그 위에 선 *"operand capability gap 없음"* · *"faithful-formalize SFT NO-GO 확정"* | `gold_opts = vs[gold_new]["options"]` 를 프롬프트에 넣고 그 dict 를 포함한 목록에서 고르라는 설계 ⇒ 100%는 구조상 보장. 부정통제 전무. **원자료 없음(문서 축자로만 존재)**. 이 판정을 뒷받침한 t71 은 생존 로그에서 **실패**했고 `op_match_gold=False` 였다. |
| 07-11 | T5-C V0 화이트리스트 | **item 129 fix / 3 break = net +126**, `disamb_sub_args` 등재 근거 | A2 필드에 **무엇을 등재할지를 gold 재현율 비교로 골랐다**(`*_ok = gold 일치`). `bank_rule_fit.py` 의 T1=30 선택과 **절차적으로 동형**. |
| 07-14 | C78 keystone REPLAY | **필터오답 0 · 81.9% · 100%** | criteria 를 gold 에서 파생하므로 gold 는 항상 자기 기준을 만족 ⇒ 필연. 그 천장이 A2 `_note_reference_filter` 를 낳았다. |
| 07-14 | C81 COMPUTE slice | **16.7% · 651** | 종점 전부 gold `action_checks`. 다음 날 C93 이 그 축이 reward 가 아님을 증명. |
| 07-14 | §8-2 재현율 | **89.4% / 73.6%** (본 회고 독립 재현 **94.4% / 72.3%**) | 정책 축자는 *"within 2 business days"* 인데 실린 값은 **30**. 30 의 유일한 근거가 gold 재현율. |
| 07-14 | §8-3 Δspurious | **오치환 27(6.3%) · 교정 375(90.6%) · 순 +348** | 저울 **양쪽이 모두 gold**. pass 로 환산되는 양이 아니다. |
| 07-14 | **§8-4 config 스윕** | **재현 94.4% · Δspurious 2.1% · 순 +366** | 스윕 축 {flat/min}×{cal/biz}×임계, **선택 기준이 gold 재현율 하나**. 자기모순 축자: *"min은 …**벤치 미적용**→flat이 더 gold-blind"* (gold 를 보고 gold-blind 를 선언). |
| 07-14 | §8-5 통합 replay | **90.9%(491/540) · +366** | 검사 대상의 임계가 gold 로 고른 T1=30 = 순환. 라이브 검정 없이 *"정량 정본"* 으로 승격. |
| 07-14 | C83 H_min | **DEBIT 4.27bit · CREDIT 2.60bit** | DEFAULT ~5필드가 gold 분포에서 읽은 상수(*"police_report 100% false"*) — **3일 전 C58 이 이름 붙여 금지한 행위**. |
| 07-16 | C96(a) | **liability 94.7% · `amount_difference=(exp-act)/100*balance`** | `bank_rule_fit.py:43` 이 `reward_info.action_checks` 를 읽고 `:65` 가 임계를 스윕. 두 산출물이 그대로 A2 로 갔고 `b220745d` 로 제거. ※ `provisional_credit_eligible` 을 base rate 65% 와 대조해 **NOT deterministic** 으로 강등한 부분만 유효. |
| 08-16 | t7297 ACT_DEMAND | **gold-write 전부 호출 9/20 → 12/20**(출시 판정선) | gold 일치율을 성적으로 읽었다. 같은 런의 pass 는 8↔9(null)이고 050 은 1/5 → **0/5**. over-action 2→8. |
| 08-17 | t7305 SUB_REQUIREMENT | **+6(gold 클래스 일치율)** | pass 는 **0/8 ↔ 0/8**, aux 는 **6 → 5(−1)**. 게다가 gold 를 낸 `[T2_DOCDECIDE]` 는 treat 0회 ↔ ctl 4회 = 기전 반증. 지연 2.4×. |

### 4.1 추가 인용 금지 (GOLD_TAINTED 밖)

- **C53 prov e2e** *"reward 0.580 > floor 0.547 (+3.3pp)"* — floor 가 다른 조건(t3·342 sim). 같은 t4·456 sim floor = **0.557** ⇒ +1.97pp, 부트스트랩 95%CI **[−3.51, +7.68]**, 태스크 단위 상승 33 / 하락 35. **`LEVER_CONSOLIDATION_2026_08_19.md:292` 가 아직 이 부풀린 수를 나른다 — 수정 필요.**
- **C60 routerv1** *"p4 +1.8pp 부분 환매·조건부 GO"* — p1 −0.66pp CI[−6.14,+4.61] · p4 +1.75pp CI[−4.39,+7.89], DISAMB 1,274회 중 switched **26회(2.0%)**.
- **asmscale 헤드라인** *"Assembled 14B(0.313) > bare 32B floor(0.281)"* — 문서 자신이 서술한 배제 규칙(6 pure-infra 만 제외)으로 계산하면 **14B 0.279 < 0.281** 로 사라진다.
- **C77/C78/C79 prevalence** — 모집단 853 이 C80 에서 오염 확정(진짜 ⋈ 159). 추가로 본 회고 확인: 853 은 **7 태스크**에서 나온 유사복제이고 최대 덩어리 task_086(27%)이 C80 이 COVERAGE 오분류로 지목한 태스크다.
- **C243 OPERAND 57.1%** — 현행 코드 재실행 시 **47.6%** 로 재생되지 않음.
- **x420 R_asis 0.426 / R_doc 0.440 / R_neg 0.445** — 원자료 부재 + `MSG_CAP=3000`·`TOTAL_CAP=60000` 절단(회수 문서는 20k~50k자)이므로 결손이 아니라 절단의 몫일 수 있다.
- **x421 상승 0 / 하락 1 / 동률 3** — n=4 는 C483 잡음 바닥(n=8 에서 ±4) 아래.

---

## 5. ★아키텍처 판정 — "LLM formalize → 엔진 calc" 는 그 형태 그대로 시험된 적이 있는가

### 5.0 답

# **없다.**

174건 중 이 아키텍처를 **그 형태 그대로**(① 상수·규칙의 출처가 정책 · ② 엔진은 formalize 된 값만 계산 · ③ 엔진이 채점되는 인자를 만들지 않음 · ④ 대조 팔과 채점되는 종점) 측정한 실험은 **0건**이다.

그 형태로 **지어진** 것은 최소 5건 있다. 그중 **효과가 대조와 함께 측정된 것은 0건**이고, 반대로 **효과가 측정된 것들은 그 형태가 아니었다.**

### 5.1 측정된 것 = 다른 형태 (엔진이 gold 로 맞춘 규칙으로 채점되는 값을 채운다)

`compute_ops` 2개(`customer_max_liability_amount`, `amount_difference`). 본 회고 4중 확인:

1. **커밋 `b220745d`(2026-08-19 09:10:40)** 축자: *"Stop the engine from writing scored arguments, and remove the gold-fitted constants"* · *"the A2 note records 'T1=2 (policy literal) 73.6% / T1=30 (proxy) 89.4%'. The policy text says 'within 2 business days of statement'; the shipped value was 30."*
2. **소스**: `bank_rule_fit.py:43` 이 `(s.get("reward_info") or {}).get("action_checks")` 에서 (days, gold_liab) 쌍을 모으고 `:65` 가 `for thr in (2, 5, 10, 30, 60)` 을 스윕한다. 본 회고 독립 재현: **T1=2 → 72.3% · T1=30 → 94.4%**.
3. **A2 자백**: `_note_compute_ops` 축자 *"2026-07-16 확장(**gold-fit 확증**·rule_fit.py): liability 94.7% · amount_difference=(exp-act)/100\*balance"*. 현재 `compute_ops = {}`.
4. **라이브 발화**: 로컬 로그 전수 `zgrep` = `[T2_RESOLVE] compute silent-repair customer_max_liability_amount` **총 100회**(-1→50 44 · 0→50 40 · 500→50 6 · 50.0→50 6 · 50→500 2 · -1→500 1 · 0→50 1). 최다 `bank_t7295_a_20260815n` 16회.

이것은 전제의 아키텍처가 아니다. 전제는 *LLM 이 formalize 하고 formalize 된 값만 계산한다* 인데, 여기서는 **formalize 결과가 없거나 틀리면 엔진이 값을 채웠고**, 그 값의 임계는 정책이 아니라 gold 재현율로 골랐다. [[23]] + [[62]] 동시 위반이며, **엔진이 채점 대상을 대필하는 순간 측정 대상이 사라진다.**

### 5.2 그 형태로 **지어졌으나 측정되지 않은** 5건

| 건 | 날짜 | 형태 적합성 | 측정 상태 |
|---|---|---|---|
| **`get_atm_fee_discrepancies`** | 08-13 (C462) | **가장 근접**. C462 축자: *"모델이 fee 라인+짝 인출액+network(rho/non_rho/foreign) 를 **복사·formalize**([[22]])·엔진은 min/max/tier 산술만"* · *"반환 = id 목록만(금액 합산은 모델 몫·[[62]] 최소 결정론)"*. 요율 출처는 정책 축자 8클래스 전수(`ATM_FEE_SCHEDULE_VERBATIM_2026_08_13.md`·gold 미접촉) ⇒ **[[23]] 클린**. | **측정 0.** 본 회고 원자료 확인: `bank_t7274w_a`(task 074·072) / `_b`(task 073) 는 **arm 이 아니라 태스크 분할**이고 총 **n=3 · pass 0**. 짝 A/B 가 존재한 적 없다. |
| ↳ **FIX-5(같은 날 C464)** | 08-13 | **형태가 깨졌다.** `return_template` 에 `${delta_total:.2f}` 추가 = 엔진이 **순 보정액**을 계산해 모델에게 건넨다. 본 회고 확인: task_073 gold `action_checks` 의 크레딧 인자는 **`amount: 9.50 / 9.00 / 1.50`** — 즉 `delta_total` 은 **채점되는 gold 인자 그 자체**다. | 미측정. 제거된 `compute_ops` 와 **구조 동형**이며 현재 `a2/banking_knowledge.gate.json:1749` 에 **살아 있다**. |
| **`group_reduce` (ACCOUNT_APY)** | 07-19 | **형태 적합**. `t2_compute.py:182` 축자: *"엔진은 combinator 2개(max1/sum)만 안다. stack_rules 규칙표=A2"* · 도메인 리터럴 0 · unknown-kind silent drop 금지(`_gr_flags` 표면화). | **측정 0.** 본 회고 확인: `grep -c "group_reduce" RESEARCH_MASTER.md` = **0**, `get_best_account_option`/`get_correct_savings_apy` = **0**. 오프라인 `test_group_reduce.py` ALL PASS 뿐. |
| **`catalog_filter` 3버킷** | 07-25 (C181/C185) | **형태 적합**. eligible/excluded/**unverified** — 미지를 주장으로 바꾸지 않는다. | 오프라인 `test_compute.py` ALL PASS. **라이브 근거 `bank_w5` 가 repo 에 없다** ⇒ 라이브 [?]. |
| **`check_cli_eligibility`** | 07-23 (C114) | 형태 적합(오프라인 판정 → 텍스트가 따르는지). | n=3 · 부정통제 없음 · **대조가 성립 안 함**(수정 전 런도 approve 도구 호출 0). 052 는 전 런 reward 0.0. |
| **C113 rate-formalize 사다리** | 07-17 | **분담선을 직접 재려 한 유일한 설계**(formalize-only → 날짜도구 → 날짜+곱셈도구). | **RAW_MISSING**(`rate_tc2_20260718.log` 0건). 게다가 *"KB 해석 100%"* 반쪽은 같은 문서 §2d 가 반증(*"temp=0에서도 서브 base_rate 오독 다수"*). n=4 거래·단일 태스크. |

### 5.3 전제를 **지지하는** 증거 — 하나 있고, 그것은 결손 측정이지 아키텍처 측정이 아니다

**x288**(2026-08-13·C462⒜·073 t0 문맥·n=8): A_LIVE **0/8** · **A_DOCS 0/8**(세 클래스 ATM 문서 13편 축자 동봉에도 gold 세 금액 미산출) · **D_NEG 0/8**(무관 문서 통제). 사전등록 문턱(A_DOCS ≤2/8) 발동.

이것은 [[62]]① 을 **바르게** 수행한 사례다 — *재료를 다 줘도 안 닫힌다 ⇒ 계산 결손이다*. 사용자 전제의 **동기**는 이 한 건이 지지한다. 그러나 그 다음 단계(*"그래서 결정기로 옮기면 pass 가 오르는가"*)는 **한 번도 재지 않았다**: FIX-5 출시 근거는 unit 12/12 + n=1 궤적 직독이었고, 라이브 짝 A/B 는 존재하지 않는다.

### 5.4 반대 방향 증거 — 정책을 주면 formalize 가 **붕괴**한 사례

같은 밤(08-17) 두 프로브가 정반대 부호를 냈다:
- **x345**(복사 충실도): 프롬프트에 축자로 있는 레코드 값 3종(`5000` · `Gold Rewards Card` · `cc_584f9c5d00_gold`)은 격리 서브가 축자 복원 — 이는 **판단 없는 복사**([[42]] induction-head 부류).
- **x347**(`param_cap`): **정책을 주면 `5000000` 으로 붕괴**. ⇒ *"한 등급에서 성공했다고 검증기를 지우면 다른 등급에서 조용히 상한을 넘긴다."*

그리고 **C215**(07-29)는 전제의 직접 반례다: 정책 상수 `monthly_threshold` 를 도구 내부에서 빼내 **LLM 인자로 만든 것**이 이 구간 −6 pass 중 유일한 레버 인과였다. 즉 *무엇을 formalize 시키고 무엇을 엔진에 고정할지*의 경계는 자유 변수가 아니며, 잘못 그으면 손해가 난다.

### 5.5 판정 요약

| 질문 | 답 |
|---|---|
| 아키텍처가 그 형태로 **지어진** 적이 있나 | **있다** (5건: fee 도구 원판 · `group_reduce` · `catalog_filter` · `check_cli_eligibility` · rewards `select_discrepant`) |
| 그 형태로 **효과가 대조와 함께 측정된** 적이 있나 | **없다 (0건)** |
| 그럼 무엇이 측정됐나 | **엔진이 gold 로 맞춘 규칙으로 채점되는 값을 채우는 형태** — 그 종점은 gold 이고, [[62]] 상 측정 대상이 소멸한다. `b220745d` 로 제거. |
| 전제의 동기(계산 결손 실재)는 지지되나 | **부분적으로 지지된다** — x288 A_DOCS 0/8 · D_NEG 0/8 (n=8, 단일 태스크 문맥) |
| 지금 같은 구조가 남아 있나 | **있다** — `a2/banking_knowledge.gate.json:1749` 의 `${delta_total:.2f}` 는 task_073 gold 인자(9.50/9.00/1.50)와 같은 양이다. **미측정 상태로 라이브다.** |

> **그러므로 "실험이 모두 잘못됐나"에 대한 정확한 답**: 실험은 결손의 *위치*에 대해 많은 것을 확립했다(§3). 잘못된 것은 ⑴ **종점**(46.6%가 gold 일치율) ⑵ **아키텍처를 시험하는 대신 아키텍처의 위조판을 시험한 것**(gold-fit 상수 + 엔진 대필) ⑶ 그 결과 **전제 자체가 3개월 동안 한 번도 검정대에 오르지 않은 것**이다.

---

## 6. 재사용 가능한 자산 (지금 그대로)

### 6.1 로컬 생존 원자료 — 재실행 비용 0

| 자산 | 내용 |
|---|---|
| `sim_results/bank_t7326_{halfA,halfB}_20260819q.results.json.gz` | 40 sim · pass 7/40 · `reward_basis` DB35/A4. 본 회고의 MATCH_COUNT 감사·x400 재분류·x384 입력이 전부 여기서 나왔다. |
| `sim_results/asmscale_{32b,14b}_0626pm_*.results.json.gz` (3.0/3.3MB) | 342 sim × 114 태스크. 배제 규칙 민감도 재계산 가능. |
| `sim_results/{fl32b_floor,prov_e2e,routerv1}_retail_t4.results.json.gz` | 456 sim × 114 태스크 **완전 짝지음** 3종. 어떤 pass 축 재분석도 비용 0. |
| `sim_results/{fl32b_floor,fl14b_floor,prov_e2e,qwq32b}` 4 arm | C43 층화 재현용. |
| `c51_disamb_results.jsonl` (400행) | `task,trial,idx,arg,gold,ncand,A,A_ok,B,B_ok` 필드 완비. |
| `bank_compute_cases.jsonl.gz` (880행) | `gold_liab·agent_liab·facts` 10종. 본 회고에서 §8-2/8-3/8-4 전 수치를 몇 초에 재현. |
| `bank_xmatch_formalize.results.json` (467KB) + `bank_xmatch_cases.jsonl.gz` | 올바른 분모(⋈ 159)로 즉시 재집계 가능. |
| `C:/tmp/traj/*_banking.json` **17개** | 사정권 계산·census 를 임의 신규 레버에 비용 0 으로 재적용. |
| `x351/x352/x353/x354/x355/x356/x357(v1·v2)/x380/x384/x400/x403` JSON | **전부 `reports/facet_rft_2026/` 에 실재**. ⚠ 2026-08-19 감사 §7 L8 이 이들을 '원 산출물 부재'로 잘못 등재 — **갱신 필요**. |
| `probe_039_join_artifacts.json` (121KB) | 입력 + gold 완비 격리 fixture. `--dump` 만 켜면 재실행. |
| `eamb6_fl32b_floor_retail_t4.jsonl` (1,614행) | C58 3팔 대조 재현용. |
| `x8_triage_rows.jsonl` (432행) + 수작업 gold 48 | X8 재채점 가능. |
| `ma/cases/tau2_{retail,airline}_cases.jsonl` (85KB/21KB) | nl·old_options·gold_options·variant_catalog 완비. |
| `ma/results/scale/multidomain_scale/*.json` (12개) | 32B/72B/235B × retail/airline × g0/g1, per-case rows 완비. |
| `bank_t7274w_{a,b}` · `bank_t7292_{a,b}` · `bank_t7299_{ctl,treat}` · `bank_t7305_{ctl,treat}` 로그 | 각각 fee 오답·중복 크레딧·`[T2_ARG_AXIS] deny`·`[T2_DOCGROUP]` 의 **유일한 로컬 근거원**. |

### 6.2 즉시 실행 가능한 도구

- **`efiso_detmatch_proof.py`** — LLM 0 · gold 4 사례 내장. 본 회고 재실행: merchant-only **슬립 8/8 검출 · gold 25/25 통과**(merchant+amount 는 24/25 — 술어 강도 선택의 실측 근거).
- **`x12_action_fail_exact.py`** — 채점 규약 오라클. 재구현 일치 **1080/1080**. 새 실패 분류를 만들기 전 필수.
- **`x9b_refiso_adjudicate.py`** — 기권 축과 값 축을 분리하는 판정기(판정기 없이 raw 를 세면 오독한다).
- **`x22_txnid_isoprobe.py`** (16,401B) — 오염 방지 4조항 내장. **오늘 실행 가능.**
- **`x28_merchant_ground_census.py`** — 가드 출시 전 **gold 없이** 술어의 구조적 오탐 표면을 센다.
- **`bank_eplan_controller.py --dag`** (35KB) — per-step 연산 분류기. [[62]] 순서를 그대로 구현.
- **`bank_xmatch_forensic.py`** — 제출집합 ↔ gold집합 3분할. 어떤 도메인에도 즉시 이식.
- **`test_group_reduce.py` · `test_compute.py` · `test_c197_inputholes.py` · `test_c278_quotepin.py` · `test_unknown_name_channel.py`** — 본 회고 재실행 전부 PASS.
- **`bank_rule_fit.py`** — STOP 헤더가 지정한 용법으로만: *"Use this only as a forensic tool: to ASK whether a field is deterministic at all"*.

### 6.3 설계 관용구

| 관용구 | 출처 | 무엇을 막는가 |
|---|---|---|
| **차단 팔 ↔ 공급 팔 분리** | §35b | 이득이 막기에서 오는지 주기에서 오는지 |
| **rand / 통계 / 원리 3팔** | C58 | A2 값이 정책 도출인가 분포 베끼기인가 |
| **agent-correct 셀을 부정통제로** | §8-3 | 레버 선택성(Δspurious) |
| **파생 정확도 × 기존 오답률 교차표** | C82 | 고칠 자리에 능력이 있는가(착수 전 무료) |
| **base rate 대조** | C96(a) | gold 스윕의 최적점은 정의상 base 를 넘는다 |
| **reward_basis census** | C93 | 내 종점이 채점되는 종점인가 |
| **잡음 바닥 선측정** | C483 | n 이 모자라면 결론이 사라진다 |
| **arm 교집합 assertion** | C215 | 병렬 분할을 대조로 착각 |
| **표적 모집단 사전 확인** | t7307/8 ⓐ · x384 | 표적이 없으면 A/B 를 읽을 수 없다 |
| **레버 마커 0/0/0 확인** | §35c | 격리가 실제로 격리인가 |
| **`would-fire but suppressed by=`** | C115 · C264 | '표적 부재'와 '상류 억제'를 가른다 (현재 `T2_HAVE_VALUE` 만 보유) |
| **per-switch/per-firing 포렌식** | C126 · C129 | 집계는 해로운 레버를 성공처럼 보이게 한다 |
| **모델 원출력과 우리 파싱을 같은 줄에** | C516 `[T2_DOCGROUP] raw=… → …` | 우리 층 산물을 모델 결손으로 오귀속 |
| **as-is / 반사실 replay 짝** | C197 | 유료 arm 전 무료 인과 검정 |
| **3버킷 계약(미지 ≠ 거짓)** | C181/C185 | 우리 도구가 모델에게 거짓을 단언 |
| **`pin_kind` 라우팅** | C278 | 열린 술어를 LLM 선언 + 엔진 닫힌-검사로 |
| **사전등록 GO/NO-GO 3조건** | C10 §7 | (여전히 미이행) 무료 instruction 이 1차 경쟁자 |
| **3기준 사전 게이트** | AXIS_DECISION | 술어 닫힘 · 처방 닫힘 · 변이 불변 |

---

## 7. 다음에 재야 할 것 — formalize 단계 격리 채점 명세

### 7.1 왜 값을 채점하면 안 되는가

값을 채점하면 세 가지가 동시에 일어난다.
1. 종점이 gold 가 된다 ⇒ [[23]] 경유. 이 코퍼스에서 81건이 그 길로 갔고 그중 37건이 폐기됐다.
2. `action_match` 는 표기 차이로 무너진다(C486: `9.50` ↔ `9.5`).
3. 값을 맞히게 도우려는 순간 엔진이 채점 인자를 대필하게 된다 ⇒ [[62]] 측정 대상 소멸(`compute silent-repair` 100회 · `${delta_total}`).

### 7.2 대신 채점할 것 — **유도식/근거** ([[22]] 근거-우선 formalize 계약, [[66]] 인용-근거)

formalize 산출을 **값이 아니라 3-튜플**로 요구한다:

```
{ rule:  { doc_id, quote },        # 어느 정책 줄을 근거로 삼았나 (축자)
  slots: { name -> {value, quote} },# 그 줄에서 뽑은 술어/상수 (각각 축자 근거)
  bind:  { name -> record_ref } }   # 그 술어에 어떤 레코드를 바인딩했나
```

채점은 **전부 닫힌 검산**이고 **gold 를 한 번도 보지 않는다**:

| 채점 축 | 검산 | 성질 |
|---|---|---|
| **G1 인용 실재** | `quote` 가 회수 문서 본문에 substring 으로 실재하는가 | 결정론 · 변이 불변 |
| **G2 상수 유래** | `slots[*].value` 가 그 `quote` 안에 실재하는가 | 결정론. **T1=30 은 여기서 즉시 걸린다**(정책 축자에 30 이 없다) |
| **G3 바인딩 실재** | `bind[*]` 가 도구 출력에 실재하는 id 인가 | 결정론. 날조 ↔ 오선택을 가른다(C472⒡ 가 못 하던 구분) |
| **G4 완결성** | 규칙이 요구하는 slot 이 모두 채워졌나, 미확정은 **명시적 null** 인가 | C197 3-값 논리 · 침묵 금지 |
| **G5 자기일관** | `slots` 만으로 계산했을 때 모델이 낸 값이 재생되는가 (엔진은 **검산만**, 값을 만들지 않는다) | [[62]] 안전 — 엔진 산출물이 채점 인자로 흘러가지 않는다 |

**G1~G5 어디에도 gold 가 없다.** 그리고 G5 가 핵심이다: 엔진은 *모델이 선언한 유도식이 모델이 낸 값을 재생하는가*만 본다. 재생 실패는 계산 결손, G2 실패는 formalize 결손, G3 실패는 참조 결손 — **세 결손이 분리된다.** 이것이 3개월 동안 한 번도 분리되지 않은 것이다.

### 7.3 팔 구성 (사전등록)

| arm | 내용 | 무엇을 답하나 |
|---|---|---|
| **A_ASIS** | 라이브 결정점 **재생**(x420 방식) — **절단 금지**(`MSG_CAP`/`TOTAL_CAP` 제거) | 라이브 상태 그대로의 formalize 품질 |
| **B_POLICY** | 결정론 검색(tf-idf·질의 = 손님 요청 + 도구 이름 + 설명·**gold 미참조**·x421 방식)으로 고른 정책 문서 동봉 | 재료 전달이 G2 를 여는가 |
| **C_NEG** | 같은 길이·같은 개수의 **무작위** 무관 문서 ([[57]]) | B 의 이득이 문서 내용인가 분량인가 |
| **D_DERIV** | 값을 **묻지 않고** 3-튜플만 요구 (계산 금지) | 계산 부하를 뺐을 때 formalize 가 사는가 |
| **E_MINUS** | 후보 집합에서 **오답 하나 제거**(x353 E_MINUS 6/8 의 재현) | [[63]] 제거 레버가 이 축에도 사는가 |
| **F_NULL** | 무내용 재시도 ([[57]]) | 재시도 자체의 몫 |

### 7.4 표본·판정선

- **n ≥ 24 per arm**(8 × 3 블록). C483: n=8 잡음 바닥 ±4/8이므로 n=8 에서 차 ≤4 는 인용 금지. 블록별 수치를 함께 보고한다.
- 판정선은 폭이 아니라 **차의 표준오차**로 세운다(두 팔 독립 시 sd≈2.0 → 차 ≥5 가 대략 2.5σ).
- 온도 0 팔을 반드시 포함한다(C483⒞: 온도 0 에서 8/8 같은 오답 — 적중이 표집 꼬리에서 나오는지 확인).

### 7.5 실행 전 필수 사전 검사 (전부 무료)

1. **종점 검정** — `reward_basis` census(C93). 이 표적 태스크들이 DB-basis 인가 ACTION-basis 인가. DB 면 `action_checks` 로 아무것도 읽지 않는다.
2. **표적 모집단** — G2/G3 가 실제로 실패하는 결정점이 몇 개인가. 분모를 **우리 마커나 우리 분류기에서 뽑지 않는다**(t7310 · C519).
3. **배선 생존** — `t2_liveness`([[67]]) 로 0단계. 발화 0 이면 음성이 아니라 **무효**(t7304 · t7324).
4. **잡음 바닥** — 동일 본문 5블록(x320 방식) 선측정.
5. **arm 교집합** — 하드 assertion(C215).
6. **정보-맞춤** — 격리 구성 전 *"라이브가 그 시점에 가졌던 것 중 무엇을 뺐나"* 를 열거([[18]] · x417/x419/x343 이 이 검사를 안 해서 무효화됐다).
7. **원자료 영속** — gzip → `reports/facet_rft_2026/sim_results/` → `git add -f`, distinct tag. **프로브 출력도 포함**(현재 32건이 RAW_MISSING 인 직접 원인).

### 7.6 이 명세가 즉시 답하는 미결

1. **`${delta_total}` 감사** — `a2/banking_knowledge.gate.json:1749` 가 task_073 gold 인자(9.50/9.00/1.50)를 대필한다. G5 로 바꾸면(엔진이 검산만 하고 값을 안 준다) [[62]] 를 안 건드리고 같은 결손을 잰다. **`compute_ops` 와 구조 동형이므로 같은 감사를 즉시 적용해야 한다.**
2. **`group_reduce` · `catalog_filter` 최초 측정** — 형태가 옳은데 3개월간 라이브 효과가 0회 측정됐다. §7.3 틀에 그대로 걸린다.
3. **C10 사전등록 3조건 이행** — *"★competitor = escalate 가 아니라 **무료 instruction**"* 팔은 2026-06-21 이후 지금까지 한 번도 돌지 않았다. 6월의 NO-GO 는 이 조건들로 판정된 적이 없다.
4. **x421 재발사** — 절단 제거 + n≥24 로 돌리면 2026-08-19 감사가 명시한 빈 칸(*"집합 內 실재하는 이름·값 중 오답 선택 — 레버 0"*)을 **처음으로** 재는 실험이 된다.
5. **t7328 완주 + 원자료 회수** — t7326 의 7/40 중 gold-fit 배선이 만든 몫을 분리한다.

### 7.7 문서 수정 부채 (본 회고에서 확정)

- `LEVER_CONSOLIDATION_2026_08_19.md:292` — C53 *"+3.3pp"* → 같은 조건 대조 **+1.97pp · CI[−3.51, +7.68] · 상승 33/하락 35**.
- `OPERAND_LEVER_AUDIT_2026_08_19.md` §7 **L8** — x351·x352·x353·x354·x355·x356·x357(v1·v2)·x380·x384·x400·x403 출력은 `reports/facet_rft_2026/` 에 **실재**한다. 살아 있는 근거를 죽은 것으로 분류 중.
- 같은 감사 §4 — 양성통제 `[DUPLICATE-READ]` 2,301 은 t7326 안에는 **14건**뿐(전 코퍼스 스캔 수치). `task_055 실제 10 ↔ 보고 34` 는 재현되지 않음(본 회고 계수 13↔10).
- `x275` HARMFUL 판정 — **방향은 원자료로 확인**(`[T2_ARG_AXIS] deny got=checking want=['business_checking','savings']` 실재)되나 **크기는 인용 불가**(t7299 두 팔 ARG_AXIS 총 25줄).

---

## 8. 부록 — 이 회고가 로컬 원자료로 직접 재현한 것

| 항목 | 결과 |
|---|---|
| t7326 pass · reward_basis | **7/40** · DB **35** / ACTION **4** / 없음 **1** (문서 일치) |
| `compute silent-repair` 라이브 발화 | 로컬 로그 전수 **100회**(t7326 8회분은 로그 미영속으로 검증 불가) |
| 커밋 `b220745d` 메시지 | 축자 확인 |
| `compute_ops` 현재 값 | `{}` (gate.json) |
| `bank_rule_fit.py` gold 스윕 | `:43` `reward_info.action_checks` · `:65` `for thr in (2,5,10,30,60)` |
| T1 스윕 독립 재현 (880행) | **T1=2 → 72.3% · T1=30 → 94.4%** |
| task_073 gold 크레딧 인자 | `amount 9.50 / 9.00 / 1.50` (전부 `action_match` False) |
| `t7274w` a/b 구조 | **arm 아님** — a=(074,072), b=(073), 총 n=3 · pass 0 |
| `${delta_total:.2f}` | `a2/banking_knowledge.gate.json:1749` 에 **현존** |
| `group_reduce` 라이브 측정 | RESEARCH_MASTER 언급 **0회** |
| `t2_compute.py:182` | *"엔진은 combinator 2개(max1/sum)만 안다"* 축자 확인 |
