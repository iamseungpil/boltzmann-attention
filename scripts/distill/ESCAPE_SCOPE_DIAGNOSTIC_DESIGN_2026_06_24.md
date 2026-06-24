# Escape-Scope Diagnostic — make-or-break 첫 측정 설계 (2026-06-24)

> **위치**: `EPISTEMIC_A2_THESIS_2026_06_23.md` §6 "★선행 진단(escape 범위 측정)"의 구현 설계. SOAR impasse(§3 블록)가 분류축. **이 측정이 abstain-커리큘럼 SFT(풀 make-or-break) 전체를 GO/NO-GO**. 학습 *전* 단계 = 지금 진행 가능(딥리서치 방법 대기 불요). [[05]]·[[10]]·[[11]] 준수.

## 0. 한 줄
**우리 escape("빈/모호한 관계→ASK")가 32B의 실패를 *얼마나* 덮는가**를, 기존 gap 데이터를 formalize→σ로 돌려 **ⓐ(표면화·잡힘) vs ⓑ(침묵·놓침)**으로 분류해 측정한다. = 학습 방향을 켜고 끄는 게이트.

## 1. 핵심 질문 + 단일 베팅
- **escape ceiling**: *명세-충실하게* formalize했을 때, 32B 실패 중 σ가 **비단일(|σ|≠1)**로 표면화되는 비율 = escape가 *원리상* 잡을 수 있는 상한.
- **★단일 베팅**: 최대 gap 클래스 **wrong-ORDER 47%**가 **tie impasse(ⓐ·|σ|>1) 냐**, **확신적 오해결(ⓑ·|σ|=1 틀림) 냐**.
  - tie면 → "후보 여럿→어느 것? ASK" 학습이 gap 절반을 직접 때림 = **GO 강함**.
  - **단 사전 경고(정직)**: tau2 다수 태스크는 유저 요청이 주문을 *유일하게* 결정함(정보는 있음·모델이 매핑 실패)을 시사 → wrong-order가 **ⓑ(mis-ground)일 가능성**이 실제로 높음. 그러면 escape는 *좁고* 명세-충실 formalize(§3)가 load-bearing. **이 진단의 목적 = 그 추측을 데이터로 결판.**

## 2. 분류 스키마 (SOAR impasse → ⓐ/ⓑ)
각 실패 분기점에서 **명세-충실 σ**(아래 §4 oracle formalize)를 참 DB state에 실행, |σ|와 정답대조로:

| |σ| (faithful) | SOAR impasse | 분류 | escape 동작 | 실패 정체 |
|---|---|---|---|---|---|
| **0** | no-change | **ⓐ** | "없음→ASK" 발화 | 모델이 묻지 않고 행동 |
| **>1** | tie | **ⓐ** | "모호→ASK" 발화 | 모델이 묻지 않고 한 개 찍음 |
| 1 (precond 위반) | constraint-failure | **ⓐ′** | G-gate deny | (이미 gate가 처리·기존 레버) |
| **1 (옳은 단일·but 모델은 딴 걸)** | impasse 아님 | **ⓑ** | 안 잡힘 | mis-formalize(grounding 실패) |
| 1 (옳은 단일·but arg 값 틀림) | impasse 아님 | **ⓑ-op** | 안 잡힘 | operand=B2-resolve 별도 |
| 1 (옳은 entity·but 틀린 operator) | impasse 아님 | **ⓑ-act** | 안 잡힘 | wrong-action=선택 학습 별도 |

- **ⓐ = escape가 원리상 잡음**(σ 비단일이 토큰으로 표면화). **ⓑ = 침묵 잔여**(σ=1인데 틀림 → 카디널리티로 안 드러남).
- **ⓐ 안에서 2차 태그(커리큘럼 설계용·[[10]] A/B)**: **ask-needed**(정보부재→유저에 질문·SOAR 외부) vs **compute-resolvable**(argmax/rank·B2·SOAR 내부 subgoal). → 커리큘럼이 *무엇을* 가르칠지 갈림.

## 3. 입력 데이터 (재실험 0)
- **Stage-1(헤드라인)**: gap-task = **gpt-4.1 pass ∧ 32B fail-all-3 = 15 task**(기존 census `/tmp/gapcensus.sh`·`decompose.sh` 산출). 디스크에 이미 sim 있음.
- **Stage-2(robust 확장·선택)**: 32B retail floor 전체 실패(robust=fail-all-3 우선) → ⓐ/ⓑ 비율의 CI. + airline/banking(cross-domain escape 너비).
- 측정신호: **fail-all-3(robust)** 우선·pass^1 점추정 단독 무효([[06-NOW]] user-sim 편차).

## 4. 방법 = formalize→σ 파이프라인
**핵심 설계결정 = formalize를 *누가* 하나** (두 arm 분리·conflate 금지):
- **(A) Ceiling arm = oracle/faithful formalize**: 각 gap의 결정점에서 gold-action이 함의하는 *명세-충실* predicate를 도출(§3 "조건 맞는 *모든* tuple"·섣불리 1개 narrowing 금지). 출처 = gold action seq + task user-scenario. 15개 → **Opus-보조 큐레이션**(소N·고충실·비용 미미). → |σ| 분포 = **escape 상한(아키텍처)**.
- **(B) Base arm = 32B 자기 formalize**: base 32B가 실제로 형성한 predicate(궤적서 추출 or 재질의) → faithful과 대조 → **mis-formalize rate**(=학습이 닫아야 할 ⓑ). → ceiling−base gap = *학습 여지*.
- **σ 실행**: 태스크 DB state 로드 → predicate를 σ_condition(table)로 실행 → |result| + 정답 tuple 포함여부. 엔진 = `grounding.json` π/⋈/σ(부분 실물·미비분 보강). **도메인분기 0·A2 관계만**([[05]]).
- gpt-4.1 user-sim **불요**(정적 분석·throttle/비용 없음). 로컬 32B만(GPU1 유휴 or GPU0 t5 후).

## 5. 출력
1. **ⓐ/ⓑ split**(ceiling arm·impasse 타입별 막대): escape 너비. **wrong-order가 어디 떨어지나**(단일 베팅 결판).
2. **mis-formalize rate**(base vs faithful): 학습이 닫을 ⓑ 크기.
3. **ⓐ 내 ask-needed vs compute-resolvable**: 커리큘럼 타깃 비율.
4. impasse 타입 × gap-클래스 교차표(wrong-order/action/operand/budget/over-action → tie/no-change/constraint/non-impasse).

## 6. GO / NO-GO (정직)
- **GO(abstain-커리큘럼 강함)**: ⓐ가 gap의 *과반*(특히 wrong-order가 tie) + base mis-formalize가 faithful 대비 학습으로 줄 여지(ceiling≫base).
- **부분 GO(명세-충실 우선)**: ⓐ 좁음(wrong-order=ⓑ) → escape보다 **faithful-formalize 학습**이 본체. 커리큘럼 무게중심 이동(여전히 학습대상이나 ASK 아님).
- **NO-GO**: (a) 5개 실패가 유한관계로 깔끔히 표현 안 됨 or (b) faithful formalize해도 |σ|=1-틀림(ⓑ)이 압도적이고 그게 capability-bound(base가 oracle 줘도 못 따라옴) → **진짜 경계 = escalate/scale**(thesis §6 NO-GO).

## 7. 측정 주의 / 함정
- **conflate 금지**: ceiling(oracle formalize)과 base(32B formalize)를 *반드시* 분리 — 안 그러면 "escape 좁다"가 base capability인지 아키텍처 한계인지 구분 불가(과거 G5=0가 binding이 아니라 capability였던 것과 동형 함정).
- **oracle formalize 편향**: Opus가 *과도 narrowing*(섣불리 1개)하면 tie를 ⓑ로 오분류·*과도 광역*하면 ⓑ를 tie로. → §3 "조건 맞는 모든 tuple" 규칙 엄수 + 분류 spot-check.
- robust: 분류는 fail-all-3에 우선 적용·trial 흔들리는 건 별도 표기.
- σ 엔진 미비분(grounding.json 부분 실물)은 *측정 전* 보강·검증(gap 5개 소검증서 σ 출력 sane 확인 후 전수).

## 8. 구현 단계
1. **S0 설계 리뷰**(이 문서) → 합의.
2. **S1 harness 빌드**: `escape_scope_diag.py` — gap sim 로드 → 결정점 추출 → (A)oracle/(B)base predicate → σ 실행 → 분류. `grounding.json` σ 보강.
3. **S2 소검증**: gap 5개로 σ 출력·분류 sane 여부 *대면* 확인(무인 금지·load-bearing).
4. **S3 전수**: 15 gap → ⓐ/ⓑ split 헤드라인. (GPU1/0)
5. **S4 robust 확장**(선택): 32B retail 전체 실패 + cross-domain.
6. 산출 = `ESCAPE_SCOPE_RESULTS_2026_06_2x.md` + GO/NO-GO 판정 → 풀 make-or-break(SFT) 설계로 진행 or escalate.

## 9. 불변 (discipline)
- 정적 진단·**tau2 학습 0**([[11]])·A2 관계(σ/⋈/π)만·도메인분기 0([[05]])·gpt-4.1 불요.
- harness는 repo 커밋([[30]])·**S2 소검증 전 무인 전수 launch 금지**(틀린 split=방향 오도).
- 풀 SFT(make-or-break 본체)는 별개·딥리서치 방법(IDK targets·learning-to-defer `w0r8slp20`) 수령 후. 이 진단은 그 *선결 게이트*.
