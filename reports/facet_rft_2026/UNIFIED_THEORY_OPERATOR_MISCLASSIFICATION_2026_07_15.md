# 통합 이론 — 모든 실패 = open/closed 경계에서 해소연산 오분류 (2026-07-15)

> 정본. 사용자 통합 통찰(경계에서 과확신/under-action·formalize=Find/Get/Ask) + [[16]] 재유도 + H_min.
> 입력: C88(voting-vacuity)·C89(4하위유형)·C90(2층)·C91(closure)·[[16]] GENERALIZED_SCAFFOLD·[[44]] SOAR·C81(compute)·reach forensic.
> 등급: 이론 [D]·구성요소 [M](각 실패 실측)·통합비율 [?](결정적 실험 대기).

## 0. 한 문장
**모든 tool-use 실패 = LLM이 각 잔여 정보-갭의 *해소연산*을 오분류한다** — closed(GET/FIND/COMPUTE로 닫힘)를 추측하거나(과확신), open(ASK만)을 추측하거나, 갭을 아예 안 닫는다(under-action).

## 1. 해소연산 = [[16]] GET→FIND→INFER→ASK (INFER→ASK 결정론화·§4c)
각 연산이 서로 다른 open→closed 갭을 닫는다:
| 연산 | 닫는 갭 | world |
|---|---|---|
| **GET** | 알지만 미조회(메모리/컨텍스트) | closed |
| **FIND** | 열거가능·미열거(검색/DB·명시제약 filter) | closed |
| **COMPUTE** | 도출가능·미도출(정책/산술) | closed |
| **ASK** | 외부만 앎 | **open** |
- [[16]] §4c: **INFER(유효후보 추측=오류원) 삭제** → FIND후 1개면 사용·**≥2면 ASK**(결정론). 당신 "나열+ASK"=이것.
- COMPUTE = INFER의 *결정론* 부분(추측 아닌 도출)만 남긴 것 = 엔진 gate/calc.

## 2. 실패 = 오분류 두 방향
- **과확신(over-action)**: closed를 추측(GET/FIND/COMPUTE 안 하고 INFER) 또는 open을 추측(ASK 안 하고 committed). = 확신에 찬 오답.
- **under-action**: 갭을 아예 안 닫음(FIND/enumerate 안 하고 종료). = 조기종료.
- 공통 = **경계 오판**: "이게 closed(내가 닫을 수 있음)인가 open(물어야 함)인가"를 모름.

## 3. ★엄밀 매핑 — 우리 모든 실패가 이 하나로 (기존 궤적 데이터)
| 실패 | 실측 | 오분류 | 맞는 연산 |
|---|---|---|---|
| **⋈ mis-anchoring** | E-REGIME plan 560·n_disputes≥2=852/853·gold∈support 0/29 | ≥2 후보(open)를 confident 1개로 committed | **ASK**(≥2→ASK) |
| **reach under-action** | reach forensic closed-world 69.2% | 미완(open 갭) done 취급 | **FIND/GET**+H_min종료 |
| **reach open-world** | reach forensic 30.8% | 진짜 open | **completeness-ASK** |
| **compute** | C81 liability 51% 오답·systematic | 도출가능(COMPUTE-closed)을 추측 | **COMPUTE/verify** |
| **hallucination** | C45 날조 67% | open을 confident closed로 | GET/ASK |
- **⋈이 결정적**: 852/853 다중후보(≥2)인데 gold가 8샘플에 0 = 애매(open)를 확신(closed)으로 처리·재샘플로도 못 뒤집음 = **지식결함 아니라 경계오판**.

## 4. H_min = 최소질문 + [[16]] over-ask 문제 해결
- [[16]] §4c 열린 문제: "≥2→항상 ASK = over-ask 비용(측정필요)".
- **H_min이 답**: GET/FIND/COMPUTE로 닫을 수 있으면 먼저(질문0)·남은 진짜 open만 **H_min bit·VOI 순서로** ASK → 최소질문.
- 종료조건 = 잔여목표 엔트로피 > floor면 continue(문제2·under-action 차단).

## 5. H_min 기반 분해 = SOAR 일반화·MAKER 포섭 ([[44]])
- **분해 경계 = H_min > floor인 지점**(정보-갭). 각 subtask = "한 갭을 GET/FIND/COMPUTE/ASK로 닫기".
- **SOAR impasse**([[44]]): 이산 "결정불가"→subgoal. 우리 "≥2→ASK"=impasse 결정론판·**H_min=연속판**(impasse=갭 하나의 특수경우) → **SOAR 일반화**.
- **MAKER**: 구조적 micro-step(과분할). H_min=정보-갭 분할(필요분할만) → **MAKER 포섭**(최소·정보정당화).

## 6. novelty (PREEMPTION_SCAN 정합)
- 부품(GET/FIND/COMPUTE/ASK 각각·엔트로피-ask·decomposition)=선점→인용.
- **미선점 = 이 통합**: 모든 실패를 *하나의 연산-오분류*로 진단 + {GET/FIND/COMPUTE/ASK} 결정론 라우터 + H_min(종료·최소질문·분해). "correlated open problem"을 연산-오분류 taxonomy로 분해(MAKER blob 대비).

## 7. 정직한 잔여 2개 (과장 방지)
1. **compute는 순수 경계 아님**: 도출가능(closed) 내 추측 = "guess vs COMPUTE"(open-vs-closed 아님)·같은 연산-오분류 프레임엔 들어옴.
2. **경계 판정 자체의 calibration**: "≥2 감지"는 결정론(스키마)이나 "언제 confident해도 되나"(INFER→ASK 보정)는 [[16]] §6·C38/C42 미확립 = 진짜 make-or-break·learn 잔여.

## 8. ★결정적 실험 — 실행결과 + [[08]] 정직 교정
**`bank_failure_operator_label.py`(로컬 17궤적·gold-dispute 4139·실패 2858) 실행:**
- 분포: **COMPUTE 36.0%(1030) · FIND 25.4%(725) · GET/ASK 19.1%(547) · completeness-ASK 11.3%(323) · ⋈-ASK 8.2%(233)**.
- ⚠**"매핑률 100%"는 tautological**(4범주 exhaustive 설계) → *증거 아님*. **정보=분포**: 실패가 4연산에 **고르게 분산**(한 연산 지배 아님)=taxonomy 비-degenerate(필요조건).
- **★진짜 검증(non-trivial·반증가능) = operator-*choice* 오류인가(연산 안 씀) vs within-operator 실행오류(연산 썼는데 실패)**:
  - ⋈: **gold∈support 0/29**(E-REGIME) → 재샘플로 못 고침 = 연산-choice(ASK/verify 필요). ✅
  - compute: **재샘플=gold 도달(실행)? vs verify-recompute만(연산-choice)?** = 트랙1 compute 프로브. ⏳
  - reach: 69% enumerable/queryable=FIND 안 씀(under-action). ✅ 부분.
- **반증 hunt(TODO)**: agent가 *올바른 연산을 쓰고도* 실패한 케이스(within-operator)를 수동감사로 탐색 → 그 비율이 통합의 진짜 반증. (예: ASK했는데 user-답으로도 실패·COMPUTE 맞는데 정책 애매.)
