# 일반화 Scaffold 아키텍처 — LOCK (2026-07-12)

> ★★이 문서 = 2026-07-12 설계토론의 박제·LOCK. **앞으로 retail-B/scaffold/상용/특허 작업은 전부 여기서 파생.** 표류 시 즉시 복귀.
> 상위 = `RESEARCH_MASTER §3`(C73) · 파생 = `TRIVIAL_REGRESSION_ABLATION`·`SCAFFOLD_STATE_ROUTER`·`INTERVENTION_LEVER_CONDITIONALIZATION`.
> 불변: [[05]] 엔진 도메인일반·A2만 · [[10]] 선택기=결정론·생성기=LLM · [[11]] learn=도메인일반·A2-swap 전이 · Δspurious≤0(모트) · gold-independence · [[09]] 무료우선.

## 0. 한 문장 원리 (LOCK)
**도메인-일반 검증 엔진(가드) + 거의-무료 A2(도구 스키마) + 일반 GET→FIND→INFER→ASK 루프 + learn(의미). 도메인 차이는 A2로만. 개입레버의 "엔진이 의도 추측"은 전부 폐기 — 대신 (구조=A2/엔진) + (의미=learn/ASK)로 분해.**

## 1. 가드 vs 개입 — 확정된 선 (C73 실증)
- **가드**(gate/prov/calc/coverage) = "정책·데이터·산술 확인" → **답이 spec 안에** → 도메인일반·Δspurious=0·상용 엔진 코어.
- **개입**(present/disamb/ground/principle/eplan as-override) = "의도 추측" → **답이 대화-순간 의도에** → 엔진 결정 시 트릭(전이깨짐) or 부작용(C73 trivial 회귀 실증). **as-override 폐기.**
- 실증(C73·task106 양방향 절단): full(개입 always-override)이 COMP-robust 6 trivial 회귀·advise가 5/6 회복(부작용원=override 기전)·106=intent 잔여.

## 2. 5개 개입레버의 일반화 분해 (LOCK·설계토론 확정)
각 레버 = **구조(A2/엔진 가드) + 의미(learn/ASK)**, **트릭 휴리스틱은 폐기**:

| 레버 | 구조 (A2/엔진·도메인일반) | 의미 (learn/ASK) | 폐기 트릭 |
|---|---|---|---|
| **disamb** | operand valid-set 탐지(스키마) + **GET→FIND→INFER→ASK 루프** | INFER 정확도·INFER→ASK 보정 | 엔진이 골라줌(most-recent/first) |
| **ground** | **미grounded 거부**(write값 ∈ tool출력∪사용자발화·보편규칙) | — (순수 가드) | 엔진이 교정/치환 |
| **principle_default** | 정책-강제 채움(=gate) | ASK(사용자 선택) | 통계 default(C58 트릭) |
| **eplan** | A2 scope 커버리지(요청타입→관련집합) | 요청-관련성 이해 | 무조건 검토(§2d 기각) |
| **present** | (스키마와 중복·불필요) | — | 후보 떠먹임(spoon-feed·C59) |

## 3. operator/operand A2 스키마 = 거의 무료 (LOCK)
모든 tool-use는 도구를 호출하려면 **이미 스키마를 가짐**(function schema/JSON Schema/OpenAPI/MCP):
| A2 조각 | 출처 | 비용 |
|---|---|---|
| operator/operand (validity·type·enum) | **도구 정의**(이미 있음) | **0** |
| provenance (미grounded 거부) | **보편 규칙**(operand-무관) | **0** |
| getter-availability 플래그 | 스키마 파생(그 필드 반환 getter 있으면) | ~0 |
| gate/정책 preconditions | **기존 정책문서** 추출 | 추출 싸고·**검증 1회/도메인** |
| coverage scope | 요청타입→scope | 소액 |
- 실증: banking A2-swap = "GB1 게이트 몇 개 + applies_when"·**엔진 리터럴 0**([[05]]). A2가 작음.
- ⇒ **가드 절반은 비용 0**·유일 실비용 = 정책-게이트 1회 검증.

## 4. GET→FIND→INFER→ASK 루프 = 출처-해소 엔진 (LOCK·기구현 C44-C48/C67)
operand 값의 출처는 고정 아님(사용자/getter/추론). A2는 **getter-available 플래그**만, 일반 루프가 해소:
```
1. GET   : A2에 getter-available → getter 먼저 (결정론)
2. FIND  : 없으면 → 사용자 발화 검색 (체크가능)
3. INFER : 없으면 → 문맥 추론 (semantic·모델)
4. ASK   : 그래도 없으면 → 질의 (일반 채널)
```
- **기구현·격리 실증**: C44-C48(4도메인·over-block 67%→0%·spurious 0)·C67(loop 0.77·**base ASK남발 21→loop ASK 0**=과질문 자동제거).
- 효과: GET/FIND=결정론 해소 → **semantic 부담=INFER 하나로 축소**·실패=ASK로 경계(날조 대신).
- task106 렌즈: GET(변형집합)→FIND("black"·사이즈X)→**INFER**("smaller"→XL)→ASK. 실패=INFER confident-wrong(black-S)+ASK 미발동.

## 5. Proven / Open 경계 (LOCK·정직)
| 조각 | 상태 |
|---|---|
| 가드(gate/prov/calc) 도메인일반 | **실증**(banking A2-swap·엔진리터럴0) |
| operand/operand A2 스키마 거의무료 | **실증**(도구정의=기존) |
| GET/FIND/ASK 루프 구조 | **격리 실증**(C67·C44-48) |
| **in-vivo 멀티턴 e2e 루프** | **미확정**(C-stage 판정) |
| **INFER 정확도 + INFER→ASK 보정** | **learn/scale·미확립**(C38/C42 SFT 퇴화·confident-wrong이 ASK 안 함) = **유일 잔여·make-or-break** |
| valid-but-wrong intent 잔여 | learn/ASK/fleet(voting=settled-neg·fleet≠voting) |

## 6. ★실험 계획 — B 78+36 (일반화 스택·다음)
> 목표: 일반화 원칙 하에서 **(a) 78 개선폭 (b) 36 무회귀(COMP-only·S0) (c) 잔여=INFER→ASK semantic만인지** 확인.
- **스택 정의 "COMP+D-v2(generalized)"**: COMP(gate+prov+calc) + ground=미grounded거부 + disamb=GET/FIND/INFER/ASK루프 + principle=gate+ASK + eplan=A2-coverage. **present 제거·개입-override 전부 제거.**
- **대조**: COMP(base) vs COMP+D-v2(generalized) vs (참고)full-override(C73 회귀본).
- **태스크셋**: 78-hard + 36-trivial. **nt=1 누적**(§0b·nt4-한방 금지) or nt≥2 rate(비결정성 커버·[[09]] 승인).
- **판독**:
  - 36: 일반화 스택서 회귀 0(=COMP·S0 무발화)이어야 함 — 개입-override 뺐으니 C73 회귀 사라질 것 예측.
  - 78: GET/FIND/coverage로 결정가능분 개선·잔여 = INFER confident-wrong(intent).
  - 잔여 per-case = **INFER→ASK 미보정**만 남는지 = learn/scale 타깃 크기(P3).
- **인프라 주의**: full-override는 subcall+gate로 sim 20-58분 pathology(C73 §8b) → 일반화 스택은 루프가 가벼워야·per-sim 타임아웃 가드·concurrency 제한.

## 7. ★Anti-drift 규칙 (LOCK·위반 시 즉시 중단)
1. **개입레버를 as-override로 재도입 금지** — 반드시 (A2-가드) + (learn/ASK)로만.
2. **도메인-특화 휴리스틱 hand-code 금지**(most-recent·first·통계 default·무조건 열거) = 트릭.
3. **present(후보 떠먹임) 재도입 금지** — 스키마와 중복·spoon-feed.
4. **"default" 개념 금지** — 정책-강제(gate) or ASK로만.
5. **엔진에 도메인 리터럴 0** — 도메인은 A2(스키마+정책)로만.
6. **pass 수치를 헤드라인화 금지**(리뷰어 공격면) — 헤드라인=가드-일반화·개입-실패·intent-경계.
7. **잔여의 정답 = ASK/learn** — scaffold로 valid-but-wrong 닫으려 하지 말 것.
8. 실험 전 이 문서 §6 + [[05]]/[[09]] 점검.
