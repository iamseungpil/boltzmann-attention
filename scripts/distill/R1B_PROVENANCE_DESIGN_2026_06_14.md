# 설계서: R1b 값-provenance 집행 (XGrammar 원천차단 + 검증기 + 학습된 복구순서) — 2026-06-14, 리뷰용

> 상위 = `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md` §2c · 규칙 = `TASKBENCH_EXPERIMENT_RESULTS.md` §10.5 R1b · 불변 = `feedback-selector-verifier-deterministic`. **구현 전 리뷰 대기.**

## 0. 동기 (τ² 실증)
plan-X 학습(SOPBench+TaskBench = 정보 upfront·ask-user 무)의 7B TBox가 τ²서 **인자값을 날조**(`jane.doe@example.com`·`#W00000000`) → 인증실패 → compliant-pass **0.10(50-up)→0.05(250-up) < base 0.17**(단조하락 = 파국적 망각). base-Instruct는 ask-user 156/160·날조0. **R1-도구이름 grounding은 전이 작동.** ⇒ 결손은 **인자-값 provenance** 한 축. 이 설계는 그 축을 *학습 + 결정론 보장* 둘 다로 닫는다.

## 1. 규칙 R1b (TB §10.5 신설)
> **모든 인자 *값*은 출처가 있어야 한다 — (a)사용자 발화 또는 (b)도구 출력. 필요한데 부재 시 *획득*(read-tool 호출 / ask-user) 후 사용. 자가생성(날조) 금지.**

- R1a(닫힌 심볼·enum) = grammar mask로 이미 집행. R1b = **열린 값**(id·email·name·금액)으로 확장.
- 두 합법 출처: **(b) tool-fetch**(DB-파생값 — 읽기도구 출력) / **(a) ask-user**(사용자만 아는 값).

## 2. ★3-레이어 아키텍처 (직교 분업)
| 레이어 | 역할 | 종류 | 보장/학습 |
|---|---|---|---|
| **L1 XGrammar 디코딩-마스크** | 인자값을 *컨텍스트-등장 후보*로 제약 → 못 지어냄 | 결정론·하드 | **날조 구조적 0**(원천차단) |
| **L2 provenance 검증기** | 값이 user/tool 출처에 없으면 reject·플래그 | 결정론·검출 | 잔여 날조 포착 + 학습 보상신호 |
| **L3 학습된 복구순서** | 값 부재 시 **fetch-우선 → 없으면 ask** | 학습(SFT/RL) | "언제 가져오고/묻나" |

**핵심**: L1이 "절대 못 지어내게", L2가 "실패 검출", L3가 "그럼 어떻게 복구"를 담당. L1이 호출을 *차단*하면 모델은 날조 못 하니 **L3(획득)로 강제**된다 — 제약이 복구를 *유도*.

## 3. 컴포넌트 상세
### 3a. L3 — 학습된 복구순서 (fetch-우선 → ask)
- 모델은 **tools= 카탈로그(A1)를 보고 분기**: 그 값을 주는 읽기도구가 *있으면* 호출(tool-fetch=R2, SOPBench gather 궤적에 이미 존재) / *없으면* ask-user.
- **SFT 데이터**:
  - tool-fetch-then-use = 기존 gather 궤적(R2).
  - **ask-user-then-use = augmentation**(`fc_askuser_augment.py`, 정보-upfront → ask-then-provide; creds=첫 tool-call 인자서 결정론 추출; user-only 키[username·id·email…]만 물음 → DB-파생값은 fetch 유지). v3서 검증 중.
  - **대조 케이스**(같은 값이 fetch가능 vs user-only)를 섞어 "카탈로그 보고 분기" 날카롭게.
- **RL/DPO(후속)**: L2 검증기 신호로 (성공복구, 날조시도) 쌍 → 복구순서 강화.

### 3b. L2 — provenance 검증기 (결정론)
- 각 tool-call 인자값 v에 대해: v가 (이전 user 메시지 ∪ 이전 tool 출력)의 부분문자열/정규화-매치인가? 아니면 **fabricated → reject**.
- 게이트(R3)와 동형 인터셉트: 날조 호출 deny + 복구 메시지("그 값을 먼저 획득하라"). day-6 A2 faithfulness 게이트와 직교·가산.
- **쉬운 첫걸음**(L1보다 단순) + L3 학습 보상신호 제공.

### 3c. L1 — XGrammar 컨텍스트-제약 디코딩 (원천차단)
- per-request 동적 제약: 인자값 생성 시 **컨텍스트서 추출한 후보값 집합**으로 `guided_choice`/동적 문법 마스크. 후보 비면 그 호출 불가 → L3 강제.
- 후보 추출 = 컨텍스트의 id/email/숫자/인용 span(정규식·NER-lite).
- ⚠️ vLLM 기본 flag 아님 = **커스텀 엔지니어링**(per-request 문법 빌드). 가장 하드.

## 4. 단계 (staging — 쉬운 것부터)
1. **L3 SFT (지금·v3)**: ask-user augmentation → 복구순서 학습. **판정 = τ² compliant-pass 0.10/0.05 → base 0.17 회복?** 회복 시 L3 작동 확정.
2. **L2 provenance 검증기**: 결정론 값-출처 검사 + 복구 deny 메시지. compliant-pass에 "날조-위반 0" 축 추가. RL 보상신호.
3. **L1 XGrammar 원천차단**: 컨텍스트-제약 디코딩. 날조 구조적 0 보장.

## 5. 정직한 한계
1. **정규화 필요 값**: user "September 29th" → tool "2026-09-29" = 부분문자열-복사 깨짐 → L1 완화문법 또는 결정론 정규화기. L2도 정규화-매치 필요.
2. **validity ≠ correctness**: L1/L2는 *날조*는 막아도 *옳은 span 선택*은 보장 안 함(틀린 id 복사 가능) — 올바른 선택은 R4(의미매칭)·gather 품질.
3. **tool-output 값은 *호출 후*에만 후보** → gather 선행(R2) 필수. 순서 의존.
4. **augmentation 품질**: 템플릿 어법 어색("apply credit card") = v1. 부족 시 frontier-rewrite 업그레이드.
5. **catastrophic forgetting 잔여**: ask-user 데이터 비율·일반 instruction 혼합·epoch 통제로 base 대화 보존 — v3 결과로 충분비율 판정.

## 6. 논문 기여 (novelty)
- **no-fabrication = 결정론 보장**(L1 마스크 = 위반 구조적 0·감사가능) + **복구순서 학습**(L3). AgentSpec/guardrail은 런타임 *규칙*은 하나 **인자-provenance 결정론 보장 + fetch/ask 복구학습**을 묶은 칸 비어있음(FIELD_GAP §5.5 위 추가).
- compliant-pass에 "값-provenance 위반 0" 축 = 새 결정론 compliance 속성.
- R1a(닫힌 심볼 mask) → R1b(열린 값 provenance) = grounding 규율의 일반화.

## 7. 검증 (eval)
- **L3(SFT)**: τ² compliant-pass(회복?) + 날조-호출 비율(궤적 census, base 0 ↔ fctbox 다수 → v3서 감소?).
- **L2**: provenance-위반 검출율(주입 날조 포착) + 복구추종율.
- **L1**: 마스크 후 날조-호출 = 구조적 0 실측 + 정규화-값 false-block율.
- 전이: SOP-Bench·τ² held-out 동일 측정(벤치-횡단 R1b 전이).

## 8. 미해결 결정 (리뷰 포인트)
- D1: L2를 게이트(R3)에 통합 vs 별도 검증기?
- D2: L1 후보추출 = 정규식 span vs 타입별(email/id/금액) 추출기?
- D3: augmentation frac(현 0.4)·일반 instruction 혼합 비율 = v3 결과 후 튜닝.
- D4: 정규화 값(날짜·금액) 처리 = L1/L2 정규화기 범위.
- D5: 대조 케이스(fetch vs ask 분기) 합성을 augmentation에 추가할지.
