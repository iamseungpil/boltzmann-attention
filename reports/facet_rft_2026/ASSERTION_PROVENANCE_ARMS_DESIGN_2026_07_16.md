# 주장-출처(assertion-provenance) 확장 — 3-arm 설계 (2026-07-16)

> 파생: `RESEARCH_MASTER` §1.3(제1원리·상쇄) · §1.5(Q1/Q5) · 원장 C45/C46/C48(출처선언 레버) · C30/[[42]](prompt-only 무효)
> 규율: [[03b]] 구현속임 금지 · [[05]] 엔진 리터럴 0·A2만 변경 · [[10]] LLM=formalize/엔진=검증 · [[08]] 궤적정독 · [[09]] 무료 先

## 0. 한 줄
C45 출처선언 레버는 **write 인자**에만 걸려 있다. banking 잔여 병목은 write가 아니라
**사용자에게 하는 사실 주장(assertion)** 에서 샌다 — 에이전트가 producer 도구를 안 부르고 "리워드 맞다"고 판단한다.
확장 후보 3안을 정의하고, **어느 것도 공짜가 아니므로**(§1.3) 상쇄를 같이 잰다.

## 1. 포렌식 근거 (t019g 3 sim 전수·[[08]])
정본 궤적: `sim_results/bank_t019g_20260716.results.json.gz` (커밋 **`48785fcc`**·이번 세션 영속화.
선행 `6e906e77`은 compliance.json을 잘못 담아 `48785fcc`가 교체). 대조: `bank_t019d_20260716.results.json.gz`.
종료 3/3 `user_stop`(crash/infra 0).

| sim | 도구호출 | `get_reward_discrepancies` | 주장 지점 | 주장자 |
|---|---|---|---|---|
| 0 | 8 | **0** | msg[16][20][24][28] "rewards ... align with the expected rates" | 에이전트 |
| 1 | 11 | **0** | msg[38] "rewards earned seem to be calculated correctly" | 에이전트 |
| 2 | 9 | **0** | msg[31] user-sim이 먼저 결론 → msg[32] 에이전트 **추인** | user-sim→에이전트 |

- **★sim 0 msg[26]**: 사용자가 밀어붙이자 "다시 확인하겠다"고 선언 → **도구 호출 0** → msg[28]서 같은 눈대중 반복.
  ⇒ soft 압박(사용자 발화)으로는 producer 호출이 유도되지 않는다([[42]] 동형).
- **★sim 2**: 실패 형태가 다르다 — *에이전트가 판단*이 아니라 *user-sim의 틀린 판단을 추인*. arm 설계는 둘 다 덮어야.

### 1b. ★기존 레버(NLNUM)는 이 병목에 **오조준** (오프라인 실측·무료)
`T2_NLNUM_PROV`(t2_gate_patch.py:1014·2566)의 검증기(`_MONEY_RE`/`_unverified_amounts`)를 두 arm 궤적에 재현 실행:

| arm | sim | 발화 고유금액 | 맞는 산술 | 근거불명(참양성) |
|---|---|---|---|---|
| t019g | 0·1·2 | 2·2·2 | **2·2·2** | **0·0·0** |
| t019d | 0·1·2 | 0·0·2 | 0·0·**2** | **0·0·0** |

- **발화 8/8 전부 *맞는 산술***: KB `doc_credit_cards_credit_cards_(general)_006` 원문 = *"1 point = $0.01 ... 250 points equals $2.50"*.
  7021 points → `$70.21`, 4839 → `$48.39` (대응 points 값이 전부 문맥에 실재). ⇒ **참양성 0 · 전량 over-block.**
- t019d sim0/1 발화 0 = 카드 조회 도달 실패(1/3 도달)와 정합 — 교차 일관.
- **정작 진짜 실패는 못 잡는다**: "align with the expected rates" 문장엔 날조 숫자 0 (points·amount 전부 도구 출력 원문 복사).
- ⇒ **값-출처(value-provenance)와 판단-출처(assertion-provenance)는 다른 술어다.** C46(날조 vs ⋈ 분리)의 동형 정련.
- 현재 banking A2에 `calc_tool` 키 부재 = NLNUM **비활성**. (=이번 t019g에 실제로 걸리지 않았음. 위는 *반사실* 재현.)

## 2. 3-arm 정의 (공통 불변량: 엔진은 어시스턴트 **텍스트를 파싱하지 않는다**)
> 텍스트 정규식으로 "판단"을 탐지하면 = 엔진-formalize + 도메인 리터럴 = **[[03b]]/[[05]] 위반 = 실험무효.**
> 이번 세션 위반 3회가 전부 이 지점. 후크는 **구조 이벤트** 또는 **LLM 자기선언**만 본다.

### arm `discovery-required` (도구-이벤트 후크)
- **A2**(ABox·도메인사실): `analysis_producers: [{data_source: <tool>, producer: <tool>, subject: <str>}]`.
- **엔진**: 사임 시점(=tool_calls 없는 어시스턴트 발화 = 턴을 사용자에게 넘김)에
  `data_source 호출됨 ∧ producer 미호출` → regen 1회(상한 1/sim). **엔진이 보는 것 = 호출 이름 집합뿐.**
- 근거: 등대 §1.5 Q5 *"읽기만 강제, 쓰기는 절대 금지"* · [[14]] read(discovery)만 강제. producer=GET(읽기).
- **예상 역효과**: 리워드와 무관한 대화(단순 거래 나열 요청)서도 발화 → over-block. **계측 필수.**

### arm `self-declaration` (C45 4지선다의 assertion 확장)
- **A2**: `assertion_operands: {<operand>: <producer tool|null>}` (예: `expected_reward → get_reward_discrepancies`).
- **엔진**: 사임 시점에 sub-call 1회 — *"직전 답변이 주장하는 operand와 각각의 출처를 {GET·FIND·INFER·ASK}로 선언(JSON)"*.
  선언 **구조체**를 결정론 검증: `source=INFER ∧ A2에 호출가능 producer 존재` → 무효 → 피드백+regen.
- [[10]] 정석: LLM=formalize(자기 주장의 operand 추출)·엔진=검증만. 엔진은 JSON 필드만 읽는다.
- **예상 역효과**: 매 사임 turn마다 sub-call = 전-궤적 비용(§1.3 thinking과 동형: F5 매도 위험) · 에이전트가 선언 회피/거짓선언 가능.

### arm `value-provenance` (기존 NLNUM·대조군)
- banking A2에 `calc_tool` 추가만(코드 0줄). §1b대로 **오조준 예측** — 그 예측 자체를 실측(제1원리 상쇄 관측).

## 3. 자기감사 ([[03b]]·제시 前 필수)
| 점검 | 결과 |
|---|---|
| 엔진에 도메인 리터럴? | **0** — 카드명·rate·"reward" 문자열 전부 A2. 엔진은 `{tool 이름 집합}`·`{JSON 필드}`만 |
| 엔진이 formalize? | **아니오** — arm1=이벤트·arm2=LLM 선언·arm3=기존(값 정규식은 *검증*이지 operand 공급 아님) |
| hack-pass? | 없음 — gold 4건을 엔진이 알지 못함. producer op-DAG는 A2 |
| 강제(forcing)? | 없음 — 전부 regen 피드백. `tool_choice=required` **미사용**(지난 세션 폐기 유지). write 강제 0 |
| 도메인-타깃 학습? | 없음(학습 0) |
| **잔존 위험** | **soft regen thrash**(핸드오프 §4 실증: deny 시 32B=동일반복·표면변형·옆-날조). arm1/arm2 둘 다 soft → 무효 가능 |

## 4. 계측 (등대 §1.3: 게이트 자신의 역효과를 반드시 같이 잰다)
1. **발화율**: sim당 fire 수·발화 지점 (stderr 태그).
2. **★over-block**: 발화가 *맞는 주장*에 걸린 건수 (NLNUM $70.21형). 궤적 정독으로 판정.
3. **★Δspurious ≤ 0**: regen이 새 날조/over-action을 유발했는가 (C45 GO 조건).
4. **thrash**: regen 후 producer 실제 호출 여부 vs 동일-주장 재발화.
5. pass / 종료사유 분포.
6. **미발화도 결과다**([[30]] 교훈: calc 31/342 미발화를 천장으로 오인).

## 5. 실행 순서 ([[09]])
1. 오프라인: t019g 궤적 replay로 arm1/arm2 발화 지점·over-block 예측 (**무료**).
2. 구현 → 오프라인 단위검증 → **스모크**(gpt-4.1-mini·nt=1~3) = 라이브 발화 확인 (단위통과 ≠ 라이브발화·[[30]]).
3. 정본: gpt-5.2 user-sim·nt≥3 — **사용자 승인 필수**([[09]]).

## 6. Caveat (정직)
- t019g = **n=3 × gpt-4.1-mini** = robust 측정 아님·**메커니즘 관측**. reward도구 0선택만 3/3 일관 + 원문 정독으로 견고.
- **sim 2형(user-sim이 틀린 결론을 먼저 줌)** 은 user-sim 품질 의존 — gpt-5.2선 다르게 나올 수 있음([[47]] 권장표준).
- arm1/arm2 모두 **soft** → 지난 세션 thrash 실증에 비춰 무효 가능. 무효면 그 자체가 [[13]] 경계 증거(scaffold 상한).
