# C49 — ASK는 금지가 아니라 최후위 위계여야 한다 (2026-07-09)

> 상위 = `RESEARCH_MASTER.md`. 사용자 지적: *"ASK 금지는 실제 ASK가 필요한 도메인/벤치에서 문제가 되지 않나? ASK는 최후로."*
> **맞다.** C44의 검증기(`producer 존재 → ASK 금지`)는 **too strong**이었다. 아래가 교정.
> 재현 = `scripts/distill/tau2/` 본 doc §4 · 대상 retail `fl32b_floor` + airline `c4ff_airline_base`.

---

## 0. 한 줄
**"producer 도구가 존재하는가"가 아니라 "producer를 *지금 호출할 수 있는가*"로 위계를 세운다.**
FIND → GET(호출가능시) → INFER → **ASK(최후)**. ASK를 막는 게 아니라 그 위 세 개를 먼저 시도하게 한다.

## 1. C44 검증기의 결함 (자기교정 #19)
- **ASK 금지 규칙 = `producer가 존재하면 ASK 금지`.** retail서 over-block 0/2650으로 측정됐으나
  이는 **retail의 gold=ASK가 0이기 때문**(벤치 특성)이지 규칙이 옳아서가 아니다.
- **producer 존재 ≠ 호출 가능.** `get_order_details(order_id)`가 있어도 `order_id`를 모르면 못 부른다.
  그 지점에서 GET을 강제하면 에이전트는 **없는 키로 조회를 시도 → 실패 → loop**(= `too_many_errors`).

## 2. ★실측 — R0가 막다른 길로 모는 지점 (전수)
"producer는 있으나 그 producer의 **입력 키가 문맥에 없어 호출 불가**"인 非-FIND 결정점:

| 도메인 | GET 호출가능 | **R0 막다른 길**(호출불가인데 R0는 GET강제) |
|---|---|---|
| **retail** (fl32b_floor) | 73 | **7** (payment_method_id 6 · address1 1) |
| **airline** (c4ff_base) | 4 | 1 |

⇒ retail서도 **7건**이 R0로는 잘못 판정된다. 우연히 gold=ASK와 겹치지 않아 over-block 0이 나왔을 뿐,
**규칙 자체가 호출가능성을 무시**한다. airline·bank·telecom·ToolDial류 clarification 벤치에선 직접 피해.

## 3. 교정된 위계 (호출가능성 기반·도메인 일반)

```
인자 x 의 출처 결정 (순서대로·먼저 되는 데서 멈춤):
  1. FIND  : norm(gold_x) ∈ {사용자 발화 ∪ 도구 출력}                → 복사
  2. GET   : ∃ producer(x) 도구 P  ∧  P의 필수 입력 키가 지금 문맥에 있음   → P 호출
             (★"P가 존재"가 아니라 "P를 지금 호출 가능"이 조건)
  3. INFER : x 가 문맥의 값들로 유도 가능(argmin·filter·계산)          → 형식화→결정론 실행
  4. ASK   : 위 셋 다 불가                                           → 되묻기 (최후)
```

### 3.1 검증기(전부 decidable · DB 내용 주입 0)
| 갈래 선언 | 통과 조건 | 위반 시 |
|---|---|---|
| FIND | 값이 {발화∪출력}에 실재 | "문맥에 없다 → 다시" |
| GET | 지목 도구가 producer(x) **∧ 그 도구 입력키가 문맥에 있음** | "그 도구 지금 호출 불가/필드 불일치 → 다시" |
| INFER | 기준이 실행 가능 | — |
| **ASK** | **더 높은 갈래(FIND/GET-호출가능/INFER)가 전부 불가** | **"P를 지금 호출 가능하다 → 먼저 GET해라"** |

**핵심 전환**: ASK 검증기가 *"producer 있으면 금지"*(R0) → *"더 높은 갈래가 지금 가능하면 그걸 먼저"*(R1).
producer가 있어도 **입력 키가 없으면 GET-불가 → ASK 정당**. 이것이 당신이 요구한 "ASK는 최후".

### 3.2 ASK 없이 여는 다른 방법 (ASK 전에 낄 갈래)
사용자에게 되묻기 전에 결정론이 시도할 수 있는 것들 — ASK를 진짜 최후로 밀어낸다:
- **GET-chain**: 입력 키가 없으면 *그 키의* producer를 먼저 GET (2-hop·P2b). 예: order_id 없음 → `get_user_details`로 주문목록 → order_id 획득 → `get_order_details`.
  ⇒ 많은 "호출 불가"가 실은 **한 단계 더 조회하면 호출 가능**. R1은 이걸 재귀로 판정.
- **DISAMBIGUATE(열거)**: 후보가 여럿이면(⋈) 사용자에게 값을 묻지 말고 **후보를 제시해 고르게** — 값 발명도 ASK도 아님.
  (C46의 FIND-wrong = 후보 2+개 지점 = 여기서 처리. 단 이건 present와 달리 **이미 조회된 것**만 열거.)

⇒ ASK 도달 조건 = **FIND 실패 ∧ GET-chain 막힘(입력키의 producer도 없음) ∧ INFER 불가 ∧ 후보 0개**.
이 정의에서 tau2 retail ASK=0은 정합(전부 GET-chain으로 도달 가능), airline PREFERENCE 17건은 진짜 ASK(cabin·baggage=사용자 선호·producer 없음).

## 3.3 ★실측 (C50·retail 전수 + airline) — R0와 R1은 이 두 벤치서 거의 같은 답

**왜 R0가 우연히 안전한가**: 조회 가능한 인자(id류)는 producer가 있고, 물어야 하는 인자(선호류)는 producer가 없다.
R0의 "producer 존재" 조건이 이 경계와 **우연히 일치**한다.

| R0가 정당한 ASK를 막는가 | 실측 |
|---|---|
| retail 32B | **0** (auth 늘 선행 · gold=ASK 0) |
| retail 14B | **7** (전부 t48 한 task · auth 실패 후 write 시도 = producer 있으나 호출 불가) |
| airline | **0** (ASK 인자 cabin·flight_type·baggage = **producer 없음** → R0 발동 안 함) |

airline write 인자 gold 거처: FIND 236 · NO-WRITE 191 · **ASK(preference·조회불가) 11** · GET 1.
**R1이 ASK로 판정한 11건 중 R0가 막는 것 0.** ⇒ 원리 결함은 실재하나 **이 두 벤치서 빈도 낮음**(14B t48류).
**진짜 차이는 clarification-heavy 벤치(ToolDial·τ²-airline 확장)서만 드러난다.**

## 3.4 ★더 큰 발견 — DISAMBIGUATE 모집단이 retail write 인자의 45.8%
후보(같은 형식 값)가 2+개인 지점 = **retail 32B write 인자의 45.8%**(전부 gold=FIND).
= C46 FIND-wrong(⋈)이 사는 곳 = **날조 닫은 뒤 남는 유일 잔여.**
- 후보 1개 → FIND(안전) · **후보 2+개 → 그냥 FIND하면 ⋈ 오선택** = F3 semantic 경계(C3b).
- ⇒ 위계의 실질 부담은 ASK가 아니라 **DISAMBIGUATE**에 있다. 그러나 이건 "값 발명"이 아니라 "옳은 값 *선택*"이고
  present/autofetch 없이 **이미 조회된 후보 열거**로만 처리(§3.2). **레버가 아니라 경계**일 가능성(C46).

## 4. 남은 검정 (무료·다음)
1. **R0 vs R1 짝비교**: retail 7건 + airline 막다른길에서 R1이 ASK/GET-chain을 옳게 고르는가. (32B·단일턴)
2. **airline PREFERENCE 17건**: R1이 정당한 ASK를 **살리는가**(R0는 producer 없어 어차피 통과 → 여기선 R0=R1, 무해).
   진짜 갈림은 §2의 "호출불가" 지점.
3. **GET-chain 재귀 판정기** 구현: "입력 키의 producer가 존재하고 호출가능한가"를 깊이 2까지.
4. **다중턴 e2e**: ASK 선택 후 사용자 답 → 그 값으로 올바른 write 하는가.

## 5. 원장 영향
- **C44/C45 정정**: "over-block 0/2650"은 유효하나 **retail-특수**(gold ASK=0). 일반 주장은 **R1(호출가능 위계)** 로 대체.
- **C46 연결**: FIND-wrong(⋈)의 처방 = §3.2 DISAMBIGUATE(후보 열거). ASK도 발명도 아닌 넷째 길.
- ASK = **최후위**. 금지가 아니라 위계. clarification 벤치(ToolDial·τ²-airline)서 이 위계가 GO 조건.
