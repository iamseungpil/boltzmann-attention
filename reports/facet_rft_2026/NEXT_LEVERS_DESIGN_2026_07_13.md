# 차기 레버 설계 — L4C(변형-선택 재설계)·L7(precondition)·L11(enum) (2026-07-13)

> v6(READALL+COV+L10) 이후 잔여 병목의 사전 설계. 근거 = `A1_V2_NT2_FORENSIC`·`A1_V3_PROBE_FORENSIC`·v4/v2 비교 3에이전트(62태스크 per-step).
> 우선순위: **D=L4C(FF 2+flip 8-9·최대)** → F=L7(FF 3+flip 3) → G=L11(FF 1+flip 1).
> 불변: [[05]] 도메인일반·A2만 · [[10]] formalize=LLM·평가=결정론 · Δspurious≤0 · G-approve(F4) 상시 · **동률=enumerate-ASK가 정도(t64 [M])**.

## 1. L4C — 변형-선택 constrained 재설계 (D 클래스)

### 1a. 실증 케이스 → 요구 연산 (전부 궤적 [M])
| task | 요청 형태 | 필요한 연산 | 현 L4a가 못 하는 이유 |
|---|---|---|---|
| t20 | "most expensive, **size stays the same**" | keep-제약 ∧ 극값 | 제약 절단(F2)→전역 argmax |
| t7 | "**less** bright" (현재=medium) | **비교급**: 순서축에서 anchor 기준 방향 이동 | 비교급 연산 부재 |
| t29 | "**shorter**" 스케이트보드 | 비교급(수치축) | 〃 |
| t106 | "one size **smaller**" (XXL→XL) | 순서 enum 산술(−1) | 순서 자체를 모름 |
| t52 | "max **zoom**" | 비-price 필드 극값 ∧ available | price-전용 사전 |
| t100 | 34"/custom 지정·재질 미지정 | **미지정 축 = anchor 승계**(bamboo) | 승계 규칙 부재→first-fit |
| t8 | "battery > USB > AC" 선호 사다리 | **순서 폴백**(가용성 평가) | 사다리 연산 부재 |
| t58 | "8bar, 없으면 9bar"+"capacity/type 유지"+"cheapest **i7 이상**" | 폴백∧keep∧(극값+속성하한)·**per-slot 분리** | F1 교차누출+F2 절단 |
| t23·46 | 발화 속성("robotic") = 후보 축 값 | 속성 eq(발화→옵션값) | L4b 보수화로 무력 |
| t64 | 복합 만족 2후보 동률 | **enumerate-ASK 외부화** | (tie-override는 근거 부족 [M]) |

### 1b. 기준 대수 (criterion algebra) — formalize 출력 스키마
```
slot_criterion := {
  anchor: <원품목 item_id>,                       # per-slot(F1): 이 슬롯의 기준만
  keep:   [axis...],                              # "same X"류 명시 keep
  set:    {axis: value|numeric},                  # 명시 변경("9 bar"·"green")
  move:   {axis, dir(-|+), steps?},               # 비교급("less bright"→(brightness,-))
  prefer: [ {set|move}... ],                      # 폴백 사다리(순서 평가)
  extremum: {field, dir} | none,                  # "cheapest"·"max zoom"
  floors: [{field, op(>=|<=), value}],            # "i7 이상"·"가격≤지불액"
}
```
- **DEFAULT-KEEP(핵심 신규 원리)**: keep∪set∪move∪extremum이 언급하지 *않은* 축은 **anchor 값 승계** = "최소 변경 원칙"(도메인일반). t100 즉시 폐쇄·t20/t7 후보공간 축소. 완화는 §1d.
- 순서축의 순서 출처: ① 수치 파싱(bar·inch·L·GB·piece = 단위 정규식·도메인일반) ② 비수치 enum(S<M<L<XL<XXL 등)은 **A2 MENU `ordinal_orders`** 선언(엔진=인터프리터·[[05]]). 미선언 비수치 축의 move → ASK 낙하.

### 1c. per-slot 결속 (F1 해법)
- 극값어/제약구의 슬롯 귀속 = **문장-스코프**: anchor 제품명 토큰(anchor record의 name·getter 출력서)과 같은 문장에 있는 기준만 그 슬롯에 적용. 무-귀속 기준 = 어느 슬롯에도 적용 안 함(보수).
- t58 재현: "cheapest i7"은 laptop 문장 → espresso 슬롯엔 keep(capacity/type)+set(9bar)만. 교차누출 구조적 차단.

### 1d. 결정 절차 (엔진·결정론)
```
candidates = variants(product(anchor)) ∧ available
C1 = candidates ∧ keep(명시) ∧ DEFAULT-KEEP(미언급 축) ∧ set ∧ floors
prefer 있으면: 사다리 순서대로 첫 비공집합
|C1|=1 → 후보 확정
|C1|≥2 → extremum 있으면 argopt → 유일해면 확정 / 그래도 동률·extremum 없음 → enumerate-ASK(메뉴·t64 정도)
|C1|=0 → DEFAULT-KEEP 축을 하나씩 완화(요청-언급 역순) 재평가 → 그래도 0 → 재형식화 1회 → ASK
확정 후보 ≠ cur:
   G-approve(cur가 대화-노출)면 no-op(관측만)
   attested(기준이 user 발화에 실재·per-slot)가 아니면 no-op
   둘 다 통과 → 치환 (T2_L4_MODE=substitute 재개 조건 = 이 절차 전체)
```
- floor-guard 교체: 구 "cur∈극값집합" → **"cur가 constrained predicate(C1) 만족"이면 keep**(정답 미파괴가 올바른 predicate 기준으로).
- formalize 경로: 결정론 추출(keep/비교급/극값/단위 사전) 先 → 잔여만 LLM formalize(스키마 §1b·retry 1) → 파싱 실패 = no-op.

### 1e. [[05]] 감사
| 항목 | 판정 |
|---|---|
| 사전(keep-구문·비교급·극값어·단위) | 닫힌 영어 일반어휘·eval-blind 고정(R2 게이트) |
| ordinal_orders | A2 MENU 데이터 |
| DEFAULT-KEEP | 도메인일반 원리(최소 변경)·측정으로 검증(§1f) |
| 치환 | F1+attested+G-approve 삼중 게이트·잔여는 ASK 외부화 |

### 1f. 검증 (무료 先·[[09]])
1. **오프라인 replay [0원]**: 기록 gz의 variant 표 + 요청문으로 §1d 절차를 t7/8/20/29/46/52/58/100/106 전수 재생 — gold 재현율 측정(치환 없이 판정만). **게이트: 9케이스 중 ≥7 gold-유일해 + 오답-유일해 0** (오답 내면 ASK 낙하가 정답).
2. 단위: 케이스별 unit(포렌식 실측값). Δspurious: t0(복합 fallback)·t95(중복 인자) 무해 확인.
3. 표적 probe(승인 후): D-클래스 11 + 가드.

## 2. L7 — precondition·상호배제 (F 클래스)

### 2a. 실증 케이스 → 기전 3종
| task | 실패 | 기전 |
|---|---|---|
| t27 | return 선실행→exchange 영구 잠김 | **mutex 사전검출**: 같은 주문에 delivered-write 의도 2개 → 첫 write 전 ASK("어느 것 먼저?") |
| t99 | bike만 exchange(배칭 위반)→puzzle 소실 + 품목취소 불가 미고지 전체취소 | **mutex-deny 피드백**: 2번째 delivered-write deny + "한 call에 전 품목 묶어라"·"품목 단위 취소 불가" 고지 |
| t57 | "gift card로 환불" 허위 확약 후 취소(gold=무행동) | **notice 게이트(기존 kind)**: cancel에 A2 notice "환불=원결제수단(X)" 강제 고지 → user 철회 유도 |
| t21·t66·t69 | pending에 exchange 고착·교차상품 modify·cancel 미제안 | **deny 피드백에 대안 도구 지목**: A2 `alternative_tool`("pending이면 modify_...") |
- 사전검출(t27)은 의도-쌍 형식화가 필요(LLM) — v1은 **deny-측(2번째 시도 차단+배칭/우선순위 ASK 지시)** 먼저(순결정론·상태 기반). 사전검출은 COV의 M-기계 재사용(intent 2개 감지) 후속.

### 2b. A2 스키마 (기존 gates 온톨로지 內)
```json
{"kind":"preconditions","applies_to":["exchange_delivered...","return_delivered..."],
 "mutex":{"scope":"order_id","policy":"one_write_per_delivered_order",
          "deny_hint":"combine ALL items into ONE call; if both return and exchange are needed, ask the user which to prioritize BEFORE acting"},
 "status_requires":{"exchange_delivered...":"delivered","modify_pending...":"pending"},
 "alternative_tool":{"exchange_delivered...→pending":"modify_pending_order_items"}}
{"kind":"notice","applies_to":["cancel_pending_order"],
 "notice":"Refunds for cancellations go to the ORIGINAL payment method ({payment_from_record})."}
```
- 엔진: GateInterpreter가 preconditions/notice kind 이미 보유(§4b 검증) — **엔진 무수정·A2 데이터만**. mutex 상태(주문별 delivered-write 이력)는 기존 게이트 상태 재구성으로 추적.
- 검증: gate unit(오프라인) → t27/57/99/21/66 표적 probe. 기대: t57(notice)·t99-2차·t21/66(대안 지목) 순결정론 / t27 완전해결은 사전검출(후속) 필요 — deny-측만으로 gold(exchange 우선) 보장 안 됨을 정직 명기.

## 3. L11 — enum-인자 carryover 가드 (G 클래스)

### 3a. 기전 (v3.2 §2b 정련)
- 탐지(순결정론): write 인자 k가 **도구 스키마 enum**(값 후보 유한·A2 불요 — 스키마가 곧 선언) ∧ 값 v가 직전 write의 같은 k와 동일 ∧ **v의 근거 토큰이 이 대상(entity)에 대한 user 발화 창에 부재**(t76: "mistake"류 0건 실증) → carryover 플래그.
- 처방: **개방형 ASK 강등** — "이 주문은 실수 주문인가요, 더 이상 필요 없으신 건가요?"(값을 user가 생산·yes/no 금지 = 확인-세탁 6건 [M] 대응). deny+피드백으로 재생성 1회·cap 1/sim.
- 근거 토큰 사전: enum 값별 파생 토큰(mistake↔accident*/wrong*·no longer needed↔don't need/cancel*…) — 닫힌 영어·eval-blind. 매칭 창 = 해당 entity 언급 ±2 user 발화.
- t38 caveat: 맥락상 양쪽 다 정당한 진성 모호 — ASK가 정답 경로(gold 회복은 user-sim 응답 의존 [?]).

### 3b. [[05]]·검증
- enum 후보 = 도구 스키마(그 자체가 A2급 데이터)·토큰 사전 = 도메인일반 영어. 엔진 리터럴 0.
- unit: t76 실측(msg51-54) 재현 reject·정당 케이스(1번째 cancel "accidentally"→mistake attested) 무발화. probe: 76·38 + 무회귀.

## 4. 구현 순서·기대치
| 순서 | 레버 | 기대(nt4 마진·[D]) | 비용 |
|---|---|---|---|
| 1 | **L4C** | FF 7·100 + flip 8·20·23·29·46·52·58·106·110 안정화 ≈ +2~5 | 중(대수+절차·replay 게이트 필수) |
| 2 | **L7**(deny-측+notice 먼저) | FF 57(·27 부분·99 부분) + 21·66·69 ≈ +1~3 | 소(A2 데이터·엔진 무수정) |
| 3 | **L11** | FF 76 부분 + 38 ≈ +0~1 | 소 |
- 전부 **오프라인 무료 검증 게이트 통과 후** 표적 probe·개별 편입(번들 금지). L4C replay 게이트(§1f-1) 미달 시 치환 재개 포기·enumerate-ASK 전면 낙하로 강등.
