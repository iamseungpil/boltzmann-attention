# 클린 nt=4 실패 전수 포렌식 — symbolic/semantic·극복 분류 (2026-07-07)

> **입력**: replay-safe 게이트 클린 재런 `asmregen{32b,14b}_regen_retail_t4`(nt=4·32768·infra=0·게이트위반0).
> **방법([[08]])**: **≥1 trial 실패한 모든 task**(32B 73·14B 83)를 자동분류(reward_breakdown DB/NL + gold-write
> vs 실제-write의 order/item/value diff + loop/error 신호) → 15 fail-all + spot-check 대표건 **단계별 궤적
> 정독**으로 검증. pass^1 point-estimate 아님·per-case 확정.
> **목적**: 하나라도 실패하는 것의 정확한 원인 + **모든 실패를 극복 레버로 분류**.

---

## 0. 한 줄 결론
클린 nt=4서도 잔여 실패의 **~75%(32B 55/73)가 결정론-decidable**(변형선택·상태추적 coverage·⋈참조·calc·
over-action)이고, **genuine semantic-scale(scaffold로 못 닫는 이해부족)은 명확히 식별되는 것 ≈0**. **32B·14B가
*같은 버킷 구조*로 실패**(14B가 더 자주) → 잔여는 규모가 아니라 **결정론 레버(현 within-order scaffold가 못
덮는 multi/cross-order + 변형-calc)**가 타깃. SEMANTIC_ERROR_FORENSIC(2026-07-06) 결론을 클린 데이터가 재확인.

---

## 1. 전수 분류 (≥1 trial 실패 task·per-task 지배 기전)

| 버킷 | 32B | 14B | 원인(기전) | symbolic/semantic |
|---|---|---|---|---|
| **within-order: 변형/값 선택** | **21** | **26** | 우주문·정확, **new_item 변형 오선택**(bigger/brighter/cheapest/i9/256GB/red 등) | 반-symbolic(calc/filter)+일부 semantic |
| **coverage-missed (상태추적)** | **17** | 16 | "**모든** X" 요청에 일부만 처리(엔티티 인지하고도 미커버) | semantic/state (decidable) |
| **loop/error (orchestration)** | 9 | **13** | 동일호출 반복·tool-error 누적 | orchestration/load |
| **wrong-order (⋈ 참조)** | 7 | 10 | 잘못된 주문에 write(참조해소 실패) | semantic/ref (decidable) |
| **NL: order-total (calc)** | 6 | 3 | 주문 총액 오산 보고 | symbolic/calc |
| **no-write (orchestration)** | 4 | 4 | 실행 아예 안 함(joint-constraint서 멈춤) | orchestration |
| **over-action** | 3+1 | 4+3 | gold=무write인데 실행 / 불필요 추가 write | semantic(feasibility·decidable) |
| **NL: report/communicate** | 4 | 3 | 정보 미전달 | communicate |
| **NL: tracking#(아티팩트)** | 1 | 0 | 벤치 범위밖 | artifact |
| all-pass (실패 0) | 41 | 31 | — | — |

## 2. 32B vs 14B 비교
- **같은 버킷 구조**: 둘 다 within-order(변형선택) > coverage(상태추적) > orchestration > ⋈ 순으로 지배.
- **14B가 전 버킷서 더 많이 실패**: within-order 26>21·orchestration loop 13>9·⋈ 10>7·over-action 7>4·
  all-pass 31<41. = **규모는 실패 *빈도*를 줄이나 *종류*는 안 바꿈**.
- ⇒ **동일 결정론 레버가 두 규모 모두에 적용**(scaffold=레버·scale=빈도). 잔여의 *성질*이 규모-불변.

## 3. 단계별 기전 확정 (궤적 정독)
- **coverage(41)**: "**모든** 주문 주소 수정" → 모델이 "two pending orders" **명시 인지**하고도 #W4082615만
  수정·#W9583042 **누락**. = 전수-열거 부재(genuine 이해부족 아님).
- **⋈(71)**: 주문을 "**DC 주소로 보낸 것**"으로 식별해야 하나 "최근 주문"으로 골라 #W5782623(gold #W5270061).
  user-sim이 오확인해 가려짐. = 참조 키 오선택(present-addressable).
- **cross-order(98)**: bike·puzzle이 **다른 주문**인데 user "같은 주문" 주장 믿고 한 주문에 conflate →
  "item not found" 에러·6회 loop. = item→order 매핑 부재.
- **변형선택(63·8·45)**: 우주문·정확, **new_item 변형만 오선택**(63: 2635 vs gold 3254·8: 7453 vs 8384).
  present-nested가 변형을 *보여주나* user-묘사→변형 매칭이 미오프로드. = calc(max/min)+attribute-filter로 닫힘.
- **over-action(12)**: gold=무write(쓰면 안 됨)인데 모델이 return 실행. = feasibility/should-not 판단 부재.

**★핵심 구조 발견**: 현 present-nested scaffold는 **한 주문 *내부*의 변형(L2/L3)**만 제시 → 실패는 전부
(a) **주문 *간* coverage**("모든 X" 전수), (b) **cross-order 참조**(item→order), (c) **변형-매칭 calc**
(user-묘사→변형) 차원 = **현 scaffold가 안 덮는 정확한 지점**. genuine 의미이해-scale 아님.

## 4. 극복 분류 (버킷 → 레버 → decidable)

| 버킷 | 32B/14B | **극복 레버** | decidable |
|---|---|---|---|
| 변형/값 선택 | 21/26 | **변형-match calc**(present-nested에 max/min + attribute-filter 결정론 주입) + verbatim-copy(주소) | ✅ 대부분 |
| coverage (상태추적) | 17/16 | **C2 coverage controller**(영향 엔티티 전수 열거·미커버 차단) | ✅ |
| ⋈ 참조 | 7/10 | **cross-order present**(order→address·item→order 매핑 제시) | ✅ |
| over-action | 4/7 | **feasibility gate**(불가능/should-not write 사전 차단) | ✅ |
| calc (order-total) | 6/3 | **calc-scope 확장**(order-total을 calc_specs에) | ✅ |
| orchestration (loop/no-write) | 13/17 | retry-guard(반복차단) + plan-execute controller | ◐ 일부 결정론·일부 load |
| report/communicate | 4/3 | report enforce(present) | ◐ |
| tracking#/artifact | 1/0 | (벤치 아티팩트·제외) | — |
| **genuine semantic-scale** | **≈0** | (scale/학습) | — 식별 안 됨 |

**decidable 합계**: 32B **55/73(75%)**·14B **62/83(75%)** = 잔여 3/4가 결정론 레버로 닫힘. orchestration
(~18%)은 혼합. genuine-scale ≈0.

## 5. 결론·다음 레버
1. **잔여=결정론 지배·scale 아님**(클린 nt=4 재확인). 현 stack(present-nested·within-order)이 못 덮는
   **multi/cross-order + 변형-calc**가 정확한 gap.
2. **다음 결정론 레버(우선순위)**: ①**C2 coverage controller**(coverage 17/16·상태추적) ②**변형-match calc**
   (변형선택 21/26·최대 버킷) ③**cross-order present**(⋈ 7/10) ④**calc-scope order-total**(6/3) ⑤feasibility
   gate(over-action). 전부 [[05]] 도메인-일반·A2-구동 가능.
3. **규모-불변 구조**: 32B·14B 동일 버킷 → 레버가 두 규모 공통. scale은 빈도만 낮춤(scaffold가 종류를 닫음).
4. **[[13]] 정합**: 결정론(위 레버) 먼저·genuine-scale은 소수/미검이라 학습·scale은 최후.

## 부록 — task 리스트 (재현)
- 32B within-order: 7,8,9,13,17,20,21,27,34,36,38,44,56,58,59,60,63,66,76,101,103 / coverage: 14,23,41,42,
  74,81,87,91,92,98,104,107,110,111,112 / ⋈: 10,12(→over-action),32,57,71,72,79,84,99,109 / calc: 19,47,54,67,68,95.
- 데이터: `sim_results/asmregen{32b,14b}_regen_retail_t4.results.json.gz`. 분류기: scratchpad `forensic_*.sh`.
