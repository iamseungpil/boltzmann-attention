# Escape-Scope Diagnostic — 층화(layered) σ 보강 설계 (2026-06-24)

> **모doc**: `ESCAPE_SCOPE_DIAGNOSTIC_DESIGN_2026_06_24.md`(rev2)의 보강. S1 harness preview가 드러낸 것 반영. thesis `EPISTEMIC_A2_THESIS_2026_06_23` §3 escape-범위·§6 make-or-break.

## 0. 동기 — S1 preview가 깬 가정 (정직)
rev2는 실패를 **order-층 disambiguation** 하나로 가정했다. **S1 preview(15 gap)가 반증**:
- **다수 gap은 32B가 *gold 주문을 이미 골랐다*** (task 8·17·34·41·101·102) → 실패는 order-층 아님.
- **단일 주문 task**(36/37/38·daiki=후보 1개)는 order-disambiguation *불가능* → 실패는 하류.
- **틀린 주문**(71·72=σ=1 ⓑ mis-ground)은 *소수*.
- ⇒ **order-층 σ만으론 gap 다수를 분류 못 함.** 실패가 어느 *층*인지부터 찾아야 한다. order-층 escape는 좁고, 진짜 게이트는 하류 층 분포에 있음.

## 1. 층 구조 (failure locus)
write tool call = (operator, order_id, item_ids, new_item_ids, options/payment). gold vs 궤적을 **field-by-field 비교 → first-divergence 층**이 실패 locus:

| 층 | divergence | 관계/σ 대상 | 예 |
|---|---|---|---|
| **L0 operator** | tool name 틀림 | (관계 아님·action) | 주소수정↔취소·교환↔반품 |
| **L1 order** | order_id 틀림 | σ(user의 orders) | task 71 DC주문 |
| **L2 item** | item_ids 틀림 | σ(그 order의 items=anchor_source) | 어느 품목 교환 |
| **L3 variant** | new_item_ids/options 틀림 | σ(product variants=candidate_source) | black lamp·medium polyester |
| **OVER** | gold에 없는 추가 write | (action·over-action) | task 101/102 여분 주문 |

- L2/L3 = `retail.grounding.json` **anchor_source(order items)·candidate_source(product variants)**가 이미 실물 → 그 σ 재사용.
- 한 task가 여러 층 실패 가능 → **first-divergence를 primary**로, 나머지는 secondary 태그.

## 2. 각 층 σ + ⓐ/ⓑ 분류 (rev2 §2를 층별로)
각 층에서 **faithful 술어**(유저 발화의 그 층 리터럴 제약) → σ(결정점 state) → |σ| + 궤적선택 대조:
- **|σ|=0** → no-change impasse → **ⓐ** ("해당 없음→ASK/불가통지")
- **|σ|>1** → tie impasse → **ⓐ-tie** *단 §3 단서 적용*
- **|σ|=1** → 유일정답: 궤적이 그걸 골랐나 → 아니면 **ⓑ**(mis-resolve·침묵)
- **L0/OVER** = impasse 아님 → **ⓑ-act**(escape 밖·operator-select/over-action)

## 3. ★핵심 정밀화 — tie의 절반은 escape 아님 (B2-resolve)
**rev2가 놓친 것**: |σ|>1(tie)이라도 **유저가 tiebreaker를 줬으면 escape-ASK가 *아니라* compute-resolvable(B2 argmax/rank)**.
- 예 task 71: "backpack medium polyester, **if multiple colors prefer grey**" → color tie를 유저가 *미리 해소*. → σ_{size=medium∧material=polyester} = {grey, ...} 다중이라도 **B2-resolve(prefer grey)**지 ASK 아님.
- ⇒ ⓐ-tie를 둘로 쪼갬: **ⓐ-ask**(tiebreaker 없음·진짜 모호→ASK) vs **ⓐ-B2**(tiebreaker 있음·내부 결정론 resolve·SOAR 내부 subgoal·[[10]] B2).
- **escape(ASK)가 잡는 건 ⓐ-ask 뿐.** ⓐ-B2·ⓑ·L0·OVER = 전부 escape 밖(B2학습/operand/action). → **escape 너비는 rev2 추정보다 더 좁아질 전망**(정직·게이트에 직접 반영).

## 4. 분류 절차 (harness)
1. gold write-actions vs 궤적 write-calls 정렬(operator+order_id 매칭).
2. **first-divergence 층** 판정(L0→L1→L2→L3→OVER 순).
3. 그 층의 candidate collection 로드(L1=orders·L2=order items[anchor_source]·L3=variants[candidate_source]).
4. 그 층 faithful 술어(큐레이션) → σ(결정점 state) → |σ|.
5. tiebreaker 유무 판정(유저 발화) → ⓐ-ask vs ⓐ-B2.
6. 궤적선택 대조 → 최종 라벨 + impasse 타입.
7. (Arm-II) ⓐ·ⓑ 케이스 후보집합 떠먹여 32B select-probe → capability.

## 5. 출력 (rev2 §5 확장)
1. **층별 실패 분포**(L0/L1/L2/L3/OVER) — gap의 어디서 깨지나.
2. 층별 **ⓐ-ask / ⓐ-B2 / ⓑ** split.
3. **escape-catchable = Σ ⓐ-ask** (전 층)= 진짜 escape 너비(헤드라인).
4. ⓑ·ⓐ-B2의 select-probe pass율(grounding-됨=학습여지 vs capability-bound).
5. impasse×층×gap-class 교차표.

## 6. GO/NO-GO 영향 (rev2 §6 정밀화)
- **escape-catchable(ⓐ-ask) 비율이 헤드라인**. preview는 이게 *작을* 조짐(다수=gold-order-picked→하류 ⓑ / tiebreaker 있는 tie=ⓐ-B2).
- ⓐ-ask 작음 + ⓑ·ⓐ-B2 큼 → **abstain-ASK 커리큘럼은 좁은 레버**. 진짜 본체 = **(i) faithful-formalize(ⓑ mis-resolve 닫기) + (ii) B2-resolve 학습(ⓐ-B2)**. = thesis §4 "대칭(결정가능→행동)" + [[10]] B2가 ASK보다 비중 큼.
- NO-GO(b 강화): ⓑ·ⓐ-B2 케이스 select-probe서 **후보 줘도 32B 틀림** 압도 → capability-bound(escalate). 후보 주면 맞힘 → self-formalize/B2 학습여지(부분 GO).
- **정직 재포지셔닝**: 이 진단이 "escape narrow + B2/formalize가 본체"를 확증하면 thesis §4 학습대상의 *무게중심*이 ASK→formalize+B2로 이동(escape는 잔여 보조). thesis 깨는 게 아니라 *정밀화*.

## 7. 구현 (rev2 §8에 추가)
- S1b: `escape_scope_diag.py`에 **layer_decompose(gold, traj)** + L2/L3 σ(grounding.json anchor/candidate_source 재사용·`t2_resolve_patch._ground` 참조) + tiebreaker 검출.
- `escape_predicates.json`을 **층별 술어**로 확장(task별 L1/L2/L3 + tiebreaker 플래그).
- S2 대면검증 항목 추가: (e) first-divergence 층 판정 정확 (f) tiebreaker 유무 판정 정확(ⓐ-ask/ⓐ-B2 갈림이 여기 달림).
- S3 정성 카탈로그 = 15 task 층별 분류. S4 비율 = retail 전체 실패 층화.

## 8. 불변
- 정적·tau2 학습0·A2 σ(grounding.json 재사용)·도메인분기0·gpt-4.1 불요([[05]][[11]]).
- **S2 (a)~(f) 대면검증 전 무인 전수 금지.** tiebreaker 오판=ⓐ-ask/ⓐ-B2 뒤집힘=헤드라인 오도.
