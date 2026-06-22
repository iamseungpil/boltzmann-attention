# 32B vs frontier 갭 — 전수 궤적 비교 (2026-06-22·사용자 지시)

> 사용자: "32B면 fetch-first 통과하니 상용=32B. 문제는 32B→frontier 능력 갭." → 전수비교.
> 데이터: `on_n32int8_floor_retail`(32B-int8·3 trial)·`retail_gpt41_nogate`(gpt-4.1 agent·1 trial). 둘 다 no-gate=순수 모델능력.

## §1. pass (★갭의 정체 = 신뢰성)
- 32B pass^1 ≈ 0.60(§35) · **32B pass-any-of-3 = 0.77** · gpt-4.1 단일시행 = **0.82**.
- ⇒ **32B 최선-of-3 ≈ frontier 첫시행.** 갭의 큰 부분 = *raw 능력*이 아니라 **신뢰성/일관성**(32B는 풀 수 있으나 첫시행 실패·재시도 필요).

## §2. GAP = frontier 풀고 32B 3시행 전부 실패 = 15 task (진짜 frontier 능력)
전수 분류(15 task·write-action 비교):
| 유형 | 수 | 정체 |
|---|---|---|
| **DIFF-ACTION** | **8 (53%)** | 틀린/불완전 행동·루프·포기 = **flow 다단계 완수 + 복구(P7)** |
| **SAME-ACTION(operand)** | 6 (40%) | 올바른 행동·틀린 인자 = variant 선택(B1)+verbatim 복사 |
| no-write | 1 (7%) | 시도 안 함 |

## §3. 궤적 구체 (직접 확인)
- **T8 (operand/B1)**: 32B new_item_ids=1270145486→4385534692(틀린 variant) vs FR 9083642334(정답). = 의미적 variant 선택. ← B1-select 타깃.
- **T17 (operand/copy-fidelity)**: 32B "123 Elm **St**" vs FR "123 Elm **Street**". = user 문자열 verbatim 복사(32B 축약). ← operand-formalize 정밀도.
- **T38 (flow/recovery)**: 32B modify_items(new_ids=[]) **4회 루프** vs FR **cancel_pending_order**(올바른 의도). = 의도파악+루프탈출.
- **T34/T37 (flow/포기)**: 32B modify 루프 후 **cancel로 fallback**(포기) vs FR modify 완수. = 다단계 완수 못 하면 도망.
- **T41 (flow/불완전)**: 32B 2 write vs FR 4 write. = 멀티-액션 누락(불완전).

## §4. 결론 — 32B→frontier 갭 = (a)flow완수·복구 > (b)operand
1. **★최대 = flow 다단계 완수 + 복구(53%)**: frontier는 올바른 시퀀스를 *끝까지 수행*·실패시 *다르게 재시도*. 32B는 *루프*하거나 *cancel로 포기*. = §35 "7B→32B 갭=DB-state/flow/복구"가 *32B→frontier에도 잔존*(작아졌으나).
2. **operand 정밀도(40%)**: variant 선택(B1·의미)+verbatim 복사. ← B1-select가 겨눈 부분(40%만).
3. **신뢰성**: 32B≈frontier@best-of-3 → 갭은 *첫시행 완수율*. flow-복구·operand 둘 다 여기 기여.

## §5. 함의 (연구 타깃 재정렬)
- **B1-select(진행중)은 갭의 40%(operand)만 커버.** 더 큰 레버(53%)=**flow 완수+복구**.
- ⇒ 32B를 frontier로 올리는 cheap-replication 타깃 = **(1)복구/완수 controller**(루프탈출·포기금지·다르게재시도=P7·이전 C8) + **(2)operand B1(variant선택+verbatim)**. flow-복구가 우선.
- 복구는 [[42]]/딥리서치가 retry-controller/reflection/최소LoRA로 지목·§35c C8서 "retry=잘못된 레버"였으나 그건 *grounding* 맥락·여기선 *완수/포기* 맥락 → 재검토 가치.
- REVERSE(32B 풀고 frontier 실패)=10 task(frontier 약점·user-sim 노이즈 가능·부차).

## §6. 다음
1. B1-select 결과 회수(operand 40% 레버 검증).
2. flow-완수+복구 census 심화(DIFF-ACTION 8개의 세부: 루프 vs 포기 vs 불완전 분해) → 복구 레버 설계.

**불변**: §35·[[42]]·`FETCH_SELECT_DIVISION`(operand=B). 상위 `RULE_LEVER_COST_EFFICIENCY_PROGRAM`(C8 recovery·C10 operand 둘 다 깊음).
