# C8 (Error-Recovery) Cheap-Replication 설계 — 결정론 retry-controller가 scale의 복구를 대체하나 (2026-06-21)

> 상위 = `EXPERIMENT_DESIGN §0★★`·`CAPABILITY_LEVER_ALLOCATION_DESIGN`(둘째 기둥=scale 능력 cheap-replication)·`ma/M_A_RESULTS §35/§35b`. 불변 [[00-thesis]][[10-roles-deterministic]][[13-absorption-priority]].
> 위치 = 논문1 둘째 기둥의 *두 번째* cheap-replication 실험(첫 = C3 grounding=engine autofetch·실증됨 §35b).

## 0. 가설 (한 줄·리뷰 위험1 reframe — C3와 *동형 아님*)
**scale이 사는 C8(error-recovery)에서 *loop 탐지*는 decidable(결정론 retry-controller로 offload)이나 *올바른 복구행동*은 모델 몫 = 부분-decidable.** ⇒ C8 = "둘째 동형 증거"가 **아니라** ★**cheap-replication의 *경계/한계* 데이터점**: scale 능력마다 decidable-비율이 다름을 특성화 (C3=완전대체 vs C8=부분대체).
- **C3 vs C8 비대칭(리뷰)**: C3=완전 decidable(producer 스키마 도출→엔진이 detect+fetch 둘 다=능력 통째 대체). C8=부분(loop 탐지만 decidable·복구행동≠decidable·엔진은 차단만). → "동형" 과주장 철회.

## 1. 동기 (실측 근거)
- scale 분해(`§35`): 7B 0.24→32B 0.60. **error-loop 종료(too_many_errors) = 7B 36 vs 32B 0**(`prompt_effect.sh`·n342) = scale이 사는 *복구* 능력.
- census(`PRIMITIVE_MATRIX:92`·06-NOW): 7B는 2-hop gather·복구 *할 줄 안다*(task6)·문제 = **반복-동일-실패호출 default**(diagnose-and-retry-differently 안 함). = capability gap 아니라 *behavior default* gap (C3 fabricate-first와 동형).
- ⇒ behavior default = decidable 가드로 교정 가능 가설.

## 2. 메커니즘 = error-loop은 decidable
- **too_many_errors loop 정의**: 모델이 *같은* 실패호출을 반복 → orchestrator num_errors 초과 → 종료.
- **decidable**: "같은 호출(tool+args)이 *이미 실패*했는가" = 결정론 술어(key=tool명+정규화 인자, 실패셋 조회). 탐지에 LLM 불요.
- ⇒ **retry-controller = offload(scaffold)** ([[10-roles-deterministic]]): 결정론이 loop 탐지·차단, LLM은 *다른 행동 선택*(다양화)만. 학습 0.

## 2.5 ★실험 전 census (리뷰 위험2·완료·gating) — too_many는 동일-반복인가
`/tmp/c8_census.sh`(기존 로그·비용0): on_n7b_floor 36 too_many sims — **동일-key 반복 = 전체 에러호출의 30%·동일key≥3회 순환 지배 sims = 25/36(69%)·전부유니크(no-op) = 6/36(17%).** (rules: 32%·33/43·14%.)
- ⇒ **혼합**: ~70% sims는 소수 key(uniq~3·maxrep~6) *순환* = 정확-반복 controller 유효(no-op 아님). ~17%는 *매번 다른 실패* = 정확-반복만으론 빗맞음.
- ⇒ **설계 보강(아래 §3 rule②)**: 정확-반복 + **연속-실패 K회 가드**로 순환·다양 둘 다 커버.

## 3. retry-controller 설계 (scaffold·`t2_gate_patch` `T2_RETRY_CONTROLLER`·리뷰 위험2 반영=2 rule)
- **추적**: orchestrator당 `failed:{call_key→에러}` + `consec_fail`(연속 실패 카운트). 출처 = provenance-deny·gate-deny·tool-exec-error 전부.
- **rule① 정확-반복 차단**(순환 ~70% 타깃): 새 호출 key ∈ `failed` → 재실행 안 함 → directive: *"이 호출 이미 실패(X). 반복말고 (a)fetch (b)ask (c)transfer 중 다른 행동."*
- **rule② 연속-실패 K 가드**(다양-실패 ~17% 타깃·신규): 연속 K회(예 3) 실패(key 무관) → escalation directive: *"연속 K회 실패. 전략 바꿔라: 빠진 값 fetch / user 질문 / transfer."* 성공 시 카운트 리셋.
- **decidable·도메인-일반**(grep if-domain=0)·무학습·무붕괴(weight 불변).
- ⚠️ num_errors 회계: 차단도 +1(정직·budget 측정)·directive가 다양화 유도해 추가반복 방지. (1차 +1 유지·측정.)

## 4. 실험 arms (한 serve·base 7B·retail n=114·C3 S-min 동형)
| arm | 구성 | 격리 |
|---|---|---|
| **c8_floor** | base 7B gate0 (patch 없음) | 절대 baseline(too_many_errors loop 존재) |
| **c8_gate** | base + gate1 (patch·retry0) | patch-내 기준 |
| **c8_gate_retry** | base + gate1 + **T2_RETRY_CONTROLLER** | ★retry 효과 격리(c8_gate 대비 env 1개 차이) |
| (참조) 32B | scale 천장 | too_many_errors=0(기지) |
- 격리 = c8_gate vs c8_gate_retry (retry만 차이). c8_floor·32B = 양 끝 참조.

## 5. 측정
- **pass^1**(db_match).
- **★too_many_errors 종료 수**(C8 핵심 신호·termination_reason) — retry가 0 쪽(32B)으로 미나.
- **failcensus_deep**: F_errorloop·A(notfound)·B(operand)·D — loop가 줄며 *어디로* 이동하나(복구→성공? or 다른실패?).
- **복구율**: 실패호출 후 *다른* 유효행동으로 전환 비율(autopsy).

## 6. 사전등록 예측 (C3 패턴 외삽)
- **H_offload(예상)**: retry-controller가 too_many_errors↓·pass↑ (C3 autofetch가 A 33→9·pass 2× 했듯). = C8도 decidable→offload로 cheap-replicable.
- **★단 정직한 한계**: retry는 *loop를 깨*지만 *올바른 다음행동*은 모델 몫. 차단 후 모델이 (a)(b)(c) 못 고르면 → **다른 실패로 이동**(A grounding or B operand) = pass 회복 제한. ⇒ retry는 loop-제거엔 충분, pass-회복은 grounding(C3 엔진)·operand(C10 학습)와 *합쳐야* 완성. (decidable-ratio: 복구의 얼마가 retry-offload vs 잔여 모델 몫.)

## 7. GO / NO-GO (리뷰 위험1 — 지표 분리·pass≥로 recovery-GO 선언 금지)
**두 결과를 *분리* 판정**(loop-제거 ≠ recovery):
- **recovery-GO (강)**: c8_gate_retry가 c8_gate 대비 **pass↑**(실제 복구=실패후 성공) ∧ too_many↓. = decidable 차단이 복구로 *전환*됨.
- **loop-only 경계 (약·예상 더 가능)**: **too_many↓ ∧ pass≈**(loop는 깨나 다른실패로 이동). = **decidable 조각(loop탐지) 단독 불충분** = C8의 부분-decidable 경계 실증. ← 이것도 *유효 기여*(thesis 경계 특성화).
- **NO-GO(설계 빗맞음)**: too_many도 ≈불변(census ~17% 다양-실패가 지배했거나 rule② 무효) → controller 타깃 빗맞음·재설계.
- ★**pass≥만으로 recovery-GO 선언 금지** — pass↑가 있어야 복구 대체. pass= = loop-only 경계로 *정직히* 기록.

## 8. 프레임워크 정합 (리뷰: 경계 데이터점)
- C8 = scale-emergent behavior → **방법집(scaffold·§5)**: C3(완전대체)와 **대비되는 *경계* 데이터점** — "scale 능력마다 **decidable-비율이 다름**(C3=100%·C8=loop탐지만)·decidable 조각만으론 부분대체." 이 *경계 특성화*가 thesis 기여(과주장 아님).
- thesis: decidable(loop 탐지)→offload·LLM=다양화 선택만·무학습·무붕괴([[10]][[13]]).
- 응용(논문2/플랫폼): retry-controller = 멀티턴 루프의 복구 안전망(L2 scaffold).

## 9. 다음 (설계 확정 후)
1. retry-controller 구현 정렬(`t2_gate_patch T2_RETRY_CONTROLLER`·초안 존재) → syntax·smoke.
2. `c8_eval.sh`(arms 4·n114·too_many_errors+failcensus 계측) 작성·실행.
3. 결과 → `M_A_RESULTS §35c`(C8 cheap-replication) 박제.
