# C8 (Error-Recovery) Cheap-Replication 설계 — 결정론 retry-controller가 scale의 복구를 대체하나 (2026-06-21)

> 상위 = `EXPERIMENT_DESIGN §0★★`·`CAPABILITY_LEVER_ALLOCATION_DESIGN`(둘째 기둥=scale 능력 cheap-replication)·`ma/M_A_RESULTS §35/§35b`. 불변 [[00-thesis]][[10-roles-deterministic]][[13-absorption-priority]].
> 위치 = 논문1 둘째 기둥의 *두 번째* cheap-replication 실험(첫 = C3 grounding=engine autofetch·실증됨 §35b).

## 0. 가설 (한 줄)
**scale이 사는 C8(error-recovery)을, *결정론 retry-controller*(반복-동일-실패호출 차단+다양화 지시)로 학습·scale 없이 cheap-replicate한다.** = C3(autofetch)와 동형의 "scale 능력=decidable→offload" 둘째 증거.

## 1. 동기 (실측 근거)
- scale 분해(`§35`): 7B 0.24→32B 0.60. **error-loop 종료(too_many_errors) = 7B 36 vs 32B 0**(`prompt_effect.sh`·n342) = scale이 사는 *복구* 능력.
- census(`PRIMITIVE_MATRIX:92`·06-NOW): 7B는 2-hop gather·복구 *할 줄 안다*(task6)·문제 = **반복-동일-실패호출 default**(diagnose-and-retry-differently 안 함). = capability gap 아니라 *behavior default* gap (C3 fabricate-first와 동형).
- ⇒ behavior default = decidable 가드로 교정 가능 가설.

## 2. 메커니즘 = error-loop은 decidable
- **too_many_errors loop 정의**: 모델이 *같은* 실패호출을 반복 → orchestrator num_errors 초과 → 종료.
- **decidable**: "같은 호출(tool+args)이 *이미 실패*했는가" = 결정론 술어(key=tool명+정규화 인자, 실패셋 조회). 탐지에 LLM 불요.
- ⇒ **retry-controller = offload(scaffold)** ([[10-roles-deterministic]]): 결정론이 loop 탐지·차단, LLM은 *다른 행동 선택*(다양화)만. 학습 0.

## 3. retry-controller 설계 (scaffold·`t2_gate_patch` `T2_RETRY_CONTROLLER`)
- **추적**: orchestrator당 `failed: {call_key → 마지막 에러}`. 실패 출처 = provenance-deny·gate-deny·tool-exec-error 전부.
- **차단+지시**: 새 호출 key가 `failed`에 있으면 → 재실행 안 함 → directive 반환: *"이 호출은 이미 실패(에러 X). 반복 말고 (a)producer서 fetch (b)user에 질문 (c)transfer 중 *다른* 행동."* = loop 깨고 다양화 강제.
- **decidable·도메인-일반**(grep if-domain=0)·무학습·무붕괴(weight 불변).
- ⚠️ num_errors 회계: 반복차단도 +1(정직·budget 소모 측정)·단 directive가 다양화 유도해 *추가 반복 방지*. (대안: 반복은 budget 면제=loop 무한방지 cap 필요 — 1차는 +1 유지·측정.)

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

## 7. GO / NO-GO
- **GO**: c8_gate_retry가 c8_gate 대비 too_many_errors 유의↓ ∧ pass ≥ (무붕괴·학습0). = "C8 복구 = decidable retry-controller로 cheap-replicate" 실증(둘째 기둥 둘째 증거).
- **NO-GO**: too_many_errors는 줄어도 pass 무변(=loop 깨도 다른실패로 이동) → retry는 *부분*(loop-제거)·pass는 C3+C10 결합 필요. = 정직한 경계(여전히 유효 발견: C8=offload 가능·단 단독 불충분).

## 8. 프레임워크 정합
- C8 = scale-emergent behavior → **방법집(scaffold·§5)**: C3(엔진 autofetch)와 함께 "scale 능력 = 분해가능·decidable 조각은 scaffold가 scale 없이 대체"의 둘째 데이터점.
- thesis: decidable(loop 탐지)→offload·LLM=다양화 선택만·무학습·무붕괴([[10]][[13]]).
- 응용(논문2/플랫폼): retry-controller = 멀티턴 루프의 복구 안전망(L2 scaffold).

## 9. 다음 (설계 확정 후)
1. retry-controller 구현 정렬(`t2_gate_patch T2_RETRY_CONTROLLER`·초안 존재) → syntax·smoke.
2. `c8_eval.sh`(arms 4·n114·too_many_errors+failcensus 계측) 작성·실행.
3. 결과 → `M_A_RESULTS §35c`(C8 cheap-replication) 박제.
