# Assembled-Deterministic 한-런 설계 (2026-06-25) — convergence checkpoint·learn 질문 종결

> 근거 = `NESTED_ARM_FAILURE_CENSUS` §4.5 정본 레버 지도(robust·stable-20·gold-diff). disentangle가 "40% capability-under-load" 기각(=user-sim 노이즈)→ learn 질문 재오픈·**operand 9건(45%)으로 환원**. 사용자 지시(2026-06-25): (a)calc_NL·(b)operand census를 *직렬 말고* **한 assembled-deterministic 런**으로 병합(gold-diff가 이미 de-confound·직렬 불요). GPU 1회로 모든 깨끗한 결정론을 max한 뒤 잔여=진짜 learn-or-capability 격리.

## 0. 목표 (왜 한 런)
정본 잔여(stable-20): operand 45%·calc_NL 20%·no-write 20%·over 15%. 직렬(a→b) 대신 **present-개선 + calc_NL-compute + 기존게이트(+disjoint)를 한 스택**으로 retail 재실행 → gold-diff 잔여 census. 이 한 수 = operand를 결정론 present로 공격(b 첫 substep·Probe-B 가설 라이브 검증) + calc_NL bank(a 동승) + over 흡수 → **남는 잔여 = 모든 결정론 max 후 = 오염 없는 make-or-break(learn-or-capability).**

## 1. 스택 구성
`present-개선 + calc_NL-compute + 전체게이트(auth·confirm·ownership·notice·preconditions) + new≠old(disjoint)`, 32B+14B, retail, **multi-trial(≥3·robust + user-sim 노이즈 평균화)**, gpt-4.1 user-sim, replay-safe.

## 2. ★Mechanism 1 — present-개선 (Probe-B 품질·operand 45% 공격)
- **현 nested present의 한계**: 필드 dump이나 (i)결정점 정렬 약함 (ii)⋈(cross-entity·다른주문 주소) 미포함 (iii)raw 완전성 부족 → present-arm이 operand "약간만" 닫은 이유(task71류 무시).
- **개선 = Probe-B 7/7 됐던 *그 형식* 복제**: 결정점(write 직전)에 **raw 후보를 명시 choice-set**으로 — 중첩 dict 그대로·**⋈ 포함**(예: address-from-other-order면 *다른 주문의 전체 주소*를 후보로 노출·101/102). 단순 fetch 아님(=무시됨).
- 구현: `gate_interpreter` present 경로 확장 — select_confirm/nested를 "결정점 raw-choice-set" 모드로(A2 present_fields→raw 레코드·⋈ source 경로 A2 지정). [[05]] 엔진-일반·A2-사실.
- **가설**: Probe-B(격리 7/7)가 라이브서도 되면 → operand 다수가 **present-형식약함=결정론-fixable**(learn 아님). 안 되면 → capability/learn.

## 3. ★Mechanism 2 — calc_NL-compute (계산/집계 offload·20% bank)
- calc_NL 실패 = {산술(환불총액)·집계(가용 변형 수)·조회(tracking#)→보고}. 모델이 available 필터 안 해 오산.
- **offload = 엔진이 파생사실 결정론 계산·주입**(read 응답 증강): 예 get_order_details→"items 총액=$X"·get_product_details→"available 변형=N개"·order→tracking#. **계산은 결정론·보고는 모델.**
- [[05]]: 엔진=일반 aggregate 연산{sum·count-where·lookup}·A2=retail이 어느 필드/조건. (별도 소-스펙 `calc_specs` in A2.)
- **★report-conversion 측정(필수·[[06]] lever≠resolution)**: 주입해도 모델이 *말 안* 할 수 있음(present order-pick과 동일 리스크). 주입↔보고 전환율 별도 측정.

## 4. 측정 ([[06]]/[[08]])
- **pass^all(robust=fail-all-3) 1차·pass^1 단독 금지**(노이즈 0.19). 결정론 action-census(escape_det_census) 병행.
- crash/infra/too_many 배제(--clean)·종료분포 먼저. user-sim 노이즈=multi-trial 평균(불일치 7류 분리).
- over-deny 체크(disjoint·present가 양성 막나).
- baseline = present+nest+g15(현 정본 `*_presentnest_g15_retail_t3`).

## 5. ★잔여 태깅 (make-or-break 종결·gold-diff)
런 후 reward=0(robust) 잔여를 배정:
| 잔여 | 의미 | 레버 |
|---|---|---|
| **present-closed operand** | present-개선이 닫음 | **결정론 승리**(learn 아님) |
| **present-but-wrong operand** | raw choice-set 줘도 틀림 | **capability-or-learn = priority-4 SFT 유일 타깃** |
| calc_NL-but-not-reported | 계산 주입했으나 보고 실패 | report-conversion(prompt/scaffold) |
| no-write/orchestration | 도달 실패 | recovery/auth·user-sim |
- **present-but-wrong operand = 이 arc 전체의 종착 질문.** 그 안에서 **C4-copy계열(전이음성·[[20]]) vs criterion-formalize(σ로 못만드는 기준)** 추가 구분 → C4면 learn도 음성=capability·non-C4면 **learn GO(priority-4·present-불가 잔여 한정)**.

## 6. NO-GO / 분기 (종결 조건)
- present-개선이 operand 대부분 닫음 → **결정론 천장↑·learn NO-GO**·헤드라인=결정론+TCO([[06]]).
- present-but-wrong이 크고 non-C4 → **learn GO**(priority-4 SFT·`A2_RULE_USE_SFT_PREP` §4.1·이 잔여만 타깃).
- present-but-wrong이 크나 C4-copy → 기약 capability(scale/escalate·[[13]]).

## 7. 선행 (구현 전)
- present-개선의 ⋈ source 경로가 순수 A2-path로 표현되나([[05]] cleanliness·payment 파생필드와 동류 위험).
- calc_specs aggregate 연산이 도메인-일반인가(sum/count-where=일반·retail 조건만 A2).
- GPU = priority-2 종료 후(현 present-nest 런 점유). 구현=GPU-free 가능(엔진·A2·드라이버).
- (이 doc은 `ASSEMBLED_STACK_CENSUS_DESIGN`의 Phase1+3을 구체화·병합 = 그 doc의 operative 버전.)
