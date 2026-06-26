# M-σ v4 — step 0: subtract-coverage 맵 (zero-cost·실험0 판독 전제) — 2026-06-16

> v4 §1 구현. zero-cost 분석([[feedback-zero-cost-diagnosis-strongest-case]])·의존 0·GPU 0. 권위 = `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(P1-P9)·`M_A_RESULTS.md §3,§11`(write-벽)·`m_sigma_transfer_eval.py`(arg 구조). 불변 = [[feedback-thesis-tbox-transfer-direction]].

## 1. ★4족 → 기존 P1-P9 매핑 (taxonomy 분기 금지·scope-partition)
v4 분업표의 4족은 *신규 primitive가 아니라* 매트릭스 P1-P9의 재그룹이다. 정확 매핑:

| v4 4족 | = 매트릭스 primitive | 매트릭스 커버상태 | 비고 |
|---|---|---|---|
| **P-gate**(정책·precondition) | **P5**(policy-gating) [+P8 auth] | SOPBench ✓(in-dist) | P6 confirm = **유보**(아래) |
| **P-thread**(passive 복사-threading) | **P2b**(gather-for-arg·출력 in-context) + **P3**(순서) | TaskBench ✓ / cfb ✓ | grounding=**P1** |
| **P-fetch**(grounded 값 proactive obtain) | **P2b + R4 proactive**(능동 getter 호출) | cfb ◐(데이터존재) | v7 autopsy 미해결층([[project-cross-bench-transfer-plan]]) |
| **P-select**(변형선택·change/keep/fallback) | **★P4**(select-from-output) + criteria-구성 | cfb ✓-데이터 / **전이 미검증(✗ ✓!)** | §5 scope-out transform과 인접 |

### ★핵심 화해 (round-3 정정 박제): P-select = 기존 P4 (신규 아님)
- 매트릭스(§2)는 **P4(select-from-output)**를 이미 열거·"cfb 데이터존재·v7 eval 전 미확정"으로 표기. **현재 ✓!(전이검증)=P1뿐**·P4는 ✓(데이터)이지 ✓!(전이) 아님.
- 06-15 census 갭 = "P2b(R4 의미전이)+P6+P7", root-cause = **P2b 'fetchable 값 날조-FIRST'**(order_id `#W0000000` 복사).
- **06-16 M-A(`M_A_RESULTS §3`)가 갭을 *이동*시킴**: anti-fab(v8/v9)이 날조($ref/P2b)를 잡자 *하류 병목이 드러남* = **new_item_ids만 틀림(order_id·payment 다 맞음) = 변형선택 = P4**. 즉 **P2b(날조)→P4(변형선택) 마이그레이션**(anti-fab 진행에 따라). 매트릭스는 이 이동에 대해 stale.
- ⇒ 화해 = "P6/P7→P-select 수렴"(세탁) 아님. **P-select=기존 P4·전이 미검증·anti-fab 후 dominant write-벽으로 격상**. P4의 criteria-구성({current⊕changes})은 매트릭스 §5가 scope-out한 transform과 인접 — 순수 arithmetic 아닌 record-override라 P4 하위로 흡수(별 P10 아님).

### ★scope-partition (측정 vs 유보·collapse 금지)
- **측정**(이 실험): **P-select(=P4)** = 단발 formalize 전이. + 진단상 **P-thread/P-fetch**($ref grounding) 회복.
- **유보**(별 트랙·§9 bridge서 정산): **P6**(confirm-gate)·**P7**(recovery) = multi-turn control-flow. v3/v4 단발 eval은 P7을 *측정조차 못 함*([[project-v9-dpo-antifab-result]]·복구 2→8 독립이동). 단발 코퍼스에 P6/P7 박지 않음.

## 2. ★τ² retail exchange — arg별 provenance 라벨 (실험0 3-way split 버킷)
`exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)` 각 arg를 provenance로 라벨(harness 버킷 = 이것):

| arg | provenance 버킷 | 소스 obs | 재추출 가능? | 실험0 예측 |
|---|---|---|---|---|
| **order_id** | passive-$ref (P-thread) | get_order_details.order_id | ✅ | **개선**(날조↓) |
| **item_ids**[old] | passive-$ref (P-thread) | get_order_details.items[].item_id | ✅ | **개선** |
| **payment_method_id** | passive-$ref (P-thread/fetch) | get_user_details.payment_methods | ✅(값-fix 후) | **개선**(현 0.07=harness 아티팩트) |
| **new_item_ids** | **$select (P-select/P4)** | get_product_details.variants by criteria | ❌(에피소드 부재) | **평탄**(synth 배타) |

- **★단발 harness 주의(설계 정밀화)**: obs를 *미리 제공*하므로 단발 모드선 **passive-$ref vs $select 2-버킷만 실측**. **proactive-gather**(getter 능동호출)는 control-flow라 단발서 안 잡힘 → harness `--withhold` 모드(특정 getter 출력 보류→"call X then $ref" 강제)로만 probe = residual #1(별 add-on·기본 실험0 밖). v4 §2 표의 3-way 중 proactive 버킷은 *withhold 모드 한정*.
- **over-$ref**: literal-arg(요청 NL에 직접)인데 $ref 시도한 비율 = M-D 음성 원인(a). passive-$ref 버킷 내 측정.

## 3. ★subtract 잔차 + anti-targeting 1차 합격선
- **subtract**: τ²-요구 {P1,P2b,P3,P4,P5,(P6,P7 유보),P8} − (SOP∪TB∪CFB 커버 {P1,P2b,P3,P5,P8 +cfb P-fetch}) = **잔차 = P4(select-from-output·전이미검증) [+ proactive-gather P2b/R4]**.
- **일반-primitive 정당화(τ²-독립·세탁 방어 1차)**: P4는 매트릭스 *자체* 열거의 도메인-독립 control/data-flow 연산(τ²가 만든 게 아님). "어떤 selection 벤치도 change/keep/fallback 변형선택을 typed-gold로 안 가르침" = field-orphan 주장 → **딥리서치(`w3l415qh5`·재실행중)가 확증/반증**. 서술-합격선은 *1차*일 뿐·진짜 증명 = **§8 multi-target blind 전이**(2차 held-out selection 분포).
- ⇒ synth 배타영역 = P4($select). 실벤치 재추출은 P-gate/thread/fetch만 공급(에피소드 한계 = 분업표를 *단단히* 만듦).

## 4. 실험0가 박을 것 (요약)
재추출 {concrete-target vs typed-target} matched 쌍 → per-provenance split:
- **passive-$ref 개선 ∧ $select 평탄** = (i)타깃-레벨이 추출가능 부분($ref grounding)의 binder (ii)P4($select)가 추출불가=synth 배타 **치수**. ★음성·치수까지만(synth 작동은 §7 factorial).
- 오독가드: 2-way 필수(passive-$ref / $select)·proactive는 withhold 모드 별도. 최소버킷($select) 기준 n.
