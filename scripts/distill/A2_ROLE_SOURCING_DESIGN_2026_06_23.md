# A2 Role-Sourcing — 1안(scaffold 추론) vs 2안(행동탐지) 비교 설계 (2026-06-23)

> 진입: [05-fixed-vs-variable] · [A2_GENERALIZATION_DESIGN](A2_GENERALIZATION_DESIGN_2026_06_23.md) · 상위 [RULE_LEVER_COST_EFFICIENCY_PROGRAM](RULE_LEVER_COST_EFFICIENCY_PROGRAM_2026_06_22.md).
> 사용자 지시(2026-06-23): **원래 하려던 "A2 최소화 + scaffold 인자/role 추론"을 1안, 행동탐지를 2안으로 두고, 실험에서 효과·비용을 비교**하라. (이전 "scaffold 추측=더 나쁘다"는 단정이 아니라 1안의 *가설적 위험*으로 두고 실험이 판정.) 본 문서=구현 전 설계(리뷰 대상).

## 0. 목표
게이트(G1-G5)가 도메인마다 도구 role(write·user-scoped·owned-entity·handoff·precond-status)을 알아야 함. 목표 = **A2를 최소화하고 role을 정확 분류**해 전이를 "거의 빈 A2-swap"으로. **그 role 출처를 두 안으로 두고 실험으로 결정**(단정 금지).

## 1. 공통 기반 (두 안 동일)
- **구조-읽기로 확정**(추측 아님·검증완료·retail+airline 일치): owned-entity(detail-getter 반환모델에 owner 필드 有=order·無=product) · user-scoped(인자에 owner-id) · ownership-path(owner-id→`get_<e>_details`→owner_field) · handoff(프레임워크 표준도구).
- **auth 분류 제거**: 인증=identity가 grounded(provenance)면 성립 → "auth 도구" 분류 불필요(retail lookup·airline user-provided 통합).
- **owner_field 이름**(`user_id`) = A2 기본값 1개(override·grep 가시).
- ⇒ 두 안의 *유일 차이* = **write 여부 + precond-status 의 출처**(tau2 미선언·구조신호 없음=환원불가 후보).

## 2. ★1안 — scaffold static inference (A2 최소·"원래 안")
- write/precond를 **도구 정의/스키마/이름에서 *정적 추론***(실행 안 함):
  - write = name prefix(modify_/cancel_/exchange_/return_/book_/update_/…) ∧ ¬read-prefix.
  - precond-status = write 이름의 status어(pending/delivered) 파싱.
- A2 = **거의 빔**(notice-text + owner_field 기본값). write/precond도 A2에 안 적음(추론).
- **[[05]] 가설적 위험(실험이 검증)**: 명명규약 가정을 *고정엔진*에 박음 → 규약 어기는 도구/도메인서 **silent 오분류→false-block**. 예: `modify_pending_order_address`(이름에 pending 有·도구는 status 미검사)→precond 오적용 가능. **이 위험이 *실제로* 발생/해를 끼치나 = 실험으로.**
- 장점(가설): build~0·transfer~0·A2 surface 0.

## 3. ★2안 — behavioral detection (행동탐지)
- write/precond를 **도구 실행 결과로 판정**(오프라인·sandbox DB):
  - write = 도구를 (gold 궤적의 실인자로) sandbox DB 깊은복사 위에서 실행 → **DB-diff 있으면 write**.
  - precond-status = 동일 도구를 *wrong-status 타깃*에 실행 → **에러나면 그 status가 precond**.
- A2 = 거의 빔(write/precond 0·자동탐지). 분류결과는 빌드시 캐시.
- **[[05]]**: A2=0·엔진=도메인-일반 프로브(실행이 ground-truth). 추측 아님.
- 위험(가설): 인자 커버리지(어떤 gold도 안 부른 도구=미분류)·side-effect 격리비용(DB 깊은복사)·프로브 자체 build복잡도.

## 4. 효과·비용 메트릭 (실험에서 측정)
방법(1안/2안) × 도메인(retail/airline/banking):
- **효과(정확도)**: 산출 write-set·precond-set이 **oracle과 일치?** precision/recall.
- **효과(다운스트림 harm)**: 그 분류로 게이트 돌렸을 때 **false-block rate**(옳은 write 차단)·pass/compliant Δ = 오분류가 *실제* 해를 끼치나.
- **build 비용**: 코드 LOC + 도메인당 compute-초/인간-분.
- **★transfer 비용(⑤일반화)**: *새 도메인*(airline/banking)을 0서 분류하는 한계비용.
- **robustness**: 도구 rename/add 시뮬 → 깨지나·*가시*(visible)냐 *침묵*(silent)이냐.
- knee = 정확도 충족 中 생애비용 최소 안. (1안이 정확도/robustness서 무너지면 비용무관 탈락·아니면 저비용으로 승.)

## 5. 실험 설계
- **oracle(ground truth)** = **전수 행동탐지**(2안의 완전판): 모든 write-후보를 *모든 gold 궤적 실인자*로 sandbox 실행→DB-diff. (gold가 실제 호출=인자확보·tau2 data 신뢰.) precond oracle = wrong-status 실행 에러여부. + retail은 검증된 hand-list(7 write)=2차 확인.
  - ⚠️ oracle 미커버 도구(어떤 gold도 안 부름)는 표기(2안 커버리지 한계 노출·보수처리 규칙 정함=§7).
- **arm**: ①1안 산출 ②2안 산출 → 각각 (a)oracle 대조 정확도 (b)그 분류로 게이트 1-trial retail e2e → false-block/pass/compliant (c)build/transfer/robustness 기록.
- **도메인**: retail(2차 oracle=hand-list)·airline·banking. 3도메인 = 1안 명명-깨짐·2안 커버리지·전이비용을 *교차*로 노출.
- **산출**: 1안 vs 2안 × 도메인 × 메트릭 표 → 채택안. = "정적추론이 충분히 정확·싸면 1안 / 전이서 깨지면 2안" *실증*.

## 6. Step-by-step build + 검정 (리뷰 후 구현)
- **S0** 본 설계 리뷰(현재).
- **S1'** `tool_roles.py` 공통기반 정리: 구조-읽기(owned-entity/user-scoped/handoff/ownership)만 + 1안용 정적 write/precond 추론을 *분리 함수*로. 검정: 구조부분 roles==hand-list·`grep`.
- **S2** `write_source(opt, domain)→(write_set, precond)`: opt∈{1안,2안}. 검정: 각 안 실행.
- **S3** oracle `role_oracle.py`(gold 인자·DB-diff·wrong-status). 검정: retail oracle==hand-list(7)·precond==검증4.
- **S4** G1 auth grounded-identity 일반화(auth분류 제거). 검정: airline 런타임 auth 작동.
- **S5** 비용효율 실험 드라이버 → §4 표(1안 vs 2안 × 3도메인). 검정: 표 산출·정확도/false-block 수치.
- **S6** knee 안 채택 → gate.json 최소 확정·전이실증.
- 각 S = 작은 단위·검정 통과 후 다음([[03]]).

## 7. [[05]] 정합 & 정직
- 두 안 다 [[05]] 합치 지향(엔진=도메인-일반·A2≈빔). **차이=write/precond 출처(정적추론 vs 행동), 우열은 실험으로**(사용자 지시).
- 1안의 진짜 위험(명명가정 엔진-박기)은 **기각 단정이 아니라 §4 정확도/robustness/transfer로 측정** → 깨지면 데이터가 기각.
- 2안의 "A2 0" 매력 뒤 숨은비용(프로브 build·커버리지·side-effect)도 §4가 드러냄.
- 미커버 도구 보수규칙(리뷰 결정): non-write 가정(놓침=under-gate) vs write 가정(과-gate)·기본=**비-게이트(under)** 두고 census로 누락노출 권장.

## 8. 리뷰 질문
1. 1안/2안 외 (M4 LLM-A2생성)을 참고-arm으로 넣을까(on-prem 반출 충돌이라 별도)?
2. oracle 미커버 도구 처리: under-gate(비게이트) vs over-gate 기본값?
3. precond를 write와 분리 측정 vs G5=oracle-probe로만?
4. 효과 동률 시 우선 기준(transfer비용 vs robustness vs surface)?
