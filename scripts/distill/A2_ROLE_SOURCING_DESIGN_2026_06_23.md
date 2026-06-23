# A2 Role-Sourcing — 1안(scaffold 추론) vs 2안(행동탐지) 비교 설계 (2026-06-23)

> 진입: [05-fixed-vs-variable] · [A2_GENERALIZATION_DESIGN](A2_GENERALIZATION_DESIGN_2026_06_23.md) · 상위 [RULE_LEVER_COST_EFFICIENCY_PROGRAM](RULE_LEVER_COST_EFFICIENCY_PROGRAM_2026_06_22.md).
> 사용자 지시(2026-06-23): **원래 하려던 "A2 최소화 + scaffold 인자/role 추론"을 1안, 행동탐지를 2안으로 두고, 실험에서 효과·비용을 비교**하라. (이전 "scaffold 추측=더 나쁘다"는 단정이 아니라 1안의 *가설적 위험*으로 두고 실험이 판정.) 본 문서=구현 전 설계(리뷰 대상).

## 0. 목표
게이트(G1-G5)가 도메인마다 도구 role(write·user-scoped·owned-entity·handoff·precond-status)을 알아야 함. 목표 = **A2를 최소화하고 role을 정확 분류**해 전이를 "거의 빈 A2-swap"으로. **그 role 출처를 두 안으로 두고 실험으로 결정**(단정 금지).

## 1. 공통 기반 (두 안 동일)
### 1a. ★A2 = 단계별 *규칙*, 도구 리스트 아님 (사용자 지시 2026-06-23)
A2에 `write_tools:[...]` 같은 **도구 enumeration 금지**. 대신 **role에 대한 단계별 선언 규칙**:
```yaml
gate_rules:                              # 각 행동이 통과하는 단계(step-by-step)
  - step: identity      on: user_scoped       require: grounded_identity
  - step: confirm       on: write             require: latest_user_confirms
  - step: ownership     on: write & owned     require: target_owner == identity
  - step: precondition  on: write             require: target_status ∈ tool.precondition
  - step: notice        on: handoff           require: sent(notice_text)
constants:
  notice_text: "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."  # 유일 도메인 문자열
  owner_field: user_id                                                         # 기본값(override 가능)
```
- **gate_rules = 대부분 도메인-일반**(retail/airline/banking 공유). role(write/user_scoped/owned/handoff/precondition)은 *도구서 도출*. ⇒ **A2 = 규칙 + 상수 1~2개·도구목록 0** = "거의 빈 A2-swap"의 실체(전이=notice_text 교체).

### 1b. role 도출 (구조-읽기로 확정·추측 아님·검증완료)
- owned-entity(detail-getter 반환모델에 owner 필드 有=order·無=product) · user-scoped(인자에 owner-id) · ownership-path(owner-id→`get_<e>_details`→owner_field) · handoff(프레임워크 표준도구).
- **auth 분류 제거**: 인증=identity가 grounded(provenance)면 성립(retail lookup·airline user-provided 통합).
- ⇒ 두 안의 *유일 차이* = **write·precond role을 "도구가 하는 일"에서 어떻게 읽나** — 1안=**코드 정독(정적)**, 2안=**실행 관찰(동적)**. 둘 다 도구 자체서 도출(이름추측·tool-list 아님).

## 2. ★1안 — 정적 auto-read (도구 정의·스키마·소스를 *읽음*·실행 안 함·"원래 안")
- 사용자 의도 = **도구 자체에서 자동으로 읽어냄**. 출처 우선순위(읽기·추측 아님):
  - owned-entity = 반환모델 owner 필드(✅검증).
  - **write = 도구 *소스/AST*에 DB-변경 존재**(`order.status=`·`order.address=`·`.append(`·db 할당). = 도구가 *실제 하는 일*을 코드서 읽음.
  - **precond-status = 소스의 가드 `if <obj>.status != "X": raise` 파싱 → X**. (이름파싱과 달리 *실제 코드*라 정확: `modify_pending_order_address`=변경O→write·status가드X→precond없음, 둘 다 정확.)
  - (이름규약 = 소스 접근 불가시 *최후 fallback*만·기본 아님.)
- A2 = **거의 빔**(notice-text + owner_field 기본값). write/precond는 소스서 도출.
- **[[05]]/취약성(실험이 측정)**: 화이트박스(소스 접근 필요)·**AST 파싱 엣지케이스**(예: 빠른 정규식이 `order.address=` 놓침=under-gate 실증). robust AST 필요·드문 변경패턴 누락 위험. → §4 정확도/robustness가 정량화.
- 장점(가설): build~0·transfer~0(소스만 있으면 자동)·A2 surface 0·실행 side-effect 0.

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
- **★oracle = 독립 hand-verified**(순환 회피): 2안(행동)을 oracle로 쓰면 2안이 자동 일치=불공정. 그러니 oracle = **수동검증 write/precond-set**(retail=이미검증 write7·precond4 / airline·banking=1회 수동검증·연구비용·배포물 아님). 1안·2안 *둘 다* 이 독립 oracle과 대조(공정). 1안/2안 *불일치 케이스*도 별도 보고.
- **arm**: ①1안(정적 소스읽기) ②2안(행동탐지) → 각각 (a)oracle 대조 정확도(P/R) (b)그 role로 게이트 1-trial retail e2e → **false-block/pass/compliant** (c)build LOC/compute·transfer비용·robustness(rename/add 시뮬) 기록.
- **도메인**: retail·airline·banking. 3도메인 = 1안 소스-AST 엣지케이스(예 address)·2안 인자커버리지·전이비용을 *교차* 노출.
- **산출**: 1안 vs 2안 × 도메인 × 메트릭 표 → 채택안. = "정적읽기가 충분히 정확·싸면 1안 / 소스-AST가 전이서 깨지면 2안" *실증*. (gate_rules·A2-형태는 두 안 공통.)

## 6. Step-by-step build + 검정 (리뷰 후 구현)
- **S0** 본 설계 리뷰(현재).
- **S1'** `tool_roles.py` 공통 구조-읽기만 잔존(owned-entity/user-scoped/handoff/ownership)·**이름추측 write/auth 제거**. 검정: 구조 roles==hand-list(retail+airline)·`grep prefix-list=0`.
- **S2 (★A2 규칙형)** GateInterpreter가 **gate_rules(단계별 선언규칙)+roles**를 소비(도구목록 아님). gate.json→`gate_rules`+`constants`(notice_text·owner_field)로 축소. 검정: `grep tool-name in gate.json=0`·`--validate` 양도메인 PassA/B=0·retail census(elig/loop) 불변.
- **S3 role 출처 모듈**: `role_static.py`(1안=소스/AST: write=DB-변경 존재·precond=`status!=` 가드 파싱) · `role_behavioral.py`(2안=sandbox DB-diff·wrong-status). 검정: 각 실행·retail서 write-set 산출.
- **S4 oracle(독립·수동검증)**: retail=검증된 write7/precond4. airline·banking=1회 수동검증 기록(`role_oracle.md`). 검정: 1안·2안 산출을 oracle과 대조표.
- **S5 G1 auth 일반화**(grounded-identity·auth분류 제거). 검정: airline 런타임 auth 작동(미인증 deny·user-id grounded시 허용)·retail 무회귀.
- **S6 비용효율 실험**: 1안 vs 2안 × (retail/airline/banking) → §4 표(정확도·false-block·build·transfer·robustness). 검정: 표 산출.
- **S7** knee 안 채택 → A2 최소형 확정·전이실증(신규 도구목록 0).
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
