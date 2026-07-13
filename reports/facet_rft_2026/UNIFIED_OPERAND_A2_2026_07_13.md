# 통일 per-operand A2 해소 — 흩어진 레버 → 도구별 A2 하나 (2026-07-13)

> 사용자 지시(2026-07-13): "모든 도메인-특화 *선택* 문제(기권 포함)는 같은 종류 → 도구별 A2로 해결."
> = `GENERALIZED_SCAFFOLD_ARCHITECTURE_2026_07_12` §4d의 **미완 "통일 TODO"** 실행. [[16]] LOCK·[[03]] anti-drift.
> **자기교정**: 이 세션에 만든 L4-tie·L10·READALL·COV·TOOLERR = LOCK이 폐기하라던 **개별 개입레버**. 본 문서로 **per-operand A2 한 곳 + 고정 인터프리터**로 흡수·통일.

## 0. 핵심 명제
모든 write는 **operand(인자) 값 해소**의 집합이다. 각 operand마다 문제는 항상 **같은 구조**:
> **후보 생산(GET) → 제약 평가(FIND) → {유일→사용 · 다수→선택기준|ASK · 공집합→재형식화|기권} → 무결성 검사 → 에러 시 복구|기권.**
도메인-특화는 "어느 getter·어느 제약·언제 기권"이라는 **정보**뿐 = A2. 로직은 하나 = 고정 엔진.

## 1. 도구별 per-operand A2 스키마 (통일형)
```
tool <T>:
  operand <arg>:
    # ── GET: 후보 생산 ──
    getter:        <tool> → <field>          # 후보 record/값 생산 (없으면 user 발화만)
    candidate_scope: product|order|account   # 스코프 앵커(L4 제품별 스코핑)
    # ── FIND: 제약 평가 (fexec = LLM formalize → 엔진 결정론 평가) ──
    filter:        fexec | none
    filterable_fields: [color, price, status, items, ...]   # 스키마 파생
    cardinality:   one | subset | all        # 요청이 커버하는 후보 수 (COV 흡수)
    tie_break:     min_price | none          # 다수 잔여 시 결정론 tie (미선언=ASK)
    # ── SELECT/ASK 행동 (도메인 MENU서 선택) ──
    resolve:
      on_ambiguous: ask_enumerate | ask_prompt_id | infer_positional   # ≥2 처방
      on_empty:     re_formalize | ask       # 0 처방
    # ── 무결성·근거 (PROV/BIND 흡수) ──
    provenance:    grounded | user_first     # 값이 출력∪발화 실재·에이전트-발명 금지(L3)
    member_of:     <entity_arg>.<container>.<id_field>   # 인자 간 정합(L10 멤버십)
    # ── 정책·필수 (GATE 흡수) ──
    policy_default: <값> | null              # 정책-강제(gate)
    required:      true|false                # 누락→ASK
    # ── 에러 처리 (TOOLERR 흡수) ──
    on_error:      [{match, class(recover|abstain), hint}]
```
- **한 곳**: getter·filter·cardinality·tie·resolve·provenance·member_of·policy·required·on_error 전부 A2. scaffold 도메인 리터럴 0.

## 2. 흩어진 레버 → per-operand 필드 흡수 매핑 (★통일의 실체)
| 이 세션 개별 레버 | 통일 필드 | 폐기/흡수 |
|---|---|---|
| L2 GET-forcing | `getter` (후보 미조회→GET 강제) | 흡수 |
| disamb(order/filter) | `filter=fexec` + `resolve.on_ambiguous` | 흡수 |
| L4 variant + L4-tie | `getter`(variant)+`filter`+`tie_break` | 흡수 |
| COV FIND-subset | `cardinality: subset\|all` | 흡수 |
| L10 멤버십 | `member_of` | 흡수 |
| L3 origin-prov / prov-rescue | `provenance: grounded\|user_first` | 흡수 |
| TOOLERR(recover/abstain) | `on_error` | 흡수 |
| L7 precondition / gate | `policy_default` + 기존 gate kind | 흡수 |
| L11 enum-carryover | `provenance`(대상-발화 attested) + `on_ambiguous:ask` | 흡수 |
| READALL | `cardinality`가 유발(subset이면 전 후보 read 필요) | **흡수·독립폐기** |
| G-noop | `filter` 결과가 현재값과 동일=무변경=재선택 신호 | 흡수 |
- ⇒ **개별 env 플래그(T2_L4/T2_COV/T2_READALL/T2_CONSISTENCY/T2_TOOLERR…) 전부 폐기** → 단일 `T2_RESOLVE=1` + per-operand A2.

## 3. 고정 인터프리터 루프 (엔진·도메인 무수정)
```
write 시도 → 각 operand a:
  spec = A2[tool][a]                              # 없으면 skip(우아한 강등)
  cand = GET(spec.getter, scope=spec.candidate_scope)         # 후보
  if spec.filter == fexec:
     pred = formalize(user_request, spec.filterable_fields)   # LLM 1회(유일 semantic)
     M    = eval(pred, cand)                                  # 엔진 결정론
  card = spec.cardinality
  # 선택/기권 (기권 = 모든 경우의 공통 종결)
  if card==one:
     |M|==1 → 값=M[0] ; |M|>=2 → tie_break? 유일화 : resolve.on_ambiguous(ASK) ; |M|==0 → resolve.on_empty
  if card in {subset,all}:  대상=M 전원 ; 미read 후보 있으면 GET 강제 ; 종료 시 M∖acted 리마인더
  # 무결성 (값 확정 후)
  if spec.provenance and 값이 grounded/user_first 위반 → prov-deny(getter 지목 or ASK)
  if spec.member_of  and 값 ∉ 대상.container → bind-deny(실제 컨테이너 지목)
  if spec.policy_default and 위반 → gate-deny/치환
tool 실행 후:
  if error matches spec.on_error → class=recover: 같은-인자 재발행/조기포기 deny+재시도지시
                                   class=abstain: 날조 deny·ASK/transfer 허용
```
- **유일 semantic 잔여 = formalize 정확도(fexec 0.79)** = learn 날개 타깃. 나머지 전부 결정론.
- **기권(ASK/transfer/abstain)이 모든 분기의 안전 종결** — 사용자 "기권 등 같은 종류" 정확히 반영.

## 4. A2 자동 도출 (거의 무료·[[05]] minimize-A2)
- getter/member_of/filterable_fields/required = **도구 스키마서 기계 도출**(거의 무료·엔진 아님).
- policy_default/on_error.class/resolve MENU = **정책 문서서 도출**(bounded opex·retail policy.md L84/L92 실증·banking GB1/GB2).
- provenance·cardinality = 요청별 formalize(LLM) — A2 아님.
- ⇒ 스키마분 무료 + 정책분 소량 = 특허 "operand-spec 스왑·엔진 무수정"의 원가 구조.

## 5. 전이 = per-operand A2 스왑 ([[11]] 키스톤·통일형이 정본 증명)
- banking/airline은 **자기 도구의 per-operand spec만** 선언 → 같은 인터프리터가 실행. 엔진 무수정.
- `test_v6_transfer`(READALL/COV/L10 banking-스왑 오프라인 5/5)가 부분 실증 → **통일 인터프리터로 승격 시 전 필드 동시 전이 증명**.
- **일반성 판정 = 로직이 A2만 읽나**(사용자 기준: 정보만 도메인-특화·로직 일반). 발화-여부 아님.

## 6. 정직 caveat (실측 반영)
- 통일이 **retail pass를 올린다는 보장 아님** — v6(개별레버)=v2 실증·banking=이득0·지배잔여=formalize(learn)/모델REACH. 통일의 가치 = **아키텍처 정합·전이 증명·특허형**(scaffold 리터럴 0·per-operand 스왑)이지 즉효 pass 아님.
- prompt-uncontrollable 결정론 잔여(§4e t71) = 통일해도 남음 → filter-substitute(결정론)가 처방·advise 아님.
- over-ask(≥2→ASK) 비용·recover/abstain 오분류 = Δspurious 계측 필수(정본 A1/실측).

## 8. ★banking gold/frontier 독립 확증 — 루프가 도메인-일반임 (사용자 지시 2026-07-13)
banking gold(97 task·독립 도메인)를 전수 분석 → 같은 GET→FIND→(select|ASK|abstain) 루프가 나타나는가. **결과: 나타남 + 핵심 일반화 발견.**

### 8a. banking gold 구조 (operator 분포·97 task)
- 지배 operator: `call_discoverable_agent_tool`(428)·`unlock_discoverable_agent_tool`(275)·`log_verification`(81)·`call_discoverable_user_tool`(62). = banking은 도구를 **KB 발견→unlock→call**하는 **간접(discoverable) 구조**.
- reward_basis: DB 88 / ACTION 9. 시퀀스 길이 1~33(median ~8-9).
- discoverable 도구 예: `update_transaction_rewards_3847`·`file_credit_card_transaction_dispute_4829`·`apply_statement_credit_8472`·`pay_credit_card_from_checking_9182`… (해시접미사 = 발견해야 하는 대상).

### 8b. ★핵심 발견: operator(도구) 선택 = operand 해소의 일반화
banking gold(task_026): `log_verification` → `unlock_discoverable_agent_tool{agent_tool_name}` → `call_discoverable_agent_tool{agent_tool_name}`.
- **`agent_tool_name`이 인자(operand)** — "어느 도구냐"가 해소 대상. retail은 operand=값(item/order), banking은 operand에 **operator(도구명) 자체가 포함**.
- 이 도구-operand가 **정확히 GET→FIND→select 루프**로 해소됨:
  | 루프 단계 | banking 실현 | retail 대응 |
  |---|---|---|
  | **GET** | `KB_search`/discoverable-목록 → 후보 도구명 | getter → 후보 값 |
  | **FIND** | 사용자 의도("rewards 안 맞음")→도구 매칭(update_transaction_rewards) | 제약→값 필터 |
  | **select/ASK** | 1매칭→unlock+call·모호→ASK·미발견→abstain/transfer | 1→use·≥2→ASK·0→재형식화 |
  | **PROV** | agent_tool_name이 KB출력에 grounded(발명 금지) | 값이 출력∪발화 grounded |
  | **GATE** | unlock-before-call(전제)·log_verification-before-access(auth) | confirm/precond gate |
- ⇒ **"intent 분석·operator 선택·계획"이 전부 operand 해소 루프의 인스턴스.** intent 분석 = 도구-operand의 FIND predicate. 계획 = operand 해소들의 의존순서(신원→도구발견→인자). 사용자 가설 정확히 확증.

### 8c. banking frontier/floor 실패모드 → 전부 루프 단계에 매핑 (BANKING_FLOOR_LEVER_FIT [M])
| banking 실패 (frontier·floor) | % | 루프 단계 |
|---|---|---|
| REACH/조립 미완 | 76.5% | **GET 실패**(도구 발견/unlock 체인 미완) |
| 도구명 날조 | 35.9% | **PROV 위반**(도구-operand 발명) |
| time_verified 날조 | 34.7% | **PROV 위반**(time-operand 발명) |
| EARLY_TRANSFER | 21.2% | **on_empty/on_error 오처방**(복구해야 할 때 조기 기권) |
| verify 무결성(누설·유령) | — | **GATE**(log_verification 유효성=B2) |
- **모든 banking 실패가 루프 단계 실패로 환원** = 루프가 banking을 *완전 덮음*(도메인-일반 강증거).
- 단 **지배 실패=GET(발견/조립)** = frontier도 동형(발견/조기중단 31%·[[47]]) = **모델 REACH 능력**(learn/scale), 루프-스캐폴드가 못 여는 축. scaffold 이득 축 = PROV(도구명/time 날조)·GATE(verify)·on_error(early-transfer).

### 8d. 결론 (사용자 질문 답)
- **같은 루프가 banking gold에서 독립적으로 나타남** — retail서 도출한 GET→FIND→(select|ASK|abstain)이 banking operator/operand/계획/intent를 완전 덮음.
- **일반화 1건 발견**: operand에 **operator(도구명) 자체**가 포함 — getter=KB/discovery. 이건 retail엔 없던 축이나 로직은 동일(값 대신 도구명을 해소). §1 스키마에 `operand=tool_name` 케이스 추가만으로 흡수(getter=KB_search·filter=intent-match).
- **abstain(기권)이 banking에서 gold의 정당 종결**(task_002/004/006 = KB 미발견→transfer가 gold) — 사용자 "기권 등 같은 종류" 확증·on_empty/on_error의 도메인-일반성.
- ⇒ 통일 인터프리터(§3)에 **도구-operand 해소**(operator 선택=KB getter+intent filter) 추가 = banking·retail 공통 루프 완성.

## 7. 구현 계획 (리팩터·번들금지)
1. **인터프리터 골격**: `resolve_operand(tool, arg, a2, msgs, am)` = §3 루프 1함수. 기존 부품 재사용(fexec_*·membership_violation·classify_tool_error·_origin_role·readall_unread·cov).
2. **A2 통일**: retail.gate.json에 per-operand spec 블록 추가(기존 variant_spec·gate_spec·tool_error_specs·eplan을 operand별로 재구성).
3. **단일 플래그** `T2_RESOLVE=1`로 개별 플래그 대체(하위호환: 개별 플래그는 deprecated alias).
4. **오프라인 검증**: 기존 retail gz replay로 per-operand 해소가 개별레버와 동치+Δspurious≤0 확인 → 그 다음 표적 probe.
5. **전이 증명**: banking/airline per-operand spec 저작 → 같은 인터프리터 오프라인 전이 unit → [[11]] [D]→[M].
