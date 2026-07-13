# 통일 per-operand A2 해소 — 흩어진 레버 → 도구별 A2 하나 (2026-07-13)

> 사용자 지시(2026-07-13): "모든 도메인-특화 *선택* 문제(기권 포함)는 같은 종류 → 도구별 A2로 해결."
> = `GENERALIZED_SCAFFOLD_ARCHITECTURE_2026_07_12` §4d의 **미완 "통일 TODO"** 실행. [[16]] LOCK·[[03]] anti-drift.
> **자기교정**: 이 세션에 만든 L4-tie·L10·READALL·COV·TOOLERR = LOCK이 폐기하라던 **개별 개입레버**. 본 문서로 **per-operand A2 한 곳 + 고정 인터프리터**로 흡수·통일.

## ★★결정화된 일반 규칙 (2026-07-13 사용자·[[00]] 두-날개 논제의 최종형)
> **tool-use = {operand, operator} 각각의 의미 해소 4지선다 루프(GET→FIND→INFER/select→ASK/abstain)를, scaffold + A2 + learn 세 자원으로 푸는 것.**

| 자원 | 담당 | 성질 |
|---|---|---|
| **scaffold** (고정 엔진) | 루프 제어 + 전 결정론 검사: select 라우팅·predicate eval·provenance·gate·membership·cardinality diff·on_error 라우팅·abstain 종결 | 도메인 무수정·리터럴 0 |
| **A2** (ABox) | 전 도메인 포인터/값: getter·filterable_fields·resolve MENU(infer/ask 모드)·policy_default·on_error class·factor keys·tie rule | **로직 일반·정보만 특화** |
| **learn** (훈련 날개) | 환원불가 의미부 2개: ① formalize 정확도(NL제약→predicate / intent→operator) ② INFER 보정 | = fexec 0.79 병목 = learn 정량 타깃 |

- **한 루프·두 해소대상(operand=값 + operator=도구·§8b banking 확증)·세 자원.**
- **경계 = [[05]]/[[10]]**: 정보=A2 · 결정론 로직=scaffold · 의미(formalize/intent)=learn. "일반이냐"의 판정 = **로직이 A2만 읽나**(발화-여부 아님·사용자 2026-07-13).
- **기권(ASK/transfer/abstain)이 모든 분기의 안전 종결** — retail·banking gold 공통(§8d).
- **잔여 = learn**: scaffold가 결정론부를 다 사도 formalize/intent 정확도가 병목(v6=v2·banking gain0 실증·§6). = 두-날개의 learn 날개가 갚을 정량 격차.

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
- **모든 banking 실패가 루프 단계 실패로 환원** = 루프가 banking을 *완전 덮음*. **★리뷰 U1: 이건 서술적 커버리지 [S·거의 정의]이지 작동적 전이 증거 아님** — scaffold가 banking 결정가능분을 실제로 닫는가(=[D]·키스톤·현 실측 이득0)는 별개. "강증거"는 서술 층에만·전이는 미증명.
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

## 9. 리뷰 (결정화 논제 검증·3건·2026-07-13)
> 논제·§6 caveat·§8 banking 실측 = 견고. 무비판 승인 대신 3건 — U1이 핵심(중심 주장의 등급).

**[U1·중대·[[08]]] §8c "루프가 banking 완전 덮음 = 도메인-일반 강증거"는 서술적 커버리지와 작동적 폐쇄를 혼동.** 두 주장이 섞임:
- (a) **서술적**: 루프 5단계(GET/FIND/PROV/GATE/on_error)가 임의의 tool-use 실패를 *라벨링*할 수 있다 — 버킷이 망라적이면 **거의 항등(trivial)**·약증거. "완전 덮음"은 이것.
- (b) **작동적**: scaffold의 *결정론부*가 banking에 전이돼 **결정가능 잔여를 실제로 닫는다** — 이게 반증가능한 진짜 주장이고 [[11]] 키스톤.
§8c 자신이 "지배실패=GET=learn/scale·scaffold 못 엶"·§6 "banking gain0"을 인정하므로 **(b)는 아직 미증명(오히려 현 실측은 이득0)**. ⇒ "강증거"는 (a)에 붙으면 tautology-과대. **수정: §8c 헤드라인을 "서술적 일반성 [S·거의 정의]"와 "작동적 전이 [D·키스톤·미증명]"으로 분리**. 이 세션 반복 색출한 "서술 relabeling을 result로 읽음"([[08]]) 패턴 — 중심 논제에 재발 방지 필수.

**[U2·중대·[[11]] 가드 누락] "learn 날개=formalize/intent 정확도"가 도메인-타깃 학습으로 미끄러질 문 열림.** §논제 표는 learn 타깃=fexec 0.79만 적고 **학습이 도메인-일반이어야 함을 명시 안 함**. formalize를 retail 실패 예로 훈련하면 [[11]] 위반(retail-타깃). **수정: learn 자원에 가드 명문** — 타깃 = **스킬-클래스 추상**(NL→predicate·intent→operator를 도메인-일반 P-primitive로·four-bench/synth 합성·[[12]] 다양성)·retail/banking 엔티티·템플릿 학습 금지. [[00]] 두-날개의 learn 날개도 four-bench→τ² swap 불변([[11]]).

**[U3·중간·operator=operand 범위]** operator=operand 통일은 **banking discoverable 아키텍처서 실재**(§8b `agent_tool_name`=명시 인자)이나 **retail/airline은 direct-dispatch**(도구 선택이 인자 아님·에이전트가 직접 dispatch). retail의 "틀린 도구"(t21 exchange-on-pending)는 GET→FIND-over-tools 루프가 아니라 **L7 GATE 체크**로 처리된다. ⇒ "operator=operand"는 *보편 환원*이 아니라 **"도구가 discoverable일 때 operator-해소가 operand-해소로 표현됨·direct-dispatch서는 GATE"**. 통일 인터프리터가 **두 모드 다 처리**해야(§8d "도구-operand 추가"가 banking엔 맞으나 retail operator-선택엔 부적용) — 스키마에 `operator_resolution: discoverable|direct` 축 명시.

**종합**: 논제는 [[00]] 두-날개의 정당한 결정화·§6 caveat 정직. 단 U1(중심주장 등급 분리)·U2([[11]] learn 가드)·U3(operator 범위)를 반영해야 "강증거"가 tautology로, "learn"이 도메인-타깃으로, "operator=operand"가 과대환원으로 새지 않는다. 셋 다 반영 시 = 특허·전이·[[00]] 정합의 정직한 결정화.
