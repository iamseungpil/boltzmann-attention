# A1-v3 구현 설계서 (코드 레벨) — 2026-07-13

> ★설계(`A1_V3_DESIGN` §2·§8)를 **코드**로. 목적 = 이 세션 반복 버그(dotted-path·shared-record·numeric-FP·unit≠라이브) 사전 차단 + per-레버 함수/배선/테스트 명세 + §8 리뷰 게이트(R1 field·R5 numeric-FP) 코드 내장.
> 최대 기대 realistic ≈ 0.72-0.75(설계 §8h·0.92 아님). 구현 목표 = **realization이 설계 상한에 gap-누수 없이 도달**하는지 레버별 실측.
> 불변: [[05]]·[[10]]·[[09]] 무료 unit 先·라이브 probe로만 발화 확정·[[08]] probe는 per-case.

## 0. ★이 세션 반복 버그 체크리스트 (구현 前 필독·재발 방지)
| 버그 | 이번 발현 | 방지 |
|---|---|---|
| **dotted-path** | fexec가 `items.name` 점경로 형식화→`_field_values` 리터럴 키 미스→8/8 폴백 | formalize 출력 field에 점경로·leaf 폴백 지원(구현됨)·**variant도 재검** |
| **shared-record** | 후보가 한 출력에만 등장→record<2→filter 무효(t71) | GET-forcing(L2)로 후보 상세 강제 |
| **numeric-FP (R5)** | "8"이 size/count/id의 8과 충돌 | **field-타입 exact-match**·substring/set 금지(CENSUS §1) |
| **unit≠라이브 발화** | 단위 14/14 통과했으나 라이브서 dotted 폴백 | 각 레버 unit 後 **반드시 라이브 probe**·발화 마커 확인 |
| **set-x trace 오탐** | `grep WRITE_CAP`가 export 라인 계수(4)→실발화(0) 오독 | 마커 grep은 `grep -av '^+'`(비트레이스) |
| **집계→결론 직행** | cap "4발화" 집계 오판 | per-case 마커·궤적 확인 |

## 1. 구현 원칙
- **격리 토글**·기존 엔진 재사용·신규 후크 최소(regen deny 경로 재사용).
- **R1/R5 게이트 코드 내장**: field-결정론성 카운터·numeric field-type match.
- **무료 unit → 라이브 probe(표적+trivial 무회귀+FP 측정) → 통과분만 스택 편입**. 번들 금지.

## 2. ★L4 fexec-variants 구현 (probe-1·최고 ROI·R1/R5 집중)
### 2a. `_variant_records(new_item_operand, msgs)` — 후보 추출 (신규)
- get_product_details 출력의 `variants`(item_id-keyed dict) → `[(item_id, variant_record)]`.
- variant_record = `{item_id, options:{...}, price, available}`. **`available==true`만** 후보.
- ★pitfall: 기존 `_candidate_record_dicts`는 order용(리스트 스캔·`_min_enclosing_record`). variants=dict-keyed라 **신규 추출기 필요**(dict.values() 순회). dotted-path 재검: `options.color` 각 variant 내 정합 확인(order와 구조 다름).
- [[05]]: variants=도구출력·A2 `variant_producer`(=get_product_details)·리터럴 0.

### 2b. L4a 극값어 사전 + ★R1 field-결정론 (핵심)
- `EXTREMUM_LEXICON` = **eval-blind 영어목록**(gold 미참조·R2): `{most expensive|priciest|dearest→(argmax), cheapest|least expensive→(argmin), largest|biggest|greatest→(argmax), smallest|tiniest→(argmin), highest→(argmax), lowest→(argmin)}`. op만 확정.
- **★R1 field 해소 (분리계상)**:
  ```
  op = lexicon_match(request)              # 극값어 → argmax/argmin (결정론)
  numeric_fields = [f for f in variant record if 값이 수치]   # price + 수치 options
  if len(numeric_fields)==1: field=그것 (결정론) ; mark field_deterministic++
  elif op word가 field 명시("cheapest GREEN"의 색 아님·"biggest SCREEN"): field=명시 (결정론)
  else: field=formalize(LLM) or ASK       # ★field 잔여 = formalize (R1 인정) ; mark field_formalize++
  ```
- **측정 훅**: `_t2_l4a_field_det` / `_t2_l4a_field_form` 카운터 → probe서 field-결정론 비율 실측(R1 게이트: 낮으면 극값 이득이 op에 한정됨을 정직 계상).
- "expensive→price"는 대개 단일 수치(price)라 결정론·"largest"는 다중 후보(size/capacity)면 formalize.

### 2c. L4b 속성매칭 + ★R5 numeric field-type match (FP 차단)
- 요청 토큰 ∩ variant option 값. **단 R5 강화**:
  ```
  for tok in request_tokens:
    for (opt_key, opt_val) in variant.options:
      if tok가 수치:  # ★"8"·"128" 등
          match = (opt_val이 수치 ∧ 단위/키가 tok 문맥과 정합)  # size:8 ○ / id·price 속 8 ✗
      else:           # green·clicky
          match = (tok == opt_val, case-insensitive 완전일치)   # substring 금지
  ```
- **field-타입 화이트리스트**: 수치 토큰은 **option-키의 값이 수치인 키**에만·문자 토큰은 문자 option에만. `item_id`·`price`·record 메타는 매칭 대상 제외(옵션만).
- **FP 가드**: 한 토큰이 ≥2 option-키에 매칭되면(모호) → 그 제약 ASK(추측 금지).
- 측정 훅: `_t2_l4b_fp_ask`(모호→ASK 카운트).

### 2d. 배선·테스트
- 배선: disamb 경로서 operand이 A2 `variant_operand`(new_item_ids)면 `_variant_records`+L4a/b → fexec_filter_decide 재사용(op·field·constraints 주입).
- **unit**(무료): (i) `_variant_records` 추출·available 필터 (ii) L4a 단일수치=결정론·다중수치=formalize 분기 (iii) L4b 색=완전일치·**숫자 "8" FP 전수**(size:8 매치·id의 8 무시·모호→ASK).
- **라이브 probe**: t20·t0·t15·t79 + trivial 무회귀. **발화 마커**(`[T2_L4]`) 확인·field-결정론 비율·FP-ASK 카운트 실측 = R1/R5 게이트.

## 3. L2 GET-forcing 구현
- `fexec_filter_decide` 반환에 **`status="need_get"`** 추가: records<2 이나 `_grounded_candidates`로 ≥2 order-후보 존재 ∧ op≠none(selection 기준) 시.
- 배선: need_get이면 미조회 후보 order-id에 **detail-read deny 피드백**(eplan-L2 텍스트 재사용·selection 문구) → 에이전트 조회 → 다음 턴 filter 재시도.
- **상한**: `T2_GETFORCE_CAP`(sim당·기존 eplan_deny_cap 동형·루프 방지).
- ★examined-safe 충돌(설계 §1a·R): op≠none이면 selection→GET-forcing·examined-safe는 op==none만. **단 §1a 리뷰반박 미해소** — 구현 前 (order op≠none ∧ 다품목 examined) 114-전수 확인.
- unit: need_get 판정(후보≥2·미조회)·cap. probe: t83·92·112·72·109.

## 4. L3 origin-provenance 구현
- `_origin_index`: ctx 구축 시 `{value: (turn_idx, role)}` 최초 등장 기록(tool/user/assistant).
- `_first_fab_call` 확장: 값∈ctx이나 `origin_role=="assistant"` ∧ producer getter 존재 → fab 취급(getter 지목 피드백). toggle `T2_PROV_ORIGIN=1`.
- ★리뷰 caveat(설계 §L3): (a) getter 호출이 write보다 늦으면 write시점 tool-first 아님→오탐 (b) user가 assistant 추측 승인=정당. **Δspurious로 (b) 오차단 계측**.
- unit: t96(assistant-first→deny)·t43(user-first→allow)·getter-재진술(tool-first→allow). probe: 주소 태스크 + Δspurious.

## 5. L7 precondition / L9 set-op 구현
- **L7**: `a2/retail.gate.json` gates[]에 preconditions 추가(엔진 무수정·GateInterpreter 재사용): exchange/return→`status==delivered`(resolver_path=[order_id,get_order_details,status]). split-payment→2 결제수단 금지. **정책→gate_spec 손저작 아닌 도출 검증**(LOCK make-or-break). 오프라인 gate unit(pending exchange deny).
- **L9**: `parse_formalize` OPS에 `complement` 추가·`execute_formalized`서 `order.items − constraint-매칭` = 잔여 item_ids. unit: "X 빼고 전부"→set-여. probe: t108.

## 6. 구현·검증 순서 (per-레버 격리·[[09]])
1. **L4** (probe-1·R1/R5 게이트) → field-결정론 비율·숫자-FP 실측 = 0.92-경로 조기판별.
2. **L2** GET-forcing (order-selection 완성) → §1a 충돌 114-확인 先.
3. **L3** origin-prov → Δspurious(정당승인 오차단) 게이트.
4. **L7** precondition → 정책→spec 도출 검증.
5. **L9** set-op.
- 각 단계: 무료 unit → 라이브 probe(표적+trivial 무회귀+Δspurious+**측정훅**) → 통과분만 편입. **A1-v2 번들 혼재 교훈**(per-레버 순효과).

## 7. 리뷰 대상 (self + user·구현 前)
- **R1 게이트**: L4a field-결정론 비율이 극값-이득의 실질을 결정 — 다중수치필드 태스크 전수 필요.
- **R5 게이트**: L4b 숫자 field-type match가 CENSUS §1 트랩 실제 차단하나 — t15 "8" FP 전수.
- **§1a 미해소**: examined-safe↔GET-forcing 직교성 114-확인.
- **L3 Δspurious**: 정당 user-승인을 origin=assistant로 오차단 계측.
- **측정훅 필수**: 각 레버가 "발화했나"(라이브)·"결정론이었나"(field)·"FP났나"를 카운트 안 하면 realization 미측정=gap-누수 재발.
