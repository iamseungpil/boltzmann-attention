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

## ★7b. Self-review 결함 (구현 前·5건·리뷰 라운드)
> §8 리뷰(R1-R5)와 같은 엄격도로 구현 설계 자체를 검증. 5건 발견 — I1·I2가 핵심(R1·R5의 코드-레벨 재확인·더 나쁨).

**[I1·중대] §2b field-결정론 의사코드가 자기모순 = L4a는 실질 op-only.** "numeric_fields = record의 수치필드; len==1이면 결정론"인데 **variant record는 price(항상 수치) + 수치 options(size·capacity·zoom) → len≥2가 거의 항상**. ⇒ "단일 수치필드=결정론" 분기가 **거의 발화 안 함** → field는 항상 formalize. "most expensive→price"조차 price+size 공존이라 이 규칙으론 formalize로 감. **R1을 코드가 더 강하게 확증** — L4a는 사실상 op만 닫고 field는 전량 formalize. **수정: 극값어→*의미 field-클래스* 매핑(expensive→price·largest→size/dim)이 필요한데 그게 R1이 지적한 field-formalize 그 자체**. ⇒ L4a 이득 = op-부하 감소분만·정직 계상(field 이득 0에 가까움). 0.92 경로 더 축소.

**[I2·중대] §2c 숫자매칭 "단위/키가 tok 문맥과 정합"이 미정의 = R5 미해결·재라벨.** "8이 size인지 count인지"를 코드가 어떻게 아나 — 명세 없음. 이건 formalize 의존을 "문맥 정합"으로 숨긴 것. **수정: 구체 규칙 = 요청서 토큰-인접 키명 파싱**("size 8"→"8" 앞 "size"→size 옵션에만 매치) or variant **옵션-키명이 요청에 등장**할 때만 그 키로 한정. 인접성 파서 명세 필요·없으면 R5 트랩 잔존.

**[I3·중대] 측정훅이 realization 아닌 *기전 발화*만 잰다.** §2b/2c/7의 카운터(발화·field-det·FP-ASK)는 **레버가 돌았나**를 셀 뿐 **정답 맞췄나**가 아님. 레버가 결정론으로 발화하고도 *틀린 field/값* 선택 가능. ⇒ 훅만으론 realization 미측정(gap-누수 재발). **수정: probe 판독 = 훅(기전) + per-case gold-diff(정답)** 병행 필수. 훅=기전 진단·gold-diff=realization.

**[I6·중대·반복버그류] variant dotted-path 미검증 = dotted-path 버그 재발 위험.** `_field_values` 점경로는 **order record서만** 테스트됨(t71). variant는 `options` 중첩·dict-keyed라 구조 다름 → `options.color` 순회가 order와 다르게 깨질 수 있음. **이 세션을 8/8 폴백시킨 바로 그 버그류.** 수정: **L4b 구현 前 variant record unit(점경로 `options.color`·`price` 추출) 필수**·라이브 발화까지 확인(unit≠라이브).

**[I7·중대] floor-guarantee 부재 = 레버가 baseline보다 나쁘게 만들 수 있음.** L4/L2가 결정론 발화 후 *틀린 값* 치환하면 **에이전트의 옳은 선택을 틀린 걸로 대체** = 순손실(T5C "레버≥floor pointwise" 위반). "동률·0통과→ASK"는 부분 커버·"결정론 픽이 틀린" 경우 미커버. **수정: 치환은 confident할 때만(단일 통과 ∧ 게이트 재검사 통과)·불확실=no-op(에이전트 선택 유지)**. Δspurious≤0의 코드-레벨 보증.

**종합**: I1이 R1을 코드로 재확인(field 이득 ~0) → **L4 이득 = op-부하 감소 + 속성filter(FP 가드 성공 시)만**·극값 field 이득 없음. **최대 기대 재하향**: L4 +3~5 → **+2~4**. ⇒ A1-v3 realistic 최대 ≈ **0.70~0.73**(0.75 상한도 낙관). 0.92는 완전 철회·L4는 여전히 최고 ROI(op+속성)지만 극값-field는 learn/formalize 몫.

## 7. 리뷰 대상 (self + user·구현 前)
- **R1 게이트**: L4a field-결정론 비율이 극값-이득의 실질을 결정 — 다중수치필드 태스크 전수 필요.
- **R5 게이트**: L4b 숫자 field-type match가 CENSUS §1 트랩 실제 차단하나 — t15 "8" FP 전수.
- **§1a 미해소**: examined-safe↔GET-forcing 직교성 114-확인.
- **L3 Δspurious**: 정당 user-승인을 origin=assistant로 오차단 계측.
- **측정훅 필수**: 각 레버가 "발화했나"(라이브)·"결정론이었나"(field)·"FP났나"를 카운트 안 하면 realization 미측정=gap-누수 재발.
