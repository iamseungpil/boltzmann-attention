# M-A 프로토타입 설계 — formalized-selector + 결정기 resolver (무재학습) — 2026-06-16

> 상위 = `ABOX_CONFIG_FORMALIZATION_DESIGN_2026_06_15.md` §9 마일스톤 M-A. 이론 = `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(P4-σ/P4-γ).
> 불변 = [[feedback-nl-formalize-llm-selection-deterministic]](LLM=NL→formalize·concrete=결정론)·[[feedback-selector-verifier-deterministic]].
> 문헌 정합 = `deepresearch/dr_nl_to_formal_interface_granularity.md`(reference-emit 전이양성·grammar=form-not-meaning·learned-SELECT+결정론-RESOLVE).

## 0. 한 줄
**τ² retail exchange 1태스크류를 "LLM이 concrete id emit" 대신 "LLM이 provenance-typed selector(추상 criteria) emit + 결정기가 concrete resolve"로 재구성해, base 모델(무재학습)서 `new_item_ids` 값-정확성이 concrete-emit 대비 오르는지 싸게 검증.** = 아키텍처 1차 분리검증(M-A).

## 1. 가설 (사전등록)
- **H1 (주)**: formal-selector + 결정기 resolver의 `new_item_ids`(=write-벽 정밀원인·[[project-tau2-write-failure-rootcause]]) 정확도 > concrete-emit. 기제 = LLM이 **원하는 variant 옵션**(clicky·Google Home)만 emit·결정기가 옵션→item_id 결정론 매칭 → "변형 오선택"(옛 id 재사용·날조) 제거.
- **H2 (보)**: order_id·payment_method_id도 selector(ref)로 두면 동일하게 정확(이미 concrete-emit서 grounded 정확 = 회귀 없음 확인).
- **H3 (전이 예고·M-A 범위 밖)**: 같은 selector 스키마가 SOPBench 등 타 도메인 config-swap서 동작(M-C/M-D).
- **반례 가드(문헌)**: xgrammar는 **type(form)만 보장·meaning 아님** → 값-정확성은 학습+resolver 몫. M-A서 정확도 이득은 **resolver의 결정론 매칭**서 와야지 xgrammar 단독서 오면 안 됨(분리 측정).

## 2. Authentic worked example (τ² retail tasks.json id=0·Yusuf Rossi)
- **NL**: "received order #W2378156, exchange the **mechanical keyboard** for same but **clicky switches**, and the **smart thermostat** for one **compatible with Google Home** instead of Apple HomeKit. **If no keyboard that is clicky+RGB+full size, go for no backlight.**" / known: Yusuf Rossi, zip 19122 / unknown: email.
- **Gold actions**: find_user_id_by_name_zip → get_order_details(#W2378156) → get_product_details(1656367028=kbd) → get_product_details(4896585277=thermostat) → **exchange_delivered_order_items(order_id, item_ids=[old_kbd,old_thermo], new_item_ids=[new_kbd_variant,new_thermo_variant], payment_method_id)**.
- **write-벽이 사는 곳**: `new_item_ids` = 원하는 옵션(clicky·Google Home)을 **해당 product의 variant 카탈로그에 매칭**·**폴백 선호규칙**(clicky+RGB+full → 없으면 무백라이트) 평가. concrete-emit이 틀리는 정확한 지점.

## 3. 데이터 모델 ground (tau2 retail `data_model.py`)
- `Product.variants: Dict[item_id, Variant]`; `Variant{item_id, options: Dict[str,str], available: bool, price}`.
- `OrderItem{name, product_id, item_id, options: Dict[str,str], price}`; `Order{order_id, user_id, items:[OrderItem], status}`.
- `User{user_id, payment_methods: Dict[id, {source,...}], orders:[order_id]}`.
- 함수: `get_order_details(order_id)→Order`·`get_product_details(product_id)→Product`·`get_user_details(user_id)→User`·`find_user_id_by_name_zip(...)`.

## 4. Formalized exchange 도구 스키마 (tools=로 제시·xgrammar guided_json)
```jsonc
{ "name":"exchange_by_intent",
  "parameters":{
    "order_ref":{"type":"entity_ref","resolver":"order","ref_by":["item_name|status=delivered"]},
    "exchanges":{"type":"array","items":{
       "old_item":{"type":"entity_ref","resolver":"order_item","ref_by":["item_name"]},
       "desired_variant":{"type":"variant_select",
          "vocab_source":"product.variants.options",       // 같은 product의 옵션空間
          "select_by":{"type":"object"},                   // {옵션키:값} 부분지정 허용
          "fallback":{"type":"array","items":{"type":"object"}}}  // 선호 순서(폴백 규칙)
    }},
    "payment_ref":{"type":"entity_ref","resolver":"payment_method","ref_by":["source"]}
  }}
```
**LLM 출력(xgrammar 강제·concrete id 0)**:
```jsonc
{ "order_ref":{"order_id_hint":"#W2378156"},
  "exchanges":[
    {"old_item":{"item_name":"mechanical keyboard"},
     "desired_variant":{"select_by":{"switch":"clicky","backlight":"RGB","size":"full size"},
                        "fallback":[{"switch":"clicky","backlight":"none"}]}},
    {"old_item":{"item_name":"smart thermostat"},
     "desired_variant":{"select_by":{"compatibility":"Google Home"}}}],
  "payment_ref":{"source":"original"} }
```
- ★LLM은 **옵션 criteria + 선호순서**만 냄(item_id 안 냄). order_id_hint는 NL에 주어진 리터럴(U-provenance) — 결정기가 검증·정규화.

## 5. 결정기 resolver 스펙 (결정론·grounded)
```
resolve(formal, env):
  order_id  = formal.order_ref.order_id_hint (NL리터럴) ; assert order∈user.orders ∧ status==delivered
  order     = get_order_details(order_id)
  for ex in exchanges:
    old = match order.items by name≈ex.old_item.item_name (정확/부분 string match)   # P4-γ
    item_ids.append(old.item_id)
    prod = get_product_details(old.product_id)
    cand = [v for v in prod.variants if v.available]
    # 선호순서 = [select_by] + fallback ; 각 criteria는 options⊇criteria 부분매칭
    chosen = first v in cand satisfying select_by ; else 순차 fallback ; else FAIL(report)
    new_item_ids.append(chosen.item_id)
  payment_method_id = match user.payment_methods by source (or 'original'=주문 payment_history)
  return exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)
```
- **분담**: provenance-SELECT(어떤 옵션·어떤 선호순서)=LLM / concrete RESOLVE(옵션→item_id·이름→item_id·source→pm_id)=결정기. = §4 worked-example(c) concrete call 재구성.
- **옵션 키 정규화**: NL "clicky"→variant options 키空間 매칭(키名 모를 수 있음→값으로 역매칭: 어느 옵션키든 값이 "clicky"인 variant). 자유텍스트 변형은 §8 위험.

## 6. xgrammar wiring (vLLM 0.11·무재학습)
- base = `Qwen/Qwen2.5-7B-Instruct` (어댑터 없음). serve GPU0(학습 GPU1 회피).
- `extra_body={"guided_json": <§4 스키마>}` 또는 `response_format=json_schema`. type-violation율 측정(0 기대=form 보장).
- ablation: (A) concrete-emit(현행 native-FC 프롬프트) (B) formal-selector+resolver(본안) (C) formal **xgrammar 끔**(학습 없이 conform하나=강제 vs 자발 분리).

## 7. 평가 프로토콜 (offline·값-정확성·무재학습)
- 대상 = τ² retail tasks.json **exchange 태스크 46개** 중 write-필요 부분집합(deterministic gold 추출 가능분).
- gold = tasks.json `evaluation_criteria.actions`서 exchange 액션의 (order_id·item_ids·new_item_ids·payment_method_id) 추출.
- **지표**: ①`new_item_ids` 정확율(주·집합일치) ②order_id·item_ids·payment 정확율(보·회귀가드) ③type-violation율(xgrammar=0) ④resolver FAIL율(매칭실패=정직 분모) ⑤end call 완전일치율. A vs B vs C.
- ★주의: **offline 값-정확성**(gold call 대비)이지 full τ² rollout 아님(user-sim·multi-turn 제외). M-A는 "값-정확성 분리검증"·rollout pass는 M-E.

## 8. Scope / 위험 (정직)
- **닫음**: controlled-vocab variant-select·entity-ref(order/item/payment)·선호순서 폴백(구조화 술어). = write-벽 정밀원인(변형 오선택) 직격.
- **안 닫음**: ①옵션 키名/값이 자유텍스트(enum 아님)면 매칭 모호(§6 값-역매칭으로 부분완화·잔여=predicate-select) ②NL이 product/옵션을 모호하게 말함(clarify 필요) ③payment "original" 의미해석 ④여러 variant가 criteria 동시충족(tie-break 규칙 필요).
- **반례 가드 재확인**: 이득이 xgrammar(form)서 오면 가설 오귀속 → ablation C로 분리(resolver 없는 formal은 여전히 값 틀려야 정상).
- **전이는 M-A 범위 밖**: single-config(retail)만. multi-config 전이=M-C/M-D.
- **gold 추출 신뢰성**: tasks.json 액션이 항상 단일 gold variant인가(폴백 분기 시 gold 모호)? task별 검수 필요(소수=수동확인).

## 9. 구현 단계 (다음 작업)
1. **`ma_resolver.py`**: retail db.json 로드 + §5 resolve() + exchange_delivered_order_items 직접 호출(tau2 environment 재사용). 단위테스트 = task 0 gold 재현(new_item_ids 일치).
2. **`ma_gold_extract.py`**: tasks.json → exchange 태스크의 (NL, gold call) 추출·write-필요분 필터.
3. **`ma_eval.sh`**: base vllm(GPU0·guided_json) → A/B/C 3-arm 생성 → resolver 적용 → §7 지표.
4. **결과 박제**: `M_A_RESULTS.md`(권위본·[[feedback_results_master_doc]]) + 설계서 §7 예측 대조.
- 전송=git push/pull·드라이버=`scripts/distill/ma/`에 커밋([[reference-remote-server-environment]]).

## 10. 성공 기준 (사전등록)
- **강 성공**: B의 new_item_ids 정확율 ≫ A·이득이 ablation C 대비 resolver서 옴(xgrammar 단독 아님)·order/payment 회귀 없음.
- **약 성공**: B≈A이나 type-violation·FAIL이 진단가능(어디서 막히는지)·predicate-select 필요범위 정량.
- **음성**: B<A(resolver 매칭이 LLM emit criteria 오류를 증폭) → criteria-emit 자체가 어려운지(=NL→formalize σ 학습 필요·M-A가 그 신호) 분석.
