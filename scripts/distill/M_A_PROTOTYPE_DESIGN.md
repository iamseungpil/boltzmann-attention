# M-A 프로토타입 설계 — formalized-selector + 결정기 resolver (무재학습) — 2026-06-16

> 상위 = `ABOX_CONFIG_FORMALIZATION_DESIGN_2026_06_15.md` §9 마일스톤 M-A. 이론 = `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(P4-σ/P4-γ).
> 불변 = [[feedback-nl-formalize-llm-selection-deterministic]](LLM=NL→formalize·concrete=결정론)·[[feedback-selector-verifier-deterministic]].
> 문헌 정합 = `deepresearch/dr_nl_to_formal_interface_granularity.md`(reference-emit 전이양성·grammar=form-not-meaning·learned-SELECT+결정론-RESOLVE).

## 0. 한 줄
**τ² retail exchange 1태스크류를 "LLM이 concrete id emit" 대신 "LLM이 provenance-typed selector(추상 criteria) emit + 결정기가 concrete resolve"로 재구성해, base 모델(무재학습)서 `new_item_ids` 값-정확성이 concrete-emit 대비 오르는지 싸게 검증.** = 아키텍처 1차 분리검증(M-A).
- **★프레이밍(리뷰 2026-06-16)**: M-A는 **"승리(B>A)를 기대하는 실험"이 아니라 root cause를 *fabrication인지 reasoning인지* 최종 가리는 *진단***이다. 설계 자신의 §4 fallback이 추론 잔여(size 분해)를 시연하고 그게 데이터서 *우연히* 통과하는 정황상, 변형 오선택의 본체는 resolver로 안 닫히고 **σ(NL→formalize) 학습**으로 갈 가능성이 높다 — §7⑥ 분해가 그 신호를 정량으로 준다. ⇒ **B>A·B≈A 둘 다 1급 결과**(승리=아키텍처 닫음 / 무차·wrong-criteria=σ 학습 필요 증명).

## 1. 가설 (사전등록)
- **H1 (주)**: formal-selector + 결정기 resolver의 `new_item_ids`(=write-벽 정밀원인·[[project-tau2-write-failure-rootcause]]) 정확도 > concrete-emit. 기제 = LLM이 **원하는 variant 옵션**(clicky·Google Home)만 emit·결정기가 옵션→item_id 결정론 매칭.
  - **★H1 기제 강등(리뷰 2026-06-16·데이터 확인)**: 아키텍처는 **fabrication(concrete id 날조)을 제거**하나 **criteria-오해석(추론)은 제거 *안* 함.** root cause([[project-tau2-write-failure-rootcause]])=order/item/payment 다 맞고 *new_item_ids만 틀림*=변형 *오선택*=**추론** 오류(날조 아님). resolver는 *틀린 criteria*도 충실히 resolve → **NL fallback을 "무엇 완화·무엇 유지"로 정확 분해**하는 어려운 추론은 *여전히 LLM*. ⇒ H1 정확 형태 = **"fabrication 제거; criteria-추론 잔여는 §7 분해지표로 별도 측정."** (설계 자신의 §4 fallback이 그 잔여 시연 — size 누락.)
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
                        "fallback":[{"switch":"clicky","backlight":"none","size":"full size"}]}},  // ★size 유지(완화 안 한 제약 보존)
    {"old_item":{"item_name":"smart thermostat"},
     "desired_variant":{"select_by":{"compatibility":"Google Home"}}}],
  "payment_ref":{"source":"original"} }
```
- ★LLM은 **옵션 criteria + 선호순서**만 냄(item_id 안 냄). order_id_hint는 NL에 주어진 리터럴(U-provenance) — 결정기가 검증·정규화.
- **★fallback 인코딩 수정(리뷰·데이터 확인)**: 원안 `{switch:clicky, backlight:none}`은 **size 제약 누락**=under-constrained(available clicky+none을 size 무관 매칭). NL "no backlight"=backlight만 완화·**clicky+full size 유지** → fallback에 `size:"full size"` 보존 必. ⚠**실 keyboard(20 variant)서 available clicky+none이 7706410293(full size) *하나*뿐 → 원안이 *우연히* gold 일치** = **worked-example 통과가 인코딩을 검증하지 *않음***(clicky+none+60% available였으면 틀림). 실데이터 키名=`switch type`(≠`switch`)→§5 값-역매칭 필수.

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
- **★⑥ B-실패 분해(리뷰·핵심)**: B 오답을 **(i) 틀린 criteria 방출=추론오류**(LLM이 fallback/criteria 오분해·예 size 누락) **(ii) resolver 매칭 FAIL**(값-역매칭 모호·키 disjoint 깨짐) **(iii) 정확 criteria인데 tie 모호**(복수 variant 충족)로 분류. ⇒ "아키텍처가 *돕나*(criteria<id) vs 추론 *재배치*만 하나" 판별. **wrong-criteria율이 root cause=reasoning인지 fabrication인지 최종 확정.**
- ★주의: **offline 값-정확성**(gold call 대비)이지 full τ² rollout 아님(user-sim·multi-turn 제외). M-A는 "값-정확성 분리검증"·rollout pass는 M-E.
- **★gold/availability 일관성(리뷰)**: resolver가 `available` 필터 → gold new_item_id가 *같은 db.json availability 상태*서 추출됐는지 확인(다르면 gold 불일치=거짓 음성). §8 gold 신뢰성과 연결.

## 8. Scope / 위험 (정직)
- **닫음**: controlled-vocab variant-select·entity-ref(order/item/payment)·선호순서 폴백(구조화 술어). = write-벽 정밀원인(변형 오선택) 직격.
- **안 닫음**: ①옵션 키名/값이 자유텍스트(enum 아님)면 매칭 모호(§6 값-역매칭으로 부분완화·잔여=predicate-select) ②NL이 product/옵션을 모호하게 말함(clarify 필요) ③payment "original" 의미해석 ④여러 variant가 criteria 동시충족(**tie-break 규칙 명시 必**: 가격? 첫매칭? gold 가정 확인).
- **★값-역매칭 disjointness 사전체크(리뷰·데이터)**: 값-역매칭(키名 모름→어느 키든 값 일치)은 **키 간 값空間이 disjoint해야** 정확(실 keyboard=switch type/backlight/size 값 분리라 작동·우연). **도메인별 cross-key 값-중복 사전 audit**(중복 시 역매칭 모호→키名 필수 or predicate-select). §8① 격상.
- **반례 가드 재확인**: 이득이 xgrammar(form)서 오면 가설 오귀속 → ablation C로 분리(resolver 없는 formal은 여전히 값 틀려야 정상).
- **전이는 M-A 범위 밖**: single-config(retail)만. multi-config 전이=M-C/M-D.
- **gold 추출 신뢰성**: tasks.json 액션이 항상 단일 gold variant인가(폴백 분기 시 gold 모호)? task별 검수 필요(소수=수동확인).

## 9. 구현 단계 (다음 작업)
1. **`ma_resolver.py`**: retail db.json 로드 + §5 resolve() + exchange_delivered_order_items 직접 호출(tau2 environment 재사용). 단위테스트 = task 0 gold 재현(new_item_ids 일치).
   - **★적대적 단위테스트(리뷰·우연-통과 가림 잡기)**: task 0 gold 재현만으로 부족(§4: available clicky+none이 full-size 1개뿐이라 *size-누락 fallback도 우연 통과*). **db.json을 인위 교란**: ①clicky+none+full을 unavailable로 막고 ②clicky+none+**non-full**을 available로 추가 → **size 누락 fallback이면 틀린 variant(non-full) 선택·size 보존 fallback이면 FAIL/정답**. = fallback 인코딩(§4 size 보존)이 *실제로* 검증되는지 분리. resolver는 교란 db서도 §4 수정 인코딩이 옳게 동작해야 통과.
2. **`ma_gold_extract.py`**: tasks.json → exchange 태스크의 (NL, gold call) 추출·write-필요분 필터.
3. **`ma_eval.sh`**: base vllm(GPU0·guided_json) → A/B/C 3-arm 생성 → resolver 적용 → §7 지표.
4. **결과 박제**: `M_A_RESULTS.md`(권위본·[[feedback_results_master_doc]]) + 설계서 §7 예측 대조.
- 전송=git push/pull·드라이버=`scripts/distill/ma/`에 커밋([[reference-remote-server-environment]]).

## 10. 성공 기준 (사전등록)
- **강 성공**: B의 new_item_ids 정확율 ≫ A·이득이 ablation C 대비 resolver서 옴(xgrammar 단독 아님)·order/payment 회귀 없음.
- **약 성공**: B≈A이나 type-violation·FAIL이 진단가능(어디서 막히는지)·predicate-select 필요범위 정량.
- **음성**: B<A(resolver 매칭이 LLM emit criteria 오류를 증폭) → criteria-emit 자체가 어려운지(=NL→formalize σ 학습 필요·M-A가 그 신호) 분석.
- **★음성=diagnostic gold(리뷰 격상)**: B≈A이고 원인=wrong-criteria(§7⑥-i)면 = **root cause가 fabrication 아닌 NL→formalize *추론*(σ-학습 필요·thesis A2 front-end)임을 *증명***. M-A의 최대 가치 = "B>A 승리"보다 **§7⑥ 분해로 root cause(fabrication vs reasoning) 최종 확정** = σ 학습 필요여부 판가름. 음성도 1급 결과.

---

## 11. ★관련연구 / 선행연구 경계 — forced-format vs reasoning (2026-06-16·정독+코드검증)
> **목적**: 우리가 *무엇을 빌리고 무엇이 새로운지* 명확히. forced-JSON 교란(§7b)·CoT arm(Acot/Bcot)·structural_tag 처방은 **전부 선행연구 영역**. 우리 신규성은 그 *밖*에 있어야 한다. (전수 문헌·신규성 = 발주 딥리서치 `wf_3f814306-3e4`가 확정 — 도착 시 본 절 정련.)

### 11.1 forced-format이 reasoning을 해친다 (선행·검증된 현상)
- **"Let Me Speak Freely?"** (Tam et al., **EMNLP 2024**, `arXiv:2408.02442`): 포맷-제약 출력이 자유생성 대비 추론 저하 — 우리 NL→SQL DR서 corroborated.
- **GCD-as-logical-parser** (ACL 2025 Industry Track 34): grammar 강제는 executable rate↑이나 **대형 모델 few-shot서 semantic accuracy *역전*** = "grammar=form-not-meaning"([[reference-nl-formal-decouple-literature]]). DR서 "consistently improves semantic"은 REFUTED.
- Ye et al. 2025(constrained-decoding KL/분포 왜곡) — DR서 언급·1차 미정독(약).
- ⇒ **우리 M-A arm A/B(`guided_json` 전체출력 강제)는 정확히 이 regime** = "reasoning 실패"가 교란일 수 있음(§7b).

### 11.2 ★CRANE — "reasoning + constrained" 선행 해법 (직접 정독)
**`arXiv:2502.09061`·ICML 2025**·Banerjee, Suresh, Ugare, Misailovic, Singh. "CRANE: Reasoning with Constrained LLM Generation".
- **이론**: 출력을 *유한* 문법(유효 최종답만)으로 조이면 답을 **single autoregressive step**으로 내야 함 → 고정깊이 트랜스포머 = 고정깊이 회로(TC⁰급·약). CoT 중간토큰이 추가 계산스텝(다항시간 흉내)을 주는데 grammar가 그 스크래치패드를 금지 → 추론력 붕괴. (한계: *유한* 문법에만 증명.)
- **방법**: 증강문법 **Gₐ → R·G**(R=자유추론·G=최종출력). delimiter `S₁`…`S₂`로 **무제약↔제약 토큰단위 교대**(밖=자유CoT·안=grammar강제). logit 접근 필요(closed-API 불가).
- **결과**: GSM-symbolic·FOLIO서 **순수제약 *및* 무제약-CoT 둘 다보다 최대 +10pp**·최종 parse 100% 유효(Qwen2.5 1.5/7/14B·Llama3.1-8B·R1-distill·QwQ-32B).
- ⇒ **"자유추론 + 제약된 최종" 일반개념 = CRANE 소유(ICML2025).** 우리 신규성 아님.

### 11.3 ★vLLM `structural_tag` — CRANE 메커니즘의 제품화 (코드검증·우리 스택)
- **vLLM 0.11.0 + xgrammar**에 `StructuralTagResponseFormat` 내장(`{type:"structural_tag", structures:[{begin,schema,end}], triggers:[str]}`). trigger 전=자유생성·`begin`~`end`=schema(xgrammar) 강제 = **CRANE delimiter-gating을 API 인자로**(커스텀 코드 0). xgrammar `GrammarMatcher`도 `accept_token`/`fill_next_token_bitmask`/`rollback` 노출 → ③ 커스텀 게이팅도 가능(불요).
- ⇒ **strict-schema + 추론보존 = 우리 스택서 즉시 가능·기성기술.** arm **Bstag**(structural_tag)로 추가 예정 = Bcot(grammar 끔)보다 엄격유효성 유지하며 추론.

### 11.4 ★신규성 경계 (정직)
| 요소 | 지위 |
|---|---|
| forced-format이 추론 해침 | **선행**(Tam·GCD-parser·CRANE 이론) |
| 자유추론 + 제약된 최종(delimiter-gating) | **선행**(CRANE ICML2025·vLLM structural_tag 제품화) |
| in-schema reasoning 필드 | **선행**(관행·OpenAI 구조출력) |
| **provenance-typed *selector*(추상 criteria·concrete id 아님) emit** | **우리 후보**(딥리서치 확인중) |
| **결정론 *resolver*(selector→concrete) + selector/verifier 결정론 분담** | **우리 후보** |
| **agentic NL→tool-action 세팅서 selector+resolver+전이(ABox-swap)** | **우리 후보**(NL→SQL DR 커버리지 갭) |
- **결론**: 디코딩-레벨 처방(structural_tag·CRANE)은 *채택*이지 기여 아님. 기여 = **"무엇을 LLM이 emit하나(추상 selector)"의 분담·전이**. M-A는 그 분담을 *진단*(reasoning vs fabrication)·M-σ가 *학습*. 디코딩 처방은 reasoning을 *살려서* σ 학습/측정을 공정하게 만드는 *도구*.
