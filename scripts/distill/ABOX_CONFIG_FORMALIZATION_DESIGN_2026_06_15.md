# ABox-config conditioned Formalization 아키텍처 설계 (리뷰용 DRAFT) — 2026-06-15

> 상태 = **리뷰용 DRAFT** (승인 후 구현). 진입점 = `HANDOFF_2026_06_15_pm.md`. 직전 진단 = `TAU2_FULLCHAIN_FIX_DESIGN_2026_06_15.md`(2-stage gate).
> 불변 = `feedback-thesis-tbox-transfer-direction`(SOPBench/TaskBench 학습·τ² held-out)·`feedback-selector-verifier-deterministic`(검증기·선택기=결정론)·`feedback-nl-formalize-llm-selection-deterministic`(LLM=NL→formalize·concrete=결정론).
> 메모리 = `reference-abox-config-formalization-architecture`.

---

## 0. 한 줄 요약
**ABox = JSON config. LLM은 config를 in-context로 읽어 NL→config-conformant formal 출력만 생성(=Function-Calling output-TYPE 학습). xgrammar(vLLM)가 type을 강제하고, 결정기가 formal→concrete를 해결하고, scaffold가 벤치 함수로 변환한다. 전이 = config(ABox) swap, LLM 재학습 0.**

---

## 1. 동기 — τ²-firefighting에서 thesis-아키텍처로 (이번 세션 진단 사슬)
1. **v8(P6)** τ² 0.10: 전수 census → 실패=Stage A **auth-날조**(find_user 식별자를 pretraining-prior `johndoe@example.com`로 지어냄·14/20). P6(write-confirm)은 무관(실패가 auth서 죽음). [`project-v8-p6-tau2-rootcause-confirmed`]
2. **v9(P6+DPO anti-fab)** τ² 0.05: DPO가 메커니즘은 옮김(날조14→10·grounded6→9·복구2→8) **but pass 불변** → "anti-fab 필요·불충분, 병목 하류 이동". [`project-v9-dpo-antifab-result`]
3. **write 벽 전수분석**: wrong_value 4건 = order_id·item_ids·payment **다 정확(grounded)**, 오직 **new_item_ids만 틀림** — 게다가 **get_product_details 호출함·쓴 id도 리턴에 있음(grounded-but-wrong)**. = 날조 아니라 **변형 *오선택***. [`project-tau2-write-failure-rootcause`]
4. **교정(사용자)**: 변형선택을 RLVR로 학습 = *offload할 정확-실행을 LLM에 내재화* = thesis 위배. **LLM은 NL→formalize까지, concrete 선택=결정론.**
5. **핵심 통찰(사용자)**: order_id `#W0000000`은 τ² **도구 스키마**(`tools.py: "order_id ... such as '#W0000000'"`)서 옴 → **학습-데이터 randomization으로 도달 불가**(출처가 평가벤치 스키마). 따라서 LLM이 배울 건 도메인지식이 아니라 **"config를 읽고 그 type에 conform하는 formalization"** 메타스킬.

**⇒ 결론**: 값-정확성 실패(order_id 날조·변형 오선택)는 전부 **"LLM이 concrete 값을 직접 emit"하는 잘못된 분담**의 증상. thesis-정합 = LLM은 formal 참조/속성만, concrete는 결정론 offload. 이를 **config(ABox)로 구동**하면 도메인 바뀌어도 전이.

---

## 2. 지배 원리
- **LLM** = NL→**formalize** (유한·저차원 추상·전이학습=TBox). *무엇을 의도하는가*.
- **결정기** = formalize→**concrete 해결 + 검증 + 게이트 집행** (무한 정확-실행·Rice/HRU 결정불가 → offload 필연). *정확히 어느 값/허용되나*.
- **오케스트레이터** = 결정론 **제어 흐름** (라우팅·순서·루프·예산·formal↔벤치 변환 scaffold).
- 세 행위자 전부 **ABox(JSON config)로 파라미터화**. **LLM만 도메인-일반(전이)**, 결정기·오케스트레이터는 config-swap으로 재구성.

**경계 원리 (어디까지 formalize):** formal 출력이 *(formal 슬롯 + ABox 데이터)의 결정론적 함수*가 되는 지점까지만. 그 너머(concrete 값)=결정기. ABox config는 각 param을 결정기가 결정론으로 풀 만큼 규정.

---

## 3. 역할 분담 — 축 × 행위자 매트릭스

| 축 | **LLM** (NL→formalize) | **xgrammar** (type강제) | **결정기** (concrete) | **오케스트레이터** |
|---|---|---|---|---|
| **계획(plan)** | 추상 plan DAG(단계·의존)=R4/R6 | plan 스키마 conform | plan 실행가능성·게이트적합 검증 | 단계실행·결과주입·상태 |
| **도구(tool)** | 추상 action 인식=의도 formalize | action enum conform | action→정확 tool명(A1)·getter 라우팅 | 현단계 허용 tool만 노출(distractor 억제) |
| **파라미터(param)** | param **의도** formalize(참조·속성·user리터럴) | param 스키마(enum/nested) conform | formal ref/변형→**concrete 값**·provenance 검증 | formal call→벤치 concrete call 변환 |
| **게이트(gate)** | (오프라인 A2) 정책 NL→GATE_SPEC | — | (런타임) GATE_SPEC 결정론 집행=R5/P5 | 비가역 write 전 게이트 호출·block 라우팅 |
| **검증(verify)** | 없음(생성기만) | — | type→arg-type→precond→provenance→replay | 실패시 차단·재라우팅 |
| **복구(recovery)** | re-plan(대안)=열린-search 잔벽 | — | 에러 결정론 분류 | 에러감지·동일호출 차단(G-loop)·구조화 전달 |

**핵심: formalize조차 2겹 분리** — TYPE conformance=**xgrammar(결정론·off-schema 불가)** / CONTENT 선택(어느 enum/ref)=**LLM(학습)**. LLM은 스키마 안에서 옳은 값 채우기만.

---

## 4. ABox config 스펙
ABox config = **JSON 함수-스키마**. A1(도구 카탈로그)+A5(출력문법)+resolver-spec 통합 단일원천(LLM·결정기 공유).

각 파라미터 type 3종:
| type | 정의 | LLM 출력 | 결정기 해결 |
|---|---|---|---|
| **controlled-vocab** | `{type:enum, values:[...]}` | NL→정규토큰(예 "남색"→`navy`) | (직접 사용) |
| **entity-ref** | `{type:entity_ref, resolver:<getter>, ref_by:[필드]}` | 참조 객체(예 `{item_name:"chair",status:"delivered"}`) | getter fetch→ref_by 매칭→concrete id |
| **variant-select** | `{type:variant_select, vocab_source:<getter>, select_by:[옵션]}` | 속성 객체(예 `{color:"navy",size:"L"}`) | vocab_source fetch→옵션 매칭→item_id |
| **user-literal** | `{type:literal, dtype:...}` | user 발화서 직접 추출 | (직접 사용) |

---

## 5. 구체 구현 (우리 스택·2026-06-15 검증)
- **vLLM 0.11.0** + **xgrammar** 사용가능(`SamplingParams.guided_decoding`·`structured_outputs` 존재). outlines 부재(xgrammar 기본). bad_words extra_body 기검증.
- **제시**: scaffold가 *formalized 도구 스키마*(ABox config)를 `tools=`로 모델에 노출. 벤치(τ²) 원본함수는 concrete id 요구하므로 **결정기 scaffold가 formal call→concrete 벤치 call 변환**.
- **강제**: `extra_body={"guided_json": <ABox 스키마>}` 또는 `response_format={"type":"json_schema",...}` → xgrammar가 conformance 보장.

### 5.1 워크드 예 (exchange)
```jsonc
// (a) ABox config — formalized 도구 (tools=로 제시)
{ "name":"exchange_by_intent",
  "parameters":{ "order_ref":{"type":"entity_ref","resolver":"get_user_details.orders","ref_by":["item_name","status"]},
                 "desired_variant":{"type":"variant_select","vocab_source":"get_product_details.variants","select_by":["color","size","material"]},
                 "payment_ref":{"type":"entity_ref","resolver":"get_user_details.payment_methods","ref_by":["type"]} } }
```
```jsonc
// (b) LLM 출력 (xgrammar 강제·concrete id 없음)
{ "order_ref":{"item_name":"office chair","status":"delivered"},
  "desired_variant":{"color":"navy","size":"L"}, "payment_ref":{"type":"credit_card"} }
```
```jsonc
// (c) 결정기 → 벤치 concrete call
exchange_delivered_order_items(order_id="#W2890441", item_ids=["8069050545"],
  new_item_ids=["1071497737"], payment_method_id="credit_card_1061405")
```

---

## 6. 학습 레시피 (현 native-FC와 차이)
- **현재 잘못**: concrete id를 emit하도록 학습(틀린 level) → full-catalog 붕괴·변형 오선택([`project-nativefc-fullcatalog-collapse`]).
- **올바름**:
  - 입력 = NL + **ABox config(in-context tools=)**.
  - 지도타깃 = **config-typed formal 출력**(슬롯·참조, **concrete id 아님**).
  - **여러 config(도메인)에 걸쳐** 학습 → "임의 config conform" 메타스킬.
  - 데이터 = SOPBench/TaskBench 궤적을 (NL, config, formal-출력) 삼중쌍으로 재구성(결정기로 gold formal 역생성).
- **xgrammar는 학습 불요**(decode 강제) — 단 학습이 *스키마 안 옳은 content*를 가르침.

---

## 7. 평가 / 사전등록
- **전이 헤드라인**: **held-out config(새 ABox)** 에서 LLM이 config 읽고 conform하는 **content 정확도**(slot-fill F1·entity-ref 정확도). = config-conformance 전이.
- **기제 지표**: ①TYPE-위반율(xgrammar로 0 보장 확인) ②CONTENT 정확도(올바른 enum/ref) ③결정기 resolve 성공률 ④end-to-end 벤치 pass(부산물).
- **ablation**: (i) concrete-emit(현행) vs formal+resolver (ii) config in-context 유무 (iii) single-config vs multi-config(전이).
- **예측**: formal+resolver가 concrete-emit 대비 값-정확성(order_id·item_id) 급상승·full-catalog 붕괴 해소. multi-config 학습이 held-out config 전이.

---

## 8. ★열린 질문 (리뷰 훅)
1. **경계 결정**: 어떤 param을 entity-ref(결정기)로, 어떤 걸 LLM-literal로? 판별 = provenance(user발화 vs tool/카탈로그). 경계가 애매한 값(예 user가 부분속성만 말함)은?
2. **ABox 규정 granularity**: config를 얼마나 세밀히? 너무 거칠면 결정기 부담↑(모호), 너무 세밀하면 config 작성비용↑·LLM 부담↑. 최적점?
3. **전이 학습성**: SOPBench/TaskBench에 "config 읽고 conform" 학습신호가 충분한가? entity-ref/variant-select analog이 두 벤치에 있나(없으면 메타스킬 학습 불가).
4. **A2(정책)은 이 틀 밖**: 정책 게이트는 type-conformance보다 깊은 의미컴파일(NL→GATE_SPEC). config-formalization으로 안 닫힘 = 별 라인(A2 front-end·thesis 유일 난제).
5. **scaffold 변환의 충실도**: formal→벤치 concrete 변환에서 결정기가 틀리면(resolver 버그·매칭 실패) 책임 분리. 결정기 검증 정밀도 선결.
6. **벤치 호환**: τ²·SOPBench·TaskBench 함수 시그니처가 다른데 scaffold가 도메인별 formal-스키마+resolver를 ABox서 자동생성 가능한가(A1 기계파생 범위).
7. **xgrammar 강제 vs 학습**: type을 xgrammar로 강제하면 학습은 content만 — 강제 없이 학습만으로도 conform하나(강제는 프로덕션 안전망·학습은 전이 본질) 분리 측정?
8. **non-enum 속성**: select_by 속성이 enum화 불가(자유 텍스트 변형)면? variant-select 적용 한계.

---

## 9. 마일스톤
- **M-A (프로토타입·무재학습)**: exchange 1개를 formalized 도구로 정의 + 결정기 resolver + xgrammar guided decoding → τ² 소수 태스크서 "formal+resolver가 concrete-emit보다 값-정확?" 확인. base 모델로(학습 전).
- **M-B (config 자동생성)**: A1(도구 카탈로그)서 ABox config(entity-ref/variant-select/resolver) **기계 파생** 파이프 — 도메인별 자동.
- **M-C (config-conditioned 학습데이터)**: SOPBench/TaskBench 궤적 → (NL, config, formal-출력) 재구성(결정기로 gold formal 역생성). multi-config.
- **M-D (학습+전이)**: config-conditioned SFT → held-out config 전이 측정(§7).
- **M-E**: 통합 + 결정기 검증기 through-line + 논문(config-conformance 전이 = TBox·offload 분담).

---

## 10. scope / caveat (정직)
- **닫히는 것**: controlled-vocab/entity-ref/variant-select로 떨어지는 값(order_id·item_id·payment) = 이 설계로 닫힘. xgrammar가 type 보장·결정기가 concrete.
- **안 닫히는 것**: ①A2 정책(의미컴파일) ②비-enum 속성 ③NL 모호→되물음(clarification dialogue). 이 셋은 별 처방.
- **전이 미보장**: config-conformance가 학습-전이되는지는 §7 측정 전 가설. 프로토타입(M-A)이 싸게 분리.
- **이 설계는 값-정확성(write 벽·order_id) 문제의 처방**이지, 상류 plan/gather 능력은 별도(기존 TBox R1-R8).
