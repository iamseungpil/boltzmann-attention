# M-σ v2 (리뷰용) — 순수 추상 selection-by-criteria 합성 → 전이 측정 (C8 깨끗한 시험) — 2026-06-16

> M-D 1차 음성(§11: cfb-threading은 τ²-selection 못 함·over-$ref·provenance 미구분)이 가리킨 3요건을 *owned 합성*으로 정면 해결. 상위 = `M_SIGMA_DESIGN_2026_06_16.md`·`THESIS_STATEMENT_2026_06_16.md`. 불변 = [[feedback-thesis-tbox-transfer-direction]](τ² 참조 금지)·[[feedback-nl-formalize-llm-selection-deterministic]].

## 0. 한 줄 + 왜 가장 깨끗한 C8 시험인가
**특정 논리기능(selection-by-criteria + provenance + 조건부 fallback)을 *순수 추상*(도메인 내용 0·랜덤 스키마) 데이터로 합성·학습 → held-out τ²(실도메인)에 무재학습 전이하는지 측정.** τ² 참조 0.
- **thesis-순정**: TBox = "논리기능"만 학습(도메인 0)·ABox-swap = 추상→실도메인. 전이되면 *학습한 게 도메인이 아니라 논리 불변량*임을 가장 직접 증명. (cfb조차 실도메인이라 덜 깨끗.)
- **M-D 음성 3요건 정면 해결**: ①selection-by-criteria *직접* 합성(orphan 해소) ②provenance(리터럴/copy/select 혼합)로 over-$ref 교정 ③harness 동반 수정.

## 1. ★합성 과제 (추상·controllable)
랜덤 "추상 도메인" 다수 생성, 각 예제:
- **스키마**: K개 속성(랜덤 중립명 `attr_<rand>`)·각 작은 값-vocab(랜덤 토큰). isotropization 내장(예제마다 새 스키마).
- **카탈로그**: M개 item = 속성값 조합 + item_id(랜덤) + available 플래그.
- **current item**: 카탈로그 중 하나.
- **NL 요청(★핵심 caveat 해결)**: "X를 [값]로 바꾸고 나머지 유지·없으면 [선호]로 완화"를 **자연어로 *패러프레이즈***(literal `attr=val` 아님). → NL→criteria *grounding*을 가르침(τ²의 "clicky switches"·synonym·"keep the rest"에 해당). 패러프레이즈 다양화.
- **gold**: {current ⊕ changes} 매칭 item_id(available)·미가용시 fallback 적용.

## 2. ★provenance 혼합 (over-$ref 교정)
한 call의 args를 3종 섞음 — 모델이 *구분*을 배우게:
- **LITERAL**: 요청에 명시된 값/id(NL서 직접) → 리터럴 emit.
- **COPY/threading**: 관측 출력의 단일필드 → `$ref`(cfb식).
- **SELECTION**: 다후보 중 기준매칭 → **selection-spec**(changes+fallback)·resolver가 item_id 선택.
- ⇒ 모델이 "언제 리터럴 vs $ref vs select"를 학습 → M-D의 over-$ref(order_id까지 $ref) 교정.

## 3. 출력 = provenance-typed formal-spec
```jsonc
{ "literal_arg": "<value>",                                  // LITERAL
  "ref_arg": {"$ref": "<obs_idx>#<path>"},                   // COPY
  "select_arg": {"$select": {"from":"<obs_idx>","by":{<attr>:<val>,...},
                             "fallback":[{<attr>:<val>}]}} }  // SELECTION-by-criteria
```
- 결정기 resolver: $select = 후보(from)서 by-criteria 매칭(+fallback)→item_id. $ref=path 해결. literal=그대로.
- 학습 타깃 = 이 provenance-typed spec(concrete 값 아님). 등방화로 표면 덮음.

## 4. 학습 + 전이 측정
- **학습**: 합성 데이터로 7B LoRA SFT(`lora_train_chat_toolcall.py`·타깃=provenance-typed spec). 추론 분포 정합(증분스텝 가능).
- **★전이 eval = held-out τ²**(M-D harness 수정판): τ² exchange를 obs+tools로 제시→모델이 $select(new_item_ids)·$ref(order/payment)·literal 혼합 emit→resolver→gold 대비 per-arg.
- **baseline 대조**: base 7B / cfb-M-σ(threading) / 추상-M-σ-v2 / 큰모델. **핵심 = 추상-v2가 τ² selection(new_item_ids)을 base/cfb-Mσ보다 올리나** + over-$ref 사라지나.

## 5. 사전등록 성공/실패
- **강 성공(C8 양성)**: 추상-synth 학습 7B가 τ² selection(new_item_ids) + provenance를 base/cfb-Mσ보다 *유의*하게 올림 = **순수 논리기능이 실도메인 전이** = thesis 헤드라인.
- **약**: in-dist(합성 held-out) selection은 배우나 τ² 전이 부분(어느 표면차원 미덮임 LODO 진단).
- **음성**: τ² 전이 0 = 추상→실도메인 grounding 갭(논리는 배우나 NL-표면 전이 안 됨) → 등방화 차원 추가 or grounding 필요. 음성도 1급(추상학습의 전이한계 박제).

## 6. 위험 (정직·리뷰 훅)
1. **★too-abstract / NL-grounding 갭(핵심 caveat)**: 합성이 `attr=val` 리터럴이면 NL→criteria *해석*을 안 가르침 → τ²의 "clicky"·synonym·"keep rest" 전이 실패. **처방 = §1 NL 패러프레이즈 필수**(추상 속성에도 자연어 요청). 그래도 실도메인 어휘(synonym "Google Home")는 ABox-제공 몫(학습 아님).
2. **합성 난이도 분포**: 너무 쉬우면(후보 2개) 천장·너무 어려우면 noise. 카탈로그 크기·기준 수·fallback 깊이 분포 설계.
3. **resolver $select 충실도**: 기준매칭+fallback 결정론 정확(tie-break 규칙)·M-A resolver 재사용.
4. **provenance 비율**: literal/copy/select 비율이 τ²와 동떨어지면 전이 약화 — 다양 비율로.
5. **전이 미보장**: 추상→실 전이는 C8 가설·이번도 음성 가능. 단 *깨끗한* 시험(도메인 0).

## 7. 구현 단계
1. **`synth_selection.py`**: 추상 도메인/카탈로그/NL-요청/gold + provenance-혼합 + 등방화 생성. round-trip 검증(spec→resolver→gold).
2. **resolver 확장**: $select(by-criteria+fallback)·$ref·literal 통합(M-A `select_variant` 재사용).
3. **SFT**(합성 데이터·7B LoRA).
4. **M-D harness 수정**(payment=값·$select 지원·n 확장) → 전이 eval(§4 baseline 대조).
5. 결과 박제(M_A_RESULTS §12).

## 8. 한 줄
**순수 추상 selection-by-criteria(+provenance+fallback·NL-패러프레이즈·등방화) 합성 학습 → τ² 전이 = C8의 가장 깨끗한 시험.** M-D 음성 3원인(selection orphan·over-$ref·harness) 정면 해결. ★성패 갈림 = §6-1 NL-grounding(추상에도 자연어 요청 필수).

---

## 9. ★합성 ablation 매트릭스 (단일 레시피 베팅 X·어느 축이 전이를 만드나) — 2026-06-16 리뷰
하나의 합성에 베팅 대신, **이미 이론화한 설계축을 통제 변주**해 *전이 구동축*을 측정. 각 config = 데이터 생성→7B LoRA SFT→M-D τ² 전이 eval. **baseline + 단일-knob-off**(전체 factorial 16개 회피·각 축 기여 격리):

| config | 등방화(iso) | NL-패러프레이즈 | provenance-혼합 | 추상도 | 측정 질문 |
|---|---|---|---|---|---|
| **B (baseline)** | ON | ON | literal+$ref+$select | random-token | 전체 레시피 전이? |
| **−iso** | **OFF**(고정 스키마명/값) | ON | mix | random | **등방화가 전이 구동? (§5.10 이론 실증)** |
| **−NL** | ON | **literal attr=val** | mix | random | NL→criteria grounding이 전이에 필수? (§6-1 caveat) |
| **−prov** | ON | ON | **$select-only** | mix | provenance-구분이 over-$ref 교정·전이? |
| (opt) +sem | ON | ON | mix | **weak-semantic**(중립영어) | 약한 의미근거가 전이 도움? |

- **결정 출력**: τ² 전이율을 config간 비교. **B 전이 ∧ −iso 미전이 → 등방화가 구동축**(이론 실증·헤드라인). −NL 미전이 → grounding 필수. −prov서 over-$ref 재발 → provenance 필수.
- **사전등록 예측**(우리 이론): iso·NL·provenance 모두 전이에 기여(특히 iso=§5.10 표면-불변)·−iso가 가장 크게 떨어짐(미덮인 표면 과적합).
- **비용**: 5 config × (빠른 합성 + 소형 LoRA SFT ~1h + M-D eval). 2-GPU 병렬 ~2.5-3h batch. 합성 작아 학습 빠름.
- **구현**: `synth_selection.py`에 knob 플래그(`--iso/--nl/--prov/--sem`)·`ma_synth_ablation_batch.sh`(config별 생성→학습→eval→집계). round-trip 검증 공통.
- ★**thesis 가치**: 등방화→전이는 현재 *이론 논증*(§5.10·analogy)·이 ablation이 **−iso vs +iso로 *실증***. 양성이면 등방화 라인이 추측→측정.
