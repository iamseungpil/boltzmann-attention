# M-σ v2 (리뷰용) — 순수 추상 selection-by-criteria 합성 → 전이 측정 (C8 깨끗한 시험) — 2026-06-16

> M-D 1차 음성(§11: cfb-threading은 τ²-selection 못 함·over-$ref·provenance 미구분)이 가리킨 3요건을 *owned 합성*으로 정면 해결. 상위 = `M_SIGMA_DESIGN_2026_06_16.md`·`THESIS_STATEMENT_2026_06_16.md`. 불변 = [[feedback-thesis-tbox-transfer-direction]](τ² 참조 금지)·[[feedback-nl-formalize-llm-selection-deterministic]].

## 0. 한 줄 + 왜 가장 깨끗한 C8 시험인가
**특정 논리기능(selection-by-criteria + provenance + 조건부 fallback)을 *순수 추상*(도메인 내용 0·랜덤 스키마) 데이터로 합성·학습 → held-out τ²(실도메인)에 무재학습 전이하는지 측정.** τ² 참조 0.
- **thesis-순정**: TBox = "논리기능"만 학습(도메인 0)·ABox-swap = 추상→실도메인. 전이되면 *학습한 게 도메인이 아니라 논리 불변량*임을 가장 직접 증명. (cfb조차 실도메인이라 덜 깨끗.)
- **★synth→*real* 이라 깨끗(synth→synth 아님)**: `EXPERIMENT_DESIGN.md:248` 경고 = "synth→synth 전이는 thesis 증거로 *약함*(vocab만 swap)". 본 설계는 **추상-synth → 실도메인 τ²** = 그 약점을 정확히 피하는 강한 형식. 음성이어도 "추상학습 전이한계"가 1급 결과.
- **M-D 음성 3요건 정면 해결**: ①selection-by-criteria *직접* 합성(orphan 해소) ②provenance(리터럴/copy/select 혼합)로 over-$ref 교정 ③harness 동반 수정.
- **이론 결박**: 추상화 = 표면군 저차원 불변량(Olver n−s·`EXPERIMENT_DESIGN.md:32`·§5.13-5.14). scale이 사는 건 암기·정확실행이지 추상화(저차원) 아님 → 소형이 저차원 selection-불변량 학습엔 용량 충분. 등방화(§9 −iso ablation)가 이 불변량을 *강제*하는지가 전이 구동 가설.

## 1. ★합성 과제 (추상·controllable)
랜덤 "추상 도메인" 다수 생성, 각 예제:
- **스키마**: K개 속성(랜덤 중립명 `attr_<rand>`)·각 작은 값-vocab(랜덤 토큰). isotropization 내장(예제마다 새 스키마).
- **카탈로그**: M개 item = 속성값 조합 + item_id(랜덤) + available 플래그.
- **current item**: 카탈로그 중 하나.
- **NL 요청(★핵심 caveat 해결)**: "X를 [값]로 바꾸고 나머지 유지·없으면 [선호]로 완화"를 **자연어로 *패러프레이즈***(literal `attr=val` 아님). → NL→criteria *grounding*을 가르침(τ²의 "clicky switches"·synonym·"keep the rest"에 해당). 패러프레이즈 다양화.
- **gold**: {current ⊕ changes} 매칭 item_id(available)·미가용시 fallback 적용.

### 1a. ★학습 타깃 = *구조적 selection 연산*, *어휘 grounding 아님* (확정 root cause 결박)
M-A 전수추적(`M_A_RESULTS.md §3`)이 확정한 write-벽 = **wrong_criteria = "변경 오계산"**("X만 바꾸고 나머지 유지" 오계산), **어휘 mis-mapping 아님**. ⇒ 합성이 가르칠 것 = **어느 속성이 changes·어느 게 keep·미가용시 fallback** 이라는 *구조적 selection 연산*. 이건 도메인-불변 논리기능 → 추상 토큰으로 충분히 표현·학습 가능(thesis-순정).
- **학습(LEARN)**: NL 구조-prose → {changes, keep-rest, fallback} criteria 추출 + provenance 판별(§2). 이게 selection-by-criteria 불변량.
- **제공(PROVIDE·학습 아님)**: 특정 값의 어휘 grounding("Google Home"→해당 item·"the black one"→color=black)은 **ABox/카탈로그가 제공**. 추상 synth는 이걸 가르치지 *않음*([[feedback-nl-formalize-llm-selection-deterministic]]·도메인-특정 사실=retrieval 몫). → 추상 토큰이 어휘갭을 안 만들어도 **정당**(어휘는 학습대상 아님). §9 −NL ablation은 이 경계의 *구조-prose 파싱* 필요성만 시험(어휘 아님).
- **함의**: 만약 τ² 전이 실패가 *어휘* 때문이면(구조 아님) → 추상 synth로 못 고침·ABox 어휘제공 라인 필요. 이 분리를 M-D autopsy(어느 arg가 구조 vs 어휘 실패)로 진단.

### 1b. ★obs-포맷 정합 (전이 표면 리스크)
모델이 보는 입력 = (NL 요청 + 관측 카탈로그/출력 as tool-obs + 도구 스키마). **합성의 obs 표현이 τ²의 obs 표현(tool 출력 JSON·관측 인덱싱)과 정합해야** $ref/$select의 `from`-인덱스 참조가 전이됨. 합성 obs 직렬화를 τ² harness 포맷으로 맞춤(§7 round-trip이 검증). 불일치=형식전이 실패 원인(M-D의 over-$ref도 일부 이것).

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

## 5. 사전등록 성공/실패 (★정량 bar — post-hoc 금지)
- **primary metric**: τ² **new_item_ids selection 정확률**(= confirmed-orphan 차원). 현 baseline: base 7B 0.41·cfb-Mσ 0.34(`M_A_RESULTS.md §11`·over-$ref로 *퇴화*). **secondary**: `all`-arg(현 base 0.41·Mσ 0.03) + over-$ref율(literal-arg를 $ref한 비율, 현 Mσ 高).
- **n / 노이즈**: 현 τ² n=29(±6pp·k/29 granularity). **bar 유효화 위해 harness n을 ≥50으로 확장**(§4·payment=값 수정 동반). bar는 확장 n 기준.
- **강 성공(C8 양성·헤드라인)**: 추상-synth 7B가 **new_item_ids ≥ base+2σ**(노이즈초과·≥+12pp 잠정) **∧ over-$ref ≈ 0**(provenance 학습) **∧ all-arg ≥ base 회복**(cfb-Mσ의 0.41→0.03 퇴화를 안 냄). = 순수 논리기능 실도메인 전이.
- **약 성공**: in-dist(합성 held-out) selection 양성(§9 round-trip) ∧ τ² new_item_ids만 부분 상승·다른 arg 미회복 → 표면차원 LODO 진단(어느 등방화 축 미덮임).
- **음성**: τ² selection 전이 0(base 미초과) = 추상→실 grounding 갭. **분기진단**: M-D autopsy로 (i)구조실패(criteria 오추출=등방화/난이도 문제) vs (ii)어휘실패(§1a·ABox-제공 라인 필요) 구분. 음성도 1급(추상학습 전이한계 + 어디서 끊기는지 박제).
- **必 음성가드**: cfb-Mσ가 base를 *망친*(0.41→0.03) 패턴이 추상-v2서 재발하면 = provenance/등방화로도 못 막음 → 합성-전이 라인 한계 박제(§9 −prov가 원인격리).

## 6. 위험 (정직·리뷰 훅)
1. **★too-abstract / NL-grounding 갭(핵심 caveat)**: 합성이 `attr=val` 리터럴이면 NL→criteria *해석*을 안 가르침 → τ²의 "clicky"·synonym·"keep rest" 전이 실패. **처방 = §1 NL 패러프레이즈 필수**(추상 속성에도 자연어 요청). 그래도 실도메인 어휘(synonym "Google Home")는 ABox-제공 몫(학습 아님).
2. **합성 난이도 분포**: 너무 쉬우면(후보 2개) 천장·너무 어려우면 noise. 카탈로그 크기·기준 수·fallback 깊이 분포 설계.
3. **resolver $select 충실도**: 기준매칭+fallback 결정론 정확(tie-break 규칙)·M-A resolver 재사용.
4. **provenance 비율**: literal/copy/select 비율이 τ²와 동떨어지면 전이 약화 — 다양 비율로.
5. **전이 미보장**: 추상→실 전이는 C8 가설·이번도 음성 가능. 단 *깨끗한* 시험(도메인 0).

## 7. 구현 단계
1. **`synth_selection.py`**: 추상 도메인/카탈로그/NL-요청/gold + provenance-혼합 + 등방화 생성. knob 플래그(`--iso/--nl/--prov/--sem`·§9). **round-trip 검증**(생성 spec→resolver→item_id == 구성한 gold; 100% 아니면 생성 reject). obs 직렬화=τ² harness 포맷(§1b).
2. **resolver 확장**(`ma_resolver.py`): $select(by-criteria match + fallback 순차 + availability 필터 + tie-break 결정론규칙)·$ref(path)·literal 통합. M-A `select_variant` 재사용·tie-break 규칙 명문화(§6-3).
3. **SFT**(합성 데이터·7B LoRA·`lora_train_chat_toolcall.py`·타깃=provenance-typed spec). config 간 #예제·step·LR 고정(§9 교란통제).
4. **M-D harness 수정**(`m_sigma_transfer_eval.py`): payment_method_id를 값으로(현 dict-키 아티팩트·§11b) · $select 채점 추가 · **n≥50 확장**(§5 bar 유효화) · per-arg-type + over-$ref율 + 구조/어휘 실패 라벨(autopsy). → 전이 eval(§4 baseline: base / cfb-Mσ / 추상-v2 / 큰모델).
5. **ablation batch**(`ma_synth_ablation_batch.sh`): B/−iso/−NL/−prov(+opt +sem) config별 생성→SFT→M-D eval→집계. 2-GPU·~2.5-3h.
6. 결과 박제(`M_A_RESULTS.md §12`)·§5 정량 bar로 판정.

## 8. 한 줄
**순수 추상 selection-by-criteria(+provenance+fallback·NL-패러프레이즈·등방화) 합성 학습 → τ² 전이 = C8의 가장 깨끗한 시험.** M-D 음성 3원인(selection orphan·over-$ref·harness) 정면 해결. ★성패 갈림 = §6-1 NL-grounding(추상에도 자연어 요청 필수).

---

## 9. ★합성 ablation 매트릭스 (단일 레시피 베팅 X·어느 축이 전이를 만드나) — 2026-06-16 리뷰
> ⚠️ **대체됨(2026-06-16)**: 이 §9의 single-knob-off(전부 ON 기준 하나씩 OFF = 조건부 marginal)는 **`M_SIGMA_V3_TRANSFER_FACTORIAL_DESIGN.md`의 2³ 완전요인 비교군**(각 축 *단독* 전이 + *조합* 전이)으로 격상. 실험설계 권위 = v3. 본 §1-8(substrate/provenance/root-cause 결박)은 v3가 substrate로 재사용.
하나의 합성에 베팅 대신, **이미 이론화한 설계축을 통제 변주**해 *전이 구동축*을 측정. 각 config = 데이터 생성→7B LoRA SFT→M-D τ² 전이 eval. **baseline + 단일-knob-off**(전체 factorial 16개 회피·각 축 기여 격리):

| config | 등방화(iso) | NL-패러프레이즈 | provenance-혼합 | 추상도 | 측정 질문 |
|---|---|---|---|---|---|
| **B (baseline)** | ON | ON | literal+$ref+$select | random-token | 전체 레시피 전이? |
| **−iso** | **OFF**(고정 스키마명/값) | ON | mix | random | **등방화가 전이 구동? (Olver 표면군 이론 실증·§0 결박)** |
| **−NL** | ON | **literal attr=val** | mix | random | NL→criteria grounding이 전이에 필수? (§6-1 caveat) |
| **−prov** | ON | ON | **$select-only** | mix | provenance-구분이 over-$ref 교정·전이? |
| (opt) +sem | ON | ON | mix | **weak-semantic**(중립영어) | 약한 의미근거가 전이 도움? |

- **★교란통제(held-constant·confound 차단)**: 모든 config 간 **#예제·SFT step수·LR·LoRA rank·카탈로그크기 분포·기준수(K)·fallback깊이 분포·M-D eval셋(동일 τ² n·동일 harness)** 고정. 변하는 건 해당 knob 1개뿐. (안 그러면 전이차가 knob 아니라 데이터량/난이도 confound.) round-trip 검증 공통 통과 필수.
- **결정 출력**: τ² 전이율(§5 primary=new_item_ids)을 config간 비교. **B 전이 ∧ −iso 미전이 → 등방화가 구동축**(이론 실증·헤드라인). −NL 미전이 → 구조-prose 파싱 필수(§1a·어휘 아님). −prov서 over-$ref 재발 → provenance 필수.
- **사전등록 예측**(우리 이론): iso·NL·provenance 모두 전이에 기여(특히 iso=표면군 저차원불변 강제)·−iso가 가장 크게 떨어짐(미덮인 표면 과적합).
- **비용**: 5 config × (빠른 합성 + 소형 LoRA SFT ~1h + M-D eval). 2-GPU 병렬 ~2.5-3h batch. 합성 작아 학습 빠름.
- **구현**: `synth_selection.py`에 knob 플래그(`--iso/--nl/--prov/--sem`)·`ma_synth_ablation_batch.sh`(config별 생성→학습→eval→집계). round-trip 검증 공통.
- ★**thesis 가치 + 이론 출처**: "등방화→전이"는 현재 *이론 논증*(표면군 저차원 불변량·Olver n−s·`EXPERIMENT_DESIGN.md:32`·§5.13-5.14·`olver_dimension_experiment.py` 1차=불변 저차원). 이 ablation이 **−iso vs B로 그 이론을 전이*결과*로 실증**(추측→측정). (주: v2 초안의 "§5.10" 참조는 M_SIGMA_DESIGN에 부재 → 위 Olver 라인이 실제 출처.)
