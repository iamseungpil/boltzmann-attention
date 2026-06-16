# M-σ 설계 (리뷰용 DRAFT) — 도메인-일반 NL→formalize 스킬 학습 + ABox-swap 전이 — 2026-06-16

> critical path #5(학습)→#6(전이=C8·thesis가 서고넘어지는 곳). 상위 = `THESIS_STATEMENT_2026_06_16.md`·`ABOX_CONFIG_FORMALIZATION_DESIGN_2026_06_15.md`. 불변 = [[feedback-thesis-tbox-transfer-direction]](SOPBench/TaskBench 학습·τ² held-out)·[[feedback-nl-formalize-llm-selection-deterministic]](LLM=formalize·concrete=결정론)·[[feedback-selector-verifier-deterministic]].

## 0. 목표 + ★v4-v7과 무엇이 다른가 (핵심)
**목표**: 작은 모델(7B)에 **도메인-일반 NL→formalize 스킬**(=M-A가 병목으로 확정한 "change-X-keep-rest + 조건부 선호 + synonym 매핑" 추론)을 학습시켜, **held-out config(τ²)에 무재학습 전이**(C8 양성)시킨다.

**★v4/v6/v7 native-FC 전이실패(τ² 0.05-0.10<base)와의 차이 — 이게 설계의 정당성:**
| | v4-v7 (실패) | **M-σ (제안)** |
|---|---|---|
| 학습 *레벨* | concrete-emit(order_id·item_id 직접) | **config-conditioned formal-spec**(typed 슬롯·참조·concrete 아님) |
| 결과 | full-catalog 붕괴·날조·과다호출 | concrete는 결정론 resolver가(LLM 안 냄) |
| 표면-불변 | 없음(naming/format 과적합) | **등방화**(tool명·값형식·속성명 randomize·§5.10) |
| 추론 구조 | 단발 emit | **typed 증분스텝**(Sstep scaffold가 inference서 검증·페어) |
| 검증 | 없음 | 결정론 per-step(Sstep) |
- ⇒ M-σ = v4-v7의 "concrete-emit 단발 monolith"가 아니라 **"formal-spec 레벨 + 등방화 + scaffold-페어"**. *같은 데이터 다른 학습타깃.*

### 0b. ★핵심 재프레임 (리뷰 2026-06-16) — 진짜 벽은 날조/표면이 *아니라* binding이다
위 4축(formal-레벨·등방화·scaffold·resolver)은 전부 **날조(fabrication)+표면 과적합** 실패족을 겨냥한다. **그러나 확정된 라이브 τ² write-벽은 그게 아니다**: [[project-tau2-write-failure-rootcause]] wrong_value 4건 = order_id·item_ids·payment 다 맞고 **new_item_ids만 틀림 = grounded-but-wrong 변형선택**. [[project-v9-dpo-antifab-result]]: anti-fab가 날조 14→10 줄였으나 **pass 불변**(병목이 grounded-but-wrong로 하류 이동). ⇒ **완벽히 typed·등방화·resolver-grounded인 spec을 내고도 *틀린 변형*을 고를 수 있다 — 4축 어느 것도 못 막는다.**
- **★5번째 축 = derivation/binding 정확성 (명시 추가)**: LLM이 concrete를 안 내는 건 맞되, **"old→new 관계 참조(derivation)"를 typed-reference로 emit**하고 resolver가 그 관계로 new_item_ids를 결정론 계산. 예: `new_item_ids := exchange_target(old_item_ids, requested_variant) via catalog` — LLM은 *관계/변형 의도*만 type-선택·concrete는 resolver가 catalog로 닫음. = grounded-but-wrong을 thesis-정합(LLM=관계선택·결정론=해결)으로 닫음.
- ⇒ **census·format·training·eval 전부 이 derivation/binding 중심으로 재정렬**(아래 §1-§6). 5번째 축 없으면 M-σ = "이해된 실패(날조/과적합)는 고치나 *실제 막힌* 실패(binding)는 안 고치는" 설계.

## 1. 학습할 것 / 안 할 것
- **학습(도메인-일반)**: (NL + config[tools/vocab/gates] + 관측상태) → **formal target-spec — slot=value 아니라 *typed-reference/derivation***(어느 *관계*로 new를 유도: `exchange_target(old, variant)`·선호순서). flat `changes={new_item_ids:X}`는 **틀린 변형선택을 그대로 재인코딩**(=binding 결함 재현) → 금지. = γ-grounding 메타스킬·"any config conform" + **관계선택**.
- **안 함**: 도메인 내용(카탈로그·특정 값)·concrete id(resolver offload)·정책 자체(ABox). 도메인 SFT/DPO/RLVR로 *도메인* 굽기 금지(과적합·미전이 실증).
- **방법 = SFT**(도메인-DPO/RLVR 아님·plan-selection DR: trajectory DPO=anti-transfer). 타깃이 도메인-일반이라 SFT가 전이 길러냄.

## 2. 데이터 = (NL, config, formal-spec) 삼중쌍·multi-config·등방화
- **소스**: SOPBench + TaskBench 궤적(τ²는 held-out·학습 금지). gold tool-call → **역생성으로 gold formal-spec**(M-A `ma_gold_extract`의 일반화: gold args를 typed-슬롯/참조로 추상).
- **multi-config**: 여러 도메인/config에 걸쳐 → "임의 config conform" 메타스킬(단일 도메인=과적합).
- **★등방화(C8 메커니즘·§5.10-5.11)**: 학습 중 *알려진 표면차원* randomize — tool명(alias)·값형식·속성명/순서·synonym surface. → 학습 스킬이 표면-불변 → config-swap 전이. (덮인 방향만 등방화·미덮인 표면은 잔여 비전이.)
- **출력 구조 = typed-reference/derivation 한 층(§0b)**: slot=관계참조(`exchange_target(old,variant)`)·concrete는 resolver. flat value-dict 금지. NL→SQL full coarse-to-fine sketch까진 불요(스키마-linking 부담을 LLM에 재적재) — **changes-dict + typed-reference 한 층이 최소충분**.
- **★학습 분포 = 추론 분포와 정합 (mismatch + P7 recovery 동시 닫음·리뷰)**: full-spec 단발 학습 ✗(추론은 검증피드백 조건부 증분스텝 = 모델이 못 본 분포·v4-v7 P7 recovery 갭). ⇒ **`(step_i | prior *verified* steps, observation, config, verifier-feedback)` 분포로 teacher-force** — recovery(에러/게이트 후 전략전환)가 바로 거기 산다. train=infer 정합. = Sstep scaffold 페어의 학습판.

## 3. ★선결 census (학습 전·싸고 필수) — ★primitive 아니라 *표면/binding* census (리뷰 재정의)
**게이트엔 강 동의(0원·최고가치·[[feedback-zero-cost-diagnosis-strongest-case]]). 단 무엇을 census하나가 결정적.**
- **★primitive-교집합 census 폐기**: 3벤치 7/9 primitive 공유였는데 τ² 전이 0([[project-cross-bench-transfer-plan]]) → **primitive census는 v4-v7을 *녹색불* 줬을 진단**(실패 예측 실패). 전이 장벽 = primitive 존재 아니라 **표면차원·binding**.
- **★재정의 census = orphan 표면 + orphan binding**: (a) **τ²의 exchange-변형 *binding*(old→new 관계유도)이 SOPBench/TaskBench 궤적에 등질로 존재하나** (b) **등방화로 덮이는 표면 vs τ²-orphan 표면**(naming/format/synonym 중 학습벤치에 없는 것) 열거. = 진짜 게이트.
- **★in-dist↑를 게이트로 쓰지 말 것**: v4 SOPBench in-dist 0.65인데 τ² 0(필요·오도신호). 게이트 = 표면/binding overlap이지 in-dist 점수 아님.
- **★orphan 발견 시 = 전이불가 *단정* 아니라 설계변수**([[feedback-no-fundamental-claims-from-convenience-data]]): orphan binding/표면 = **등방화 차원 *추가* 대상**(덮으면 전이)·범위 명시. "교집합 비면 중단"은 binding이 *구조적으로* 부재(유도관계 자체 없음)일 때만.

## 4. 평가 (M-D 전이 = 헤드라인)
- **in-dist**: M-σ가 학습-도메인(SOPBench/TaskBench) formalize 정확도 올리나(학습 작동 확인).
- **★전이(헤드라인·C8)**: held-out **τ²(retail/airline) config-swap·무재학습** → M-σ-7B의 NL→formalize(+Sstep scaffold) 정확도가 **base 7B 능가·큰모델 근접·도메인-SFT(과적합 baseline) 능가**.
- **baseline 대조**: base 7B / τ²-도메인-SFT(미전이 예측) / 큰모델(천장) / M-σ-7B(본안).
- **지표**: target-spec 정확도(slot/ref)·**★grounded-but-wrong write-pass**(옳은 변형/binding 선택율·날조율 아님·v9 대비 부가가치 핵심·§6-#6)·end-to-end resolved call 정확도·전이 Δ(held-out − base).

## 5. 사전등록 성공/실패 (정직)
- **강 성공**: M-σ-7B가 held-out config 전이·큰모델 근접·도메인SFT 능가 = **C8 양성·thesis 헤드라인**.
- **약**: in-dist↑·부분 전이(어느 표면차원서 막히나 LODO 진단).
- **음성**: 전이 0(C8 음성 지속) → 스킬이 표면-불변 아님/등방화 불충분 → **미덮인 표면차원 진단**(§5.10 coverage). 음성도 1급(어디가 비전이 핵인지 박제).

## 6. 위험 (정직·리뷰 훅)
1. **C8 현재 음성**(τ² 0.05-0.10) — M-σ가 뒤집는 시도·고위험. v4-v7과 다른 건 §0 4축이나 *전이 보장 아님*.
2. **스킬 공유성**(§3) — 선결 census 없이 학습=낭비 위험. *반드시 먼저.*
3. **scaffold work** — multi-config 공통 formal-spec 포맷 + gold 역생성 일반화(SOPBench/TaskBench)가 비-trivial. 
4. **등방화 coverage** — 미덮인 표면(τ²-고유 naming/format)은 전이 안 됨 → 알려진 차원만 덮음·잔여 정직 보고.
5. **결정론 분담 유지** — concrete는 절대 학습타깃 아님(resolver). 학습이 concrete로 새면 v4-v7 재현.
6. **★grounded-but-wrong 변형/binding 선택 (확정-블로킹·v9 대비 부가가치)**: 라이브 벽은 날조 아니라 *옳은 변형 관계추론*([[project-tau2-write-failure-rootcause]]·[[project-v9-dpo-antifab-result]] pass 불변). typed·등방화·resolver-grounded spec도 *틀린 관계* emit 가능. **5번째 축(typed-derivation·§0b)이 이걸 닫는 *유일* 장치.** ⇒ **사전등록 = M-σ의 v9 대비 부가가치 = "grounded-but-wrong write-pass" 지표**(날조율 아님): M-σ가 v9 대비 *옳은 변형 선택*을 올리는지로 측정. 안 오르면 = derivation-emit으로도 관계추론이 안 되는 것(=binding이 capability-bound·scale 필요·floor와 합류).

## 7. 구현 단계 (순서·외출 후 수동)
1. **재정의 선결 census**(§3·0원) — *표면/binding* overlap(primitive 아님). τ² exchange-binding이 학습벤치에 등질? orphan 표면 열거. binding 구조적 부재면 중단·orphan이면 등방화 차원 추가.
2. **`m_sigma_data.py`**: SOPBench/TaskBench 궤적 → (NL, config, formal-spec) 삼중쌍 + 등방화 증강. gold 역생성.
3. **multi-config SFT**(`lora_train_chat_toolcall.py` 재사용·7B LoRA·타깃=formal-spec, concrete 아님).
4. **M-D 전이 eval**: held-out τ² + Sstep scaffold → §4 baseline 대조.
5. 결과 박제(M_A_RESULTS / THESIS_STATEMENT §6 갱신).

## 8. 열린 질문 (리뷰)
1. formal-spec 포맷 = Sstep "changes/relax" dict로 충분한가, 더 풍부한 typed-IR 필요한가(NL→SQL DR: coarse-to-fine sketch)?
2. 등방화를 *데이터 증강*으로(쉬움) vs *학습 목적함수*로(IRM류·강함)?
3. SOPBench(정책게이트)·TaskBench(tool-DAG)·τ²(exchange-변형) 셋의 공통 formal-spec 추상이 자연스러운가, 억지인가(§3 census가 답)?
4. Sstep scaffold-페어가 학습-추론 mismatch 없나(학습=full-spec·추론=증분스텝)?
5. in-dist 향상 없이 전이만 볼 순 없음 — in-dist부터 확인하는 게이트.
