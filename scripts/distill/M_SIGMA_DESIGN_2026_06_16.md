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

## 1. 학습할 것 / 안 할 것
- **학습(도메인-일반)**: (NL + config[tools/vocab/gates] + 관측상태) → **formal target-spec**(어느 슬롯을 어떤 값으로·참조·선호순서). = γ-grounding 메타스킬·"any config conform".
- **안 함**: 도메인 내용(카탈로그·특정 값)·concrete id(resolver offload)·정책 자체(ABox). 도메인 SFT/DPO/RLVR로 *도메인* 굽기 금지(과적합·미전이 실증).
- **방법 = SFT**(도메인-DPO/RLVR 아님·plan-selection DR: trajectory DPO=anti-transfer). 타깃이 도메인-일반이라 SFT가 전이 길러냄.

## 2. 데이터 = (NL, config, formal-spec) 삼중쌍·multi-config·등방화
- **소스**: SOPBench + TaskBench 궤적(τ²는 held-out·학습 금지). gold tool-call → **역생성으로 gold formal-spec**(M-A `ma_gold_extract`의 일반화: gold args를 typed-슬롯/참조로 추상).
- **multi-config**: 여러 도메인/config에 걸쳐 → "임의 config conform" 메타스킬(단일 도메인=과적합).
- **★등방화(C8 메커니즘·§5.10-5.11)**: 학습 중 *알려진 표면차원* randomize — tool명(alias)·값형식·속성명/순서·synonym surface. → 학습 스킬이 표면-불변 → config-swap 전이. (덮인 방향만 등방화·미덮인 표면은 잔여 비전이.)
- **출력 구조 = Sstep과 정합**: formal-spec을 typed 증분스텝(changes→[fallback])으로 = inference서 Sstep scaffold가 per-step 검증. 학습=생성기·scaffold=검증기.

## 3. ★선결 census (학습 전·싸고 필수) — 스킬이 *공유*되나?
**위험: SOPBench/TaskBench가 τ²와 *같은* NL→formalize 스킬을 안 가지면 전이 원리적 불가**(스킬 미공유 → 학습해도 전이 0). ⇒ 학습 전에:
- SOPBench/TaskBench/τ² 각각의 NL→formal 결정을 **공통 primitive(P1-P9·"change-keep"·조건부선호·synonym-map)로 분류** → 교집합 스킬 존재 확인. orphan(τ²에만 있는 스킬)=전이 불가 신호.
- 0원 census(매트릭스 재사용). **교집합 비어있으면 M-σ 중단**(벤치 재선정).

## 4. 평가 (M-D 전이 = 헤드라인)
- **in-dist**: M-σ가 학습-도메인(SOPBench/TaskBench) formalize 정확도 올리나(학습 작동 확인).
- **★전이(헤드라인·C8)**: held-out **τ²(retail/airline) config-swap·무재학습** → M-σ-7B의 NL→formalize(+Sstep scaffold) 정확도가 **base 7B 능가·큰모델 근접·도메인-SFT(과적합 baseline) 능가**.
- **baseline 대조**: base 7B / τ²-도메인-SFT(미전이 예측) / 큰모델(천장) / M-σ-7B(본안).
- **지표**: target-spec 정확도(slot/ref)·end-to-end resolved call 정확도·전이 Δ(held-out − base).

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

## 7. 구현 단계 (순서·외출 후 수동)
1. **선결 census**(§3·0원) — 스킬 교집합 확인. 비면 중단.
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
