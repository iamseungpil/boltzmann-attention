# M-σ v3 (리뷰용) — 전이 메커니즘 *비교 실험군* (2³ 완전요인설계) → held-out τ² 전이 — 2026-06-16

> ★설계 전환: v2(`M_SIGMA_V2_SYNTH_DESIGN.md`)는 **단일 레시피 + single-knob-off**(전부 켜고 하나씩 끄기 = marginal contribution)였다. 사용자 교정 = 이때까지 검토한 전이 방안(등방화·provenance·NL-grounding…)을 **각각 독립 축/실험군**으로 두고 *단독 전이* + *조합 전이*를 **비교**한다. ⇒ 단일-knob-off → **완전요인(factorial) 비교군**으로 격상.
> 상위 = `M_SIGMA_DESIGN_2026_06_16.md`·`THESIS_STATEMENT_2026_06_16.md`. substrate 합성 상세 = v2 §1-3 재사용. 불변 = [[feedback-thesis-tbox-transfer-direction]](τ² 참조/학습 금지)·[[feedback-selector-verifier-deterministic]]·[[feedback-nl-formalize-llm-selection-deterministic]].

## 0. 한 줄
**"전이를 만드는 게 무엇인가"를 한 레시피에 베팅하지 않는다. 검토해 온 전이 구동 후보(등방화 ISO·NL-grounding·provenance PROV)를 *직교 축*으로 두고, 순수 추상 selection substrate 위에서 *완전요인(2³)*으로 각 축의 *단독 전이* + *조합 전이*(가산/시너지/잉여)를 held-out τ²로 측정.** = 어느 축이, 단독으로/합쳐서, 실도메인 전이를 구동하는지 *비교 실험군*으로 확정.

## 1. ★왜 비교군(요인설계)인가 — single-knob-off의 한계
- **single-knob-off**(v2 §9): "전부 ON 기준 X를 빼면?" = X의 *조건부 marginal* 기여만. 한 레시피(FULL)에 암묵 베팅·X가 *단독으로* 전이하는지/조합이 *가산인지*를 못 답함.
- **완전요인(2³)**: 8셀 = {각 축 OFF/ON}³. → **main effect**(축이 단독·평균적으로 전이?) + **interaction**(축들이 합쳐질 때 시너지/잉여?) 둘 다 분리. 사용자 의도("각각 전이? + 합쳐서?")와 정확히 일치.
- 비용 비교: knob-off 5 arm → factorial 8 arm(+2 ref). +3 arm으로 *비교군 전체* 획득(§9 비용 정당).

## 2. ★전이 구동 후보 = 직교 축 (이때까지 검토한 방안 정리)
| 축 | OFF | ON | 가설(왜 전이 구동?) | 출처 |
|---|---|---|---|---|
| **ISO**(등방화) | 고정 스키마명/값(전 예제 동일 표면) | 예제마다 랜덤 스키마/도구명/필드명/값-vocab | 표면군 저차원 불변량 강제 → 표면 과적합 차단·논리만 남음 | Olver n−s(`EXPERIMENT_DESIGN.md:32`·§5.13-5.14)·v1 iso=도구명만(부분) |
| **NL**(grounding) | literal `attr=val` 직접 제시 | "X 바꾸고 나머지 유지·없으면 Y 완화"를 자연어 패러프레이즈 | NL 구조-prose→criteria 파싱 학습(τ²의 "keep rest"·조건부) | input-formalize DR·v2 §1a(★구조-파싱이지 어휘 아님) |
| **PROV**(provenance) | $select-only(타입 미구분) | literal/$ref/$select 혼합·"언제 ref vs literal" | over-$ref 교정(M-D서 order_id까지 $ref해 base 망침) | M-D §11 음성 원인(a)·NL→SQL decouple DR |

- **고정(held)**: substrate = **순수 추상 selection-by-criteria**(도메인 0·v2 §1)·deterministic resolver offload(concrete는 학습 아님·[[feedback-nl-formalize-llm-selection-deterministic]])·학습량/step/LR/난이도분포(§6 교란통제).
- **확장 축(옵션·budget)**: SEM(추상 vs 약한-의미 중립영어 substrate)·SUBSTRATE(selection vs cfb-threading — R1이 부분대리). 핵심 결론 후 4번째 축으로만(2³→2⁴ 회피).

## 3. ★실험군 = 2³ 완전요인 + 2 참조 (10 arm)
substrate 고정(추상 selection), 축 {ISO,NL,PROV} 8셀:

| arm | ISO | NL | PROV | 역할 |
|---|---|---|---|---|
| **R0 base** | — | — | — | 무학습 floor(현 τ² all 0.41·new_item 0.41) |
| **R1 cfb-Mσ** | (도구명) | off | off | 직전 음성 레퍼런스(threading substrate·all 0.03) |
| **M0** (000) | OFF | OFF | OFF | 추상 selection substrate-only(고정스키마·literal·select-only) |
| **A-iso** (100) | ON | OFF | OFF | ISO 단독 전이? |
| **A-nl** (010) | OFF | ON | OFF | NL 단독 전이? |
| **A-prov** (001) | OFF | OFF | ON | PROV 단독 전이? |
| **C-in** (110) | ON | ON | OFF | ISO+NL 조합 |
| **C-ip** (101) | ON | OFF | ON | ISO+PROV 조합 |
| **C-np** (011) | OFF | ON | ON | NL+PROV 조합 |
| **FULL** (111) | ON | ON | ON | 전체 레시피(=v2 baseline B) |

- 각 arm = 합성 생성(축 플래그) → 7B LoRA SFT → **held-out τ² 전이 eval**(§5). R0=무학습·R1=기존 어댑터(재학습 불요).
- **비교군 판독**(§7): main effect(축 단독) = ON셀 평균 − OFF셀 평균. interaction = FULL − (M0 + ΣΔsingle). 합쳐서가 가산↑(시너지)/평탄(잉여)/음(상쇄)인지.

## 4. ★공유 substrate (추상 selection·v2 §1-3 요약 + root-cause 결박)
각 예제: 랜덤 추상 스키마(K속성·값-vocab) → 카탈로그(M item = 속성조합+item_id+available) → current item → **요청**(변경할 속성 + fallback) → **gold** = {current ⊕ changes} 매칭 available item_id(미가용시 fallback).
- **학습타깃 = provenance-typed spec**(v2 §3): `literal` / `{"$ref":"<obs>#<path>"}` / `{"$select":{"from","by":{attr:val},"fallback":[...]}}`. concrete item_id 아님 → resolver가 결정론 해결.
- **★확정 root-cause 결박**(v2 §1a·`M_A_RESULTS §3`): write-벽 = **구조적 변경-오계산("X만 바꾸고 유지" 틀림), 어휘 아님**. ⇒ substrate가 가르치는 건 *구조적 selection 연산*(어느 속성 change/keep/fallback). 특정값 어휘 grounding("Google Home"→item)은 **ABox 제공 몫·학습 아님**. → 추상 토큰이 어휘갭 안 만들어도 정당(NL 축은 *구조-prose 파싱*만 시험·어휘 아님).
- **obs-포맷 정합**(v2 §1b): 합성 obs 직렬화 = τ² harness 포맷(tool출력 JSON·관측 인덱싱) — $ref/$select의 `from`-인덱스 전이 위해 필수.

## 5. ★전이 측정 = held-out τ² (v2 §4-5 정량 bar 승계)
- **eval**: `m_sigma_transfer_eval.py` 수정판 — τ² exchange를 obs+tools로 제시 → 모델이 literal/$ref/$select 혼합 emit → resolver → gold 대비 per-arg-type. **harness 수정**: payment_method_id=값(현 dict-키 아티팩트·§11b)·$select 채점·**n≥50 확장**(bar 유효화)·over-$ref율·구조/어휘 실패 라벨(autopsy).
- **primary metric**: τ² **new_item_ids selection 정확률**(confirmed-orphan 차원·R0=0.41·R1=0.34 퇴화). **secondary**: all-arg(R0 0.41·R1 0.03)·over-$ref율.
- **arm별 성공 bar**(사전등록·post-hoc 금지): arm이 **new_item_ids ≥ R0+2σ**(노이즈초과·잠정 ≥+12pp·n≥50기준) **∧ over-$ref↓** **∧ all-arg ≥ R0 회복**(R1식 퇴화 안 냄). FULL이 이걸 만족하면 C8 양성 헤드라인.

## 6. ★교란통제 (factorial 타당성)
- arm 간 **#예제·SFT step·LR·LoRA rank·카탈로그크기/K/fallback깊이 분포·τ² eval셋(동일 n·harness)** 고정. 변하는 건 해당 축 플래그뿐. (안 그러면 전이차 = 축 아니라 데이터량/난이도 confound.)
- 모든 arm round-trip 검증(spec→resolver→gold == 구성 gold) 100% 통과분만 학습(정직 분모).
- **power**: n≥50·±~5pp → main effect는 2개 OFF셀 vs 2개 ON셀(각 ~n)로 평균 → 노이즈 1/√2 감소. interaction은 약하면 미검출 가능(정직 명시·헤드라인은 main effect).
- **seed**: ISO 랜덤성·합성 랜덤성 seed 고정·arm 간 동일 seed 패밀리(축 외 변동 차단).

## 7. ★판독 = 비교군 결론 (사전등록 예측)
- **main effect(각 축 단독 전이?)**: ΔISO = mean(100,110,101,111) − mean(000,010,001,011). 동일하게 ΔNL·ΔPROV. **예측**: 셋 다 +(ISO 최대=표면불변)·단 *단독으론* 약할 수 있음(특히 NL·PROV는 ISO 없으면 표면 과적합).
- **interaction(합쳐서?)**: FULL − [M0 + ΔISO + ΔNL + ΔPROV]. **예측**: ≥0(가산~약시너지) — provenance는 ISO와 보완(ISO=표면, PROV=참조타입), NL은 ISO와 보완(grounding+불변). 음(상쇄)이면 라인 재검토.
- **결정표**:
  - FULL 양성 ∧ 단일 음성 ∧ interaction 양 → **전이는 *조합*이 만든다**(어느 단일도 불충분·합쳐야 함) = 강한 thesis 주장.
  - 한 단일(예 A-iso) 양성 → **그 축이 단독 구동**(등방화가 핵심·Olver 이론 실증·헤드라인).
  - 전 arm 음성(FULL 포함) → 추상→실 grounding 갭 = 음성도 1급(추상학습 전이한계·M-D autopsy로 구조 vs 어휘 끊김 진단·§4).
- **R1 대조**: cfb-Mσ(threading substrate)가 음성인데 추상 selection substrate arm이 양성이면 = **substrate(selection vs threading)도 구동축**(확장축 SUBSTRATE 입증·orphan 해소 확인).

## 8. 위험 (정직·리뷰 훅)
1. **too-abstract grounding 갭**: substrate가 `attr=val`이면 NL→criteria 해석 안 가르침 → NL 축이 처방(§4). 실도메인 어휘는 ABox 몫(§4 root-cause 결박).
2. **합성 난이도 분포**: 너무 쉬움(후보2)=천장·너무 어려움=noise. 카탈로그/K/fallback 깊이 분포 설계·arm 간 고정.
3. **resolver $select 충실도**: 기준매칭+fallback+availability+tie-break 결정론(M-A resolver 재사용·tie-break 명문).
4. **interaction 검정력 부족**: n≥50도 2차효과 약하면 미검출 → 헤드라인은 main effect·interaction은 방향만.
5. **factorial 비용**: 8 SFT — budget 초과시 §9 tier1(M0·A-iso·A-nl·A-prov·FULL=5 arm·main effect 근사)로 축소·조합셀은 tier2.
6. **전이 미보장**: 추상→실은 C8 가설·음성 가능. 단 *깨끗한* 시험(synth→**real**·v2 §0).

## 9. 구현 단계 + 비용 tier
1. **`synth_selection.py`**: 추상 도메인/카탈로그/NL요청/gold + provenance 혼합 + 축 플래그 `--iso/--nl/--prov`(+옵션 `--sem`). round-trip 검증. obs 직렬화=τ² 포맷. (v1 `m_sigma_data.py`는 cfb-파생/threading — 별 파일로 신규: 순수 합성.)
2. **resolver 확장**(`ma_resolver.py`): $select(by-criteria+fallback+availability+tie-break)·$ref·literal. M-A `select_variant` 재사용.
3. **harness 수정**(`m_sigma_transfer_eval.py`): payment=값·$select 채점·n≥50·over-$ref율·구조/어휘 autopsy 라벨.
4. **factorial batch**(`ma_factorial_batch.sh`): 8셀 생성→SFT(교란통제 고정)→τ² eval→집계(main effect/interaction 표). 2-GPU 병렬.
   - **tier1(핵심·~3-4h)**: M0·A-iso·A-nl·A-prov·FULL(5 arm) + R0·R1 = main effect 근사 + FULL 판정.
   - **tier2(조합·+~2h)**: C-in·C-ip·C-np(3 arm) = interaction 완성(2³ 전체).
5. 결과 박제(`M_A_RESULTS.md §12`·main effect/interaction 표)·§5 정량 bar 판정.

## 10. 한 줄
**전이를 한 레시피가 아니라 *비교 실험군*으로: ISO·NL·PROV를 직교 축으로 두고 순수 추상 selection substrate 위 2³ 완전요인으로 *각 축 단독* + *조합* 전이를 held-out τ²로 측정 → main effect(단독 구동축) + interaction(합쳐야 전이?)을 분리 확정.** C8 = synth→real 가장 깨끗한 시험·M-D 음성 3원인(orphan·over-$ref·harness) 정면 해결.
