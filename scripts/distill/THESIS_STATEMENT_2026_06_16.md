# THESIS STATEMENT (crystallized·2026-06-16) — 5 딥리서치 + floor 실험 수렴본

> **★★심화 (2026-06-17) = 이론 정초 + thesis 직접 실측 · 권위 = [`NL_PROCEDURE_OFFLOAD_THEORY_2026_06_17.md`](NL_PROCEDURE_OFFLOAD_THEORY_2026_06_17.md)**: 아래 명제(분담)의 *왜*가 형식화됨. **(a) 역할분담의 근거** = 언어=얕은-병렬 진화 인터페이스(LLM=TC⁰ forward pass가 잘함)·깊은 절차주머니(most/best·**표기-깊이 d(e)**)는 인간도 외부도구로 offload → **LLM=얕은연상+절차-타입 분류 / 결정론=깊은 실행**. **(b) 유계 절차예산 B(L,width)**(§7d-bis)=고정모델 직렬-깊이 유계·d(e)>B 하락·scale은 내부암기로 키움(유계)·결정론 offload는 B=∞ → "binding 벽≠scale" 근본이유. **(c) 직접 실측**: 정적 $select=해로움·**연산-IR**(LLM이 연산 *명명*·엔진 실행)로 **7B가 rank 0.17→1.00 극복**(thesis 직접 증거). **(d) §9 선행연구 정초**(semantic automata·절차의미·학습되는 라우팅·Roman=비용교정). ⇒ 아래 "두 날개"는 이 역할분담의 *특수예*(capability날개=깊은실행 offload·비용/전이날개=얕은 분류 학습+ABox). 다음 실험 = `B_BUDGET_SCALE_DESIGN_2026_06_17.md`(연산-IR thesis × 스케일).

> 5 딥리서치(NL→SQL·det-vs-learned TCO·plan-selection·constrained-decode·input-formalize·small-model-reasoning) + M-A/floor 실험이 한 명제로 수렴. 권위 컨텍스트: `deepresearch/`(5 보고서)·`ma/M_A_RESULTS.md`·`DECOMPOSITION_OPTIMALITY.md`·`MIN_CONTEXT_FORMALIZER_DESIGN.md`. 불변: [[feedback-thesis-tbox-transfer-direction]]·[[feedback-selector-verifier-deterministic]]·[[feedback-nl-formalize-llm-selection-deterministic]].

## 1. 명제 (한 문단)
**작은 on-prem LLM이 큰 모델 수준의 tool-use(NL→action) 성능에 도달하는 법 = 기능을 분담해 각자 잘하는 메커니즘에 맡기고 협업시키는 것.** 구체적으로: **결정론 scaffold가 LLM을 *typed 증분 스텝*으로 몰고 *결정론 per-step 검증*(자유 CoT·내부 self-correction 아님)을 가해 누적·오분해를 차단하고, 도메인 지식은 *제공*(ABox/retrieval)하며, LLM이 *학습*하는 건 도메인-일반 스킬(추론 + config-conditioned NL→formalize)뿐이라 ABox-swap으로 무재학습 전이한다.**

## 2. 두 날개
- **A. capability 날개 (작은→큰 reasoning 닫기)**: **분해 + 결정론 per-step 검증.** 자유 CoT는 일부만(7B 0.438→0.656≈14B·plateau)·외부 검증이 강한형(DR5: self-correction *악화*·소형은 외부 verifier 필요·Math-Shepherd 7B 89.1%·770M>540B).
- **B. 비용·전이 날개**: **MSC**(입력 minimal-sufficient formalize=비용-Pareto·토큰 절반·단 scale 대체 아님) + **ABox/retrieval**(도메인 지식·암기 아님) + **도메인-일반 학습**(전이=ABox-swap·도메인 SFT는 과적합·미전이=v4/v6/v9 실증).

## 3. 분담 기준 (라우팅)
| 함수 성질 | 메커니즘 | 근거 DR |
|---|---|---|
| 정확-명세가능(decidable) | **결정론**(resolver·gate·**per-step verify**) | det-vs-learned·plan-selection·small-model(external>self) |
| 도메인-불변 추론 | **LLM 학습-일반**(typed 증분스텝) | small-model(decompose)·NL→SQL(decouple) |
| 도메인-특정 사실 | **retrieval/ABox**(제공) | input-formalize·NL→SQL(schema-as-input) |

## 4. 최적성 주장 (전이와 독립)
협업이 monolith를 **비용·성능 Pareto-지배** — monolith는 LLM에게 *못하는* 일(날조=exact resolution·환각=knowledge·과다호출=tool-select; 전부 dist/M-A 실증)을 시켜 성능↓+비용↑. 분담은 각 기능을 최강 메커니즘에 → 오류클래스 소멸 + LLM은 환원불가 추론만 → recurring 최소·소형(주권). (`DECOMPOSITION_OPTIMALITY.md`·조건#1-5.)

## 5. ★신규성 (5 DR이 *모두* 같은 gap을 가리킴)
각 조각은 **문헌 검증**(분해+외부검증·decouple-then-resolve·input-formalize↑·det gate·process>outcome) — *발명 아님*. 신규 = **이 검증된 조각들의 *미점유 교차점***:
- **검증이 PRM(학습)도 self-correct도 아닌 *결정론 scaffold***,
- **typed 증분 action**(math reasoning 스텝 아님),
- **tool-use / NL→formalize 세팅**(전 PRM/CoT 문헌은 math/QA),
- **ABox-swap 무재학습 전이**.
- + **방어가능 측정 기여**: min-info **floor 측정**으로 info-limited / reasoning-limited / knowledge-limited 분리(Sufficient Context ICLR25 근접·우리=typed-DAG closure + tool-use).

## 6. 실증 현황 (정직)
- **양성/지지**: 정보 floor 실재·scale-불변(+16pp); formalize=비용-Pareto; 자유 CoT가 7B→14B 거의 닫음; 5 DR이 메커니즘 검증.
- **음성/미완**: MSC≠scale대체(7B reasoning 천장 ~0.53·단 단일샷); selector offline 음성; **ABox-swap 전이 양성 미증명**(C8·τ² 현 음성); 강한형(scaffold 증분+결정론검증) **미측정**.
- **caveat**: "소형>대형" 헤드라인=task-narrow·math·cherry 잦음 → 우리 exchange서 *측정*으로(보편주장 금지). 32B=Int8(coworker bf16 대기).

## 7. 증명 경로 (남은 실험)
1. **강한형(Sstep)**: 7B + 결정론 scaffold 증분 typed스텝 + per-step 결정론 검증 → 자유 CoT 0.656 넘어 32B-L2b 0.844 닿나? = A날개 입증.
2. **scale(coworker)**: 32B-bf16/72B floor → reasoning 천장 위치·Int8-cap 제거.
3. **전이(M-σ→M-D)**: 도메인-일반 학습 → held-out config 전이 = B날개·C8 양성.
4. floor 측정이 (1)(2)서 reasoning-limited 비중 확정 → "학습 vs 제공 vs scale" 라우팅 검증.

## 8. 한 줄
**기여 = 발명이 아니라 *통합·측정*: 문헌이 각각 검증한 [분해+외부검증 / 입력 formalize / decouple-then-resolve / 결정론 gate / 도메인일반 학습]을 *결정론-scaffold·typed증분·tool-use·ABox-swap전이*라는 미점유 교차점에 묶고, floor 측정으로 capability/cost/knowledge 경계를 긋는다.**
