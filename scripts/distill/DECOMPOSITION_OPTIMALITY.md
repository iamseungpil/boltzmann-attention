# 분담 기준 + 협업의 비용·성능 최적성 (기여 진술) — 2026-06-16

> 전이의 실용가치와 *독립적인* 주장: 기능을 각자 잘하는 메커니즘에 분담하고 협업하면 monolith 대비 **비용·성능 Pareto-최적**. 근거 = 우리 실증(dist 과다호출·M-A 날조·knowledge 오류). 상위 = `ABOX_CONFIG_FORMALIZATION_DESIGN_2026_06_15.md` §6.6·`MIN_CONTEXT_FORMALIZER_DESIGN.md`. 불변 = [[feedback-thesis-tbox-transfer-direction]]·[[feedback-selector-verifier-deterministic]].

## A. 분담 기준 (함수 → 메커니즘 라우팅 규칙)
함수 f의 3 속성으로 결정:
1. **정확-명세가능(decidable)?** 규칙/lookup으로 정확 명세되나 → **YES: 결정론**(코드/config).
2. (NO) **로직 도메인-불변 + 추론 필요?** → **YES: LLM 학습-일반**(multi-config).
3. **큰/변하는 사실?** → **retrieval/ABox**(제공).

**규칙 한 줄:**
| f 성질 | 메커니즘 | 성능 | recurring 비용 | 전이 |
|---|---|---|---|---|
| 정확-명세가능 | **결정론**(resolver·gate·verify·MSC build) | exact(100%) | ≈0 | 무관 |
| 도메인-불변 추론 | **LLM 학습-일반**(TBox·NL→formalize 메타스킬) | LLM capability | LLM 추론비 | ABox-swap |
| 도메인-특정 사실 | **retrieval/ABox**(vocab·카탈로그·상식) | exact(권위소스) | retrieval(모델무관) | 데이터-swap |

## B. 최적성 주장 — 분담 협업이 monolith를 Pareto-지배
**monolith(단일 큰 LLM이 전부)** vs **분담 협업**:
- **성능**: monolith는 LLM이 *못하는* 일을 강제 → ①**날조**(exact resolution: order_id/item_id) ②**환각**(large knowledge) ③**과다호출**(tool selection). 분담은 각 f를 *최강* 메커니즘에 → 결정론부 exact·LLM은 강점(추론)만 → **그 오류 클래스 소멸**.
- **비용**: recurring(GPU·latency·throughput)이 on-prem TCO 지배. 분담은 결정론·retrieval=0-recurring-model·**LLM은 환원불가 추론만·최소입력(MSC)** → recurring 최소 + **모델 축소(주권)**.
- ⇒ **monolith는 비용·성능 *양쪽* dominated** = 분담 Pareto-최적.

**★성능 우위의 실증 근거(우리 데이터)**: 세 실패가 모두 "LLM이 더 못하는 일":
- dist full-catalog 과다호출(6.02 vs 2.97·SFT 유발) = LLM이 결정론 gate가 할 tool-select.
- M-A new_item_ids 날조/오선택 = LLM이 결정론 resolver가 할 exact-resolution.
- "Google Home"→옵션값 미매핑 = LLM이 retrieval/ABox가 할 knowledge-lookup.
→ 각각을 결정론/retrieval로 옮기면 그 오류가 *원리적으로* 사라짐(LLM이 안 하니까).

## C. 최적성의 조건 (정직 — 무조건 정리 아님)
1. **올바른 분담**: 경계 함수(예 "관련 옵션키 투영"=NL→스키마 soft grounding)는 오분류 시 깨짐 → 충분성=full closure로 보장·최소화=형식으로([[reference-abox-config-formalization-architecture]]).
2. **결정론부 정확 명세**(resolver fidelity): resolver 버그=책임전가. 검증 정밀도 선결.
3. **인터페이스 깨끗**: LLM 출력이 결정론 소비가능(formalize 계약·xgrammar type 보장).
4. **잔여 추론 ⊆ 작은모델 capacity**: 초과(reasoning-limited)면 *그 슬라이스만* scale → "작은모델 최적"은 reasoning floor 아래일 때만.
5. **recurring-dominated on-prem 가정**: compute 공짜면 monolith도 무방(분담 이점은 비용제약서 발생).

## D. 검증 = 우리 실험이 조건을 *측정*
- **floor(L0–L3)** → 조건#4(잔여추론 small-model-doable? = reasoning-limited 비중) 판정.
- **cost 계측(토큰·호출·모델크기)** → Pareto 곡선 실측(MSC@작은모델 vs 큰모델: scale 대체?).
- **transfer(ABox-swap)** → 학습부(B)의 일반성 — 별 축·실용가치쪽(이 문서 주장과 독립).

## E. 한 줄
**분담 기준 = "정확-명세가능→결정론 / 도메인-불변추론→LLM학습 / 도메인-사실→retrieval". 협업이 monolith를 비용·성능 Pareto-지배 — LLM이 *못하는* 일(날조·환각·과다호출)을 안 시키고 *오직* 환원불가 추론만 시키기 때문(우리 실증). 단 조건#1-5 하에서·#4(잔여추론이 작은모델 capacity 내)는 floor 측정이 판정.**
