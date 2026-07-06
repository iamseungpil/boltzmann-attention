# Semantic 에러를 scale 없이 넘는 방법 — 외부 문헌 정본 (딥리서치#1 회수·2026-07-06)

> **딥리서치#1(semantic non-scale)** 회수·정리. 원본 raw(`tasks/wlsm7q3si.output`·196KB)는 디스크 정리로 소실되었으나, **전체 워크플로 트랜스크립트가 보존됨**: `…/subagents/workflows/wf_f6c32fcf-443…/`(219 agent JSONL). fetch/verify 에이전트 출력에 findings·검증판정(투표수 포함) 전부 잔존 → **거기서 복구**. 아래 인용번호·수치·투표수는 트랜스크립트 검증판정에서 직접 회수.
> **★정직 경계([[08]])**: CONFIRMED = 적대검증 생존(투표 명시). REFUTED = kill(인용 금지). **CRITICAL CAVEAT**(§2c) = 증거 대부분이 symbolic/synthetic state-tracking·고전 entity-linking = **부하-유발(load-induced) semantic 잔여가 아님**. genuine semantic서 "소형+방법 정량승리"는 **ED crossover + intent-extraction 둘뿐·load축 미검**.
> **목적**: DR#1은 "학습·scale 없이(또는 소형+보조로) semantic/entity-reasoning 에러를 넘는 외부 방법"을 물었다. 우리 `SEMANTIC_ERROR_FORENSIC_AND_OVERCOMING_2026_07_06.md`이 **내부 궤적 포렌식**(τ²-retail)이라면, 본 doc은 그 **외부-문헌 짝**이다.
> **포맷**: `AGENTIC_AI_ADOPTION_RESEARCH_2026_07_06.md`(DR#2·adopt/adapt/avoid/already-ahead)와 동일.

---

## 0. 판정 요약

**한 줄 Summary**: scale 아닌 방법으로 semantic 잔여를 이기는 가장 강한 실증 = **(1) 고정파라미터로 state/entity-tracking·intent 스킬을 설치하는 도메인-일반 학습** + **(2) genuine entity-reference resolution서 대형-맨몸을 이기는 구조화-grounding/retrieval scaffold**. 단 대부분 증거가 symbolic/synthetic·load축 미검.

| 외부 메커니즘 (문헌) | 검증 | 우리 컴포넌트 | 판정 |
|---|---|---|---|
| **★KG entity-disambiguation crossover** [2505.02737] | ✓ 3-0/2-1 | ⋈ present-scaffold·φ_finite | **already-ahead 수렴 + 인용 핵심** |
| **code 계속사전학습이 state-tracking 설치** [2405.21068·2409.04556] | ✓ 3-0 | C2 state-tracking·Track3 학습 | **adapt**(규모≥13B) |
| **★단 scale-gated (7B 무효)** [2405.21068] | ✓ | 우리 7B under-completion | **경계 확정·인용** |
| **state-tracking = 중간학습으로 steerable(비-emergent)** [2503.02854] | ✓ | Track3 학습 | **adapt** |
| **외부메모리 Self-Notes** [2305.00833] | ✓ | C2 execution-state store | **adapt** |
| **의미파싱**(Wang/Lapata/Titov) | ✓ **4/4 not-refuted=최강** | present-formalize | **adapt·최강 confirmed** |
| self-critique로 semantic 회복 [2402.08115 Kambhampati] | ✗ KILL | 게이트 필요(Kalai) 정합 | **avoid·인용금지** |
| embedding-scale 대체 re-ranking [2506.00049] | ✗ KILL | — | **avoid·인용금지** |
| PAW compiler crossover | ✗ KILL | — | **avoid·인용금지** |
| clarifier=671B 매칭 | ✗ KILL | — | **avoid·인용금지** |
| neuro-symbolic +9.36pp CLUTRR [2408.13654] | ✗ KILL(도메인 mismatch 과대) | — | **avoid·인용금지** |
| intent-extraction crossover 일부 [2509.12423] | ✗ KILL(부분) | — | **avoid·인용금지** |

---

## 1. CONFIRMED — 검증 생존 (인용 가능)

### ★1.1 KG entity-disambiguation crossover — "소형+보조 > 대형-맨몸"의 가장 깨끗한 실증 (최고 가치)
- **발견** [2505.02737]: **GPT-3.5 + YAGO 클래스계층 pruning**이 **10개 ED 데이터셋 weighted-F1 81.1%**로 **GPT-4 기반 ChatEL(79.3%)을 >20배 싸게 이김**. 증강 사다리: 비증강 75.4 → DBpedia 78.8 → **YAGO 81.1**. **투표 3-0/2-1**.
- **= semantic(⋈)서 "소형+결정론 보조 > 대형-맨몸"의 가장 깨끗한 정량 실증.**
- **단서(정직)**: 고전 **mention-to-KB linking**·**curated KG(YAGO/DBpedia) 의존**·**load축 미검**. 우리 τ² ⋈처럼 부하-유발 잔여에서의 crossover는 아님(§2c).
- **우리와의 수렴**:
  - **⋈**: operand_controlled·p_debias 실측(⋈ wrong-match=위치편향 지배·genuine 소수·`SEMANTIC_ERROR_FORENSIC §2c-1·§3.3`)과 같은 방향 — scale 아닌 **결정론 graph present + debias**.
  - **DR#2 GraphRAG**: `AGENTIC_AI_ADOPTION_RESEARCH §1.1`(DB서 결정론 entity-graph present)와 **삼중 수렴** — 깨끗한 DB(τ²)선 LLM 추출 불요·링크 traverse만 결정론.
  - **이론**: `THEORY_BOUNDARY_MAP` **φ_finite**(코도메인이 요청마다 𝒜-내 유한집합으로 환원=유한 선택·매칭·scaffold-reachable·scale 아님) 정확 대응 → **Cor 4** 외부 앵커.
- **판정 already-ahead 수렴 + 인용 핵심**: 규율([[03]]·[[46]]) — "소형=대형" 아님·"**대형에 버금가는(결정론 보조로)**". *특정 태스크(ED)*의 crossover.

### 1.2 code 계속사전학습이 state/entity-tracking을 설치 — 단 scale-gated
- **발견** [2405.21068·2409.04556]: **Code Llama 13B 60.8 vs Llama-2 13B 25.2 = +35.6pp(동일 파라미터)**. 학습(code 계속사전학습)이 상태추적 능력을 *설치*. **투표 3-0**.
- **★scale-gated** [2405.21068]: **7B선 무효 — Code Llama 7B 13.7 vs Llama-2 7B 15.0**(오히려 미세 하락). → 문턱 아래 규모에선 학습해도 안 올라옴.
- **단서**: **~500B 토큰(경량 아님)**·synthetic boxes 태스크.
- **우리와의 수렴**: **7B=orchestration 바닥**(핸드오프 settled·43%만 write)·**φ_open scale-gated** 실측과 정합. [[13]](scale→학습→scaffold)·[[11]](학습=학습벤치·전이=swap). **함의**: state-tracking 학습설치는 **≥13B**서만·7B τ²는 결정론 controller(coverage 강제·`SEMANTIC_ERROR_FORENSIC §3.1 A`)로 우회.
- **판정 adapt(조건부)**: Track3 학습 타깃이나 **규모 문턱 + make-or-break 게이트**(§A2_RULE_USE 4조건) 통과 시.

### 1.3 state-tracking = 중간학습으로 steerable(비-emergent)·외부메모리·의미파싱
- **steerable** [2503.02854]: state-tracking은 emergent 아니라 **중간학습으로 조종 가능** → 학습레버 유효성의 독립 지지.
- **외부메모리 Self-Notes** [2305.00833]: 추론 중 상태를 외부 노트로 기록 → C2 execution-state store 근거(DR#2 MAGE "state≠semantics"와 이중).
- **★의미파싱**(Wang/Lapata/Titov 계열): **4/4 not-refuted = 본 DR에서 최강 confirmed**. NL→구조 파싱은 학습으로 안정 설치 → present-formalize(NL 스펙→𝒜 매칭)의 문헌 backbone.
- **판정 adapt**: C2 execution-state·present-formalize 설계 확정 지지. [[05]] 도메인-일반 엔진·ABox 도메인.

---

## 2. REFUTED + CRITICAL CAVEAT

### 2a. REFUTED — 적대검증 kill (인용 금지)
| 주장 | 문헌 | kill 사유 |
|---|---|---|
| LLM self-critique로 semantic 회복 | [2402.08115 Kambhampati] | 자기비판이 무너짐(자기보장 불가) |
| embedding-scale이 구조-scaffold 대체 (re-ranking) | [2506.00049] | 임베딩 규모확대가 대체 가능 → crossover 보편 아님 |
| PAW compiler crossover | — | 미생존 |
| clarifier = 671B 매칭 | — | clarification이 초대형 매칭이나 "소형이 이긴다" 아님 |
| **neuro-symbolic +9.36pp CLUTRR** | [2408.13654] | **도메인 mismatch 과대주장** — CLUTRR(합성 친족)≠부하-유발 semantic |
| intent-extraction crossover 일부 | [2509.12423] | 부분 kill |

- **★내 이전 초안 정정**: neuro-symbolic +9.36pp를 CONFIRMED/adapt로 잘못 넣었으나 **REFUTED(도메인 mismatch 과대)**. Cor 5(solver>CoT) 지지 인용에서 **제외**. self-critique kill = 우리 **"모델 자기보장 불가·게이트 필요"(Kalai [2509.04664])** 정합·강화.

### 2b. CRITICAL CAVEAT — 증거의 성격 (헤드라인 right-size)
- 본 DR 증거의 **대부분**이 **symbolic/synthetic state-tracking(boxes)·의미파싱·고전 entity-linking(curated KB)** = **부하-유발(load-induced) semantic 잔여가 아니다**.
- **genuine semantic + 정량 "소형+방법 승리"** 로 좁히면 남는 것 = **(1) ED crossover [2505.02737]** + **(2) intent-extraction**(단 후자는 부분 kill) — 그리고 **둘 다 load축(부하 유발 조건) 미검**.
- **함의([[08]]·[[03]])**: "구조로 semantic scale 넘는다"는 **문헌 전체선 태스크-의존·부분적**. 우리 인용은 **KG-crossover의 구체 데이터점**·**state≠semantics**·**의미파싱 4/4**에 한정. "소형=대형" 일반화·load축 주장은 근거 없음.

---

## 3. 인용·positioning (relwork/특허)
- **지지 인용(생존)**:
  - **[2505.02737] KG-crossover** = paper1 §5 Cor 4(⋈)·특허 A §3.4/B §3.2(비용순 배분)의 **외부 데이터점**(결정론 보조가 대형 맨몸을 태스크별 저비용 초과).
  - **[2405.21068] scale-gated** = 7B=바닥·φ_open scale-gated 실측의 문헌 짝. state-tracking 학습설치=규모 문턱.
  - **의미파싱 4/4** = present-formalize backbone. **Self-Notes/steerable** = C2·학습레버.
- **양보·구별**:
  - "구조로 소형=대형" 일반주장·embedding-scale 대체·self-critique 회복 = **우리도 미채택/반증** → 헤드라인 "대형에 버금가는"·전이=schema-conditioning 양보. **모트=결정론 게이트(준수보장)+오프로드**는 이 문헌들이 다루지 않음.
  - self-critique kill(Kambhampati) + Kalai = **"모델 자기보장 불가→외부 결정론 게이트 필수"** 논거 강화.
- **[[46]] 정합**: KG-crossover=부품/전제로 인용·양보. 모트=scale-invariant compliance residual — DR#1 어느 발견도 선점 안 함. neuro-symbolic·self-critique 등 kill은 **인용 금지**(DR#2 KILLED "구조로 소형=대형"과 동일 규율).

---

## 4. 종합 — DR#1이 우리에게 주는 것
1. **외부 독립확증**: 우리 SEMANTIC_ERROR_FORENSIC(내부 τ² 궤적) 결론 — entity/semantic 잔여=대부분 결정론+보조로 닫힘·genuine만 scale — 이 외부 문헌서 독립 재현(KG-crossover·state≠semantics·의미파싱). 단일 벤치 아티팩트 아님.
2. **φ_finite 좌표**: KG-scaffold=φ_finite를 결정론에 넘기는 문헌 실증 → THEORY_BOUNDARY_MAP §5 Cor 4 외부 앵커.
3. **학습 레버 경계**: state-tracking은 학습-설치 가능하나 **규모-gated(7B 무효·13B 유효·~500B토큰)** → [[13]] 순서·7B=바닥 지지. τ²-retail 7B는 결정론 controller 우선.
4. **헤드라인 규율**: self-critique·neuro-symbolic·embedding-scale kill → "구조로 소형=대형"은 문헌서도 미생존. 우리 "대형에 버금가는"·모트=게이트+오프로드가 whitespace.
5. **★가장 큰 gap(정직)**: genuine·load-induced semantic서 "소형+방법 정량승리"는 문헌서도 **거의 미검**(ED crossover 하나가 사실상 유일한 깨끗한 점) → 우리 τ² crossover 실험이 채우는 **진짜 whitespace**.

**경계 준수([[05]][[11]][[03]])**: 채택은 전부 결정론 scaffold/ABox 구조 층(KG present=엔진·링크=ABox) 또는 규모-조건부 학습(학습벤치·전이=swap). 도메인-타깃 학습·"소형=대형" 위반 0. 학습 레버 불변.

---

## 5. 상태·후속·복구경로
- **raw 소실**: DR#1 원본 196KB(`tasks/wlsm7q3si.output`) 복구 불가. **복구 경로 = 워크플로 트랜스크립트** `subagents/workflows/wf_f6c32fcf-443…/`(219 agent JSONL·fetch/verify 출력에 findings·투표 잔존). 추가 tally 필요 시 거기서 재추출(무료·유료 재-DR 불요).
- **후속 pending**(핸드오프 §0.2·§0.3): infra/retry 포렌식 정본화·pass^3 지표 수정. 본 doc(§0.1)로 최우선 pending 해소.
- **연결**: `SEMANTIC_ERROR_FORENSIC_AND_OVERCOMING_2026_07_06.md`(내부 짝)·`THEORY_BOUNDARY_MAP_2026_07_06.md §5`(이론 좌표)·`AGENTIC_AI_ADOPTION_RESEARCH_2026_07_06.md`(DR#2·GraphRAG 삼중수렴).
