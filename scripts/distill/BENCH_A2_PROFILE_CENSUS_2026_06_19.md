# 벤치 A2-프로파일 Census — 25벤치 확장 + scaffold-고정성 판정 (2026-06-19)

> **자립 문서**(리뷰용·확장 골격). 질문(사용자 2026-06-19): "A2(grounding-spec) 불필요한 벤치는 TBox+scaffold로 다 되나, 아니면 scaffold 수정해야 하나?" → **이 문서가 답을 검증가능한 census로 박는다.**
> 상위 = `A2_MINIMIZATION_FRONTIER_DESIGN`(모델크기×A2·이 census가 "도메인/벤치" 축을 구체화) · `A2_GROUNDING_WIRING_DESIGN`(grounding-spec=A2 컴포넌트 1개·관계대수 닫힘) · `ABSTENTION_AS_DECIDABLE`(P-조건). 권위 census = `PRIMITIVE_COVERAGE_MATRIX §2`(P1-P9 per-bench) · `GENERATOR_ALGEBRA §3,§8`(content-op 5도메인·25벤치 3층). 불변 = `05-fixed-vs-variable`(scaffold 고정·`grep "if bench"=0`).

## 0. ★질문에 대한 답 (한 단락)
**"A2 불필요" 벤치는 없다 — 정확히는 "grounding-spec 불필요"다.** A2는 단일물이 아니라 **컴포넌트 프로파일**이고 벤치마다 다른 부분집합을 켠다. in-scope 벤치(25−4≈21)는 전부 **고정 scaffold + 벤치별 A2-프로파일 swap**으로 처리·**scaffold 코드 수정 0**. grounding-spec이 안 켜지는 벤치(content-light)도 *여전히 다른 A2*(gate_spec·tool-schema)는 켠다. **scaffold 수정이 필요한 건 out-of-scope 4축뿐**(closure 밖 능력=code-exec/CSP/GUI/long-plan·A2-swap 아님=선언된 경계).

## 1. A2 = 컴포넌트 프로파일 (P-primitive ← A2 매핑)
| A2 컴포넌트 | 담당 P-primitive | 무엇 | 형식 |
|---|---|---|---|
| **tool-schema** | P1·전부 | 함수 시그니처·출력 타입·enum/attr-vocab | `05-fixed §2bis ①` |
| **dep-map** | P2b(which-producer) | "어느 producer가 needed-type 생산" | grounding-spec ①(`A2_FORMAT §1`) |
| **grounding-spec** | **P4(select)**·P2b(catalog/anchor 투영) | tool출력→relational rows(π/unnest/⋈/σ·닫힘) | `A2_GROUNDING §2` |
| **gate_spec** | **P5·P6·P8**(policy/confirm/auth) | `precond⇒allowed`·irreversible·scope | `A2_FORMAT §1 ③`·`t2_gate GATE_SPEC` |
- **고정(swap 안 됨)** = TBox weights(op-naming·NL→formalize·도메인일반) + scaffold 엔진(resolve/gate/verify/sequence/provenance/fetch-ask). **변경=A2 프로파일 swap만**(`05-fixed`).
- P3/P9(sequence/par)=scaffold orchestration(도메인 A2 없음·tool-schema 의존). P2a(branch)=모델(관측후). **P7(recovery)=반응형·RL·A2 아님**(하드 잔여·`PRIMITIVE_MATRIX:95`).

## 2. 3-class 판정 (scaffold-고정성)
| class | 켜지는 A2 | scaffold | 예 |
|---|---|---|---|
| **selection-rich** | grounding-spec + gate_spec + tool-schema (+dep-map) | **고정** | retail·airline·CFB |
| **content-light/command** | gate_spec + tool-schema (**grounding-spec 없음**) | **고정** | telecom·banking·SOPBench(정책) |
| **flow-pure(FC)** | tool-schema + dep-map (gate 약·selection 약) | **고정** | TaskBench·BFCL·RestBench·Seal-Tools |
| **out-of-scope 4축** | — | **불충분**(엔진 확장 필요) | AppWorld·TravelPlanner·WebArena·OSWorld |

## 3. Census 표 (25벤치 — 권위본 매핑·확장 골격)
범례: A2 컴포넌트 ✓필요/–불요/?미검증. scaffold: **F**=고정충분 / **F\***=엔진 일반화 후 고정(grounding-spec식) / **X**=out-of-scope. status: **C**=concrete(실증) / **M**=권위 census 매핑 / **V**=to-verify.

| 벤치 | 축 | tool-schema | dep-map | grounding-spec | gate_spec | scaffold | status | 근거 |
|---|---|---|---|---|---|---|---|---|
| **τ²-retail** | FLOW+CONTENT | ✓ | ✓ | ✓ | ✓ | F | **C** | `A2_GROUNDING §3` |
| **τ²-airline** | FLOW+CONTENT | ✓ | ✓ | ✓(explode) | ✓ | F | **C** | `A2_GROUNDING §4` |
| **τ²-telecom** | command | ✓ | ◐ | – (single-field set) | ✓✓ | F | M | `GEN_ALG §3` |
| **τ²-banking** | command/auth | ✓ | ◐ | – | ✓✓ | F | M | `GEN_ALG §3` |
| **τ²-mock** | mixed | ✓ | ◐ | ◐ | ✓ | F | M | `GEN_ALG §3` |
| **SOPBench** | policy-flow | ✓ | ✓ | ◐ | ✓✓(P5/P8) | F | M | `PRIM §2:76` |
| **TaskBench** | flow(DAG) | ✓ | ✓(symbolic) | – | – | F | M | `PRIM §2:77` |
| **ComplexFuncBench** | flow+select | ✓ | ✓(2-hop) | ✓ | ◐ | F | M | `PRIM §2:78` |
| **SOP-Bench** | policy-flow | ✓ | ? | ? | ✓(P5/P8) | F | V | `PRIM §2:80` |
| **BFCL V3** | flow | ✓ | ◐(state-val) | ◐ | – | F | M | `PRIM §2:81` |
| **RestBench** | flow(REST) | ✓ | ✓(inferred id) | ◐ | – | F | M | `PRIM §2:82` |
| **Seal-Tools** | flow | ✓ | ✓(=TaskBench) | – | – | F | M | `PRIM §2:84` |
| **NESTful** | flow+math | ✓ | ✓(P2b수학=offload) | – | – | F | M | `GEN_ALG §9` |
| **API-Bank·ToolBench·API-BLEND·ToolAlpaca·ToolEmu·WorkBench·MetaTool·AgentBoard-tool** | flow/agentic | ✓ | ? | ? | ? | F? | **V** | `GEN_ALG §8.1`(닫힘 매핑·실데이터 미분석) |
| **AppWorld** | G_loop(code) | — | — | — | — | **X** | M | `GEN_ALG §8.1` |
| **TravelPlanner** | G_csp | — | — | — | — | **X** | M | `GEN_ALG §8.1` |
| **WebArena·Mind2Web·VisualWebArena** | G_ground(GUI) | — | — | — | — | **X** | M | `GEN_ALG §8.1` |
| **OSWorld·AgentBoard·Aquawar** | G_plan | — | — | — | — | **X** | M | `GEN_ALG §8.1` |
- **정직(anti-drift)**: concrete=retail·airline뿐. SOP/TB/CFB·τ²-도메인=train/autopsy census 매핑(M). FC 8벤치=닫힘 *문헌 매핑*이지 실데이터 grounding-spec 미작성(**V**·작성 시 가짜커버리지 금지). out-of-scope 4축=scaffold 불충분 선언.

## 4. ★Falsifiable 주장 (이게 thesis의 칼날)
- **고정 scaffold 불변식**: in-scope 전 벤치서 `grep -n "if .*bench\|if .*domain" <scaffold>` = **0**. CI 가드(`A2_MIN §8`).
- **벤치가 scaffold 수정을 강요하면** → 둘 중 하나(벤치별 분기 추가 금지):
  - (a) **엔진 primitive 불완전** → **A2로 일반화**(resolve→grounding-spec가 정확히 이 수술·`A2_GROUNDING §2b`). 일반화는 *모든* 그 primitive 쓰는 벤치로 1회.
  - (b) **closure 밖** → **out-of-scope 재분류**(+1 X행·`PRIM §2:100`).
- **포화 곡선 = 유한성 실증**(`PRIM §4`): 벤치 순차 추가 시 *새 A2-컴포넌트 종류 → 0*. 현재 4종(tool-schema/dep-map/grounding-spec/gate_spec)·새 벤치가 5번째 종류 강요하면 미닫힘. **적대적 추가 필수**(서비스-API만이면 포화 자명·무정보·`PRIM §2:102`).
- **scaffold-고정성 ≡ grounding-spec 포맷-닫힘의 상위판**: §3 census가 "F* 없이 F"로 채워지면(엔진 일반화 0회) 강증명·F*가 반복되면 엔진이 N에 맞춰 자라는 중.

## 5. 기록 노트 (리뷰 record-only·데이터 후 격상·지금 패치 금지)
> 리뷰(2026-06-19) "블로커 아님·과설계 방지 위해 기록만". 데이터로 P-기저가 서면 census로 격상.
1. **abstention-trigger 닫힘 census**: `ABSTENTION §2` P-조건 매핑표를 닫힘 census로 격상 가능 — **"orphan abstention-trigger = 0(모든 막힘 ∈ P-조건 ∪ 의미잔여)"**는 P-기저 닫힘을 상속(grounding-spec §2b·이 문서 §4 포화와 같은 구조). 데이터 후.
2. **P5/P6 decidable = gate_spec 완전성 조건부**: 불완전 gate_spec → 미탐지 정책위반(구조-sound인데 위반). = grounding-spec **PROVIDE-완전성**과 같은 의존. 이미 "구조적≠correct"(`ABSTENTION §1`)에 흡수되나, **gate_spec 완전성을 명시 측정**은 데이터 후.

## 6. 한 줄
**A2=컴포넌트 프로파일(tool-schema/dep-map/grounding-spec/gate_spec)·벤치마다 부분집합. in-scope 25−4는 고정 scaffold + 프로파일 swap으로 scaffold 수정 0(grounding-spec 안 켜져도 다른 A2는 켜짐). scaffold 수정 강요 = (a)A2로 일반화 or (b)out-of-scope 재분류 — 벤치별 분기는 thesis 실패. out-of-scope 4축만 엔진 확장 필요(선언된 경계).** = 25벤치 확장의 검증가능 골격.
