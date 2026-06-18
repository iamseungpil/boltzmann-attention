# ★ EXPERIMENT DESIGN — MASTER (단일 권위·2026-06-18 전면 개편·LOCKED)

> **이 문서 = theory + 아키텍처 + 실험의 *고정본*.** 방향이 흔들리면 이 문서 §0–§4 + [`FIXED_VS_VARIABLE.md`](FIXED_VS_VARIABLE.md)만 다시 읽는다. 세부는 §8 문서지도.
> 고정/변경 경계는 [`FIXED_VS_VARIABLE.md`](FIXED_VS_VARIABLE.md)(단일 권위)·메모리 `05-fixed-vs-variable`이 정의 — **여기서 재정의하지 않는다.** 06-04~06-17 역사 블록(SOPBench-rung·dirgraph-emit·두날개 등)은 §9로 강등(detail/근거).

---

## §0. 목표 (한 문장·불변)
**자연어 멀티턴 tool-use 요청을, 작은 on-prem LLM이 *학습된 도메인-일반 스킬*(NL→formalize)로 풀되, 깊은 실행·구체값·정책집행은 *고정 결정론 엔진*에 offload하고, 새 도메인·새 벤치는 *ABox config 교체만*으로 재학습0 전이한다.** = 작은 모델 + 구조가 큰 모델급 tool-use에 비용·주권·신뢰로 도달.

## §1. 이론 (LOCKED — `NL_PROCEDURE_OFFLOAD_THEORY`·`DECOMPOSITION_OPTIMALITY`)
1. **𝔤 / G / 구체 삼분**: 모든 연산 = 유한차원 **대수 𝔤**(생성원·추상·도메인-불변·저차원) + exp이 생성하는 **군 G**(궤도·실행·무한) + **구체 좌표**(이 카탈로그·값). **LLM은 𝔤 명명(NL→formalize)에 강하고 군-실행에 약하다. 결정론이 군을 실행. 구체는 ABox.**
2. **왜 분담**: 언어=유계-깊이 병렬 인지를 위해 진화한 압축 인터페이스 → 대부분 얕은 병렬 연상(TC⁰ forward pass가 잘함). NL 속 깊은 절차(most/best·산술·다단 = **표기깊이 d(e)**)는 인간도 외부도구로 offload. ⇒ **LLM=얕은연상+절차-타입 분류 / 결정론=깊은 실행.**
3. **유계 절차예산 B(L,width)**: 고정모델 forward는 직렬-깊이 유계 → d(e)>B 하락. scale은 B를 내부암기로(유계·비쌈)·CoT는 외부토큰(오차누적)·**결정론 엔진은 B=∞**(정확·저비용). = "binding 벽 ≠ scale"의 근본.
4. **offload는 수학적 필연**: 임의 계산=Rice·임의 정책게이트=HRU(1976) 결정불가 → 유한 학습 불가 → 결정론 도구가 *정확히* 함. (빅모델도 내재화하면 근사·환각.)
5. **학습 대상 = 𝔤-식별** (저차원·도메인-불변 → 학습·전이가능). 군-실행은 학습 안 함(엔진).
6. **생성원 2축 closure** (`GENERATOR_ALGEBRA_DESIGN`): **flow 생성원 P1-P9**(`PRIMITIVE_COVERAGE_MATRIX`·Böhm–Jacopini) + **content 생성원 8 op**(filter/argmax/argmin/rank/comparative/substitute/create/project·Codd+functional). **transactional tool-orchestration scope서 유한·닫힘**(census orphan=0). 정직: finiteness/closure만·**minimality 금지**(Kozen-Tseng).

## §2. 아키텍처 = FIXED vs VARIABLE (LOCKED — 전문 = `FIXED_VS_VARIABLE.md`)
| | 정체 | 학습/전이 |
|---|---|---|
| **고정: TBox (LLM weights)** | 𝔤-식별 = **NL→formalize**(4 facet)+환원불가 추론 | 1회 학습 후 **FROZEN** |
| **고정: Scaffold (결정론 엔진 코드)** | 군-실행 = **GateInterpreter**(GATE_SPEC 해석)·resolve·per-step verify·step-orchestrator·**per-bench 분기 0** | 학습 아님·절대 미수정 |
| **변경: ABox (데이터)** | 구체 = A1(카탈로그)·**A2(정책 NL→GATE_SPEC·유일 난제)**·A5(문법)·attr-type/vocab. **실무 정의 = `FIXED_VS_VARIABLE §2bis`**(config 데이터·DAG 아님·온톨로지 아님·GATE_SPEC dict가 실물) | per-bench **swap** |

- **NL→formalize 4 facet** (LLM이 *intensional* selector emit·concrete 아님): (1) flow-타입/순서[SOPBench]·(2) data-flow threading[TaskBench]·(3) content op-명명[Synth]·(4) operand formalize(attr/set·keep-rest)[Synth/CFB]. ("오케스트레이션"=facet 1뿐.)
- **offload 경계 = 둘**: ①스텝내부 typed-selector[LLM] │ concrete-resolution+assembly+deep-exec[엔진] ②스텝사이 결합 verdict→emit/defer(decidable-비율 측정).
- **전이 ≡ ABox만 swap·TBox·Scaffold unchanged.** TBox나 Scaffold per-bench 수정 = bench-베이킹 = thesis 실패.
- **★현 위반(고칠 것)**: `tau2/t2_gate.py RetailGate`가 retail 도구·정책을 코드에 하드코딩 = 위반 → **`GateInterpreter(gate_spec)` 통일이 prerequisite keystone**(§4 조건③).

## §3. 학습 (LOCKED)
- **TBox = 4벤치(SOPBench+TaskBench+CFB+Synth) native function-calling 1회 학습 → FROZEN.** 각 벤치가 다른 facet 학습(§2 표). 공통표현=표준 OpenAI FC(`tool_calls{name,args}`·vLLM hermes·표현 발명 불요).
- **분해 멀티-스페셜리스트** (단일 merged LoRA 폐기): 기존 per-bench LoRA = facet 스페셜리스트·머지 안 함. 얽힌 결정=typed 스텝 분해·결합=결정론 우선·잔여만 consensus LoRA. (`INTEGRATED_TBOX_DESIGN`)
- **학습 = 𝔤-식별만**(op-명명·flow-type·threading·operand formalize=intensional). 실행/구체/binding은 학습 안 함(§23D: binding 학습→라우팅 퇴행).
- 보상 = 결정론(rule-oracle·graph-F1·DB∧NL∧comm). **LLM-judge 보상 금지**(`feedback-selector-verifier-deterministic`).

## §4. 실험 (LOCKED) = maximal thesis 가능성 = 4조건
**명제(최강형)**: ⟨TBox 고정 + Scaffold 고정⟩ + ABox-only swap → transactional 벤치 전부. 가능성 = 아래 4조건 전부 통과. (`ISOLATION_EXPERIMENTS_2026_06_18.md`가 ②③ 실험 권위.)

| 조건 | 내용 | 실태 | 실험 |
|---|---|---|---|
| **① closure** | 벤치 primitive ⊆ P1-P9 ∪ 생성원 | transactional 닫힘(census)·**AppWorld/WebArena/TravelPlanner/OSWorld scope-out**("25 전부"=과주장) | census·적대탐색 |
| **② TBox formalize-전이** | 고정 TBox가 새 벤치서 NL→formalize 정확(ABox swap) | facet 3 cross-bench **증명**(§21)·1·2 within-bench 전이증거·4 최약 | **4-facet 격리실험**(formalize 출력 채점·confound 제거·재학습0) |
| **③ scaffold 벤치-일반성** | 한 엔진+ABox-only·per-bench 분기 0 | **현재 FALSE**(RetailGate=retail 하드코딩) | **GateInterpreter 통일 keystone**(prerequisite)·설계=`GATE_INTERPRETER_UNIFICATION_DESIGN_2026_06_18.md`(유한 5 gate-kind·gate_spec ABox·분기0 grep·ABox-ablation·retail+airline+SOP unchanged) |
| **④ ABox 충분성** | per-bench ABox 기계도출 | A1=기계·**A2(정책→GATE_SPEC)=유일 난제** | A2 front-end 자동화·LODO |

**측정 규율(불변)**: formalize *출력* 채점(e2e/STACK/net 아님 — 이게 전이 해석 진동의 원인)·ABox-swap 재학습0·per-bench 분기 0(grep `if bench`)·결정론 보상·in-domain ceiling+base floor 동시(전이 gap 정의).

**facet 전이 증거 (궤적 census 후·권위=`reports/facet_rft_2026/`·과주장 금지)**:
- facet 3 content-op = 격리 cross-bench **증명**(§21 synth-only LoRA→retail+airline 0.03→0.44·held-out 1.00). **단 op-IR 포맷→native 생존이 keystone**(§23E·`facet3_native.sh` 가동중).
- facet 1 flow = SOPBench cross-domain 양성(Exp-5a 77.3%·Exp-4a 학습TBox 게이팅전이)·단 STACK(LEARN/scaffold 분할=LODO).
- facet 2 threading = 규율 전이 실증·어휘간섭에 가려짐(`TASKBENCH:184`)·간섭은 grounded-copy/ABox 제거(:192).
- facet 4 operand = 진짜 최약(§21 천장 0.44=성분 B)·**격리전이 미측정 = theory(LLM) vs 리뷰어(offload) 판별 실험**.
- **전부 cross-bench→τ² ABox-swap은 측정 대상**(기반 가정 아님).

## §5. 벤치마크·메트릭 (settled·detail 참조)
- **학습 = SOPBench+TaskBench+CFB+Synth / 테스트 = τ²·SOP-Bench(Amazon)** — transactional·ABox-swap 재학습0. (`CROSS_BENCH_TRANSFER_PLAN`·`BENCH_PORTFOLIO_FRAMEWORK_DESIGN`.)
- **scope-out**(closure 밖): AppWorld·WebArena·TravelPlanner·OSWorld.
- **메트릭 = §1.6 배터리**(`reports/facet_rft_2026/research_framework_metrics_2026_06_12.md`): 헤드라인 tier=각 벤치 네이티브 공식(TB F1·SOPBench success·τ² pass^1/pass^k)·framework tier=교차벤치 census(F1 어댑터비용·F2 전이보존·F3 일관성·F4 무위반·F5 회수율·F6 abstain·F7 비용). 집계평균 금지·per-bench 개별·리더보드 4-tuple(user-sim·judge·trials·split) 명시.
- 보상 전부 결정론(LLM-judge 0).

## §5b. 비용 회계 (LOCKED — thesis 핵심 = 싸게 먹혀야·이미 committed)
**비용-효율이 헤드라인의 절반**(나머지=전이). 두 축 + 3 비교대상 모두 *측정*(추정 금지). 권위 = §1.6 배터리 F1·F7·`DECOMPOSITION_OPTIMALITY §B`·`FIELD_GAP` 비용축.

- **축1 = ABox 생성 비용**(one-time·벤치당·amortize) = **F1**: per-part 정직(`FIXED_VS_VARIABLE §2bis`) — **A1=전 tool-use 공통(차등 아님·상쇄)·A5=파생(0)·attr-vocab=소량 수동(비숫자 ordinal 순서)·A2(정책→GATE_SPEC)=차등 비용 본체·front-end 자동화 타깃**. F1 장부 상시 갱신. 실측 "airline/telecom 0줄"=*A2 컴파일*(Fable-5)이 0·A1/attr-vocab은 별도. thesis = A2가 front-end로 소거.
- **축2 = 추론 비용**(recurring·쿼리당) = **F7**: 토큰(모델-불변) + USD(스냅샷) + **cost-of-pass=E[비용]/R** + accuracy×cost Pareto. 소형 7B + 결정론 엔진(0-model-recurring) + MSC(최소입력).
- **TCO**(DECOMP §B): on-prem은 recurring 지배 → 분담이 최소화.
- **★3 비교대상**: (a) monolith 빅LLM = 추론비용 높음·도메인별 프롬프트 / (b) run_scripted(결정론 전부) = 도메인별 절차 손코딩 비용 높음·**전이 불가** / (c) 우리 = A2만(자동화)·7B 추론·ABox-swap. **= monolith를 추론비용서·run_scripted를 전이서 Pareto-지배.**
- 정직: F1=우리 발명 지표(문헌 무표준)·novelty 플래그. cost는 per-bench 개별·가격 스냅샷 날짜 명시.

## §6. 성공 게이트 (사전등록)
- **G-②(facet 전이)**: 고정 TBox+ABox-swap으로 facet formalize-출력 held-out≈in-domain·base 초과. facet별 판정.
- **G-③(scaffold)**: 한 GateInterpreter+ABox-only가 retail+airline+SOP 작동·grep `if bench`=0·ABox-ablation(빈/틀린 spec→붕괴).
- **G-전이(헤드라인)**: 같은 TBox+엔진 unchanged·ABox swap만으로 held-out 벤치 작동·재학습0.
- **G-Pareto**: 협업(TBox+엔진+ABox) > base·monolith(탐지가능 오차클래스서 상대 지배).
- **G-비용(§5b)**: 새 벤치 ABox 생성비용(F1) 하향·추론비용(F7) < monolith·cost×accuracy Pareto서 우리가 지배점. A2 자동화로 벤치당 수동 LOC→0 추세.
- **정직**: 절대수 약속 금지(라우팅 천장 ~0.44·§5b Risk B)·헤드라인=상대 Pareto+전이.

## §7. 인프라 불변 (`30-remote-env`·반복사고 방지)
- 모든 실험 = woori 리모트 GPU. 로컬=git 미러. 전송=git만(SFTP 금지). 시크릿 커밋 금지.
- 실행=`ssh_run.py`(--cmd 큰따옴표 금지·stdin 파이프). seka_env(py3.12). 측정만 "실측" 인용·**fabrication 금지**.
- vLLM kill=PID직접·GPU별 격리·2잡/48GB OOM주의. SSH EXIT-1=검증 후 재시도.

## §8. 문서 지도 (마스터=이 문서; 나머지=detail)
**현 권위 (2026-06-18):**
| 문서 | 역할 |
|---|---|
| **`FIXED_VS_VARIABLE.md`** | ★★고정/변경 경계 단일 권위(§2 전문) |
| **`ISOLATION_EXPERIMENTS_2026_06_18.md`** | ★★조건 ②③ 격리 실험(4 facet + 스캐폴드 통일 §S) |
| **`GATE_INTERPRETER_UNIFICATION_DESIGN_2026_06_18.md`** | ★★조건 ③ keystone: RetailGate/SOP gate → 하나의 GateInterpreter(유한 gate-kind·gate_spec=ABox·분기0) |
| **`ABOX_CONFIG_FORMALIZATION_DESIGN_2026_06_15.md`** | ★ABox=JSON config로 출력형식 고정(xgrammar TYPE강제·rigid-forcing 아님)·4-way(TYPE=xgrammar/CONTENT=LLM/concrete=결정기/변환=scaffold)·config-conditioned FC output-type 학습=전이메커니즘 |
| **`INTEGRATED_TBOX_DESIGN_2026_06_18.md`** | ★분해 멀티-스페셜리스트 + resolve_selection + ABox-swap |
| **`NL_PROCEDURE_OFFLOAD_THEORY_2026_06_17.md`** | ★★이론(𝔤/G/구체·B-budget·offload 필연·§10 에너지-Lie) |
| **`GENERATOR_ALGEBRA_DESIGN_2026_06_17.md`** | ★★생성원 2축 closure·25벤치 적대 딥리서치 |
| **`THESIS_STATEMENT_2026_06_16.md`** · **`DECOMPOSITION_OPTIMALITY.md`** | thesis 수렴본·분담 라우팅·Pareto-지배 |
| **`PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`** | P1-P9 분류·census·도출닫힘 |
| **`CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md`** · **`R1B_PROVENANCE_DESIGN_2026_06_14.md`** | 4벤치 배치·native 재학습·R1b provenance |
| **`EXPRESSION_DIVERSITY_TRANSFER_DESIGN_2026_06_17.md`** | 다양성 D\*·K-sweep |
**결과 권위본:**
| `ma/M_A_RESULTS.md §1-27` | synth/생성원/facet 실증(§15-23 핵심) |
| `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` | ★SOPBench 전이(Exp-5a 77.3%·Exp-4a)·천장 34/48 |
| `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` | ★TaskBench(§8 census 두-힘·§9.5b grounded-copy) |
| ⚠️ **전이/궤적 판정 = 반드시 `reports/facet_rft_2026/`도 grep**(`scripts/distill/`만 보면 over-correction 재발·2026-06-18 실사고) |

## §9. 역사 (대체됨·근거/detail만·재론 금지)
SOPBench-rung 라인(`RUNG1_*`·`RESIDUAL_*`·`TASK_CONSTRAINT_*`·dirgraph-emit·xattn/steering·LLM-선별기 SEL-2/4/5·diffusion·ⓟ1 결정론 핵심) = **전부 superseded**. 두날개(06-16)·plan-X(06-14)·primitive 격상(06-15) = 현 라인에 *흡수*. 상세·결과는 `HANDOFF_2026_06_*`·결과 권위본. 진행로그=`AUTONOMOUS_PROGRESS_2026_06_14.md`.

## §10. 정직 (과대주장 금지·anti-drift)
- closure=transactional만("25 전부" 금지). minimality 금지. Lie geometry 실증·에너지 정식화-only.
- facet 전이 = facet 3만 cross-bench 증명·1·2·4=측정 대상(과주장/과소진술 둘 다 금지).
- scaffold "전이"라는 말 금지(scaffold=불변·ABox가 전이).
- 헤드라인=상대 Pareto+전이(절대수 약속 금지).

**규칙: 목표·이론·아키텍처·실험 변경은 이 문서 §0–§4 + `FIXED_VS_VARIABLE.md`에서만. detail 문서는 구현 세부만. 고정/변경 경계 재론 금지.**
