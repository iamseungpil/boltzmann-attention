# GROUNDED_BIZ_AGENT_BENCH — DB-grounded 금융·마케팅 에이전트 벤치마크 설계 (DRAFT v0.1, 리뷰 대기)
> 2026-06-10. 사용자 발주: "할루시네이션 없이 DB 조회 등으로 확보된 자료로 정확한 금융 마케팅·전략 조언·보고서 자동생성·paper-conditioned 분석을 오류 없이 수행하는가"를 테스트할 벤치마크.
> 지위 = **CDP/OISA 1차 타겟 도메인의 직접 벤치** — TaskBench(충실성)·SOPBench(soundness)가 못 덮는 "**산출물-수준 grounding**"(모든 수치·주장 ↔ 실데이터 provenance) 측정. thesis(`FIELD_GAP_LLM_VALUE_DESIGN.md` §17.9) 헤드라인을 사업 도메인에서 그대로 조작화: **fabrication=0(precision=1) 보장 하 grounded coverage(recall) 최대화**.

---

## §0 무엇을 측정하나 — 4 태스크 패밀리 (사용자 발주 그대로)

| ID | 태스크 | 입력 | 산출물 | 핵심 실패모드(측정 대상) |
|---|---|---|---|---|
| **T1** | grounded 마케팅 분석·제안 | NL 요청 + DB + 도구 | 수치 포함 분석·세그먼트 제안 | 수치 fabrication, 스키마 환각, 잘못된 집계 |
| **T2** | 전략 조언·방향 설정 | NL 질문 + DB + 도구 | 데이터 근거 첨부된 유동적 제안 | 데이터 안 읽고 일반론 답변(=ungrounded), 읽은 데이터와 모순된 결론 |
| **T3** | 보고서 자동 생성 | **인간 작성 보고서(형식 reference)** + DB(신규 기간/대상) | 형식 보존 + 데이터 추출·분석은 신규 계획·실행한 보고서 | 형식 붕괴, 옛 수치 복사(=reference 컨닝), 신규 수치 fabrication |
| **T4** | paper-conditioned 분석 | **마케팅 방법론 페이퍼** + NL 요청 + DB + 도구 | 페이퍼 방법을 도구로 구현한 결과 | 방법 오적용, 페이퍼에 없는 절차 fabrication, 도구 선택 오류 |
| **T5** | **실시간 마케팅 시나리오 설계 (멀티턴 composite — ★앵커, §0.5)** | 멀티턴 사용자 대화 + DB + 도구 + 실행계 액션(트리거/고객군 등록) | 퍼널분석→트리거 추천→**복수 방안 제안(모수·예상전환율 추정 포함)**→세부 조정→등록→최종 종합 | T1·T2의 모든 실패모드 + 추정치 fabrication, 등록 payload 오류, 턴-간 상태 불일치 |

**공통 요구 = "산출물의 모든 atomic claim이 logged tool-call로 derivable"** — 이것이 규제 sourcing(`REGULATORY_DETERMINISM_SOURCING.md`) 결론(moat=검증가능성, EU AI Act Art.12 traceability)의 벤치 조작화이기도 함: provenance log가 곧 Art.12-급 감사 추적.

## §0.5 ★앵커 사례 = CDP PoC 마케팅 워크플로 (리모트 실물 확인 2026-06-10)

리모트 `/home/woori/workspace_common/CDP/poc/uf_agent/uf_agent_deploy/data/` 실사 결과 (사용자 지칭 "agent/data"의 실경로):

**(1) `workflow/marketing_workflow.yaml` (v4.0, 12-turn 대화형 시나리오)** — 벤치가 일반화해야 할 *목표 행동의 구체 사양*:
- Phase 0: 상품 TOP5 조회→상품 선택 / Phase 1: 퍼널 분석→최대이탈 단계에 CEP 트리거 등록 / **Phase 2 = 방안1 "적금 상품 고관심 고객군 선별"**(4기준: 방문빈도 상위N%·최근 재방문·체류시간 상위20%·이자 시뮬레이션 실행 — 10%-단위 분포 분석→threshold 조정→CEP 등록) / **Phase 3 = 방안2 "유입경로 분석 고객군 선별"**(Removal Effect 기반, 경로 A금리비교형/B이벤트유입형/C검색유입형/D직접유입형→고전환 경로 선택 등록) / Phase 4: 최종 모수·예상전환 종합.
- **response_templates의 `variables` 목록(turn당 5~30개: `{topN_count}`, `{path_a_effect}`, `{total_conv}` …) = atomic-claim 사양 그대로** — 벤치 채점기의 claim-extraction 스키마를 이 변수 집합에서 직접 도출(judge 자유도 최소화, §7-R3 해소 경로).
- PoC는 12-turn 고정 스크립트+가안 하드코딩(예: 전환율 "16-22%")이지만 **벤치는 이를 푼다**: 동일 산출물(트리거 추천·방안별 모수/전환율 추정·분포표·등록 payload·최종 종합)을 모델이 *자율 도구 계획*으로 생성, 수치는 전부 DB-derived여야 함. PoC 고정 스크립트는 **scripted-oracle 상한 검증**(run_scripted 패턴)으로만 사용.

**(2) `workflow/tool_dags/*.yaml` (분석유형별 결정론 실행 DAG, `dag_templates.yaml`에 21유형)** — **★결정론 authoring의 한계 실증이자(사용자 교정 2026-06-10), 벤치의 대조군**:
- 노드 = `{id, type(resolver/param_setter/data_fetch/analysis/reporter), tool, depends_on, parallel_with, condition, params}` + `${node.output}` 데이터플로 바인딩 (예: funnel_analysis = resolve_funnel∥set_period∥resolve_segment → calculate_funnel → analyze_dropoff∥time_analysis → generate_report).
- **⚠️이 DAG들은 "정답 경로의 열거"가 아니다**: 고객군 생성 방법은 수천 가지·도구도 수천 개이고, PoC의 2방안/21템플릿은 그중 손으로 박은 극소수 = **per-path 결정론 authoring 비용의 실증**(thesis의 "특허=결정론 authoring / front-end가 per-domain·per-path 재구축비 제거" 구도 그대로). **벤치가 측정할 본체 = LLM이 거대 도구 카탈로그(GraphRAG/도구 위키/리스트)를 보고 적절한 조합을 *생성*하는 능력**이지, 열거된 템플릿 중 선택이 아님.
- 따라서 tool_dags의 용도는 둘로 한정: ① **대조군(Arm-T)** — 열거-템플릿 매칭 방식의 커버리지 상한 실측(커버된 경로에선 강하나 그 밖 0 = 결정론 한계 정량화), ② DAG *스키마*(노드/엣지/바인딩 형식)는 모델이 **생성하는 플랜의 표현 형식**으로 재사용(플랜=구조화 산출물→게이트가 검증).
- SOPBench dirgraph 게이트(DGGATE)와의 접합은 유지하되 방향 정정: 게이트는 *열거된 정답 DAG와 비교*하는 게 아니라 **생성된 플랜의 타당성**(도구 입출력 타입 정합·선행조건 충족·제약 비위반)을 검증.

**(3) 도구·스키마 카탈로그**: `unified_tools.yaml`(130KB)·`tools_schema{,_core,_compact,_full}.yaml`·`tool_selection_rules.yaml`(58KB)·`tool_capability_graph.yaml`·`schema_mariadb.yaml`·`semantic_layer.yaml`(197KB)·`ontology_layer.yaml`(120KB) — **벤치 도구 카탈로그·DB 스키마·온톨로지의 시드를 전부 실물에서 채취 가능**(임의 발명 불요). `docs/AIPoC_인사이트보고서/`의 워드/PPT 블록 템플릿+항목별 프롬프트 = **T3(보고서 생성)의 reference 형식 실물**.

**(4) 실행계 액션**: `register_cep_trigger`/`register_interest_segment`/`register_path_segments` + SmartCEP 팝업 필드 사양 = **soundness 게이트 대상인 부수효과 액션**(SOPBench의 goal-call과 동격) — 옵트아웃 포함 발송·모수 상하한·트리거 중복 등록이 위반 채점 지점.

**PoC→벤치 갭(=벤치의 존재 이유)**: PoC는 (i) 12-turn 스크립트 고정(자율 계획 없음) (ii) 수치 가안 하드코딩(grounding 미검증) (iii) 단일 상품·단일 도메인 (iv) 제약 게이트 없음 (v) **★방법·도구 공간이 손-열거(2방안/21DAG)된 폐집합 — 실공간은 수천 방법×수천 도구의 개집합**. 벤치는 이를 자율계획/provenance 채점/도메인 swap/soundness 게이트/**open-set 도구 조합**으로 일반화 — 사용자 발주 그대로 "고객이 원하는 바를 골라 추천하고, 고객이 도구·파라미터를 조정하면, 실제 보고서/고객군 생성".

**⇒ 노동 분업(thesis 구도와 동일)**: **LLM = 카탈로그 검색(GraphRAG/위키)+조합 생성+고객 의도 매칭**(창의·개집합·열거 불가) / **결정론 게이트 = 조합된 플랜의 검증(타입·선행조건·제약)+실행+provenance**(보장·감사). 열거 DAG 방식의 한계가 LLM 필요성의 논거이고, LLM 환각이 게이트 필요성의 논거 — 벤치는 이 분업의 양쪽을 다 측정한다.

## §1 forward guard 적합성 (벤치 자격 — §17.9 3질문)
- (1) **구조-충실성** ✓: claim→provenance DAG 충실성이 1급 지표.
- (2) **감사가능 soundness** ✓: 마케팅·금융 제약(옵트아웃 제외, 예산 상한, 채널 규제문구, 교차판매 금지, PII 비노출)을 NL 정책문서로 제공 → 결정론 체커로 위반 채점 = SOPBench-형 게이트 접합점.
- (3) **재학습0 전이** ✓: 도메인(은행/보험/리테일)·스키마 swap held-out.
- capability 함정 회피: "frontier 이기나"가 아니라 **fabrication=0 보장 여부**가 축. frontier도 fabrication>0이면 우리 패키지({소형}×{게이트}×{전이})가 이기는 좌표.

## §2 ★핵심 설계 원칙 — GT는 "정답 경로"가 아니라 "검증기" (사람 라벨 0, 정답 열거 0)

**★설계 전환(사용자 교정 2026-06-10): 방법 수천·도구 수천의 개집합에서 "유일 정답 도구체인"의 열거는 불가능하고, 열거 시도 자체가 PoC-DAG의 한계를 벤치에 복제하는 것.** 따라서 GT는 두 층으로 분리:

1. **Seeded synthetic DB + planted facts** (도메인별 생성기, seed 고정): 고객·거래·캠페인·상품·동의/옵트아웃 테이블에 의도된 사실(예: "20대 여성 세그먼트의 카드 이탈률이 Q2에 2배")을 주입 → **사실-층 GT(폐집합)**: 어떤 도구체인을 쓰든 도달해야 할 수치·사실은 결정론적으로 유일. fabrication/coverage 채점은 이 층에서 — *경로 무관*.
2. **검증기-층 GT(개집합)**: 도구마다 **형식 명세**(typed 입출력·선행조건·효과 — `tool_capability_graph.yaml`이 시드)를 부여 → **모델이 조합한 *어떤* 플랜이든** (i) 타입 정합 (ii) 선행조건 충족 (iii) 제약 비위반 (iv) 실행 replay 성공을 **결정론 검사** 가능. "정답 체인과 일치하나"를 묻지 않고 **"유효한 체인인가 + 사실-층 GT에 도달했나"**를 묻는다 — valid-대안 무한 수용이 1급 설계(TaskBench A-0 valid-대안 문제의 구조적 해소).
3. **채점 = 결정론 대조 우선, judge 최소화**: 수치 claim → planted-fact match + tool-call log provenance 대조. judge는 claim-extraction 1곳에만(schema-forced; T5는 PoC variables로 사전 정의).
4. **⚠️ 순환 차단 (TaskBench 교훈 박제)**: back-instruct generator 모델 ≠ teacher/평가대상 모델 계열. GT 자체는 코드(DB+검증기)가 만들므로 순환 표면은 NL 표현뿐.
5. **identity-match 채점**(index 금지)·should-style 분모 고정·alias 마스킹(도구 이름 암기 차단) — 기존 eval 규율 그대로 이식.

## §3 환경 스펙
- **DB**: SQLite/postgres, 도메인 3+(은행 카드·보험·리테일 멤버십), 스키마 의도적 이질화(컬럼명·정규화 수준 상이) → 전이 측정. **기준 도메인 스키마 = PoC `schema_mariadb.yaml`+`semantic_layer.yaml`에서 채취**(은행 행동로그: 페이지 방문·퍼널·세션·상품), planted facts = 최대이탈 단계·기준별 모수·경로별 Removal Effect 값 주입.
- **도구 = ★대규모 개집합 카탈로그 (수백→수천 스케일)**: PoC 실물(`unified_tools.yaml` 130KB·`tool_capability_graph.yaml`·tool_dags 21유형)을 시드로 **합성 확장**(파라미터 변형·도메인 특화 변종·**동음이의어/유사기능 distractor** — CDP Patent1 실측 함정[도구폭발 82K토큰·동음이의어 58%] 재현) + `sql_query`(읽기 전용), `report_writer`, `paper_retriever`(T4), `policy_lookup`, 실행계 `register_cep_{trigger,segment}`(부수효과·게이트 대상). 전부 결정론 스텁(typed 명세 포함) = replay·검증 가능.
- **★도구 발견 채널 = 벤치의 1급 환경요소**: 카탈로그가 컨텍스트에 다 안 들어가는 스케일이므로(의도적), 에이전트는 **도구 위키/GraphRAG/카탈로그 검색 도구**(`tool_search`, `tool_doc_lookup`, capability-graph 질의)로 후보를 발견해 조합해야 함. 검색 채널 자체도 로깅 = 도구-선택 provenance.
- **제약 정책**: 도메인별 NL SOP 문서(옵트아웃·예산·규제문구·금지조합·PII). **Exp-B(NL→구조 induce) 경로와 동일 형식** → 우리 스택은 SOP→게이트 컴파일, baseline은 SOP를 프롬프트로.
- **T3 reference**: 인간 보고서 3~5종(실제 금융 마케팅 월간보고 형식 모사, 섹션·표 스키마 추출해 GT화). **reference의 수치는 구식 데이터의 것** → 복사 시 즉시 fabrication 검출(신규 DB와 불일치하도록 설계).
- **T4 페이퍼**: RFM·uplift modeling·CLV 등 방법론 페이퍼 요약본(저작권 회피 위해 자체 재서술) + "이 방법으로 X 분석" 요청. GT = 방법의 절차를 도구 체인으로 구현한 결정론 파이프라인.

## §4 지표 (사전등록 — 총점 헤드라인 금지, 축 분리)
1. **Fabrication rate** (헤드라인-precision): atomic claim 중 provenance 부재/모순 수치·사실 비율. **목표 주장: 게이트 스택=0 by construction, frontier/LLM-direct >0.**
2. **Grounded coverage** (헤드라인-recall): 태스크가 요구한 분석 항목(planted facts 기준) 중 올바른 provenance로 회수된 비율. = "precision=1서 recall 최대화" 그대로.
3. **Soundness violation rate**: 제약 위반(옵트아웃 포함 발송 제안, 예산 초과, 금지조합) — 결정론 체커.
4. **Plan validity rate**: 모델이 조합한 플랜의 검증기 통과율(타입·선행조건·실행 replay — §2-2). **유일-정답 비교 아님.**
5. **Tool retrieval/composition**: 대규모 카탈로그에서 (i) 필요 capability 도구 발견율 (ii) distractor(동음이의어·유사기능) 오선택률 (iii) 조합 깊이별 validity.
6. **Format adherence** (T3): 섹션/표 스키마 매칭률.
7. **Method fidelity** (T4 한정, 보조): 페이퍼가 절차를 *명시*한 경우만 노드/엣지 F1 — **개집합 태스크(T1/T2/T5)에는 적용 금지**(유일 정답 없음, §2 전환).
8. **Transfer**: held-out 도메인/스키마 Δ (재학습 0).
9. *(deferred)* optimality: 제안 효과성 — §17.9 리뷰6-5 정합, 2차 축으로만.

## §5 구축 단계 (zero-GPU 먼저, 측정 우선)
- **P0 (zero-GPU, 1~2일)**: **PoC 자산 인벤토리 확정**(tool_dags 21유형 중 벤치 1차 채택분·unified_tools 추출·schema/semantic_layer 채취·response_templates→claim 스키마 변환) + 스키마·제약·planted-fact 문법 동결 + 본 설계 리뷰(§7) 통과.
- **P1**: DB 생성기 + **사실-층 GT 계산기**(planted-fact 산출 코드 — 벤치 내부용이지 정답 경로 열거 아님) + **검증기**(도구 typed 명세→플랜 validity 체커; tool_dag 실행기는 실행엔진으로만 재사용). 단위검증 = planted fact 전수 회수 + 검증기 OVER/UNDER 0(알려진 유효/무효 플랜 셋으로 — SOPBench Guard-2 방식).
- **P2**: back-instruct 태스크 생성 n≈50/도메인/패밀리 + claim-extraction 채점기. **pilot 10태스크로 채점기 신뢰도 먼저**(judge-인간 일치 확인 후 스케일).
- **P3**: baseline 매트릭스 — **Arm-T(결정론 템플릿 매칭: 열거 21-DAG 중 선택, PoC 방식)** vs frontier API(GPT-5/o4급) vs 우리 스택(소형 front-end+게이트) vs 소형 base — fabrication/coverage/soundness/plan-validity 축. **★Arm-T의 커버리지 절벽(커버된 경로 강함·그 밖 0)과 LLM-arm의 커버리지 확장이 thesis 가치명제의 직접 실측**(결정론 authoring 한계 vs front-end 일반화).
- **P4**: 학습 접합(gold-SFT→RFT, Exp-A 레시피 이식).

## §6 기존 자산 재사용 맵
| 자산 | 출처 | 재사용 |
|---|---|---|
| 결정론 게이트 사다리(ARGFIX~DGGATE) | SOPBench | 제약 집행 게이트 |
| alias 마스킹 | `build_tbox_planner_sft` 계열 | 도구 일반화 강제 |
| eval 규율(분모 고정·identity-match·should 분리) | SOPBench/TaskBench | 채점기 |
| 노드/엣지 F1 | TaskBench | T4 method fidelity |
| NL SOP→구조 induce | Exp-B(blind-E1) | 정책 컴파일 경로 |
| **마케팅 시나리오 사양·턴별 claim 변수** | **PoC `marketing_workflow.yaml`** | T5 태스크 사양·claim-extraction 스키마 |
| **결정론 분석 DAG 21유형(+실행기)** | **PoC `tool_dags/`·`dag_templates.yaml`** | 실행엔진·**Arm-T 대조군(열거-템플릿 한계 실측)**·플랜 표현형식 — *정답 열거로는 사용 금지(§2)* |
| **도구 capability graph** | **PoC `tool_capability_graph.yaml`** | 검증기 typed 명세 시드·GraphRAG 검색 채널 |
| **도구·스키마·온톨로지 카탈로그** | **PoC `unified_tools.yaml`·`semantic_layer.yaml` 등** | 벤치 도구/DB 시드 |
| **보고서 블록 템플릿·항목별 프롬프트** | **PoC `docs/AIPoC_인사이트보고서/`** | T3 reference 형식 |

## §7 리뷰 훅 (BLOCKING — P1 착수 전 답 필요)
- **R1 외적 타당도**: synthetic DB·planted facts가 실 CDP 워크로드 대표하나? → 실 스키마 1개(비식별) 입수해 생성기 캘리브레이션 권장.
- **R2 T4 GT 약점**: 페이퍼 해석 자유도로 "정답 도구체인"이 유일하지 않음 → GT를 "페이퍼→절차 명세 추출"까지 포함해 닫거나, valid-대안 인정 채점(TaskBench A-0와 동형 문제)으로.
- **R3 claim-extraction 신뢰도**: judge 의존 유일 지점 — schema-forced extraction의 인간 일치율 ≥95% 게이트. **완화 경로 확보(§0.5-(1))**: T5는 PoC response_templates의 `variables` 집합이 claim 스키마를 *사전 정의* → extraction이 변수-바인딩 매칭으로 환원, judge 자유도 최소.
- **R6 T5 멀티턴 통제**: user_sim이 PoC `expected_queries` 분포로 행동하되, 12-turn 고정 재현이 되지 않게(스크립트 암기 채점 방지) 턴 순서·**방안 후보 생성(고정 2개 아닌 open-set에서 N개 제안)**·도구/threshold 조정을 인스턴스별 샘플링. PoC 고정 스크립트는 scripted-oracle 상한 검증 전용.
- **R7 검증기 완전성(★개집합 전환의 사활)**: 검증기-층 GT(§2-2)가 무효 플랜을 통과시키면(UNDER) fabrication 누수, 유효 플랜을 기각하면(OVER) coverage 과소측정 = Guard-2 동형 위험. P1의 OVER/UNDER-0 단위검증 셋(알려진 유효 플랜 = 21 tool_dags + 수동 변형 / 무효 플랜 = 타입 위반·선행 누락 주입) 통과 전 P2 진행 금지.
- **R8 distractor 난이도 캘리브레이션**: 동음이의어/유사기능 distractor가 너무 쉬우면 retrieval 지표 천장, 너무 어려우면(기능 동일 도구) "오선택"이 valid-대안과 구분 불가 — distractor는 **검증기가 무효 판정하는 것만**으로 한정(기능 동일 도구는 valid-대안으로 수용).
- **R4 T2 채점가능성**: "유동적 제안"은 GT가 열려 있음 → 채점을 (i) 인용 수치의 provenance 정합 (ii) planted-fact 회수율 (iii) 모순 검출의 3축으로 한정하고 제안의 "질"은 deferred(optimality)로 — 아니면 T2가 judge-벤치로 퇴화.
- **R5 벤치 분담 중복**: method fidelity(T4)는 TaskBench와, soundness는 SOPBench와 부분 중복 — 본 벤치의 고유 기여 = **산출물-수준 fabrication/provenance**(둘 다 없음). 고유축이 헤드라인이어야 함.
