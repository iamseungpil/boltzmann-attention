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

**공통 요구 = "산출물의 모든 atomic claim이 logged tool-call로 derivable"** — 이것이 규제 sourcing(`REGULATORY_DETERMINISM_SOURCING.md`) 결론(moat=검증가능성, EU AI Act Art.12 traceability)의 벤치 조작화이기도 함: provenance log가 곧 Art.12-급 감사 추적.

## §1 forward guard 적합성 (벤치 자격 — §17.9 3질문)
- (1) **구조-충실성** ✓: claim→provenance DAG 충실성이 1급 지표.
- (2) **감사가능 soundness** ✓: 마케팅·금융 제약(옵트아웃 제외, 예산 상한, 채널 규제문구, 교차판매 금지, PII 비노출)을 NL 정책문서로 제공 → 결정론 체커로 위반 채점 = SOPBench-형 게이트 접합점.
- (3) **재학습0 전이** ✓: 도메인(은행/보험/리테일)·스키마 swap held-out.
- capability 함정 회피: "frontier 이기나"가 아니라 **fabrication=0 보장 여부**가 축. frontier도 fabrication>0이면 우리 패키지({소형}×{게이트}×{전이})가 이기는 좌표.

## §2 ★핵심 설계 원칙 — GT는 back-derivation (사람 라벨 0)
1. **Seeded synthetic DB** (도메인별 생성기, seed 고정 = 결정론 재생성): 고객·거래·캠페인·상품·동의/옵트아웃 테이블. 분포에 의도된 사실(예: "20대 여성 세그먼트의 카드 이탈률이 Q2에 2배")을 **주입**(planted facts) → 정답이 *존재함*이 보장.
2. **결정론 GT 파이프라인**: planted fact마다 정답 산출 쿼리·집계 체인을 코드로 보유 = (정답 수치, provenance DAG). 태스크 NL은 그 위에서 역생성(back-instruct).
3. **채점 = 결정론 대조 우선, judge 최소화**: 수치 claim → exact/허용오차 match + tool-call log 대조(claim의 모든 수치가 어떤 호출 결과에서 왔는지). judge는 claim-extraction(산출물→atomic claims 파싱) 1곳에만 — 그리고 그 extraction도 schema-forced output으로 좁힘.
4. **⚠️ 순환 차단 (TaskBench 교훈 박제)**: back-instruct generator 모델 ≠ teacher/평가대상 모델 계열. GT 자체는 코드가 만들므로 TaskBench보다 순환 표면이 작음(NL 표현만 모델 생성).
5. **identity-match 채점**(index 금지)·should-style 분모 고정·alias 마스킹(도구 이름 암기 차단) — 기존 eval 규율 그대로 이식.

## §3 환경 스펙
- **DB**: SQLite/postgres, 도메인 3+(은행 카드·보험·리테일 멤버십), 스키마 의도적 이질화(컬럼명·정규화 수준 상이) → 전이 측정.
- **도구**: `sql_query`(읽기 전용), `segment_builder`, `campaign_cost_calc`, `uplift_stub`(결정론), `chart_writer`, `report_writer`, `paper_retriever`(T4), `policy_lookup`. 전부 결정론 스텁 = replay 가능.
- **제약 정책**: 도메인별 NL SOP 문서(옵트아웃·예산·규제문구·금지조합·PII). **Exp-B(NL→구조 induce) 경로와 동일 형식** → 우리 스택은 SOP→게이트 컴파일, baseline은 SOP를 프롬프트로.
- **T3 reference**: 인간 보고서 3~5종(실제 금융 마케팅 월간보고 형식 모사, 섹션·표 스키마 추출해 GT화). **reference의 수치는 구식 데이터의 것** → 복사 시 즉시 fabrication 검출(신규 DB와 불일치하도록 설계).
- **T4 페이퍼**: RFM·uplift modeling·CLV 등 방법론 페이퍼 요약본(저작권 회피 위해 자체 재서술) + "이 방법으로 X 분석" 요청. GT = 방법의 절차를 도구 체인으로 구현한 결정론 파이프라인.

## §4 지표 (사전등록 — 총점 헤드라인 금지, 축 분리)
1. **Fabrication rate** (헤드라인-precision): atomic claim 중 provenance 부재/모순 수치·사실 비율. **목표 주장: 게이트 스택=0 by construction, frontier/LLM-direct >0.**
2. **Grounded coverage** (헤드라인-recall): 태스크가 요구한 분석 항목(planted facts 기준) 중 올바른 provenance로 회수된 비율. = "precision=1서 recall 최대화" 그대로.
3. **Soundness violation rate**: 제약 위반(옵트아웃 포함 발송 제안, 예산 초과, 금지조합) — 결정론 체커.
4. **Format adherence** (T3): 섹션/표 스키마 매칭률.
5. **Method fidelity** (T4): GT 도구체인과의 절차 일치(노드/엣지 F1 — TaskBench 지표 재사용).
6. **Transfer**: held-out 도메인/스키마 Δ (재학습 0).
7. *(deferred)* optimality: 제안 효과성 — §17.9 리뷰6-5 정합, 2차 축으로만.

## §5 구축 단계 (zero-GPU 먼저, 측정 우선)
- **P0 (zero-GPU, 1~2일)**: 스키마·제약·도구 스펙·planted-fact 문법 동결 + 본 설계 리뷰(§7) 통과.
- **P1**: DB 생성기 + GT 파이프라인 (코드, 결정론). 단위검증 = planted fact 전수 회수(coverage 상한 100% 확인 — SOPBench Guard-2 방식).
- **P2**: back-instruct 태스크 생성 n≈50/도메인/패밀리 + claim-extraction 채점기. **pilot 10태스크로 채점기 신뢰도 먼저**(judge-인간 일치 확인 후 스케일).
- **P3**: baseline 매트릭스 — frontier API(GPT-5/o4급) vs 우리 스택(소형+게이트) vs 소형 base — fabrication/coverage/soundness 3축.
- **P4**: 학습 접합(gold-SFT→RFT, Exp-A 레시피 이식).

## §6 기존 자산 재사용 맵
| 자산 | 출처 | 재사용 |
|---|---|---|
| 결정론 게이트 사다리(ARGFIX~DGGATE) | SOPBench | 제약 집행 게이트 |
| alias 마스킹 | `build_tbox_planner_sft` 계열 | 도구 일반화 강제 |
| eval 규율(분모 고정·identity-match·should 분리) | SOPBench/TaskBench | 채점기 |
| 노드/엣지 F1 | TaskBench | T4 method fidelity |
| NL SOP→구조 induce | Exp-B(blind-E1) | 정책 컴파일 경로 |

## §7 리뷰 훅 (BLOCKING — P1 착수 전 답 필요)
- **R1 외적 타당도**: synthetic DB·planted facts가 실 CDP 워크로드 대표하나? → 실 스키마 1개(비식별) 입수해 생성기 캘리브레이션 권장.
- **R2 T4 GT 약점**: 페이퍼 해석 자유도로 "정답 도구체인"이 유일하지 않음 → GT를 "페이퍼→절차 명세 추출"까지 포함해 닫거나, valid-대안 인정 채점(TaskBench A-0와 동형 문제)으로.
- **R3 claim-extraction 신뢰도**: judge 의존 유일 지점 — schema-forced extraction의 인간 일치율 ≥95% 게이트.
- **R4 T2 채점가능성**: "유동적 제안"은 GT가 열려 있음 → 채점을 (i) 인용 수치의 provenance 정합 (ii) planted-fact 회수율 (iii) 모순 검출의 3축으로 한정하고 제안의 "질"은 deferred(optimality)로 — 아니면 T2가 judge-벤치로 퇴화.
- **R5 벤치 분담 중복**: method fidelity(T4)는 TaskBench와, soundness는 SOPBench와 부분 중복 — 본 벤치의 고유 기여 = **산출물-수준 fabrication/provenance**(둘 다 없음). 고유축이 헤드라인이어야 함.
