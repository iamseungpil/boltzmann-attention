# LB — 레버 베이스 엔진 7개 (새 코드베이스 · 2026-09-08)

정본 `research_base/NEW_RESEARCH_BASE.md` §2 의 LB1~LB7 을 **엔진 7개 + 조정기 1개**로 구현한 독립 코드베이스다.
구 `scripts/distill/tau2/`(4.2만 줄 · 레버 185개 · 플래그 358개)를 import 하지 않는다. 도는 실험(rep1·rep2·base)은
리모트의 구 트리를 쓰므로 이 디렉토리는 그 조건을 건드리지 않는다.

| 파일 | 기전 | 하는 일 (규칙 하나) | 선언(A2) 섹션 |
|---|---|---|---|
| `lb1_requirements.py` | F1 | 정책이 명명한 선행조건이 다 돌기 전엔 그 호출을 열지 않는다 · 진행 중 절차의 금지를 지킨다 | `LB1.prerequisites · gates · procedures` |
| `lb2_decision.py` | F2 | 정책이 고정한 산수·비교는 엔진이 레코드 위에서 한다(모델은 키만) | `LB2.computations` (ratio_cap · distinct) |
| `lb3_citation.py` | F2 | 모델이 쓰는 값·이름은 출처(레코드·손님·레지스트리·스키마)에 실재해야 한다 | `LB3.grounding · names · schema` |
| `lb4_coverage.py` | F4 | 요청 집합 − 처리 집합 | `LB4.sets` (ledger · follow_up · settled_rows · once) |
| `lb5_resignation.py` | F5 | 나가려는 턴에 아직 열린 의무를 이름으로 보인다 | `LB5` |
| `lb6_load.py` | F6 | 말하지 않고 생성 뷰만 줄인다(중복·압축·주석) | `LB6` |
| `lb7_material.py` | 전제 | 못 본 후보 집합을 전문으로 배달한다(순위 0) | `LB7` |
| `lb_coordinator.py` | — | 순서 `LB3 > LB1 > LB2 > LB4 > LB5 > LB7` · 같은 표적엔 명령 하나(사실 합집합) · 창·지문·예산 · `[LB_CONFLICT]` 기록 | — |
| `lb_runtime.py` | — | tau2 훅 3개: 생성(LB6 뷰 → 생성 → 평가 → 출구 → 비커밋 재생성) · 실행(LB2 검증기 도구 · 파생 사실) · 서브콜 문 `ask` | — |
| `lb_a2.py` | — | `a2/<domain>.lb.json` 적재 · 구 3층 A2 → 새 단일 선언 **데이터 이전**(`python lb_a2.py migrate banking_knowledge`) | — |
| `lb_run.py` / `lb_report.py` | — | 레인이 넘기는 인자 그대로의 러너 · 사이드카를 태스크×LB 표로 | — |

## 규칙

- **플래그는 7개**: `T2_LB1`~`T2_LB7`(기본 켬). 경로 2개(`LB_SIDECAR`·`LB_DOCS_DIR`)는 하네스다. `tests/test_lb.py` 가 강제한다.
- **엔진에 도메인 이름·수치·문장 0 · 정규식 0.** 태스크별 예외는 전부 `a2/<domain>.lb.json` 의 선언이다.
- **선언이 판정을 하면 잘못 놓인 엔진이다.** 049 의 `eplan.intent_chains[0].phrase`(SKIP …)는 그래서 버렸고, 신호는
  `procedures[].enter_when.signals` 로, 필수 read/write 는 `nodes` 로 옮겼다. 절차는 손님 발화(`role=user` 만)로도 열린다 — 048.
- **충돌은 데이터다.** 같은 표적에 둘이 말하면 `[LB_CONFLICT] target=… winner=LB1:procedure(E2) losers=LB4:…(E5)` 가
  stderr 와 사이드카에 남고 `lb_report.py` 가 접는다. 고정 산문(E5)은 계산 결과(E1·E2)에 이기지 못한다.

## 실행

```bash
python tests/test_lb.py                                  # 자기검정 + 이전 + 048/049 형태 + 플래그 7개 검사
python lb_a2.py migrate banking_knowledge                # a2/banking_knowledge.lb.json 재생성(데이터→데이터)
cd $GO_TAU2 && PYTHONPATH=src:$REPO/scripts/distill/lb python $REPO/scripts/distill/lb/lb_run.py --domain banking_knowledge \
  --agent_model Qwen/Qwen3.8-27B-FP8 --agent_base http://localhost:9143/v1 --user_llm openrouter/openai/gpt-5.2 \
  --user_temp 0.0 --user_reasoning_effort low --task_ids task_048 --num_trials 4 --max_concurrency 4 --save_to lb_task_048
```

## 구 185 레버 → 7 칸 (요약)

구 L군 → LB: L5 선행강제·L8 부재종결·L13 출구 → LB1 · L11 계산이관·L1 쓰기근거(계산형)·L10 형식화 → LB2 ·
L2 이름원장·L3 서술출처·L1(접지형) → LB3 · L4 회수집합차·L6 완결 → LB4 · L6 사임시점·L9 국면배치·L7 유도 → LB5 ·
뷰·중복·정체 → LB6 · L12 재료배달 → LB7 · L14 형식채널 → 삭제. 낱개 대응표는 구 명부
`reports/facet_rft_2026/LEVER_ROSTER_CANONICAL_2026_08_19.md` 와 조사 산출 `flags_inventory.tsv` 에 있다.

## 서브콜 레버 (2026-09-08 이전 완료)

| 구 레버 | 새 자리 | 형태 |
|---|---|---|
| `scaffold_get_tools` 검증기 도구 10개 (`t2_scaffold_get` 3,435줄 + `t2_compute` 1,123줄) | `LB2.tools` + `lb2_decision.evaluate_op`(op 21종) + `lb_runtime.inject_tools/execute` | 선언 그대로(변이 `ledger`·`ratefix` 적용) · 접지(`ground`) · 격리 `fetch_formalize` |
| `derived` DAG + `ledger_metrics`(`t2_factdag`·`t2_ledger`) | `LB2.derived` + `lb2_decision.derived_facts` — read 출력 뒤 `[FACTS]` | 형식화 서브콜(행·날짜·주어) + 결정론 연산 8종 |
| `reference_filter`(`t2_resolve`) | `LB2.computations[kind=select]` | 손님 기준 형식화 1회 → 레코드 필터 → 다른 id 면 deny |
| PROV_REGEN(날조 인자 재생성) | `LB3.identifying` | 식별 인자·id 형 값은 레코드/손님 발화에 실재해야 한다(결정론) |
| CLAIM_PROV / WRITE_PROV | `LB4.sets[kind=claims]` | 사임·이관 턴에 주장 목록 서브콜 1회 → 원장 집합차 |
| HAVE_VALUE / VALUE_ACQUIRE | `LB7.have_value` | 생산자 출력의 값 재사용 · 없으면 획득 도구 지목 |

옮기지 않은 것: `resolve_write`·`resolve_action_operator`·`resolve_recommendation`·`formalize_intent_tool`(발견형 도구
이름 해소·의도 형식화 — 이름은 LB3 레지스트리 검사, 절차는 LB1 이 대신한다) · `catalog_arg_docs` 서브콜 · E-PLAN replan
서브콜(LB4 claims 가 같은 자리) · `ledger_metrics` 의 다단 프롬프트(diagnose·rederive·objective — 태스크별 저작).

리모트 e2e 는 아직 0회다(도는 런의 조건을 바꾸지 않는다 — 승인 후 `lane_rep3` 형식의 별도 트리로).
