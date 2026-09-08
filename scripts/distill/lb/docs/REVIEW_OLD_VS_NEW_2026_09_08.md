# 구 코드베이스(`tau2/`) 대 신 코드베이스(`lb/`) 정밀 대조 — 2026-09-08

질문: 일곱 레버는 거의 독립인데 왜 버그가 계속 나오는가.
답: 레버끼리 얽혀 난 버그는 **0건**이다. 여섯 결함 전부 **엔진과 바깥의 경계**에서 났다 — 구 선언을 그대로 들여온 것,
구 술어는 옮기고 그 술어를 감싸던 상한·정규화를 안 옮긴 것, 인자 뷰를 잘못 고른 것. 그리고 공정 실수 하나:
라이브에 올리기 **전에** 실제 궤적으로 재현 검사를 안 돌렸다. 이 문서 이후 재현 검사(`lb_replay.py`, base 54 태스크 216 sim)는
배포 전 관문이다.

## 1. 라이브에서 잡힌 결함 여섯 — 어디서 났나

| # | 결함 | 경계 | 고침 |
|---|---|---|---|
| 1 | LB3 schema 가 디스패처 페이로드를 래퍼 인자로 읽어 280건 오거부 | 인자 뷰 | 래퍼 인자만 (`as_dict(call.arguments)`) |
| 2 | LB4 follow-up 이 decision_tools 없는 사슬에서 IndexError | 선언 결손 | 표적 대체 |
| 3 | 절차 도구를 지목하는 write-evidence 선언 5건(구 HARMFUL) 그대로 이전 | 구 선언 | 이번에 종류 자체 삭제(§2 D2) |
| 4 | claims 감사가 문장을 바꿔가며 13회 발화 | 예산 단위 | 규칙당 예산 |
| 5 | 절차가 read 를 10회 거부(048) | 술어만 이전 | `write_tools` 에만 deny |
| 6 | base 는 `max_tokens` 없음, 신 러너는 8192 | 실험 조건 | 기본 None (§2 P1) |

## 2. 구 가드 전수 판정 — 세 칸

**① 일반화되어 규칙 하나로 흡수** (코드에 들어간 것)

| 구 코드 | 신 코드 | 왜 일반인가 |
|---|---|---|
| 레버별 상한 25종 (`T2_PROCEDURE_CAP=6` `T2_VERIFY_DENY_CAP` `T2_FOLLOWUP_CAP=3` …), 소진 = 통과 | `lb_coordinator.DENY_BUDGET = 6` 규칙당·sim당, 초과분은 `lb-release` 로 기록하고 통과 | "같은 모델에게 계속 지는 규칙은 진 것이다, sim 은 살아야 한다" 는 하나의 문장 |
| `T2_REGEN_BUDGET=12` | `lb_runtime.REGEN_BUDGET = 12` sim당 (턴당 `ROUNDS=3` 은 유지) | 재생성은 sim 단위로 유한해야 한다 |
| `_ctx_has` 의 `#` 제거 | `lb3.norm()` — 대소문자·구두점 제거 | 렌더링은 증거가 아니다 |
| `_wev_expand` 의 `%g %.1f %.2f int` | `lb3.renderings()` — 수치는 관용 표기 전부 | 같은 값의 다른 표기 |
| `_hint_hit` (인자명 토큰 힌트) | `identifying.args` 에 토큰 매칭 (`txn_ref` ⊃ `txn`) | 인자 **이름** 규약; 값 모양 추정은 삭제(048 에서 날짜·금액 오거부) |
| `_ctx_fits` ((hist+len)/3.5 ≤ cap−gen−1024−11000) | `lb7.room()` — 남은 자리만큼 전달, 자리 없으면 문서 **이름만** | 컨텍스트에 안 들어갈 것은 넣지 않는다 |
| viewmax2 `T2_VIEW_COMPACT_MINTOTAL=344064` (=0.75·131072·3.5) | `lb6.FOLD_AT=0.75` × `a2.model_context` | 문턱은 모델 컨텍스트에서 유도, 상수 금지 |
| `_SRC8` 선점 사슬 | `LB_ORDER` + 표적당 명령 하나 (이미 있음) | 순서 하나로 충분 |

**② 실험 조건 동등성** (레버가 아님, base 와 같아야 하는 것)

| 항목 | base | 신 코드 |
|---|---|---|
| `max_tokens` | 없음 | 없음 (P1) — `--max_tokens` 를 명시할 때만 |
| `temperature` | 0.0 | 0.0 |
| user-sim | gpt-5.2 temp 0 low | 동일 |

**③ 옮기지 않은 것** (과제·도메인 관용구, 또는 이미 다른 규칙이 덮음)

| 구 코드 | 이유 |
|---|---|
| `DEFAULT_ARG_HINTS = ("email","name","zip",…)` | 도메인 리터럴. 필요하면 `identifying.args` 데이터로 |
| `_dup_stub_content` + bm25/kb 재검색 유도 산문 | LB6 뷰 dedup 이 부하는 이미 접는다; 산문은 검색 관용구 |
| `T2_SUPPRESS_AUTH` `T2_SG_TRUTH` `T2_KB_NOHIT_K` `T2_SEARCH_EXHAUST_TH` `T2_TRANSCRIBE_CAP` `T2_TOOLLIST_CAP` `T2_REPEAT_GOV` `T2_REGEN_KEEP_MUTATING` `T2_REGEN_WRITE_GATES` | 과제 관용구 또는 예산 하나로 흡수됨 |
| `T2_REQUIRE_DOC_DELIVER_CAP=3` | 조언 예산 2/규칙 이 덮음 |

## 3. 재현 검사가 추가로 잡은 것 (base 216 sim, 라이브 전에)

| 규칙 | 발화 | 판정 | 처리 |
|---|---|---|---|
| LB1 `absent` (핀) | 180 | 043 msg5 = 모델이 신원 확인을 **묻는** 턴에 다음 단계를 핀. 핀은 창·지문·예산을 전부 우회했다. 인계 시 미완 단계는 LB5 `steps-open` 이 이미 본다 | **D1 삭제** |
| LB3 `tokens` (write-evidence) | 53 | 049 base 통과 sim 이 `CLOSURE_OK` 없이 close 하고 reward 1.0. 판정 문자열 요구는 인용이 아니라 특정 검사의 **처방** | **D2 종류 삭제**, `_dropped` 에 기록 |
| LB3 `identifying` 값 모양 추정 | 5 → 1 | 날짜·금액 오거부 | 선언 인자만 |
| LB5 `uncalled-unlock` | 28 (빈 슬롯) | 구 `tool_unlock_hint` 문장을 빌려 `{tools}` 슬롯이 비었고 뜻도 반대(잠금 해제 **안 된** 도구용) | 규칙 자체 문장으로 교체 |

검사 후 잔여 발화: LB1 requirement 164 (그중 137 = `log_verification` 전에 `verify_identity` — LB2 검증기 요구, sim당 1회, **판단 보류: 결손 측정 전이라 유지하되 표시**), LB4 follow-up 71(조언·예산 2), LB5 28, LB7 17, LB3 2(플레이스홀더 `YOUR_USER_ID` = 정당).

## 4. 배포 전 관문 (이 순서, 매번)

1. `python tests/test_lb.py` 23/23
2. `python lb_replay.py sim_results/bank_x806_base_nt4_task_*.results.json.gz` — base 통과 sim 에서 deny 가 늘면 그 규칙을 읽는다
3. 8141 프로브: 레버별 대표 태스크 nt=1, 1분 단위 `lb_tick.py`
4. 그 뒤에만 라이브
