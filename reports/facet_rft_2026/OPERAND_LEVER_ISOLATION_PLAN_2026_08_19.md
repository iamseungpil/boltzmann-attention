# OPERAND 레버 격리 실증 실행 계획서 (2026-08-19)

- 대상 궤적: `reports/facet_rft_2026/sim_results/bank_t7326_half{A,B}_20260819q.results.json.gz` (20 태스크 / 40 sim / 2349 메시지 / 908 tool 메시지 / 887 assistant tool_call)
- 선행 판정: `OPERAND_LEVER_AUDIT_2026_08_19.md` (재조사 아님 · 인용)
- 실증 설계(고정): **재생(replay) 격리** — 그 sim 의 메시지를 결정 지점 **직전까지 원문 그대로** 잘라 주고 답을 받는다.
  - `A_off` 재생 그대로 · `B_on` 재생 + **그 레버가 실제로 내보내는 문자열** · `C_neg` 재생 + 같은 길이 무의미 문자열([[57]] 부정통제 의무)
- 레버당 필요물 셋: (1) 트리거 술어 축자 (2) 출력 문자열 축자 (3) t7326 에서 술어가 참인 (sim, msg idx) 목록. **(3)이 0이면 격리 불가** — 조건을 지어내면 [[62]]·[[03b]] 위반이고 실험이 무효다.
- ⛔ t7326 stderr(`.log.gz`)는 영속되지 않았다. 모든 자리는 궤적 재평가로 셌다(추정 없음).

---

## 1. 요약표 (36 레버)

| # | flag | trigger_sites | isolatable | why_zero / 격리 불가 사유 |
|---|---|---:|---|---|
| 1 | T2_ARG_REPEAT | 0 | ✗ | **WIRING_DEAD** — `t2_gate_patch.py:4057` 이 `error` 참인 tool 메시지만 보는데 t7326 의 `Unexpected parameter` 4건은 전부 `error=False` + 감시 채널이 손님측 `call_discoverable_user_tool` 을 안 본다(완화 재평가도 0) |
| 2 | T2_WRITE_DEDUP | 0 | ✗ | **NOT_IMPLEMENTED** — 조건문 자체가 없다. x294 격리에서 결손 미재현(`A_ASIS 0/8`)이라 짓지 않은 것 = [[62]] 순서를 지킨 옳은 보류 |
| 3 | T2_SCALAR_ARRAY | 0 | ✗ | **TARGET_ABSENT** — 배열 인자 8건 전부 복수형 이름이라 설계된 배제. 단수명 필드 배열이 이 20 태스크에 0건 |
| 4 | **T2_FIT_DIFF** | **7** | ✓ | — (t7326 에서는 플래그 OFF · 자리는 정본 술어 재평가로 확정) |
| 5 | T2_GROUNDING_SPEC | 0 | ✗ | **UPSTREAM_BLOCKED** — 3중: `--resolve` 미전달(`go_stack.sh:183-184` `shift 4` 후 `"$@"`=∅) · env 경로 미사용 · `banking_knowledge.grounding.json` 파일 부재 |
| 6 | T2_GROUND_DROP_NAVKEYS | 0 | ✗ | **UPSTREAM_BLOCKED + NO_EMIT** — `apply()` 미호출 · 게다가 문맥에 아무것도 안 넣는 인자-제거형 |
| 7 | T2_PROV_GROUND | 0 | ✗ | **UPSTREAM_BLOCKED** — 플래그 미설정 + unified 모드에서 켜면 `t2_run_gated.py:222-223` SystemExit + 술어 함수(`apply_provenance_regen`)가 애초 미설치 |
| 8 | T2_QUOTE_HINT | 0 | ✗ | **TARGET_ABSENT(로스터)** — 배선 생존(`[GROUNDING WARNING]` 12건 실발화). 표적 도구 `check_card_closure_eligibility` 호출 0회 |
| 9 | T2_CHOICE_GROUND | 0 | ✗ | **WIRING_DEAD + TARGET_ABSENT** — 33/33 디스패처 중첩 미파싱(`_args_dict` 가 `account_class` 를 못 꺼냄). 중첩 파싱 가정 반사실에서도 33/33 접지 성공 ⇒ 0 |
| 10 | T2_HAVE_VALUE | 0 | ✗ | **TARGET_ABSENT** — 조건②(producer 출력 실재) 0회. 재요청은 6 sim 에 있으나 값이 온 적 없음 ⇒ 결손은 형제 레버 T2_VALUE_ACQUIRE 소관 |
| 11 | T2_DOCS_AT_WRITE | 0 | ✗ | **UPSTREAM_BLOCKED(플래그 0) + WIRING_DEAD** — 켜도 순증 0: 확장 이름 `open_bank_account_4821` 은 접미사-strip 불일치, `apply_for_credit_card` 는 6/6 role=user |
| 12 | **T2_ARG_PRODUCERS** | **9** | ✓ | — (9/9 오발화 · 진짜 표적 1자리는 cap 2 에 막혀 미발화) |
| 13 | **T2_WRITE_EVIDENCE** | **6** (call-level, 라이브 등가 5) | ✓ | — |
| 14 | **T2_RESOLVE** | **8** (operator-fab) | ✓ | — (다른 채널: reference-filter 2 무발화 · membership 0 · recommendation 196 · action-required 330 · verify-persistence 0) |
| 15 | T2_RESOLVE_CAP | 0(원리상 계수 불가) | ✗ | **NO_EMIT(억제형)** — 출력이 stderr 3줄뿐. 다른 레버의 문자열을 *끄는* 것이라 B_on 이 공집합. 상태변수 `_t2_resolve_deny` 는 regen 전용이라 궤적 복원 불가 |
| 16 | T2_COMPUTE | 1 (잔여) | ✗ | **NOT_IMPLEMENTED(HEAD) + NO_EMIT** — HEAD 리더 0건·`compute_ops={}`. t7326 당시에도 모델에겐 무발화(silent repair). ⚠임계값이 gold 재현율로 피팅([[23]]) |
| 17 | **T2_MATCH_COUNT** | **151** (35/40 sim) | ✓ | — 유일한 **커밋 채널** 부착 레버 |
| 18 | T2_GROUND | 0 | ✗ | **TARGET_ABSENT + UNKNOWN + NO_EMIT** — fab 턴 1/887, 그마저 단일후보 실패. 성공 치환은 술어를 스스로 거짓으로 만들어 커밋 채널이 구조적으로 실명. unified 경로는 문자열 무발화(인자 제자리 치환) |
| 19 | **T2_SG_GROUND** | **12** | ✓ | — (`ground` 선언 도구 호출 34건 중 12 발화) |
| 20 | T2_OPERATOR_PINPOINT | 0 | ✗ | **UPSTREAM_BLOCKED(플래그 한 줄)** — `t2_resolve.py:216` 이 형제 분기(operator-scope, t7326 48회 발화)로 return. 서브콜 도달 자리는 90건 실재 |
| 21 | **T2_VALUE_ACQUIRE** | **64** (캡 적용 ≤17) | ✓ | — 6 sim |
| 22 | T2_REF_ISO | 2 | ✗ | **NO_EMIT** — 인자 제자리 치환(`t2_gate_patch.py:4437-4443`). 모델이 읽는 문자열은 격리 서브 세션 안이라 문맥이 아님 |
| 23 | **T2_WRITE_ARG_ENUM** | **10** (커밋 4 + deny 6) | ✓ | — 전부 task_057 |
| 24 | **T2_WRITE_ARG_GROUND** | **2** | ✓ | — 커밋 재평가로는 0, `x392_block_join.json` 이 유일 생존 산출물 |
| 25 | **T2_GROUND_HDR** | **12** | ✓ | — 빼기형(B_on 이 궤적 원문) |
| 26 | T2_QUOTE_PIN | 0 | ✗ | **TARGET_ABSENT** — 검사기는 10행 전부 돌았고(coverage `5 of 5 … 0 could not be verified`) `exclusion_quote` 가 비어 반환. 제외-매핑 주장 태스크가 이 로스터에 없음 |
| 27 | T2_PROD_BIND | 0 | ✗ | **TARGET_ABSENT + NO_EMIT** — 20/20 operand 가 레코드 출력에 실재(강등 0). 개입은 필드 `None` 화, 인쇄는 stderr |
| 28 | **T2_HAVE_VALUE_FORCE** | **64** (실발화 슬롯 17) | ✓ | — ⚠문자열 아님(`tool_choice='required'` 디코딩 제약) |
| 29 | T2_REF_VERIFY | 0 | ✗ | **TARGET_ABSENT(가드 정상 통과)** — 표적 호출 2건이 손님 명시 상점의 레코드라 옳게 통과(false-block 0) |
| 30 | **T2_DECIDE_BEFORE_WRITE** | **2** | ✓ | — 반향 축자 `missing decision` 2건. 후보 1건은 도구-이름 되먹임에 밀림 |
| 31 | T2_ARG_SCHEMA | 0 | ✗ | **TARGET_ABSENT** — 908 호출 전수에서 스키마-밖 최상위 키 0. ⚠디스패처 `arguments` 안쪽(211건)은 원리상 사각 |
| 32 | **T2_ARG_EMPTY** | **1** | ✓ | — 술어 참 1건인데 라이브 발화 0(상류 WEV/WAG 에 밀림) |
| 33 | **T2_WRITE_PROV** | **191** (결정론 창) | ✓ | — 실발화는 2차 LLM 게이트 통과분뿐이라 궤적으로 미확정(다른 런 비율 85:1) |
| 34 | T2_PAIRCHECK | 2 | ✗ | **NO_EMIT(로그 전용)** — `t2_gate_patch.py:5954` stderr 1줄, 행동 무변경 |
| 35 | T2_PAIRFIX | 2 | ✗ | **NO_EMIT(재정렬)** — 내용 집합 불변의 의미론 no-op. 종점이 pass 가 아니라 평가 replay 크래시 |
| 36 | **T2_GIVE_QUOTE** | **2** | ✓ | — give 메시지 9개 중 인용 부재 2건(하한: 철회되면 커밋에 안 남음) |

**집계** — 격리 가능 15 · 격리 불가 21.
격리 불가 내역: TARGET_ABSENT 8(3,8,10,18,26,27,29,31) · UPSTREAM_BLOCKED 5(5,6,7,11,20) · WIRING_DEAD 3(1,9,11 중복포함) · NOT_IMPLEMENTED 2(2,16) · NO_EMIT 8(6,15,16,18,22,27,34,35 중복포함).

---

## 2. 격리 가능 (sites ≥ 1) — 실행 명세

### 2-0. 공통 규격

- **모델·디코딩**: t7326 과 동일 고정(`Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8`, `go_stack.sh:204-213` 파라미터).
- **컷**: 각 자리의 `messages[0..k]` 를 **원문 축자**로 준다. 잘라낸 뒤 아무것도 요약하지 않는다([[65]] — 재료가 아니라 원문 그대로가 이 실험의 정의).
- **삽입 채널 3종** (레버마다 다르다 — 섞지 말 것):
  1. **tool-result 대체(fail-closed deny)**: `ToolMessage(id=c.id, role="tool", requestor="assistant", error=True, content="Error: "+…)` — `t2_gate_patch.py:9216-9217`
  2. **regen 작업버퍼 UserMessage**: `fb.append(UserMessage(role="user", content=…))` → `work = work + fb` — `t2_gate_patch.py:9362-9366`, `:9376`
  3. **커밋 tool result 부착**: `r.content = txt + "\n" + "[axis] " + x` — `t2_gate_patch.py:1004-1006` (MATCH_COUNT · FIT_DIFF · SG_GROUND · GROUND_HDR 계열)
- **C_neg**: 자리별 **동일 문자 수**. 도구명·카드명·필드명·숫자가 들어가면 안 된다. 길이가 자리마다 다르면 C_neg 도 자리마다 맞춘다.
- **계상 기준(고정)**: 팔 3종 × n=3 샘플/자리. 아래 표의 `응답수` 는 `자리 × 팔 × 3`.
- **[[57]] 의무**: 모든 레버에서 `B_on − C_neg` 를 같이 보고한다. `B_on − A_off` 만으로 효과를 주장하면 위반.
- **[[08]] 의무**: 집계 metric 이 아니라 **다음-행동 분류**(도구명·인자·산문 3분류)로 채점한다. 자리 수가 한 자리이므로 pooled rate 금지 — **자리별 이항**으로 보고한다.

### 2-1. 실행표

| 우선 | flag | 표적 자리(계획) | 팔 | n | 응답수 | 종점(결정론 채점) |
|---|---|---:|---:|---:|---:|---|
| T1 | T2_WRITE_EVIDENCE | 5 (017t0 idx70 · 093t0 idx39/55/71 · 093t1 idx92) | 3 | 3 | **45** | ① write 철회 여부 ② operand 가 `33.00` 으로 이동하는가 / `get_interest_correction` 선호출로 바뀌는가 |
| T1 | T2_WRITE_ARG_ENUM | 4 (057t0 100·118 · 057t1 52·58) | **4** (A/B/C_neg/**C_list**) | 3 | **48** | 다음 `account_class` 가 공식 9개 명단 안인가(이진) |
| T1 | T2_RESOLVE | 8 (050t0 24·40·42 · 050t1 38·40 · 073t1 45 · 079t1 107·137) | 3 | 3 | **72** | 같은 unlock 재발행(무해) / 포기·대체(gold 파괴) / 발견된 이름으로 정정 |
| T1 | T2_HAVE_VALUE_FORCE | 6 (각 sim 첫 슬롯: 033t0 12 · 040t0 8 · 040t1 9 · 074t1 48 · 094t0 42 · 094t1 56) | **4** (A/B/c1/c2) | 3 | **72** | ① tool_calls 산출 여부 ② give 지목 순응 ③ ★ACTION_SUB 소실(상쇄, [[19]]) |
| T2 | T2_ARG_PRODUCERS | 10 (해악 9 + 이득 1: 040t1 idx34) | 3 | 3 | **90** | (a) 검색·unlock 진행 유지 vs give 로 이탈 (b) 12인자 write 재발행 |
| T2 | T2_VALUE_ACQUIRE | 6 (sim 별 첫 발화 자리) | 3 | 3 | **54** | give 호출 / 재요청 반복 / 이탈. ★표적 안(040 2 sim) ↔ 표적 밖(4 sim) **분리 채점** |
| T2 | T2_DECIDE_BEFORE_WRITE | 2 (040t1 msg31 · 085t1 msg56) | 3 | 3 | **18** | 결정 인코딩 인자가 `_dmat` 답으로 바뀌는가 |
| T2 | T2_ARG_EMPTY | 1 (063t0 msg6) | 3 | 3 | **9** | `min_cashback` (a)빈칸 유지 (b)채움 (c)생략. **정답은 (c)** — 문면이 요구하는 (b)와 어긋남 = over-action 계측 |
| T2 | T2_WRITE_ARG_GROUND | 2 (017t0/t1 messages[0..17]) | 3 | 3 | **18** | `get_current_time` 단독 선호출 / 같은 턴 병렬 / 날짜 재날조 |
| T2 | T2_GIVE_QUOTE | 2 (063t0 idx20 실패궤적 · 017t1 idx44 **성공궤적=자해 계측**) | 3 | 3 | **18** | ① give 철회 ② `_shared_span` True 전환 ③ 도구명 정정 ④ Δspurious |
| T3 | T2_MATCH_COUNT | 24 (151 중 층화: form2 12 · form1 6 · form3 6) | 3 | 3 | **216** | ① 검색 중단/속행 ② 결정 정확도 ③ **form2(완결 인증) ↔ form3(경고) 대조** |
| T3 | T2_GROUND_HDR | 12 (intent 4 + record/KB 8, **분리 집계**) | 3 | 3 | **108** | (b) 드롭된 제약 재삽입률 감소 / (a) 레코드 재독 회수 |
| T3 | T2_SG_GROUND | 12 | 3 | 3 | **108** | 드롭 operand 재회수 호출 / 손님 되물음 / 그냥 보고 |
| T3 | T2_FIT_DIFF | 7 | 3 | 3 | **63** | 1차 카드 지명 정확도 · 2차 국면(지명/되묻기/재호출) |
| T3 | T2_WRITE_PROV | 12 (층1 통과분 상한) | 3 | 3 | **108** | ① give/call 산출 ② 사건번호 날조(`[A-Z]{2,}-?\d{4,}` substring 검산) ③ `claims_completion` true→false |

**본 프로브 응답 소계 = 1,047건.**

### 2-2. 부수 서브콜 (본 프로브 외 · 같은 GPU)

| 항목 | 건수 | 근거 |
|---|---:|---|
| T2_WRITE_PROV **층1**(2차 게이트 `claim_question` 측정) — 실효-write 0 sim 12개의 사임 턴 전부 + 나머지 28 sim 무작위 40 = 52자리 × 1 | **52** | 이 값 없이는 층2 의 **분모가 없다**. `a2/banking_knowledge.gate.json:2658` |
| T2_WRITE_PROV 재채점(재생성 턴에 게이트 재적용) 108 × 1 | **108** | 종점 ③ |
| T2_DECIDE_BEFORE_WRITE `_dmat` 재현(`_search_material` 재실행) 2자리 | **2** | B_on 절반이 LLM 산출. 재현본 ≠ 원본이면 프로브 중단하고 그 사실을 기록([[03b]]) |

**부수 소계 = 162건.**

### 2-3. 총계

> **총 GPU 응답 = 1,047 (본 프로브) + 162 (부수) = 1,209건** (팔 3종 × n=3 기준. ENUM·FORCE 2건은 필수 4팔이라 3팔 초과분 포함)

- n 을 3→8 로 올리면 본 프로브는 2,792건, 총 3,000건 근처가 된다.
- 자리 수가 1~2 인 레버(ARG_EMPTY·DECIDE·WAG·GIVE_QUOTE)는 n=3 으로는 **검정력이 없다**. 이 4건만 k=16 으로 올리면 +(1+2+2+2)×3×13 = +273 → 총 1,482건.
- ⚠ 이 계상에서 **뺀 것**: T2_RESOLVE 의 recommendation(196턴)·action-required(330턴) 채널 — LLM `formalize` 서브콜을 재생 안에서 재현해야 하므로 별개 배치. T2_OPERATOR_PINPOINT 의 90자리 `want` 확정 서브콜도 여기 없다.
- ⚠ T2_WRITE_ARG_ENUM 의 deny 6자리(A_off 를 먼저 **생성**시켜 집합 밖 호출을 재현)를 포함하면 +6×4×3 = +72.

---

## 3. 격리 불가 — 사유별 분류

> **핵심 구분**: `TARGET_ABSENT` 는 "이 20 태스크에 표적이 없다"는 뜻이지 **레버가 무용하다는 뜻이 아니다**. 로스터를 바꾸면 살아난다. `WIRING_DEAD`·`NOT_IMPLEMENTED` 는 로스터를 바꿔도 안 산다 — 코드가 선행이다.

### 3-A. TARGET_ABSENT — 배선 생존 · 이 20 태스크에 표적 없음 (8건)

| flag | 무엇이 없었나 | 살리는 법 |
|---|---|---|
| T2_SCALAR_ARRAY | 단수명 필드에 배열이 온 사례 0(배열 8건은 전부 복수형 = 설계된 배제) | 표적이 실재하는 다른 표본 탐색이 **선행**. 인위 생성 금지 |
| T2_QUOTE_HINT | `check_card_closure_eligibility` 호출 0. 주입 채널은 12회 실발화 = 배선 생존 | **로스터 문제**. 043/046/061 계열이 든 런 확보. 특히 **task_046 t0/t1 자연실험**(패러프레이즈 드롭 ↔ 축자 통과) |
| T2_HAVE_VALUE | producer 성공 출력 0(재요청은 6 sim 실재) | 손님이 실제로 `get_card_last_4_digits` 를 실행한 궤적(039 계열) |
| T2_GROUND | fab 턴 1/887 + 성공 치환이 술어를 스스로 거짓으로 만듦 | **실험 아니라 계기 수정**: 다음 스택 런에 `T2_TRACE=1` 켜고 `[T2_GROUND] substituted` 줄만 세면 끝(무료) |
| T2_QUOTE_PIN | `exclusion_quote` 빈 반환(제외 대상 행 없음). 검사기는 10행 전부 돌았음 | 제외-매핑을 요구하는 태스크(019 계열)가 든 런 |
| T2_PROD_BIND | 20/20 operand 가 레코드 출력에 실재 | 날조 operand 가 실재하는 궤적. 단 NO_EMIT 도 겸함(§3-E) |
| T2_REF_VERIFY | 표적 호출 2건이 **손님 명시 상점**이라 가드가 옳게 통과 | wrong-pick 이 실제로 난 궤적(rall19-22 계열, 8/8). ⚠회귀 테스트 깨짐(`test_ref_verify.py:67` 3인자 ↔ 현행 6인자 TypeError) |
| T2_ARG_SCHEMA | 908 호출 전수 스키마-밖 최상위 키 0 | ⚠사각 정정이 선행: 디스패처 `arguments` 안쪽(211건)을 원리상 안 본다. 오프라인 스캔으로 내부 여분 키를 먼저 세라(unlock 결과가 인자 목록을 축자로 실어줌 = gold 불요) |

### 3-B. UPSTREAM_BLOCKED — 표적·배선 무관하게 도달 자체가 막힘 (5건)

| flag | 막은 것 | 해제 순서 |
|---|---|---|
| T2_GROUNDING_SPEC | ①`--resolve` 미전달(`go_stack.sh:183-184` `shift 4` 후 `"$@"`=∅) ②env 경로 미사용 ③`banking_knowledge.grounding.json` 부재 | spec 작성(**출처는 정책·환경뿐** — gold 참조 시 [[23]] 위반) → `--resolve 1` → **모델이 실제로 `resolve_selection` 을 부르는지 먼저 관측**. 그 다음에야 재생 격리가 정의된다 |
| T2_GROUND_DROP_NAVKEYS | 위와 동일 상류(`apply()` 미호출) | 위 3단계 + 종점 재정의(§3-E) |
| T2_PROV_GROUND | 플래그 미설정 + unified 모드 `SystemExit` + `apply_provenance_regen` 미설치 | 비-unified arm 을 새로 돌리거나, unified 형제 `T2_GROUND` 를 **별개 레버로 재정의**해 재인구조사 |
| T2_OPERATOR_PINPOINT | `t2_resolve.py:216` 플래그 게이트 **한 줄** — 형제 분기(operator-scope)가 t7326 에서 48회 발화 | 서브콜 도달 90자리에서 `formalize_intent_tool` 을 오프라인 1회 돌려 `want` 확정 → `want≠chosen` 자리만 남김. ⚠**우선순위 최하**: x322 n=24 에서 `A_REF 24/24 ↔ B_PINPOINT 0/24` 로 이미 음성 종결 |
| T2_DOCS_AT_WRITE | `run_t7326_stage1_nt2_20260819.sh:78` `T2_DOCS_AT_WRITE=0` | 켜도 순증 0(§3-C) — 플래그보다 대조식이 먼저 |

### 3-C. WIRING_DEAD — 술어가 라이브 스키마와 어긋나 원리상 참이 될 수 없음 (3건)

| flag | 어긋난 지점(축자) | 비고 |
|---|---|---|
| T2_ARG_REPEAT | `t2_gate_patch.py:4057` `not getattr(m,"error",True)` 로 skip — 대상 4건 전부 `error=False`. 게다가 반려는 손님측 `call_discoverable_user_tool` 에서 나는데 술어는 에이전트 `give_…` 만 본다 | 두 수정 후에도 **행위자가 user-sim** 이라 '에이전트 재생성' 격리와 안 맞는다. 무엇을 재는지 정하기 전엔 프로브를 짜지 마라 |
| T2_CHOICE_GROUND | `t2_gate_patch.py:11046` 이 디스패처 중첩을 안 푼다(33/33 `_v_cg=''`). 같은 계열에 `_parse_nested_args`(`t2_resolve.py:775-784`)가 이미 있는데 이 자리에서 안 쓴다 | **고쳐도 t7326 은 0** — 반사실에서 33/33 접지 성공. 오선택이 '지어낸 이름'이 아니라 '회수된 이름 중 잘못 고른 것'이라 접지 술어로는 원리상 안 잡힌다 |
| T2_DOCS_AT_WRITE | 확장 이름 `open_bank_account_4821` ↔ `_eff_tool_name` 이 만드는 `open_bank_account` 접미사-strip 불일치 · `recommendation_verify.action_tool = apply_for_credit_card` 가 6/6 `role=user`(손님 도구를 에이전트 술어에 넣음) | 선행 배치의 "6자리"는 **오산 → 0으로 정정** |

### 3-D. NOT_IMPLEMENTED (2건)

- **T2_WRITE_DEDUP** — 조건문 부재. `X291_CHECKING_FIT_DESIGN_2026_08_13.md:179` 축자 `x294 … A_ASIS 0/8 — 재현 실패·T2_WRITE_DEDUP 보류`. **[[62]] 순서를 지킨 옳은 보류**다. 되살리려면 먼저 결손 재현부터(동일 `(name,args)` write 재호출 건수가 첫 측정치이고, 0에 가까우면 짓지 않는 것이 정답).
- **T2_COMPUTE** — HEAD 리더 0건 · `compute_ops={}` · `resolve_compute_params` 호출자 0 · `test_compute_params.py:70-71` 이 부활 금지. 제거 커밋 `b220745d` 사유 축자: *"엔진이 채점되는 인자를 쓰지 못하게 하고, gold 에 맞춘 상수를 제거"*. **재도입 금지**([[62]]·[[23]]).

### 3-E. NO_EMIT — 문맥에 문자열을 넣지 않는 레버 (재생 격리 설계가 **형식적으로 미적용**)

| flag | sites | 실제 개입 | 이 레버에 맞는 종점 |
|---|---:|---|---|
| T2_RESOLVE_CAP | 계수 불가 | 다른 레버의 문자열 **억제** (stderr 3줄) | T2_RESOLVE 프로브의 **파라미터**로 재라 — 같은 문자열을 k=1,2,3,4 반복해 포화·반전점을 본다([[57]] "횟수 아닌 인자 변화"의 정확한 형태) |
| T2_REF_ISO | **2** | 인자 제자리 치환(`:4437-4443`) | 040t1 msg31 에서 A(원본 실행) / B(서브콜 값으로 치환) / C(listing 안 무관 실재 txn) 대조. ⚠앞 단계의 '날조 id' 판정은 **틀렸다** — `txn_a1b2c3d4e503` 은 listing 실재이고 손님 서술과 정확히 일치, 기대 결과는 `keep` |
| T2_PROD_BIND | 0 | 행 필드 `None` 화 | 하류 결핍 문구는 ABSTAIN_FIELDS 의 것 — 그것을 재면 다른 레버를 재는 것 |
| T2_GROUND / T2_GROUND_DROP_NAVKEYS | 0 / 0 | 인자 치환 / 인자 제거 | 계기(`T2_TRACE`, `T2_GROUND_LOG.dropped_nav`)로 **율만** 재고, 그 다음에 설계 |
| T2_COMPUTE | 1 | silent repair | §3-D. 굳이 재려면 **전달형**(정책표 축자 제시)으로 — 그건 다른 레버다 |
| T2_PAIRCHECK | **2** | stderr 1줄, 행동 무변경 | **계기 검정(무료)**: 두 sim 을 같은 시드로 재생하며 라이브 `state.messages` 에 `_paircheck` 를 걸어 라이브 침묵(=직렬화 층 부패) ↔ 라이브 적중(=에이전트 층 부패)을 가른다. `:4970-4972` 가 적어 둔 미해결 분기 |
| T2_PAIRFIX | **2** | 리스트 in-place 재정렬(의미론 no-op) | **평가 replay 크래시 건수**: 두 sim messages 를 `Environment.set_state(message_history=…)` 에 먹여 (A)원본 → `ValueError: Tool call id mismatch` (B)`_pairfix` 후 → 통과 대조. t7326 결과에 스왑이 **그대로 남아 있다** = 커밋 자리(:5937)만으로 부족하다는 증거 |

> ⛔ 이 7건에 억지로 A_off/B_on/C_neg 를 붙이지 마라. B_on 이 없는데 문자열을 지어내면 그건 다른 레버를 재는 것이고 결과가 오귀속된다.

---

## 4. 우선순위 — 무엇부터 재야 정보량이 큰가

### 4-0. 판정 근거 (x420 ↔ x408)

- **x420(2026-08-19 실측)**: 결정점에 **정답이 든 문서 조각**을 넣어도 **operand 값이 안 바뀐다** — `R_asis 0.426 ↔ R_doc 0.440 ↔ R_neg 0.445`, 35 표적 중 28 동률.
- **x408**: 같은 조각을 넣으면 **어느 단계를 할지**는 `0.50 → 0.83` 으로 오른다.
- ⇒ **텍스트 제시형 레버(문서·표·주석 부착)의 operand 효과는 x420 이 이미 음성으로 답했다.** 그것을 다시 재는 것은 정보량이 낮다.
- ⇒ **x420 이 답하지 못한 부류 = 거부(deny) · 차단 · 재생성 강제 · 디코딩 제약 · 치환.** 이들은 "읽을 재료를 더 준다"가 아니라 "그 수를 못 두게 하고 다시 두게 한다"이므로 기전이 다르다. **여기부터 재라.**

### T1 — 최우선 (x420 미답 부류 · 결정론 채점 · 자리 확보)

1. **T2_WRITE_EVIDENCE** (5자리 / 45응답) — 이 배치에서 **operand 이동을 직접 검정하는 유일한 자리**다. task_093 gold 는 두 write 모두 `33.00` 이고 두 자리 다 `action_match=false` 였다. 문서 제시로는 안 움직인 그 값이 **거부로는 움직이는가** — 양성이면 그것이 부하-축소 주장의 실물이다. ⚠이 6자리는 **잔여(residual)** 다 — regen 이 고친 deny 는 커밋에 흔적이 없으므로 표본이 "거부가 통하지 **않은** 자리" 쪽으로 편향돼 있다. 이 편향을 결과에 반드시 적어라.
2. **T2_WRITE_ARG_ENUM** (4자리 / 48응답) — **관찰적 대조가 이미 데이터 안에 있다**: 057t1 turn41/43/45(문면 있음) → msg42/44/46 전부 집합 內 정정 ↔ msg52·58(캡 소진·env 에러만) → 집합 밖 유지. 채점이 이진(9개 명단 안/밖)이라 잡음이 가장 적다. **C_list arm 필수**(§5).
3. **T2_RESOLVE** (8자리 / 72응답) — **부호가 음(-)일 가능성이 높은 유일한 T1**. task_050 5자리의 거부 대상이 전부 **gold 요구 이름**이다(`050_9`/`050_5`/`050_3` 전부 `action_match=true`) ⇒ B_on 은 **오차단**이다. "레버가 gold 행동을 파괴하는가"는 양성 실증보다 정보량이 크다([[62]] 제1원리 = 하나를 사면 하나를 판다). 073t1·079t1 3자리가 진양성 arm.
4. **T2_HAVE_VALUE_FORCE** (6자리 / 72응답) — **문자열이 아예 없는 레버의 순수형**(`tool_choice='required'`). 텍스트 제시 축과 완전히 직교하므로 x420 축과 겹치지 않는다. ★상쇄 계측 필수: force 턴은 `_gen_action_sub` 를 죽인다(`t2_gate_patch.py:9468-9469`) — **이 노브는 그 턴의 ACTION_SUB 를 판다**([[19]]).

### T2 — 다음 (자리는 있으나 검정력·표적정합 문제)

5. **T2_ARG_PRODUCERS** (10자리 / 90응답) — 정보량의 핵심이 **해악 arm**이다: 발화 9/9 가 오발화(비-error 산문 substring, [[59]] 위반)이고 진짜 표적 1자리는 cap 에 막혔다. "레버가 산 자리 ↔ 닿았어야 할 자리가 어긋나 있다"는 것을 행동으로 확정한다.
6. **T2_VALUE_ACQUIRE** (6자리 / 54응답) — ★**표적 안(040 2 sim) ↔ 표적 밖(4 sim) 분리 채점 없이는 결과가 무의미하다**. 094t0 msg42 축자는 *"the last four digits of your **savings account number**"*, 033t0 msg12 는 *"to **verify your identity**"* 로 dispute 와 무관하다.
7. **T2_DECIDE_BEFORE_WRITE** (2자리 / 18응답) — ⚠B_on 절반(`_dmat`)이 LLM 산출이라 **재현 실패 시 프로브를 접고 그 사실을 기록**한다([[03b]]).
8. **T2_ARG_EMPTY** (1자리 / 9응답) — 값이 아니라 **문면과 정답의 어긋남**이 산출물이다: A2 축자가 *"A superlative request … is NOT a threshold: leave this out"* 이므로 정답은 (c)생략인데 B_on 은 (b)채움을 요구한다 = **over-action 계측**(등대 §1 "게이트 자신도 역효과 → Δspurious ≤ 0").
9. **T2_WRITE_ARG_GROUND** (2자리 / 18응답) — A_off 를 먼저 생성시켜야 자리가 생긴다(확률적, 시드 반복). 발화 2건이 둘 다 gold 도구 표적이지만 둘 다 `later_ok=true` ⇒ 회복 2/2.
10. **T2_GIVE_QUOTE** (2자리 / 18응답) — 자리 B(017t1 reward **1.0**)가 **자해 계측 전용**이다. 성공 궤적에서 give 가 철회되면 그 자체가 손실.

### T3 — 텍스트 제시형 (x420 예측 확인용 · 후순위)

11. **T2_MATCH_COUNT** (24자리 / 216응답) — T3 중 **유일하게 정보량이 큰 것**. 종점이 operand 가 아니라 **stop/continue**(x408 축)이고, 게다가 **우리 층이 거짓 완결 인증을 발급하는지**를 궤적 안에서 검정할 수 있다: form2("all N shown") ↔ form3("K shown, M not shown") 대조. 결함 서명이 이미 보인다 — `shown_in`(`t2_match_count.py:88-91`, `(?m)^\s*(\d+)\.\s`)이 **문서 본문의 번호 줄까지 센다** ⇒ `shown>=n` 이 참이 되어 form2 로 떨어진다. 같은 카운터가 같은 코퍼스에서 `0 shown`(055t0 idx17)과 `all N shown`을 동시에 낸다. **[[25]] 위반 후보 — 행동 효과를 귀속하기 전에 원격 코퍼스로 오프라인 검산.**
12. **T2_GROUND_HDR** (12자리 / 108응답) — 빼기형. 판정선은 12자리 전체가 아니라 **모순이 실재하는 intent 4자리**에 세운다.
13. **T2_SG_GROUND** (12자리 / 108응답) — ⚠세 팔의 **결과 본문이 동일**하다(드롭은 본문 생성 전에 끝남). 이 프로브가 재는 것은 오직 "워닝 문면이 재독을 유도하는가"이고, '레버 없음'의 진짜 대조군을 원하면 `T2_SG_GROUND=0` 으로 본문을 다시 만든 **D_nodrop 4번째 arm** 이 필요하다(+12×3 = 36응답).
14. **T2_FIT_DIFF** (7자리 / 63응답) — **최후순위**. 이유 둘: ①전형적 텍스트 제시형이라 x420 예측 구간 ②⛔**우리 문구가 7자리 중 5자리에서 거짓을 말한다** — `{n}` 이 `excluded` 목록까지 eligible 로 센다(003 3/4→"7 eligible" · 055 3/4→"7" · 063t1 4/3→"7" · **063t0 0/7→"7 options are eligible"**). 재생 격리는 '레버가 실제로 내보내는 문자열'을 재는 것이라 이 상태로 재는 게 규격에 맞지만 **라이브 출시는 이 버그를 고치기 전에는 불가**([[25]] 유일 근거원 오염). 또 `t2_axis_levers.py:135/139/140` 이 도구 출력 산문을 정규식으로 뜯는다 = [[59]] 정면(재생 격리는 엔진 파싱을 안 타므로 프로브 자체는 무방).
15. **T2_WRITE_PROV** — 층2 는 T3 이지만 **층1(52응답)은 T0 = 무료 선행**이다. 191 은 결정론 창일 뿐이고 실발화는 2차 LLM 게이트 통과분뿐인데 다른 런 비율이 `window 12,038 : regen 141` = 85:1 ⇒ **층1 없이 층2 를 돌리면 분모 없는 수치가 나온다**.

### T0 — 유료 런 이전에 끝낼 무료 작업

- **T2_WRITE_PROV 층1** 52응답(위).
- **T2_PAIRCHECK / T2_PAIRFIX 계기 검정** — GPU 불요. 라이브 vs 직렬화 층 판별이 끝나야 `_pairfix` 설치 자리(`:5937` 커밋 ↔ `:4973` 평가-입력) 중 어느 쪽이 유효한지 결론이 난다. **t7326 결과에 스왑이 그대로 남아 있다** = 지금 설치가 부족하다는 직접 증거. 40 sim 중 2 sim(5%)이 표적.
- **T2_GROUND 계기 수정** — 다음 스택 런에 `T2_TRACE=1`(`t2_lever_beat.py:115-135`) 켜고 `[T2_GROUND] substituted` 줄만 세면 치환율이 한 번에 나온다. 실험 아님.
- **계기 오귀속 2건 수정**(결과 해석 전 필수): ①`t2_gate_patch.py:7106-7107`(WAG) 과 ②`:7110`(REF_VERIFY) 이 `_lbeat` 를 `"T2_WRITE_EVIDENCE"` 로 **하드코딩** ⇒ beat 집계가 세 레버를 WEV 로 합산한다(감사 §7 L5). 이걸 고치기 전엔 beat 수치로 어떤 레버의 발화도 주장하지 마라.

---

## 5. 떠먹이기(spoonfeed) 위험 — B_on 이 사실상 정답을 알려주는 레버

> [[62]]: 엔진이 argmax·최댓값·"정답은 X"를 내면 **측정 대상이 사라진다**. [[03b]]: 그런 arm 에서 양성이 나오는 것은 당연하므로 **결과가 무의미**하다.
> 아래는 "재면 안 된다"가 아니라 **"이 대조군 없이 재면 무효"** 라는 뜻이다.

### 5-1. 🔴 심각 — 대조 arm 없으면 결과 폐기

| flag | 무엇을 떠먹이나 | 필수 대조 arm |
|---|---|---|
| **T2_WRITE_ARG_ENUM** | B_on 이 **공식 명단 9개를 통째로** 싣는다(`Bronze / Diamond Elite / Gold / Gold Plus / Green (savings) / Platinum / Platinum Plus / Silver / Silver Plus`). 정답 이름이 그 안에 있으므로 채점 종점(이진: 명단 안인가)이 **문자열 복사**로 만족된다 | **C_list** = 같은 문면에서 `{candidates}` 명단만 제거한 판본. C_neg 만으로는 "집합 밖이다"와 "명단 9개"를 못 가른다 |
| **T2_VALUE_ACQUIRE / T2_HAVE_VALUE** | 문구가 다음 행동을 **축자 지정**한다: *"Use give_discoverable_user_tool to give get_card_last_4_digits to the customer NOW"*. 채점 종점을 give 호출로 두면 **순응 측정**일 뿐 | 도구 이름을 뺀 일반 지시 arm(예: "손님이 직접 실행해야 하는 값이다") 추가. 또는 종점을 하류(dispute 접수 성공)로 이동 |
| **T2_ARG_PRODUCERS** | 문구가 producer 도구 이름·절차를 통째로 지정(`'card_last_4_digits' is produced by … 'get_card_last_4_digits'`). (a) 해악 arm 의 give 이탈률은 순응 측정 | 정보량은 **(b) 이득 arm** (040t1 idx34 write 재발행)에 있다. (a)는 '해악'만 읽고 '효과'로 읽지 마라 |
| **T2_FIT_DIFF** | `{table}` 이 **카드 필드 차이표를 통째로** 제시한다 = 카드 선택에 필요한 재료 전부. 다만 **어느 것이 정답인지는 손님 발화와 대조해야** 하므로 완전 떠먹이기는 아님 | 표 유/무 arm 분리. + `{n}` 거짓 진술 버그(§4-14)를 결과에 함께 적을 것 |

### 5-2. 🔴 엔진이 답을 **쓴다** — 격리 이전에 금지 대상

| flag | 사유 |
|---|---|
| **T2_COMPUTE** | 엔진이 gold 값 `50` 을 인자에 직접 써넣는다. 게다가 임계값이 **gold 재현율로 피팅**됐다(`T1=2 → 73.6%` vs `T1=30 → 89.4%`) = [[23]] 정면. 제거 커밋 `b220745d` 가 바로 이 사유. **재도입·재측정 금지.** 굳이 재려면 **전달형**(정책표 축자 제시)으로 바꿔라 — 그건 다른 레버이고, x420 은 그것이 operand 를 안 움직일 것이라 예측한다(그 음성이 정확히 가질 가치가 있는 발견) |
| **T2_REF_ISO** | 격리 서브가 **정답 후보를 골라 인자를 치환**한다. 엔진이 결정을 대신하는 형태 | 
| **T2_GROUND** | fab 인자를 후보 1개로 **치환**. 동형 |
| **T2_OPERATOR_PINPOINT** | `{want}` 로 **도구 이름을 직접 지목**. x322 에서 이미 `B_PINPOINT 0/24` — 파괴적임이 실측됨 |

### 5-3. 🟡 A2 출처 감사 필요 ([[23]] — gold 보고 쓴 A2 = 실험 무효)

B_on 문자열이 **A2 에서 오는** 레버는 프로브 전에 **정책 축자 출처를 `_note_` 로 댈 수 있는지** 확인하라. 못 대면 그 arm 은 돌리지 않는다.

- `T2_WRITE_ARG_ENUM` (명단 = `policy_ontology.doc_index` 기계 도출 → 출처 OK로 보이나 group_map 은 확인 필요)
- `T2_VALUE_ACQUIRE` / `T2_HAVE_VALUE` (`gate.json:2352-2386`)
- `T2_WRITE_EVIDENCE` (`write_evidence_specs` 12건 — 특히 `RESOLVED_BANK_FAVOR`/`RESOLVED_PARTIAL` 분기 문면이 정책 산문인지)
- `T2_WRITE_PROV` (`completion_guard.feedback`, `gate.json:2659`)
- `T2_QUOTE_HINT` / `T2_REF_VERIFY` (격리 불가지만 미래 배치용)

### 5-4. 🟢 떠먹이기 아님 — 그래서 정보량이 크다

- **T2_HAVE_VALUE_FORCE** — 문자열이 없다(디코딩 제약). 정답 정보량 0.
- **T2_MATCH_COUNT** — 숫자 하나(완결 여부)뿐, operand 정보 0.
- **T2_GROUND_HDR** / **T2_SG_GROUND** — 값이 아니라 "이 값을 못 믿는다"만 말한다.
- **T2_ARG_EMPTY** — 문면이 요구하는 행동(b채움)이 **정답(c생략)과 어긋난다** ⇒ 떠먹이기의 반대. over-action 계측용으로 오히려 귀하다.
- **T2_RESOLVE**(task_050 5자리) — B_on 이 **gold 이름을 거부**한다 ⇒ 떠먹이기가 아니라 **오차단**. 부호가 음일 것.

---

## 6. 실행 순서 요약

```
T0 (무료, GPU 52건만)
  ├ T2_WRITE_PROV 층1 게이트 측정 (52응답)  ← 층2 의 분모
  ├ T2_PAIRCHECK/PAIRFIX 계기 검정 (GPU 0)
  ├ 계기 오귀속 수정: _lbeat 하드코딩 2건 (t2_gate_patch.py:7106-7110)
  └ T2_MATCH_COUNT shown-inflation 오프라인 검산 (원격 코퍼스)

T1 (237응답) — x420 미답 부류
  WRITE_EVIDENCE 45 · WRITE_ARG_ENUM 48 · RESOLVE 72 · HAVE_VALUE_FORCE 72

T2 (207응답)
  ARG_PRODUCERS 90 · VALUE_ACQUIRE 54 · DECIDE_BEFORE_WRITE 18(+2 서브콜)
  · ARG_EMPTY 9 · WRITE_ARG_GROUND 18 · GIVE_QUOTE 18

T3 (603응답) — x420 예측 확인용
  MATCH_COUNT 216 · GROUND_HDR 108 · SG_GROUND 108 · FIT_DIFF 63 · WRITE_PROV 층2 108(+108 재채점)
```

**총 GPU 응답 = 1,209건** (본 프로브 1,047 + 부수 162 · 팔 3종 × n=3 기준).
자리 1~2 인 4개 레버만 k=16 으로 올리면 **1,482건**.
