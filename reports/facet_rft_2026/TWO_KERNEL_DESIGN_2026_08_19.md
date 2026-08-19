# TWO_KERNEL_DESIGN — 결정점 전달 커널 + operand 검산기 (설계 권위본)

- 날짜 2026-08-19 · 대상 런 t7326(banking_knowledge · 40 sim · nt2)
- 짝 산출물(기계가독) `C:/workspace/ba-frft/reports/facet_rft_2026/two_kernel_lever_map_2026_08_19.json`
- 상위 등대 `RESEARCH_MASTER.md` · 분류 정본 `UNIFIED_TAXONOMY_2026_07_09` · 배치 정본 `GENERALIZED_SCAFFOLD_ARCHITECTURE_2026_07_12`
- 표기 규약 — 문장 단위 [S](직접 실측) / [M](간접·소표본) / [D](유도·추정) / [?](미측정). 새 실패-코드 이름은 만들지 않는다([[48]]) — 아래의 `K1.*`·`K2.V*` 는 실패 분류가 아니라 **설계 필드 좌표**이고, 레버 이름은 전부 기존 `T2_*` 를 그대로 쓴다.

---

## 1. 한 줄 요지

**60+ 레버를 ①결정점에서 재료를 다시 세우는 pull 서브콜 하나와 ②모델이 낸 계약 위의 닫힌-술어 검산기 하나로 접고, 재료를 세울 때 닫힌 술어로 빼는 필터를 그 부속으로 둔다. 엔진은 정답을 만들지 않는다 — 이름·값·성분·수량은 전부 LLM 이 근거와 함께 내고 엔진은 대조만 한다.**

### 이 설계가 답하는 실측 3개

| | 실측 | 이 설계의 응답 |
|---|---|---|
| **A. 실패 구조** | 미매치 gold 유효 111행(원 147 − generic 동반행 33 − 통과 sim 3행) [S]. 보정 분할 = 완료 허위주장 **3** · 의도만 말함 **43** · 계획 부재 **22** · 오선택 **43** [M]. 원 x396 의 "완료 허위주장 29" 는 귀속 누수로 **−90%** 붕괴하고 진성 사례는 `073 t0` 하나뿐이다 [S] | 커널①은 ①b(43)+②(22)=**65행**을 겨냥하고, 커널②는 ③(43)을 겨냥한다. **false-success 를 t7326 라이브 근거로 주장하지 않는다**([[46]] 금지문 준수) |
| **B. 이행 격리** | `A_min` exact **0.58** ↔ `B_full` 0.39 ↔ `C_neg` 0.19, plan 모드 `in_plan` 0.86 [M] · 용량-반응 **비단조** 0.58 → 0.08 → 0.25 → 0.17 → 0.31 → 0.39 [S] · 온도 무효 6팔 전부 [S] | **해로운 것은 문맥의 양이 아니라 형태다** — B_* 계열 안에서는 문맥↑ ⇒ 성능↑ 이고 `A_min` 만 그 곡선 밖에 있다 [S]. ⇒ 커널①은 잘라내기(edit)가 아니라 **재구성(rebuild)** 이다. 단 `A_min` 의 절차 블록은 **gold 도구 이름으로 고른 오라클**이므로(§2-3) 0.58 은 명세가 아니라 **상한**이다 [S] |
| **C. 계산·형식화** | `x394` **전량 무효** — `records_for()` 가 `db.json` 에서 계좌를 못 찾아 **72/72 프롬프트의 레코드 블록이 빈 `{}`** 였다 [S]. 유효한 것은 라이브 실물 3종뿐: 093 구간 오선택(2.5 ↔ 4.0), 094 출처 오선택(손님 주장 1.0 ↔ 문서 0.75), 094 성분 부분집합(카드 3장 중 1장만 선언) [S] | "인용은 맞는데 형식화를 틀린다"는 진술을 **철회**한다([[08]]). 대신 세 실물에 대응하는 닫힌 술어 셋(**V11 구간 소속 · V10 출처 자격 · V12 성분 완결성**)을 검산기에 추가한다. 셋 다 정책 축자 출처가 있다 [S] |

**추가 전제 정정 (설계의 바닥)** — 보상은 `action_checks` 가 아니라 `db_check` **하나**다. `017 t1`·`050 t1` 은 미매치 gold 를 갖고도 reward 1.0 이다 [S]. ⇒ ⒜ 읽기 gold 미매치는 보상과 무관하고 ⒝ **미매치를 하나도 안 만드는 초과 write 가 sim 을 죽인다**. 따라서 어떤 커버리지 산술도 pass 예측으로 쓰지 않는다.

---

## 2. 두 커널 명세

### 2-0. 왜 2 가 아니라 2+1 인가

재료를 세울 때 **닫힌 술어로 빼는 필터**가 없으면 이미 실측된 실패가 그대로 돌아온다: `C496` 은 날짜를 맞바꿔도 답이 안 바뀌는 것을 24/24 로 보였고(`D_FLIP` SKY 24/24 = 모델이 날짜를 안 쓴다) [S], `C417` 은 만료 창에서 `W_EXPIRED` 0/8 이었다 [S]. 그리고 이 필터는 **제거만** 한다 — `C511` 은 2지선다(`C_PAIR`) 2/8 ↔ 오답 제거(`E_MINUS`) 6/8 로 더하기와 빼기를 갈랐다 [M]. 그래서 커널①의 재료 조립 안에 `K1.M6` 을 부속으로 둔다.

---

### 2-1. 커널① — 결정점 전달 커널 (`K1`)

#### (a) 트리거 — pull 이고, 술어는 전부 코드다

**신설이 아니다.** 트리거 자리는 `t2_gate_patch.py` 의 `T2_WRITE_SUB=3` 술어이고, **t7326 은 이미 그 값으로 돌았다**(`run_t7326_stage1_nt2_20260819.sh:76` 축자 `T2_WRITE_SUB=3`) [S]. 프롬프트도 이미 있다 — A2 `write_initiation.instructions` 축자:

> "You are handling ONE decision in isolation. Below are the customer's request and the verbatim results of the audit tool already run in this conversation. Using ONLY those results, carry out the correction the policy requires now."

트리거는 **두 항의 논리합**이고 두 항 모두 카운트·집합 멤버십뿐이다. 의도·국면 분류 0([[66]]·⛔0④).

| 항 | 술어 | 근거 |
|---|---|---|
| **D1 착수 결정점** | ①`gate.check(t,{},last_user,transfer_sent)[0]` 가 참인 `t ∈ a2.action_tools` 존재 ∧ ②그 `t` 가 `_executed_dispatch_names(msgs,a2)` 에 없음 ∧ ③`hash(recent_tool_text(msgs))` 가 직전 호출 때와 다름 ∧ ④`_last_assistant_did_write(msgs,W)` 거짓 | 기존 배선 축자 재사용 |
| **D2 실행 직전(operand 결정점)** | 메인 초안에 tool_call 이 있고 그 **실효 이름 ∈ W** | 아래 W 정정 |
| **D3 IDLE(★신설 항·2안에서 이식)** | 손님-가시 응답 직전이고 `assistant_turns_since_last_tool_call ≥ 2` | 실효 write 가 **0회인 sim 이 12개**이고 그 12개가 미매치 gold **53/147** 을 갖는다 [S]. D2 단독은 이 53행에 **영영 발화하지 않는다** |

**W 의 출처 정정 [S]** — 1안 초안이 지목한 `_confirm_write_tools(a2)` 는 banking 에서 **공집합**이다: `banking_knowledge.gate.json` 의 `gates[].kind` 는 `['auth','notice','precheck']` 뿐이고 `'confirm'` 이 없다. `eplan.write_tools` 도 `['file_credit_card_transaction_dispute','file_debit_card_transaction_dispute','submit_cash_back_dispute']` 3종뿐이라 gold 표적(`apply_checking_account_credit_*`·`open_bank_account_*`·`close_debit_card_*`)이 **하나도 없다**. 실제 주소 형식은 평평한 이름 집합이 아니라 **디스패처 + applies_when** 이다 — `write_evidence_specs` 12건이 전부 `{"applies_to":"call_discoverable_agent_tool","applies_when":{"arg":"agent_tool_name","prefix":…}}` 형태다 [S].

⇒ **W = `a2.action_tools`(8종) ∪ `write_evidence_specs[].applies_when.prefix`(12) ∪ `write_arg_enum[].applies_when.prefix` ∪ `eplan.write_tools`(3)**, 그리고 이름 대조는 `_exact_tool_name`(축자 동일성)으로만 한다. `_eff_tool_name` 의 `_SUFFIX_RE = re.compile(r"_\d+$")` 는 쓰지 않는다 — `t2_procedure.py` 도입부 축자가 같은 이유를 이미 적어 두었다:

> "An earlier version normalised a numeric suffix away, which is the kind of pattern rule this project has already retired for producing quiet mismatches (C279)"

**예산·실패 모드**: sim 당 커널 호출 ≤ 12, 동일 3중키 `(진입술어, 표적도구집합, 근거서명)` 은 1회만([[57]] 무내용 재시도 금지). 초과·예외·파싱 실패·재료 블록 공백은 **무발화 fail-open**(종전 경로). `040 t0` 형 자해(gold write 4회 차단·회복 0) 재발 금지 [S].

**push 금지**: 이 술어들이 참일 때만 존재한다. `C543` 은 `VERDICT_CARRY` 를 항상 켜면 073 이 **1.0 → 0.0** 으로 죽는 것을 보였고(무정보 발화 `후보 10·OK 10·VIOLATES 0`·`WRITE_SUB` 31↔19 밀어냄) 범위 술어 세 갈래가 전부 막혔다 [S]. ⇒ **pull 이 유일 해다.**

#### (b) 재료 — 각 블록의 런타임 결정론 경로와 gold 미경유 증명

**증명 규약 두 개(전부 무료·오프라인)** — ⒜ **T1 태스크 불변성**: 태스크를 바꿔도 블록이 바이트 동일한가. 동일하면 per-task gold 가 원리적으로 들어갈 수 없다. ⒝ **T2 gold-스왑 민감도**: gold 만 셔플하고 블록을 재생성해 바이트 비교. 바뀌면 gold 경유 확정.

| 필드 | 블록 | 런타임 경로(결정론) | T1 | T2 | 판정 |
|---|---|---|---|---|---|
| `K1.M1` | 호출 가능한 도구 | env tool registry ∪ `agent_discoverable_names` ∪ `_retrieved_unlockables` — **그 턴에 실제 호출 가능한 것만** | ✗(턴 의존) | 불변 | **G0** — 도구 우주 50개를 주면 우주 밖 이름이 **4/129** 로 떨어진다 [S]. ★`x395` 는 문서 정규식 채굴이었고 `TOOLNAME_RE` 에 도구명 8개가 리터럴로 박혀 있었으며 unlock 전 discoverable 까지 다 줘 **시점 누설**이 있었다 [S] ⇒ 라이브 커널은 리터럴 0 |
| `K1.M2` | 손님 요청 | `state.messages(role=user)` **전량 축자**, 각 발화에 `[손님 주장]` 라벨 | ✗ | 불변 | **G1** — `x395` 는 첫 발화 400자만 실었다 [S]. ⚠반대 위험: 094 는 `96000/408/5.0/6.0/480/72` 가 전부 요청문 안에 있고 모델이 12/12 로 전사했다 [S] ⇒ 라벨 + `must_be_ledger`(V10)가 짝이어야 한다 |
| `K1.M3` | 실행 원장(값 포함) | `t2_subcall.recent_tool_text(msgs, cap, scope='all')` — (도구, 인자, **성공 결과 본문 축자**) | ✗ | 불변 | **G1** — 판단 0·문자열 수집만. ★이름+id 만 실으면 write 표적에서 진다: `apply_checking_account_credit_5829` 에서 `A_min` **0/3·0/3** ↔ `B_full` 2/3·3/3 [S]. ⛔"요약"·"관련순 정렬"을 붙이는 순간 선택이 곧 답이 된다 — 축자·시간순·앞자르기만 |
| `K1.M4` | 레코드 id 집합 | `K1.M3` 결과 본문에서 A2 `id_shape` 정규식으로 수집 | ✗ | 불변 | **G1** — V3 의 기준집합을 겸한다. ⚠12표적 중 3건이 수집 결과 공집합이었다(055 t1·085 t0/t1) [S] ⇒ 공집합이면 계약 대신 `absent` 를 요구 |
| `K1.M5` | 정책 재료(2단 회수) | ⓐ `t2_search.action_index_note(a2)` 로 A3 `policy_ontology.action_index` **43줄** 인쇄 → ⓑ LLM 이 doc id 지목 → ⓒ `t2_search.docs_for/read_docs/material_for(per_doc=400)` 가 **축자로 읽는다** | **✓ 43줄은 태스크 불변** | 불변 | **G0(ⓐ) + G1(ⓑⓒ)** — A2 `_note_action_index` 축자: *"★출처 = 환경 파일뿐(문서 `title` 필드 + 본문이 대는 레지스트리 도구명 정규식·`t2_index_build`). 저작 0·gold 무접촉·판정 0 — 엔진은 이 목록을 **그대로 인쇄만** 하고 무엇을 부를지는 LLM 이 정한다([[62]] ④)"*. 실측: 도움 없음 10/24 → **action 문서 제목 43줄 24/24** [M]. ★★이것이 `x395` **오라클의 대체물**이고 그 recall 은 **0회 측정** [?] |
| `K1.M6` | 닫힌 술어 재료 필터 | `t2_search.drop_expired(read, declared_windows(a2, read), now)` + `t2_search.eligibility_line(...)` 를 **상류**에 | ✓(선언은 불변) | 불변 | **G0** — `C517`: 상류 배치 시 요청 밖 군 11→6 · `business_*` 오선택 3→0 · 적중 27/27 불변, **하류는 답 바뀐 축 0개** [M]. 뺀 것은 이유와 함께 표기 |
| `K1.M7` | 미실행 빼기 | `_executed_dispatch_names(msgs,a2)` 를 도구 우주에서 뺀다 | ✗ | 불변 | **G1** — 더하기가 아니라 빼기다 |
| — | **⛔ 싣지 않는 것** | 어시스턴트 산문 **0자** · 대화 전문 · gold · 태스크별 힌트 · 우리 A2 훈계 산문 | | | 자기오염 루프가 실측되었다: no-op 기권 87건이 **전부 대화 팔**에서 나오고 비대화 팔은 **0/72**(표적 단위 p=0.008), 그 기권의 근거가 축자 *"The credits have already been applied and confirmed"* [S]. `C490` 은 우리 현행 `ask` 문구가 위반을 **0/24 → 24/24** 로 제조함을 보였다 [S] |

**누출도 경계 진술** — **G0**(태스크 불변) → 힌트 불가 · **G1**(손님 발화·원장 조건부) → 정당한 전달 · **G2**(선택자가 gold·태스크 메타를 봄) → 떠먹이기. `x395` 의 `proc_lines` 는 선택 기준이 `if tool not in d["content"]: continue`(tool = 빠뜨린 gold 도구)라 **G2** 다 [S]. 코퍼스는 **698 문서·7,367 절차줄**이고 그것이 2~5줄로 줄었다 = 축소비 **1/1,500 ~ 1/3,700 을 정답이 해줬다** [S]. **이 설계의 M5 는 그 자리를 G0 재료(43줄)로 대신하며, 그것이 성립하는지가 곧 §8-M2 다.**

**보정된 상한 [D]** — 이름·시그니처가 다 인쇄된 11표적의 `A_min` exact = **18/33 = 0.545**. 즉 답을 축자로 깔아줘도 45% 가 안 부른다. 라이브 상한 ≤ recall × 0.545, 바닥 `C_neg` 0.19. recall 0.6 가정 시 순증 0.14 → 순수 잔여 39행 × 0.14 ≈ **5~6행** [D]. **"15~21행"은 오라클 계수로 계산된 값이라 인용 금지.**

#### (c) 출력 스키마 (`K1.O`)

JSON 1개, 마크다운 펜스 금지. 형식 부담을 최소로 둔다 — `C528` 이 봉투 지시가 행동량을 부풀리고 검증기가 기준선으로 되돌리는 것을 짝비교 36 에서 실측했다(B→C 호출 감소 27/4 · 증가 **0**) [S].

```json
{"decision":"act|ask|none",
 "targets":[{"id":"<레코드 id>","quote":"<원장 축자 ≤120자>"}],
 "count_claim":{"n":3,"quote":"<손님/문서 축자 — 왜 n 개인가>"},
 "calls":[{"step":"unlock|give|call",
           "tool":"<레지스트리 실명·접미사 포함>",
           "executor":"agent|user",
           "arguments":{...},
           "evidence":[{"arg":"<인자명>","value":<값>,
                        "source":"ledger|policy|user_claim|derived|expects:<seq>",
                        "quote":"<축자>"}],
           "applies_when":[{"arg":"<열거 인자>","value":"<값>","quote":"<정책 축자>"}],
           "components":[{"kind":"<A2 선언 kind>","value":<수>,
                          "source":"policy|ledger|user_claim","quote":"<축자>"}],
           "identity":"<A2 identities id>|null"}],
 "declined":[{"tool":"<부르지 않기로 한 후보>","why_quote":"<근거 축자>"}],
 "reply_claims":[{"claim":"<손님에게 할 완료·의도 주장>","evidence":"exec:<call_id>|absent"}],
 "absent":[{"need":"<빠진 값>","blocked":"<막힌 항목>","ask":"<손님에게 물을 질문>"}],
 "reason":"<≤200자>"}
```

규약 8:

1. **모든 인자에 evidence 1행 강제** — 단, **검산기가 실제로 쓰는 param**(id·enum·derived-number)에만 요구하고 자유 텍스트 인자에는 요구하지 않는다(스키마 무게의 역효과 회피·`C528`).
2. `targets` 는 **LLM 이 선언하는 대상 집합**이고 엔진은 `len(calls)` 와의 자기정합만 본다. 엔진이 원장에서 대상을 유도하면 그 순간이 정답 발급이다([[62]]).
3. `count_claim` 이 **엔진의 수량 파싱을 대체**한다 — 3안 실측이 출하 `_parse_qty` 의 오염을 확정했다: `094 qty=6 ← "6.0%"`, `063 qty=8 ← "$8,000"`, `024 qty=5 ← "1.5%"`, `057 qty=4 ← 리스트 마커 "4)"` [S]. 수량 해석은 LLM 이 인용과 함께 내고 엔진은 substring 검산만 한다(사용자 확정 2026-08-12 "인용-근거로 대체"·C45 동형).
4. `step:"unlock"|"give"` 를 **1급 시민**으로 둔다 — gold 는 `unlock_discoverable_agent_tool`·`give_discoverable_user_tool` 을 독립 action 으로 채점한다(`task_055 t0` gold 8행 중 3행) [S]. 부수 효과로 `F.inner_name()` 이 unlock 을 호출로 계수하던 배선 오탐 **11건**(040 t0 8 + 073 t1 3)이 구조적으로 사라진다 [S].
5. `reply_claims` 는 **산문면 검산의 유일한 입구**다 — 이행 최대 범주(완료·의도 주장)는 정의상 tool_call 이 아니라 산문이다.
6. `declined` 가 있어야 후보 무시가 침묵이 아니라 기록이 된다.
7. 못 정하면 `decision:"ask"` + `absent` — 침묵·추측 금지([[64]]). 없으면 모델이 id 를 지어낸다(085 t1 `tx123456`·`unknown`·`account_id=user_id`) [S].
8. 메인에 올라가는 것은 **검산 통과한 `calls`(또는 deny 한 줄)뿐**이다. 재료·후보·중간계산은 서브 안에서 끝난다([[65]]) — `C397` 4% ↔ 100%, `x187` 2모델×20셀 전부 파레토 지배 [S].

#### (d) 부정통제 (`K1.N`) — 코드에 상설

| 팔 | 무엇을 비우나 | 실패 판정 |
|---|---|---|
| `KERNEL_NEG` | `M5`·`M6` 을 **같은 길이 무내용 문구**로 치환. 지연·토큰·턴 동일 | 처치 − NEG < 문턱이면 커널 내용은 값이 없다 |
| `KERNEL_DISCARD` | 진짜 재료·진짜 서브콜, **답만 버린다**(메인 부착 0자) | 이득의 50% 이상을 재현하면 커널을 폐기하고 **빈 턴**으로 대체(훨씬 싸다) |
| `KERNEL_SHUF` | 원장 id ↔ 값 뒤바꿈 | 기각률이 안 오르면 검산기는 死배선이다(`t7290` 에서 `T2_SEARCH_AGENT` **0 전달/18 침묵**으로 하루를 태운 전례) [S] |
| `KERNEL_ORACLE` | `M5` 를 `x395` 의 `proc_lines`(gold 선택자)로 되돌린 **상한 참조 팔** | 라이브 팔과의 낙차가 곧 회수 recall 의 값이다 |

---

### 2-2. 커널② — operand 검산기 (`K2`)

**원칙**: 출력은 **불리언과 기각 목록뿐**이다. 옳은 값을 알 필요가 없고, 알려고 드는 순간이 ⛔0 위반이다.

**기수(cardinality) 규칙 [설계 하드룰]** — 엔진 필터의 잔여 후보수 |S| 를 매 호출 계측하고, **|S| = 1 이 되는 순간 그 검사는 검산이 아니라 해답**이다. 그 사건을 로그하고 해당 행은 성적에서 제외하거나 그 필터를 금지한다. 실물 선례가 이미 있다: `t2_compute.group_reduce` 의 reducers `{"checking":"max1","card":"max1","relationship":"sum","tier":"sum"}` 가 카드 3장 위에서 max 를 취해 **6.85** 를 만들고, A2 주석이 축자로 *"gold 095 재현 6.85"* 라 적었다 — **gold 로 엔진을 검증했다** [S].

| # | 검산 | 술어 | 왜 닫힌 술어인가([[22]]) | 표적(실물) | 사정거리 |
|---|---|---|---|---|---|
| **V1** | 이름 실재 | `call.tool ∈` 그 턴 레지스트리(접미사 포함 축자 동일) | 집합 멤버십. 표현 변이에 불변 | 055 t1 `open_green_fee_free_account` · 072 t1 `apply_fee_refund` — 둘 다 **손님에게 건네져** sim 을 죽였다 [S] | 도구명 날조 전량 |
| **V2** | 값의 코퍼스 실재 | `val_grounded` — 형식 불문 수치·날짜 대조 | 문자열/수치 동일성 | 074 `txn_001~004` 날조 배열 [S] | **날조만.** `C46`: `FIND-wrong` **3/30 안 닫힘**(FAB·CLEAN 공통) [M] |
| **V3** | 식별자 자리 | A2 `id_args` 선언 인자의 값 ∈ `K1.M4` id 집합 | 집합 멤버십 | 085 t1 `tx123456`·`unknown`·`account_id=user_id` · 074 t1 `account_id:"Purple Account"` [S] | F3-ID 19 중 실물 확인 4 ⇒ **[4, 19]** [D] |
| **V4** | 열거 자리 | `write_arg_enum` 소속 ∧ 그 값 축자가 **배달된 재료**에 실재 | 집합 멤버십 + substring | — | ★**F3-ENUM 11 중 0 을 닫는다** — 11/11 전부 모델 값이 정책에 실재하는 정당한 열거값이다(003 `Platinum`·024 `Business Platinum`·055 `Green Fee-Free`/`Gold`·063 `Bronze`/`Gold`·057 `Light Blue`·079 `CLASSIC`·004 `customer_demands_after_unavailable_offer_refusal`) [S]. **2안의 "V5 = 11/11 원리적 전량" 주장은 원자료로 반증된다.** V4 가 닫는 것은 비실재 열거값뿐이고, 적격 오선택은 `K1.M6` 빼기의 몫이다 |
| **V5** | 인용 실재 + 정규화 | 모든 `quote` 가 **그 호출에 배달된 블록**의 정규화 substring | substring | — | ★정규화 규약이 본체다: 마크다운 강조 제거(`C510` 제안 7→통과 0), 따옴표 통일(`C533`), em dash ↔ `--`(`C534` 가 098 핵심 요구를 거짓 기각), 공백 접기, 수치 십진 비교(`C486` 9.50↔9.5) [S]. 규약마다 왕복 회귀 테스트 동봉 |
| **V6** | 항등식 자기정합 | LLM 이 선언한 `identity` 에 `components` 를 대입해 재계산 → 인자와 형식-불문 비교 | 산술 동일성 | `get_interest_correction.return_template` 축자 *"Correction amount = principal x (expected-actual)/100 / 12"* [S] | F2 순수 11 중 ≈4~6 [D]. 엔진은 식도 성분도 고르지 않는다 |
| **V7** | arity·커버리지 자기정합 | `len(calls) == len(targets) == count_claim.n` ∧ `target.id ∈ M4` | 카운트 + 멤버십 | 073 t1 gold 4 vs 서로 다른 호출 1 · 040 t1 gold 9 vs 2 · 085 t1 gold 4 vs 2 [M] | 부분이행 **20행** — 이 설계에서 새로 열리는 최대 칸 |
| **V8a** | 응답 내 완전중복 | 같은 응답 안 동일 (실효이름+인자) 2회+ ⇒ **strip·무발화** | 카운트 | — | `T2_STALE_STRIP` 축자 술어 그대로 |
| **V8b** | committed write 재호출 | committed 에서 이미 성공한 **write** 재호출 ⇒ deny + `prior_result` 동봉 | 카운트 + 집합 | `050 t0` 동일 인자 2회(fail) ↔ `t1` 1회(pass), 양쪽 gold 12/13 매치 [S] · 정확중복 write 초과 **164회·12 sim** [S] | ★**pass 를 뒤집는 것이 [S] 로 증명된 유일한 술어** |
| — | ⚠**read 의 committed 재조회는 어느 쪽도 아니다** | | | `T2_STALE_STRIP` docstring 축자: *"read의 committed-재조회는 미포함(상태변화 존중·over-fire 방지)"* [S] | read/write 무차별 dedup 은 `C542` 가 실측한 `READ_MISS 11:11` 을 우리 손으로 제조한다 |
| **V9** | 실행자 역할 | `executor` ↔ A2 `dispatcher_role_check` 채널 | 집합 멤버십 | 016 손님 도구 `submit_transaction` 본문 언급 0 · 063 t1 *"Log in to your Rho-Bank account … Select 'Open New Account'"* [S] | |
| **V10** | 출처 자격 | A2 `must_be_ledger` / `must_be_policy` 필드에 `source:"user_claim"` 금지 | 집합 멤버십 | **094**: 모델이 `checking boost 1.0`(손님 주장) + `base 5.5`(문서) 를 섞었다. 태스크 축자: *"thinks base rate is 5.0% (actual 5.5%) and checking boost is 1.0% (actual 0.75%)"* [S] | 인용 실재 검산은 손님 발화를 출처로 **통과시킨다** — 이 술어가 없으면 094 는 원리적으로 안 닫힌다 |
| **V11** | 구간 소속 | 선언 임계표에 대해 `bracket(quoted_operand) ∋ declared_value` | 수치 구간 비교 | **093**: 정책 축자 *"- Balances under $25,000: 2.5% APY"* / *"- Balances at or above $25,000: 4.0% APY"*, 태스크 잔액 $144,000 인데 모델이 2.5 선언 [S] | ⚠**엔진이 티어를 고르면 안 된다** — LLM 이 `applies_when{arg, value, quote}` 로 술어와 피연산자를 **둘 다 선언**한 뒤에만 대입한다 |
| **V12** | 성분 완결성 | `|declared[kind]| == |ledger_candidates[kind]|` | 카운트 | **094**: 정책 축자 *"Only the HIGHEST applicable credit card APY bonus is applied"*, 손님 카드 3장인데 모델이 card 성분 1개만 선언 [S] | 엔진은 max 를 취하지 않는다 — **개수 불일치만 말한다** |
| **VP** | 산문면(`reply_claims`) | ⒜ 산문의 도구명 토큰 ⊆ (registry ∪ 원장) ⒝ 완료 주장 → 실행 이벤트 id 참조 필수 | 집합 + 참조 존재 | `073 t0` *"These corrections have been applied to your accounts."* → 호출 0 [S] | ★계약만 검산하면 최대 이행 범주를 **구조적으로 못 본다** |
| **VC** | 손님-채널 왕복 | give 산문에 실린 인자 == user-sim 실제 호출 인자 | 문자열 동일성 | user gold **24 중 미매치 18(75%)** ↔ assistant 265 중 129(48.7%) [S]. `017 t0` user-sim 이 `user_id:"Kenji Tanaka"`·스키마 밖 `recorded_rewards`/`correct` 를 냈고 `057 t0` 은 `navigate_to_section` 을 냈다 [S] | 계약 → 산문 → user-sim 왕복을 보는 유일한 자리 |
| **DENY** | 거부 렌더러 | [[64]] 3슬롯 | — | `C413`/`C414`: 이름 없는 *"먼저 해소하라"* 가 3회↑ 나온 6 sim = **6/6 실패** [S] | |

#### deny 문면 3-슬롯 규약 (이름은 대되 정답은 흘리지 않는다)

허용 슬롯은 셋뿐이다 — ⒜ *무엇이*(검사 id + 필드명 + **모델이 낸 값 되읽기**) ⒝ *왜*(A2 **정적** 선언의 축자 요구) ⒞ *무엇을 하면*(그 근거를 **만드는 행위의 이름** — 도구명·문서 id·필드명, **값이 아니라 이름**). 여기에 V8b 는 `prior_result` 슬롯을 더한다 — `t2_scaffold_get.py` 축자: *"스텁이 이전 결과를 재제시하지 않으면 \"earlier output 참조\" 지시가 재호출 유인을 못 끊는다(020 실측: 동일 인자 5회=창 29%)"* [S].

**금지**: 올바른 값·올바른 성분·후보 목록·개수·순위·"대신 X 를 써라".

**불변 조건 2개(오프라인·무료·전수)** — **L1 gold-스왑 불변성**: 같은 모델 출력 + 셔플된 gold 로 deny 를 렌더해 **바이트 동일**해야 한다. **L2 회수 불가능성**: `deny-only` 팔(최소 재료 + deny 문면만)의 표적 적중이 `C_neg + 0.10` 을 넘지 않아야 한다.

예시(094·V10): *"[V10 source] actual_apy=5.0, source=user_claim. A2 must_be_ledger 에 actual_apy 가 있다(출처 축자: «…»). 손님 발화는 이 필드의 출처가 될 수 없다. 통과 조건: source=ledger 로 하고 그 값이 등장하는 도구 결과를 인용하라. 원장에 없으면 값을 만들지 말고 absent 에 적어라. 이 필드의 producer 로 A2 가 선언한 도구: get_bank_account_transactions_9173, get_all_user_accounts_by_user_id_3847."* — 도구명은 A2 `arg_producers`(태스크 불변)에서만 오므로 L1 을 통과하고, **5.10 도 어느 거래인지도 말하지 않는다**.

---

## 3. 배치 결정

### 채택 = **1안(매 결정점 pull 서브콜 + operand 검산기)**, 단 트리거에 2안의 IDLE 항을 이식한다.

| 이유 | 근거 |
|---|---|
| **신규 코드가 압도적으로 적다** | 커널①의 네 요소가 이미 라이브에 있다 — 프롬프트 `write_initiation`, 트리거 `T2_WRITE_SUB=3`(t7326 에서 이미 ON), 전송·파싱·V1+V2 `t2_resolve.sub_write_proposal` + `t2_subcall.{sub_generate,parse_contract,grounded_calls}`, 재료 조립 `t2_search.{action_index_note,docs_for,read_docs,drop_expired,material_for,eligibility_line}` [S]. 되돌리기가 **플래그 한 개** |
| **상태기계가 없다** | 2안의 계획 저장소·executed 마킹·일탈/재계획 카운터는 전부 신규이고, **그 자리를 이미 시도한 `T2_EPLAN`/`T2_EPLAN_WALK` 가 라이브에서 값을 못 샀다**([[14]] 미해결) [M] |
| **실패 반경이 한 턴** | 2안은 스스로 R-G 에 적었듯 관문 1회가 sim 당 평균 20회(총 561/1338 = 42%)를 통치한다 [S] |
| **[[62]] 보존** | 1안은 결정점마다 **동시대 원장**으로 재구성한다 — `x395` 가 잰 것과 같은 조건. 2안은 그 재구성을 최대 20회 재사용하므로 전이 계수가 미측정이다 [?] |
| **3안은 자기 실측이 기각했다** | (listed−examined)는 **36/40 sim 에서 전 구간 공집합**이고, (qty−executed)는 숫자 오염이 확정적이다 [S]. orphan 101/139 로 능력 상실도 최대다 |

### 패자에서 이식하는 것

| 출처 | 이식 항목 | 왜 |
|---|---|---|
| **2안** | **IDLE 트리거 항(D3)** | 0-write sim 12개가 미매치 gold **53/147** 을 갖는다 [S]. write 트리거 단독은 이 53행에 영영 발화하지 않는다 |
| **2안** | `must_be_ledger`(→V10) · `closed_filters{kind:threshold}`(→**V11 로 배선**) | 2안만 기계를 갖고 있었으나 재료 필터로만 쓰고 성분 검산에 연결하지 않았다. 093/094 는 그 배선으로만 닫힌다 |
| **2안** | `arg_evidence.source` 4지선다 | C45 계보(날조 67%→0%·over-block 0/2650·Δspurious 0) [M] |
| **2안** | `arguments:null` + `source:"expects:<seq>"` 지연 바인딩 | 040 형 교차채널(손님만 생산 가능한 값)의 유일한 표현 수단 |
| **3안** | `count_claim`(수량 해석을 LLM 인용으로) | 엔진 `_parse_qty` 오염 4건 실물 [S] |
| **3안** | `declined` 필드 | 후보 무시를 침묵이 아니라 기록으로 |
| **3안** | 미결 후보 집합(C_name)을 **재료가 아니라 계측**으로 | 정밀도 0.24(후보 93 중 gold 관련 22) [S] ⇒ 재료로 실으면 over-action 축을 태운다. 트리거 커버리지 진단용으로만 유지 |

### 단계적 이행

**1단계 — 커널 0. 배관 수리 + 계측만.** ([[55]] 0단계)
- 대상: 우리 층 미회복 11건 fail-open(분포는 §7), 런처 통일(go_stack 로 흡수), `_ap_regen` 29 사이트 선점 로거, `F.inner_name()` unlock 계수 수리, **`shell` OFF**(`C487` 0/8·t7292 2/8·075 2/2→0/2 전 태스크 하락 [S] 인데 t7326 에서 **41회/16 sim** 살아 돌았다 [S] — 커널 재료 ⑤와 같은 기능의 검증 안 되는 두 번째 회수 채널이라 켜 둔 채로는 recall 이 측정 불가다).
- 통과 조건: 같은 시드 재실행에서 gold-block 미회복 **11 → 0~2**, `READ_MISS` 비악화. **pass 는 종점이 아니다.**

**2단계 — 커널① D1+D3 + 검산 관측 전용(deny 0).**
- `KERNEL_VERIFY=observe|enforce` 노브 하나. `KERNEL_NEG`/`KERNEL_SHUF`/`KERNEL_DISCARD` 동시 필수.
- **그 전에 무료 격리로 M2(회수 recall)를 먼저 판다** — 이것이 0순위다([[09]]).

**3단계 — enforce, 단 write 만 · fail-open · over-block 계측.** read 는 `허용+마크`. `C50` 문턱(over-block 6 > TP 5 면 NO-GO) 사전등록 [M].

**4단계 — 레버 이관, 한 번에 하나.** 순서 = 중복 계열(x392 에서 gold 차단 0건으로 가장 안전) → 이름 우주 → 근거 계열 → 계산 계열(`T2_RESOLVE` 는 라이브 최대 차단자라 **마지막**). 매 단계 같은 시드 재실행 + 래칫 5종.

---

## 4. 레버 이관표

전량은 짝 JSON `two_kernel_lever_map_2026_08_19.json` 에 있다 — **177 행**(t2_levers 레지스트리 132 ∪ go_stack ∪ run_t7326 ∪ run_one), 상태 분포 **흡수 112 / 보존 65 / 잔여 0**. 엔진이 실제로 읽는 `T2_*` 는 **267 종**이므로 전제의 "60+" 는 과소계상이다 [S] — 미선언 잔여는 런처 통일(1단계)로 먼저 좁힌다.

**⛔ 이 표에 "끈다" 상태는 없다([[60]]).** 실측 음수가 나온 레버도 **기능은 커널 필드로 옮기고 문면만 제거**한다.

| 커널 필드 | 흡수되는 기존 레버(발췌) | 근거 |
|---|---|---|
| `K1.T` 트리거 | `T2_WRITE_SUB` `T2_ACTION_SUB` `T2_NOW_SELFCALL` `T2_DELIVER_PRECOMMIT` `T2_SG_ISOLATE` `T2_DECIDE_ANY` `T2_TOOLGATE` `T2_PHASE_OWNER` `T2_ENVELOPE_GUARD` `T2_DECIDE_BEFORE_WRITE` `T2_FORCE_ACTION` `T2_ACT_DEMAND` `T2_SPEAK_PROHIBIT` | `C473` 073 첫 1.0·65 msgs ↔ 사후 배선 202 msgs [M] · `C439` `T2_DECIDE_BEFORE_WRITE` **P0 발화 0회**(가드가 `not _t2_search_done` 이라 실패와 정반대 조건) [M] · `C492` 촉구 배선 231↔0 통과인데 over-action 2→8·050 pass 1→0 [M] |
| `K1.M1` 도구 우주 | `T2_TOOLLIST` `T2_CALLABLE_HINT` `T2_DISCOVERY_NAMES` `T2_DISCOVERY_STEP2` `T2_PENDING_DISCOVERED` `T2_TOOL_SIGNATURE` | 우주 50개 제공 시 우주 밖 이름 4/129 [S] |
| `K1.M3` 원장(값) | `T2_LEDGER` `T2_DISPATCH_LEDGER` `T2_SCAFFOLD_GET` `T2_SG_REQREADS` `T2_VALUE_ACQUIRE` `T2_HAVE_VALUE` `T2_HAVE_VALUE_FORCE` **`T2_WRITE_EVIDENCE`(차단→사전 재료 배달)** | `A_min` 0/3 ↔ `B_full` 3/3 [S] · `T2_WRITE_EVIDENCE` 는 x392 에서 gold 22건 차단·**미회복 5** [S] |
| `K1.M5` 정책 재료 | `T2_PREKB` `T2_PROCEDURE` `T2_PIN_READ` `T2_PIN_READ_STEPS` `T2_PROC_PIN_REARM` `T2_REQUIRE_DOC` `T2_SEARCH_AGENT` `T2_SEARCH_ON_PROCEED` `T2_PROCEED_DOCBODY` `T2_DOCS_AT_WRITE` `T2_ACTION_INDEX` `T2_SUB_REQUIREMENT` | x319 10/24 → 43줄 24/24 [M]. **검색은 `t2_search` 하나로 고정**([[67]]) |
| `K1.M6` 재료 필터 | `T2_ELIG_LINE` (+ 신설 `closed_filters` 소비) | `C517` 상류 11→6·오선택 3→0 / 하류 0 [M] |
| `K1.O` 계약 | `T2_EPLAN` `T2_EPLAN_WALK` `T2_DECISION_CARRY` `T2_VERDICT_SURFACE` `T2_VERDICT_CARRY` `T2_VERDICT_GATE` `T2_SOURCE` `T2_SELF_DECLARATION` `T2_DECLFIRST(_GUIDE_FIX)` `T2_WRITE_PROV` `T2_GUIDED` `T2_USER_TOOL_NOTE` `T2_GIVE_*_NUDGE` `T2_UNLOCK_NAME` `T2_UNLOCK_PROV` `T2_PROC_ABSENT` `T2_ABSTAIN_FIELDS` `T2_UNAVAIL_PROMISE` `T2_TRANSFER_LEAVES_STEPS` `T2_HANDOFF_PREDICATE` | `C543` VC 단독이 073 을 1.0→0.0 [S] ⇒ **push→pull** · `C529` 술어 도달 75%인데 pass 2/12↔2/12·지연 1.90×·CWE 13↔0 [S] ⇒ **상시 문구 폐기·기권 채널 유지** |
| `K2.V1~V12` | `T2_UNKNOWN_NAME_BL` `T2_GROUND` `T2_SG_GROUND` `T2_WRITE_ARG_GROUND` `T2_TRANSCRIBE` `T2_REF_VERIFY` `T2_SG_BYREF` `T2_FAB_STRIP` `T2_CHOICE_GROUND` `T2_ARG_SCHEMA` `T2_ARG_AXIS` `T2_ARG_EMPTY` `T2_CALL_FORM` `T2_WRITE_ARG_ENUM` `T2_BRANCH_REGROUND` `T2_TRANSFER_TIER` `T2_QUOTE_PIN` `T2_QUOTE_HINT` `T2_GIVE_QUOTE` `T2_GROUND_HDR` `T2_COMPUTE` `T2_RESOLVE` `T2_PRESCRIPTION` `T2_SG_TRACE` `T2_COVERAGE_FOLLOWUP` `T2_FOLLOWUP_*` `T2_WITHDRAWN_ROW` `T2_COV_MIDDRIVE` `T2_STALE_STRIP` `T2_READ_DEDUP` `T2_READ_NEARDUP` `T2_SG_DEDUP` `T2_NOTICE_REPEAT` `T2_UNVERIFIED_FOLLOWUP` `T2_REPEAT_CAP` `T2_DISPATCH_ROLE(_ENVSET/_NOTE)` | 각 행 근거는 JSON `evidence` 필드 |
| `K2.VP` 산문면 | `T2_CLAIM_PROV` `T2_CLAIMPROV` `T2_CLAIM_VERIFY` (+ `T2_TOOL_CHANNEL` 부분) | CLAIMED_DONE 은 정의상 산문 [S] |

**보존(커널이 접지 않는다) 65 행 — 접으면 조용히 죽는 능력**

| 보존군 | 레버 | 왜 접으면 안 되나 |
|---|---|---|
| 선언면 | `T2_A2_VARIANT` | `C186` record 날조 **46%** · grounded **0/24** · A2 설명레버 **0/24** · 라이브 PROV 미포착 ⇒ 축자 *"슬롯 삭제만 남음"* [S]. **근거검산 계보가 실측으로 못 잡은 유일 자리**이고 도구 스키마 변이로만 닫힌다 |
| 산문면 상설 | `T2_UNKNOWN_REPEAT_GUARD` `T2_UNINSTRUCTABLE` `T2_TOOL_CHANNEL` | 술어가 `am.content` 를 본다 — `T2_UNINSTRUCTABLE` 은 축자로 `not getattr(am,"tool_calls",None)` 조건이다 [S] |
| 우리 도구 정직성 | `T2_SG_TRUTH` `T2_SG_WINDOW_ABSTAIN` `T2_NOREC_BRANCH` `T2_RETURN_EMPTY` | [[25]] 우리 도구는 100% 정답 의무. 074 에서 우리 계산 도구가 `NOT_VERIFIED` 날조 배열 위에 권위 판정을 발급했다 [S]. `T2_NOREC_BRANCH` 축자: 라이브 v1 은 *"…then call this tool again"* 으로 닫혀 **종료 분기가 없고** t7313 `task_040` 이 그 모양으로 turn 104 를 태웠다 [S] |
| 계기 | `T2_MATCH_COUNT` `T2_SEARCH_EXHAUST_NUDGE` `T2_KB_NOHIT_SURFACE` `T2_TRACE` `T2_PROV_OURS` | `T2_MATCH_COUNT` 축자 *"KB_search 회수 경계 표면화(\"N개 걸림 중 K개 표시\" ↔ \"전부 표시\")"* — **M5 recall 결손의 유일한 계기**다. 접으면 recall 이 계기 없이 0.58 에 곱해진다 |
| deny 재제시 | `T2_DUP_REPRESENT` | 차단만 남기면 [[64]] 위반 |
| 터미널 유예 | `T2_TERM_GRANT(_USERDEMAND)` | 커널의 fail-open 은 '레버 없음'이지 '1턴 유예'가 아니다 |
| 기구 | `T2_SURFACE_BUS` `T2_ARBITRATE` `T2_WINDOW` `T2_SUPPRESS_AUTH` `T2_FB_SIDECAR(_TEXT)` `T2_DD_FB` `T2_KEEP_DENY_BODY` | 레버가 아니라 배관 |
| 하네스 22 | `T2_AGENT_MAX_TOKENS` `T2_PAIRFIX` `T2_PAIRCHECK` `T2_TRUNC_GUARD` `T2_OVERFLOW_GUARD` `T2_DYN_MT` `T2_MAXPROMPT` `T2_VIEW_*` 외 | 축자 *"T2_AGENT_MAX_TOKENS=8192 — 한 번 빼봤다가 첫 런에서 폭주가 재현돼 되돌린 값이다(C271)"* [S]. **커널은 프롬프트를 키우므로 절단 위험이 오른다** — x394 가 출력 잘림으로 수치 인용 불가가 된 전례 [S] |
| 노브 21 | `T2_*_CAP` 계열 | deny 캡·재발화 캡은 fail-open 안전장치로 남는다(`C526` 거짓 deny 49건/11 sim) [S] |

---

## 5. 엔진에 남는 것 (최소 목록)

| # | 항목 | 닫힌 술어인 근거 | 도메인 리터럴 판정 |
|---|---|---|---|
| E1 | 트리거 판정(D1 4항 ∧ D2 ∧ D3) | 집합 멤버십 + 카운터 3개. 텍스트를 읽지 않는다 | ✔ 리터럴 0 — 단 W 는 A2 파생이어야 하고 `_SUFFIX_RE` 는 쓰지 않는다 |
| E2 | 재료 조립(레지스트리 읽기·원장 문자열 수집·43줄 인쇄·지목 문서 축자 read·`drop_expired`·미실행 빼기·부정통제 치환) | 전부 복사·집합 연산·날짜 비교. **도메인 텍스트 파싱 0**([[59]]) | ⚠`t2_search.to_iso(fmts=("%m/%d/%Y","%Y-%m-%d"))` 는 로케일 리터럴 ⇒ A2 `date_formats` 로 민다 |
| E3 | 전송·계약 파싱 | `t2_subcall.sub_generate`/`parse_contract` 정본 그대로(사본 금지·[[67]]) | ✔ |
| E4 | V1~V12·VP·VC 검산 + 정규화 규약 | 집합 멤버십 / substring / 산술 항등식 / 카운트 / 수치 구간. 출력은 불리언과 기각 목록뿐 | ❌ 3건이 현재 위반 — **V3**(엔진 `DEFAULT_ARG_HINTS = ("email","name","zip","user_id",…)` · `sp.get("record_key_field","account_id")` · `g6.get("user_id_arg","user_id")`), **V8**(`_READ_PREFIX_RE`·`_PROCEDURAL_RE` 의 `^kb_`·`^shell$`·`transfer_to_human`), **V9**(소비처 4곳이 `== "give_discoverable_user_tool"` 하드코딩). **셋 다 A2 로 미는 것이 처방**이고 엔진에 남기면 [[05]] 위반이다 |
| E5 | deny 렌더러 | 템플릿은 A2 문자열, 채우는 값은 검산 결과 | ✔ 엔진 리터럴 0 |
| E6 | 예산·중복 3중키·재발화 1회 상한·fail-open | 카운트 | ✔ |
| E7 | 계측(`t2_liveness` 배선 생존·전달 자수·기각 사유 분포·|S| 기수·무발화 사유) | 카운트 | ✔ |
| — | ⛔ **엔진에 없는 것** | argmax·최댓값·"정답은 X"·후보 순위화·산술 해답 생성·도메인 텍스트 값 추출·의도 분류·정책 해석·수량 파싱 | 하나라도 들어오면 측정 대상이 사라진다(⛔0) |

**기수 규칙 재확인**: `t2_compute.group_reduce(max1)` · `closed_filters(threshold)` 를 후보 위에 직접 적용 · 조건부 enum(`personal → {Silver,Gold}`)은 전부 |S|=1 을 만든다 ⇒ **금지**. 대신 V11/V12 는 **모델이 선언한 술어를 모델이 선언한 피연산자에 대입**해 자기모순만 본다.

---

## 6. A2/A3 선언 스키마

### 6-1. 신설 5키(그 밖은 기존 키 재사용)

| 키 | 내용 | 출처(축자 필수) | 왜 신설인가 |
|---|---|---|---|
| `id_args` | {도구 → 식별자 인자명 목록} | **도구 스키마의 인자명·설명뿐**(`tool_signatures` 에서 기계 도출·opex 0) | 기존 `identifying_arg_types` 는 `["time_verified"]` **하나뿐**이고 [S], 1안이 근거로 든 `name_rules` 는 실물이 **제품명 동일성 산문**(축자 *"the identity is the leading name phrase"*)이지 id 규칙이 아니다 [S] |
| `id_shape` | 도메인 id 형태 정규식(`^[a-z]{2,5}_[A-Za-z0-9]{4,}$` + bare-hex) | env 레코드 키에서 기계 도출 | 형태 정규식은 닫힌 술어가 아니라 **도메인 판별 정규식**이므로 엔진이 아니라 A2 에 둔다 |
| `must_be_ledger` / `must_be_policy` | user_claim 으로 채우면 안 되는 param·성분 kind | 정책 산문의 '확인하라/조회하라' 절차 축자. 실물: doc_044 절차 4) *"Review documentation for the savings account type to determine all applicable APY components"* | V10 의 유일 출처 |
| `closed_filters` | `[{name, kind ∈ {date_window, membership, threshold}, fields, source_quote}]` | 정책 축자. 실물: *"- Balances at or above $25,000: 4.0% APY"* | M6(재료 필터) 정의 + **V11(구간 소속)의 임계표**. ⚠ 후보를 직접 고르는 데 쓰면 금지(§5 기수 규칙) |
| `identities` | {id → {식, 축자 출처, tolerance}} | 정책 return_template 축자. 실물: `get_interest_correction` *"Correction amount = principal x (expected-actual)/100 / 12"* [S] | V6 의 유일 출처. **못 대면 넣지 않는다 — 그 자리는 레버 없음으로 남긴다**([[23]]) |

**기존 재사용(신설 0)**: `action_tools`·`write_evidence_specs`·`write_arg_enum`(트리거 W·V4) · `dispatcher_role_check`(V9) · `arg_producers`(deny 의 producer 이름·V-CHANNEL) · `policy_ontology.{action_index(43), doc_index, doc_windows(17)}`(M5·M6) · `eligibility_line_template`(M6) · `write_initiation`(커널 프롬프트) · `derived`(V6 후보) · `nonlisted_tool_feedback`·`discoverable_name_check`(V1).

### 6-2. 출처 강제 — 문구가 아니라 **로더로 집행**한다

`must_be_ledger`·`closed_filters`·`identities`·`write_arg_enum` 의 **모든 항목에 `source_quote` 필수**이고, `load_domain_a2()` 가 각 `source_quote` 를 정책 문서 코퍼스에 대해 `_norm_ground`+`_tok_in` 으로 검사한다. **불통과 항목은 경고가 아니라 드롭**(A2 에 없는 것으로 취급). 이렇게 하면 gold 를 보고 쓴 값은 정책에 축자가 없어 자동 탈락한다 — [[23]] 이 *"실패 사례 보고 고칠 때가 가장 위험(거의 항상 gold 경유)"* 라고 경고한 자리의 기계적 이행이다.

**[[24]] 양방향 동기화 의무**: 정본 층(`a2/split/*.core.json`)과 `gate.json` 의 `gates[]` 는 **주석까지 바이트 동일**. 편집 후 `load_domain_a2()` 병합 확인 + `test_a2_three_layer.py` 통과가 배선 전제다(정본만 고치면 직접-read 사이트가 옛 값을 쓴다·2026-08-03 실증).

**⛔ 넣으면 무효**: 태스크별 정답 도구·정답 열거값(003 `Silver`·024 `Business Bronze`·055 `Purple`)·gold 금액·실패 사례를 보고 만든 힌트. 그리고 `eplan.intent_chains.signals` 의 케이스 열거를 **확장하지 않는다**([[66]] 죄목 ①케이스 열거 ②공유 상류 노드).

### 6-3. 기존 `t2_procedure` 와의 관계

`t2_procedure.py` 는 A2 `procedures`(6건 DAG)를 걷는 walker 이고, 그 도입부 축자가 이 설계와 같은 분담을 이미 선언하고 있다:

> "It contains no tool name, no field name and no number: those come from the declaration. … It never rewrites an argument and never blocks a call — the engine states what the policy requires and the model acts"

그리고 A2 `_note_procedures` 축자: *"⚠**현재는 데이터만이다** — 이 DAG를 걷는 엔진이 아직 없으므로 라이브 발화는 0이다"* [S].

**관계 확정**: `procedures` DAG 는 **M5 의 대체물이 아니라 상류**다. ⒜ DAG 는 "이 도구 전에 이 노드"라는 **선언**이고, M5 는 "그 노드가 어느 문서에 있나"의 **회수 경로**다. ⒝ 커널 안에서 DAG 는 **차단 술어가 아니라 재료**로만 쓴다 — `unmet_nodes` 의 결과를 `K1.M5` 블록 머리에 축자로 얹고, 실행 여부는 계약이 정한다. ⒞ `enforce:true` 는 정책 축자(*"These steps MUST be followed in the exact order listed."*)가 있는 항목에만 있으므로 V-계열이 아니라 **deny 문면의 ⒝ 슬롯 출처**로만 쓴다.

---

## 7. 못 닿는 잔여 — 각각의 별도 표적

| # | 잔여 | 실측 | 왜 커널 밖인가 | 별도 표적 |
|---|---|---|---|---|
| **R1** | **코퍼스에 실재하는 틀린 값** | `C46` `FIND-wrong` **3/30 안 닫힘** [M] · `C472(f)` 라이브에서 `account_id="Blue Account"` 를 검산이 통과 [M] | V2 는 **실재 검산**이라 실재하는 오답을 통과시킨다. 잡으려 드는 순간이 [[62]] 위반 | MISCALLED 중 날조 몫의 비율 자체가 **미측정** [?] ⇒ §8-M5 |
| **R2** | **적격 오선택(F3-ENUM)** | 11/11 전부 정책에 실재하는 정당한 열거값 [S] | 멤버십도 인용 실재도 전부 통과시킨다 | `K1.M6` 빼기 + 자격줄 상류(`C511` `E_MINUS` 6/8 ↔ `C_PAIR` 2/8 [M]). **커널②가 아니라 커널①의 재료 문제** |
| **R3** | **격리에서도 죽은 표적 2건 — 둘 다 모델 결손이 아니다** | ⒜ `055 deposit_check_3847` : `A_min` 이 낸 `open_bank_account_4821` 은 **그 sim 의 또 다른 미매치 gold** 이고 절차상 **선행 단계**다 ⇒ **채점 결함** [S]. ⒝ `040 get_card_last_4_digits` : gold 가 `give_discoverable_user_tool` + `call_discoverable_user_tool(requestor=user)` 이라 **에이전트는 이 도구를 호출할 수 없다**. 프롬프트 질문은 축자 *"다음에 호출할 도구 하나를 정하라"* ⇒ **정답이 표현 불가능한 질문형** [S] | 커널 사정거리가 아니라 **프로브 형식·채점 결함** | ⒜ `hit_exact` 를 순서-허용으로 고치고 ⒝ 질문을 "다음 **계약** 하나(unlock/give/call · executor 포함)"로 바꾼다. **0.58 자체가 채널을 못 표현하는 형식에서 나온 수다** |
| **R4** | **교차-채널 producer 그래프** | user gold 24 중 미매치 **18(75%)** [S] · `file_credit_card_transaction_dispute_4829` 의 `card_last_4_digits` 는 **손님만 생산 가능** | 원장은 "실행된 (도구,인자,결과)"뿐이라 "이 인자의 생산자는 손님-채널 도구 T 이고 지급→대기→회수 순서가 있다"를 표현하지 못한다 | `arguments:null` + `expects:<seq>` 지연 바인딩 + **VC** |
| **R5** | **레지스트리 근접 이름** | 040 오답 `get_debit_dispute_status_7483` ↔ 정답 `get_user_dispute_history_7291` [S] | V1 멤버십은 **둘 다 통과**시킨다 | R1 의 도구-이름판. 레버 없음으로 남긴다 |
| **R6** | **2단 회수(부스트 퍼센트)** | `Linked Checking Account APY Boosts` 는 18쌍의 적격 여부만 주고 축자 *"For the exact APY boost percentages for each pairing, please refer to the specific savings account documentation."* 로 넘긴다 [S] | 틀린 문서의 **실재하는 숫자**를 인용하면 V2·V5·V10·V11·V12 모두 통과 | (pairing → 문서 id) 매핑을 A2 가 적으면 **도메인 지식 이식**이 된다 ⇒ [[13]] 최후. 정직한 형태 = A3 `doc_index` 에 *"이 쌍 표는 값을 갖지 않는다"* 라는 **회수 힌트 1줄**(문서 축자 그대로)만 |
| **R7** | **초과 write 중 서로 다른 인자 153건** | 실패 33 sim: 동일-키 52(12 sim) vs **비-gold 신규 키 153(28 sim)**. 통과 7 sim 은 7건(1.0/sim ↔ 실패 4.6/sim) [S]. 실물 057 t0 `open_bank_account` 신규 7 + 반복 15(gold 2) · 079 t0 close 4·order 5 · 074 apply 8+8 [S] | V8 은 **동일-키**만 닫는다 ⇒ 205 중 52(25%). **gold 초과는 gold 를 봐야 알 수 있으므로 레버가 될 수 없다**([[23]]) | **레버 없음으로 남기되 상한 계산에서 뺀다.** 그리고 `db_check` 인과를 오프라인 리플레이로 잰다(§8-M3) — 세 설계의 커버리지 산술은 전부 "이게 안 닫혀도 pass 가 난다"는 가정 위에 서 있었다 |
| **R8** | **우리 층 자해 11건** | x392 deny **173**건 / gold 표적 56 / 회복 45 / **미회복 11**. 분포: `T2_RESOLVE` deny **92(53%)**·gold 24·미회복 4(093 t0 turn36 `operator-scope` · 072 t1 turn26 `operator-fab` · 079 · 040) · `T2_WRITE_EVIDENCE` deny 25·gold 22·미회복 5 · `T2_UNLOCK_PROV` 1 · `T2_DISPATCH_ROLE` 1 [S] | 커널이 아니라 **수리**([[55]] 0단계) | ★차단된 이름이 **전부 `_NNNN` 접미사**다 ⇒ V1 정규화 규약이 이 11건의 인과와 같은 자리이고, 규약을 안 정하고 커널로 옮기면 11건이 그대로 이식된다 [S]. 또한 `T2_RESOLVE` 는 비-gold 를 65건 미회복 차단하고 있어 **완화하면 R7 이 늘 수 있다**(부호 미상) [?] |
| **R9** | **`079 t1` — 분모 자체에 없다** | `term=context_window_exceeded`·gold 0건·calls 64·msgs 150. 초과 write 신규 20·우리 층 deny 17 [S] | 미매치 gold 0 이라 147 에도 111 에도 없다 | `C529` 의 CWE 13↔0 은 가설이 아니라 **이 런에 이미 실현된 사인**이다. CWE 를 종점이 아니라 **중단 규칙**으로 |
| **R10** | **행 수준 열거 부하** | `task_074` 축자 *"four checking accounts … with 22 total fee errors (5 each for Purple, Dark Green, and Evergreen; 7 for Light Blue)"* 인데 gold action 은 13행 [S] · `task_073` *"9 total fee errors (3 per account)"* | V7 은 **호출 수 대 대상 수**만 본다 | 행-단위 조인의 주인이 없다. [[49]] 선행(2404.09593 축자 *"더 뽑으라 해도 같은 것을 반복"*) |
| **R11** | **국면 판단(give 가족)** | `C525` `P_HINT` 7/8(답 주면 도구·인자 정확) ↔ `A_LIVE` 3/8 ↔ `B_NOLEAK` 1/8 · `D_NEG` 0/8 [M] | 격리에서 실패하는 것이 **판단**이라 엔진이 대신하면 [[66]]·⛔0④ 위반 | 허용 형태는 **표면화·ASK 뿐** |
| **R12** | **077~097 블록** | `C523` 파일럿 8/8 전부 0.0 · 채널 99회 살아 돌았는데도 전부 0점 [M] | 원인 미규명 | 두 커널의 표적인지조차 모른다 [?] |

---

## 8. 측정 계획

### 8-0. 사전등록(그대로 복사 가능)

- **검정 단위**: 1차 = **(task,trial) sim**, 2차 = **표적 클러스터**(같은 도구 반복은 1로 접는다). **행 단위 보고 금지** — 행으로 세면 커 보이고 클러스터로 세면 사라진다(`A_min` vs `B_full` 표적 단위 4:2, **p=0.688**) [S].
- **팔 4개**(같은 시드·같은 sim 집합): `K` · `K_neg`(무내용 치환) · `K_shuf`(원장 id↔값 섞기) · `CTL`. 필요 시 `K_oracle`(상한 참조).
- **0단계 게이트**(실패하면 나머지 종점 해석 금지): **L1** 커널 발화 시 재료 블록이 비지 않았다 = 100%(x394 가 72/72 빈 `{}` 로 죽은 자리) · **L2** `K_shuf` 기각률이 `K` 대비 유의 상승 · **L3** `K_neg` 기각률 ≈ `K` 기각률.
- **1차 종점(순서 고정)**: P1 결정 turn 최소 재료 도달률 ≥ 0.90 / P2 계약 산출률 ≥ 0.95 ∧ 파싱 실패 0건 / P3 **왕복 100건 거짓 기각 = 0**(1건이라도 나오면 배선 중단) / P4 기각 사유가 사전 명시 V-id 밖 ≤ 5%. **네 종점 모두 통과해야 통과**(교집합·보정 불필요).
- **2차(효과)**: `K` vs `K_neg` 의 sim 단위 pass 불일치 McNemar 단일 검정 하나. `CTL` 비교는 참고 — `C542` 가 양팔 대칭(`READ_MISS` **11:11**·`WRITE_MISS` 2:2·`ARG_MISS` 5:4, fail 35 중 22=63% 가 조회 누락)을 실측했다 [S].
- **세금 종점(켤레와 분리 계상·C293)**: 지연 배수 · steps · **CWE 건수** · **초과 write 신규 키 수(Δspurious)** · over-block(비-gold deny 중 미회복).
- **중단 규칙(사전)**: ⑴ P3 위반 ⑵ CWE 가 `CTL` 대비 +2건 이상 ⑶ 지연 배수 ≥ 1.5 ⑷ 신규-키 초과 write 증가. **넷 중 하나면 그 자리에서 중단하고 이득 항목은 보지 않는다.**
- **측정 보존 조항([[62]]③)**: 커널 직전 메인의 자기 발화·보류 호출을 **커널에 넣지 않고 따로 로깅**해 '이름 ↔ 실행' 분리를 유지한다. 안 하면 knowing-doing 축(이름 18/24 ↔ 방출 2/24 ↔ `D_EARLY` 0/24)이 **측정 불가**가 된다 [M].

### 8-1. 무료로 먼저 잴 것 (유료 런 전 전량 완료·[[09]])

| # | 무엇 | 왜 | 종점(사전 고정) | 비용 |
|---|---|---|---|---|
| **M0** | 배관 수리 후 같은 시드 재실행 | `C542` 짝 t7313 이 이미 있다 | `READ_MISS` **22 → 감소** · gold-block 미회복 11 → 0~2. **미달이면 커널 착수 보류** | 재실행 |
| **M1** | **빼기 ↔ 재구성 동일-컷 짝 대비** | `C331` 빼기 5/24 ↔ `x395` 재구성 0.58 인데 **컷·태스크·질문이 전부 다르다** ⇒ 이 설계의 핵심 전제가 **[D]** | `A_rebuild` vs `B_edit` 표적 클러스터 McNemar, **바 = 불일치 ≥ 5:0(p ≤ .05)**. 미달이면 "재구성이 빼기보다 낫다"는 주장 **철회** | 격리·무료 |
| **M2** | **M5 회수 recall (0순위)** | `x395` 절차줄은 gold 도구명으로 골랐다(축소비 1/1,500~1/3,700) [S]. 라이브 recall 이 **0.58 에 곱해지는데 0회 측정** | ⒜ `recall@8` ≥ 0.70 ⒝ recall 성공 부분집합에서의 exact. 미달이면 **커널① 상한을 recall 로 곱해 다시 쓴다** | 격리·무료 |
| **M3** | **초과 write 의 `db_check` 인과** | 신규-키 153건/28 sim 이 미매치 gold 를 하나도 안 만든다 [S] | 오프라인 리플레이로 초과 write 만 제거 후 DB 재적용 → **뒤집히는 sim 수 / 28**. ≥5 면 over-action 을 계측 축이 아니라 **켤레 항목으로 승격** | 오프라인·LLM 0 |
| **M5** | **성분 검산 3술어(V10·V11·V12) 상한** | 093/094 셋 다 정책 축자 출처가 있고 닫힌 술어인데 세 설계 검산기로는 **0/3** [S] | 계기 수리한 x394 위에서 술어별 on/off. **바 = over-block 0**, TP ≥ 1/술어. over-block ≥ 1 이면 그 술어 폐기 | 격리·소액 |

**x394 재실행 최소 수정(계기)**: ⑴ 레코드 소스를 `tasks/task_09{3,4}.json` 의 `initial_state.initialization_data` 로 교체 + **0단계 어서션 `assert accts and tx`** ⑵ 손님 발화 숫자 처리 분리 팔(`A_rec` vs `A_rec+claim`) ⑶ `max_tokens` 2000→6000 · `finish_reason`·`usage` 저장 ⑷ `raw` 전문 저장 ⑸ `parsed=False` 행은 분모 제외 ⑹ `arith_ok` 3식 동시 계상 ⑺ `quote_ok` → `quote_present`(전문 일치).

### 8-2. 유료 런 최소

M0~M5 통과 후 **1회**, 팔 = `K`/`K_neg` 2팔·같은 시드·40 sim. `CTL` 은 t7326 을 재사용한다(재측정 금지). **M1·M2·M3 이 끝나기 전에 라이브 커널 런을 돌리면 네 번째 null 을 산다** — `C488`(재료 5→31·성적 1/16↔1/16·072 소요 2262→4003초 1.8×) [M] · `C492`(배선 231↔0·8/20↔9/20·over-action 2→8) [M] · `C529`(도달 75%·2/12↔2/12·1.90×·CWE 13↔0) [S]. **세 번 모두 배선은 통과했고 성적은 0 이었으며 비용만 샀다.**

---

## 9. 폐기 조건 (사전 고정 신호 3개)

**D1 — 선택자 검정 실패.** `M2` 의 라이브 회수(`action_index` 2단)가 `K_neg` 무내용 팔을 사전등록 문턱(표적 클러스터 불일치 ≥ 5:0 · 또는 recall@8 ≥ 0.70) 이상 못 이긴다 ⇒ **커널①(전달) 폐기, 커널②만 남긴다.** 근거: `A_min` 0.58 은 12표적 39줄 중 **35줄(90%)이 gold 도구 이름을 축자 인쇄**한 팩 위의 수치였고, 그 팩은 라이브에 존재하지 않는다 [S].

**D2 — 검산기가 답을 낸다.** 다음 중 **하나라도** 발생하면 그 검사를 즉시 제거: ⒜ **L1 gold-스왑 불변성** 위반 1건 ⒝ `deny-only` 팔 > `C_neg + 0.10` ⒞ 엔진 필터의 **잔여 후보수 |S| = 1** ⒟ 후보 위의 순위·최대·문턱 연산 재도입(`group_reduce(max1)`·`closed_filters(threshold)` 직접 적용·조건부 enum). 실물 선례가 둘이다 — 엔진이 6.85 를 만들고 A2 주석이 *"gold 095 재현 6.85"* 로 자기검증한 것, 그리고 그 강제(`T2_WRITE_EVIDENCE`)가 094 t1 turn71 의 gold write 를 차단·미회복시킨 것 [S].

**D3 — 순손익 음수 또는 바닥 지배.** ⒜ 거짓 기각으로 죽은 gold write + over-action(신규-키 write) 증가가 산 행을 넘으면(`C50` 문턱 재현: over-block > TP) 폐기 ⒝ **M0 이 `READ_MISS` 감소를 못 보이면 커널 판정 자체를 보류**한다 — 양팔이 같은 바닥에서 먼저 죽으면 커널은 측정되지 않는다 [S] ⒞ 중단 규칙 4항(P3 위반·CWE +2·지연 1.5×·신규키 증가) 중 하나 발생.

---

## 10. [?] 목록 (미측정·이 설계가 그 위에 서 있다)

1. **`K1.M5` 2단 회수의 recall** — `action_index` 43줄 → 문서 지목 → 절차줄 체인이 `x395` 오라클(1/1,500~1/3,700 축소)을 대신하는가. **0회 측정.** 이 값이 0.545 상한에 곱해진다. → M2
2. **빼기(문맥 편집) ↔ 재구성(재료 조립)의 동일-컷 낙차** — 이 설계의 핵심 전제이며 등급 **[D]**. → M1
3. **MISCALLED 중 날조 몫의 비율** — 커널②의 **실효 표적 크기**가 여기 달렸는데 미측정. `C46` 의 상한(FIND-wrong 3/30)이 그대로 남는다.
4. **초과 write 신규 키 153건의 `db_check` 인과** — pass 를 얼마나 정하는지 미측정. → M3
5. **`T2_RESOLVE` 완화의 부호** — 비-gold 65건 미회복 차단이 R7 을 억제하고 있을 수 있다. 완화하면 초과 write 가 늘 수 있다.
6. **계약 스키마 무게의 in_plan 비용** — `x395` 의 `in_plan 0.86` 은 `{"plan":[{"tool":"<이름>"}]}` = **이름만**의 수치다. `arguments`·`evidence`·`applies_when` 을 얹은 스키마의 in_plan 은 미측정이고 `C528` 이 스키마 무게의 역효과를 실측했다.
7. **후속 user 발화를 넣었을 때의 값** — `x395` 는 첫 발화 400자만 실었다. 넣으면 값이 달라진다.
8. **077~097 블록 전부 0.0 의 원인** — 두 커널의 표적인지조차 모른다.
9. **`x395` `C_neg` 0.19 · `A_min plan` 0.58/0.86 · `B_full plan` 0.61/0.69** — 원자료가 용량-반응 런에 덮여 **콘솔 로그가 유일 출처**이고 행 단위 재감사 불가 ⇒ [M] 로 강등. 라이브에서 부정통제를 다시 세운다.
10. **`017 t1`/`050 t1` 이 미매치를 갖고도 통과한 정확한 경로**(손님 도구 실행 여부).

---

### 부록 — 이 문서가 근거로 쓴 원자료 경로

- 로컬 재분석 산출물 `C:\Users\승원\AppData\Local\Temp\claude\C--workspace\11f635b1-74d5-4292-a765-0ef71eb7f3d3\scratchpad\rv\`(`dump.json` 40 sim 전량 궤적 · `repro.json` · `recls4.json` · `final.json` 보정 111행 · `aud{1,3,4,5}.py`)
- 브리핑 원자료 `…\scratchpad\x392_block_join.json` · `x393_policy_reach.json` · `x384_fail_anatomy.json`
- 원격 `/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2/x39{4,5,6}_*.py` · `…/reports/facet_rft_2026/x39{2,4,5,6}_*.json` · `/home/woori/scratch/logs/x395_{iso,dose}.log` · `/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/{db.json,tasks.json,documents/}`
- 코드(읽기 전용) `C:/workspace/ba-frft/scripts/distill/tau2/{t2_gate_patch.py, t2_levers.py, t2_search.py, t2_procedure.py, t2_subcall.py, t2_compute.py, t2_scaffold_get.py, go_stack.sh, run_t7326_stage1_nt2_20260819.sh}` · A2 `a2/banking_knowledge.gate.json` · `a2/split/banking_knowledge.core.json`
