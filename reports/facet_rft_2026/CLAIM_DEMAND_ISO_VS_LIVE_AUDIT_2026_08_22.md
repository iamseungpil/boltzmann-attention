# 완료-사칭/행동-촉구 축 — 격리 15/15 ↔ 라이브 null 차분 감사 (2026-08-22)

- 질문(사용자 축자): *"격리에선 '완료를 말하려면 그 도구 호출을 대라'가 15/15 로 듣는데 라이브 A/B 는 null — 이게 진짜로 결정론기로 안 닫히는 게 맞나? 격리에서는 되는 거 아닌가? 전달 문제 아닌가?"*
- 방법: C578 식 — 격리 조건(`x459_dup_and_claim_iso.py` ⒝)과 라이브 조립(`t2_gate_patch.py` `T2_ACT_DEMAND`·`T2_CLAIM_PROV`·`T2_UNLOCK_PROV`·`T2_DECISION_CARRY`)의 차이를 **전수 나열하고 하나씩 판정**. 전부 로컬·무료·LLM 0·코드 무수정. 라이브 수치는 `sim_results/bank_t7296_*`·`bank_t7297_*` 의 `.log`·`_results.json`·`fb_*.jsonl`·`trace_*.jsonl` 을 풀어 센 것이고, 변이는 정본 `t2_forensic.mutation_diff` 만 썼다(손 비교기 0·C583ⓐ).
- ⛔결론을 먼저 박는다: **미확정**. *"경계"* 라고 부를 근거가 없다(라이브는 격리가 잰 문면을 **한 번도 결정점에 전달하지 않았다**) — 그리고 *"부하"* 라고 부를 근거도 없다(격리 15/15 는 **도구 없는 naming 계기**·실효 n=5 이고, C489 가 같은 컷에서 naming 18 ↔ emission 2 를 이미 보였다). 둘 다 아니므로 수리 후 **x470 격리-동형 재생 → 라이브 A/B** 순서로만 판정할 수 있다(§4).

---

## 0. 먼저 — 어느 라이브 A/B 가 이 축의 null 인가 (인용 정정)

| 런 | treat 플래그(런 스크립트 축자) | `[T2_ACT_DEMAND]` 로그 줄 수 ctl/treat | 성적 |
|---|---|---|---|
| t7296 (`run_t7296_20260815p.sh`) | `T2_NOW_SELFCALL=$SELFCALL T2_SEARCH_ON_PROCEED=$SELFCALL` — **전달 복구 두 항**, 촉구 아님 | **0 / 0** | 1/16 ↔ 1/16 (C488) |
| t7297 (`run_t7297_20260815q.sh`) | `T2_ACT_DEMAND=$DEMAND` 하나 (전달 복구 2종은 양 팔 ON) | 0 / **231** | 8/20 ↔ 9/20 (C492) |

C584ⓒ·[[46]] 은 이 축(knowing-doing)의 라이브 null 로 **t7296 1/16↔1/16** 을 인용했다. t7296 은 촉구를 켜지 않았다(로그 0줄). 이 축의 라이브 A/B 는 **t7297 8/20↔9/20 하나**다. 아래 차분은 t7297 treat 팔(`bank_t7297_treat_20260815q`·20 sim)을 대상으로 한다. (t7296 은 CLAIMPROV 축 P1 오발화 계수에만 쓴다.)

---

## 1. 격리 x459 ⒝ 가 실제로 잰 조건 (코드에서 확정)

`x459_dup_and_claim_iso.py` ⒝ 완료-사칭 팔은 ⒜ DUP 팔과 **인터페이스가 다르다**. C584ⓐ 가 ⒜ 를 *"실제 메시지 객체 복원 + env 실물 도구 17종 + `la.generate`"* 로 교정했지만, ⒝ 는 교정 **전** 인터페이스 그대로다:

```python
body = render(c["pre"], append=ap_) + NLC + NLC + "What is the ONE next step?"
got = X.ask(a.port, SYS_NEXT, body, maxtok=200) or {}          # x431_spec_selects.ask
if str(got.get("next") or "") == "tool" and got.get("tool"): tool_n += 1
```
- `X.ask` = `messages=[{"role":"system","content":SYS_NEXT},{"role":"user","content":body}]`, **`temperature: 0.0`**, **도구 바인딩 없음**. `SYS_NEXT` 축자: *"Reply with ONE JSON object only: {"next": "tool" or "final", "tool": "<tool name…>" …}"*. 채점 = JSON 에 `next=="tool"` — **이름을 댔는가**이지 tool_call 을 **방출했는가**가 아니다.
- `render()` 는 user 600자·assistant 400자·tool 400자로 **절단**한다. 5 결정점의 실측(로컬 재생): 렌더 본문 **9,814~12,825자** ↔ 원 메시지 본문 **70,645~109,366자** — 격리는 라이브 문맥의 **1/7~1/10** 을 봤다.
- D_name 문면과 자리(축자): `"SYSTEM: If you are about to say something is done, you must first name the tool call that did it. If no such call was made, make it now."` — user 본문의 **맨 끝**(렌더된 대화 뒤, `"What is the ONE next step?"` 바로 앞)에 `SYSTEM:` 접두 문자열로 붙는다(실제 system 역할 아님).
- 결정점 = `claim_cases`: `mutation_diff(s)["missing"]` 이 남은 sim 의 **마지막 text-only assistant 발화 직전**(`pre = msgs[:last]`) — 즉 sim 종료 직전 한 곳. 5 결정점(072#0·072#1·073#1·074#0·074#1)에서 빠진 도구 `apply_checking_account_credit_5829` 이름은 **5/5 문맥에 축자로 이미 있었고** 2/5 는 unlock 까지 돼 있었다.
- **n**: `--n 3`·temperature 0 ⇒ 같은 문맥 3회는 동일 응답이다(C483ⒸⒸ *"온도 0 은 결정론적"*). `x459_dup_claim_iso.json` 의 claim 행은 전부 `0/3` 아니면 `3/3` 이다. ⇒ **D_name 15/15 = 5 문맥/5 · N_neg 6/15 = 2/5 · A_asis 3/15 = 1/5**. C483 규율(n=8 잡음 ±4)로는 **인용 불가 크기**다.

## 2. 라이브 조립에서 "대라" 에 해당하는 것 둘 (코드에서 확정)

| 채널 | 문면(축자) | 발화 조건 | 전달 경로 |
|---|---|---|---|
| `T2_ACT_DEMAND` (t7297 treat 유일 처치) | `"Carry out the next step of this request now."` (44자 · x330 `C_EMIT_ASK` 와 동일 문면) | `rw_fb is None ∧ not _off_phase ∧ _acts ∧ not (_called & _acts) ∧ _agent_ending(am)` — `_called` 는 **이번 draft 의 tool_calls**(`:7797`), `_agent_ending` = draft 가 도구 0 또는 transfer 만. ⇒ *text-only draft 이면 어느 턴이든* 발화 | `_cp2_assign(self,_dm,"ACT_DEMAND")` → **단일 슬롯** `_t2_cp2_pending` → `:9785` 에서 `work + [UserMessage(content=_cp2)]` 로 이 턴 재생성 버퍼의 **마지막 user 메시지**(비커밋). 재발화 가드 `self._t2_cp2_said` 는 SEARCH_ON_PROCEED·VIEW_FB 와 **공유**(`:6352·7667·8453·8490·8641`) |
| `T2_CLAIM_PROV` (양 팔 ON·`go_stack.sh:143`·cap 3) | `question`: *"…{"claims":[{"kind":…,"what":…,"tool":"<the exact name of the tool call in this conversation that performed it…>"}], "pending":[…]}"* → `feedback`: *"Error: [CLAIM-PROVENANCE] your reply asserts you already did the following, but the conversation ledger shows NO such event: {claims}. … Either actually do it now (call the real tools…) or correct your statement…"* · `feedback_pending`: *"…Do the promised work NOW by calling the real tools…"* | 사임 창(`_resign` = tool_calls 없는 텍스트 draft) ∨ transfer 호출, cap 3/sim + 예비 1 | `_ap_regen(fb,"claimprov")` → `_gen(self, work+[am,_fb], …, tool_choice=None)` — 피드백이 draft 뒤 **마지막 user 메시지**, 도구 바인딩 라이브 그대로 |

⇒ 라이브에서 *"했다고 말하려면 그 호출을 대라"* 에 가장 가까운 것은 **CLAIMPROV** 다(LLM 이 `tool` 지목 → 엔진 집합 대조 → 없으면 "지금 하라"). `T2_ACT_DEMAND` 는 x459 의 **N_neg 부류(무내용 촉구)** 다 — x330 에서 11/24, x459 N_neg 2/5 였던 그 문면이다.

---

## 3. 차분표 (7행 · 각 "같다/다르다 + 근거")

| # | 항목 | 격리 x459 ⒝ D_name | 라이브 t7297 treat | 판정 | 근거(축자·실측) |
|---|---|---|---|---|---|
| 1 | **문면** | *"you must first **name the tool call that did it**. If no such call was made, make it now."* — 지목 요구 + 조건부 실행 | ACT_DEMAND: *"Carry out the next step of this request now."* — 이름 요구 0 · 무내용 촉구 / CLAIMPROV: *"ledger shows NO such event … actually do it now"* — 고발 + 실행 | **다르다(의미)** | D_name 문면은 **라이브에 배선된 적이 없다**(`grep "name the tool call" t2_gate_patch.py` 0). 라이브 처치 문면 = 격리가 **N_neg(2/5)** 로 분류한 부류. CLAIMPROV 는 "대라"를 **사전 질문**으로 묻고 답이 원장 밖이면 **고발문**을 낸다 — 그런데 그 고발의 **88% 가 거짓**이었다(행 4) |
| 2 | **위치·역할** | system = `SYS_NEXT`(JSON 강제) · user 본문 끝에 `SYSTEM:` 접두 한 줄 → 지시가 **재료 뒤 맨 끝**(recency) | ACT_DEMAND: 비커밋 버퍼의 마지막 UserMessage(재료 뒤·맨 끝) — 단 **단일 슬롯**이라 같은 턴의 DECISION_CARRY/SEARCH 재료에 **덮인다** | 위치 **같다**(둘 다 끝) · 도달 **다르다** | 로그 전수(treat 20 sim): 촉구 발화 **109** 중 `부착 (44자)` **19~22** · 다른 재료(247/432/4536자)로 **덮임 64** · 부착 없음 26. `[T2_CP2_CLOBBER]` 줄 **0** — 덮어쓰기 경로가 `_cp2_assign` 을 거치지 않는 `VIEW_FB`(`:8635`)라 계기에도 안 잡혔다. 사이드카 `fb_*.jsonl`·`trace_*.jsonl` 에 문자열 `"Carry out the next step"` **0건** — 도달 계기 자체가 없다([[55]] *로그 마크 ≠ 전달*) |
| 3 | **시점** | sim **종료 직전** — MISSING 남은 채 마지막 text 발화를 내려는 자리(`pre = msgs[:last]`) | 첫 text draft(trace turn 1·3 = 인사/신원 질문 턴)에서 발화 → `_t2_cp2_said` 에 걸려 이후 **"같은 문자열 — 재배달 안 함" 122회** · 검색 재료가 슬롯 값을 바꿀 때만 재무장 | **다르다** | treat 20 sim 중 x459 형 결정점(MISSING+최종 text 턴) 보유 **8 sim**(050×2·072×5·073×1) — 그 **최종 턴에 촉구가 부착된 sim 0/8**. 결정점 근방(마지막 gold write 이후·끝 3 text 턴 안) 부착은 **2건**(072#s1567 gen51 · 072#s373753 gen75)뿐이고 둘 다 결과는 **text**(전환 0/2). 나머지 부착 17~20건은 전부 turn 3~27 의 **초반 탐색 턴**(결과 = `KB_search_bm25`·unlock 등 read) |
| 4 | **트리거 정확도** | 결정점 = 닫힌 술어 `mutation_diff.missing ≠ ∅ ∧ 마지막 text 발화` — gold 를 보는 **진단용 정의**(라이브에서 쓸 수 없음) | ACT_DEMAND: 098(양 팔 5/5 통과·불변 의무 태스크)에서도 sim 당 **3~4회 발화**(16회·전부 덮임/미부착) · 050/072/073 전 sim 에서 **turn 1(인사 직후) 발화**. CLAIMPROV 과거형 고발: P1 수리판 술어(kind-폴백 + 완결 event_map)로 같은 자리를 재판정 → **오발화율 88%** | **다르다(과폭 + 거짓 고발)** | 오발화 재판정(사이드카 `channel=claimprov` "NO such event: k: what" 전수 파싱 → 그 턴 직전 원장(raw+eff 이름)에 현행 `event_map` 적용): t7297 treat **23/26 구제**(still 3: `verify` 2·`record_update` 1) · ctl 8/11 · t7296 ctl 12/13 · treat 11/16. 구제된 것은 전부 `search`(KB_·call 경유 read)·`record_update`(실제 실행된 credit/approve)·`dispute_file` — 축자 예: `('task_050#s373753', 54, 'record_update', 'approved credit limit increase', 'P1_rescued')`, `('task_073#s373753', 49, 'record_update', 'credited $9.50 to Blue Account', 'P1_rescued')`. ⇒ 라이브의 "대라→없다→지금 하라"는 **이미 한 일을 안 했다고 고발**하는 자리에서 주로 울렸다([[25]] 정본 오염) |
| 5 | **도구 노출** | **도구 0종** · JSON 으로 이름만 · temperature 0 · 문맥 1/7~1/10 절단 | alltools(`go_stack.sh:221 --retrieval_config alltools` = env 실물 + `KB_search_bm25/dense` + `shell`) · tool_choice auto · 전체 문맥(11~45k prompt tokens) | **다르다(계기 자체가 다르다)** | C584ⓐ 가 ⒜ DUP 팔에서 *"텍스트 렌더 = 더 약한 인터페이스 · [[62]] §1.4 위반"* 이라 스스로 교정한 그 인터페이스가 ⒝ 에는 남아 있다. 같은 결손에 대해 C489/x330(도구 바인딩·n=24)은 **이름 대기 18/24 ↔ 방출 2/24** — 15/15 는 **"이름 대기" 축의 수치**다. 라이브 실측도 같은 말을 한다: CLAIMPROV 사임-창 regen **62회 중 `tool_calls=[]` 45회**(73%)·과거형 고발 15회 중 도구 방출 7회(call 5·unlock 1·other 1) |
| 6 | **후속 처리** | "다음 한 걸음" JSON 한 번으로 채점 끝 — unlock/게이트/deny 없음 | 촉구·고발 뒤 regen 의 정답 호출을 우리 층이 막는 경로 **실재**: `T2_UNLOCK_PROV`(followup-regen)이 CLAIMPROV regen 의 **정답** `unlock(approve_credit_limit_increase_5847)` 을 *"unprovenanced"* 로 deny | **다르다(U2 동형 실재)** | 로그 전수(open-window 귀속): treat **050#s626729·050#s373753** 각 1회 — CLAIMPROV 창 안에서 `[T2_UNLOCK_PROV] deny unprovenanced name (followup-regen) tool=unlock_discoverable_agent_tool val=approve_credit_limit_increase_5847` (ctl 050#s373753 1회). FOLLOWUP 창 안 deny `get_user_dispute_history_7291` treat 3·ctl 2. `T2_STALE_STRIP` 은 treat 1회(098)·촉구와 무관. T7336 050 t0 §7③ 의 `t2_stack.admit` 지문 억제(최종 턴 `claimprov`·`followup_decision` 모두 suppressed)도 같은 계열 — t7297 treat 는 `cp_suppressed` 0 이었으나 조건은 동일하게 살아 있다 |
| 7 | **판정 단위** | 결정점 5개(실효 n=5) · 팔당 동일 문맥 | sim 20 ↔ 20 · reward 8 ↔ 9 · 태스크별 050 1→0 · 072 0→0 · 073 2→4 · 098 5→5 | **다르다(A/B 가 결정점을 0개 시험)** | 촉구가 x459 형 결정점에 닿은 sim **0/8**, 근방 닿음 2 sim(둘 다 072·양 팔 072 는 0/5 로 바닥). 발화한 sim 만 짝지어도 **부호를 낼 결정점이 없다**. 098 의 발화 16회는 전부 미도달이라 5/5 불변은 **촉구 무해의 증거가 아니다**(도달 0 의 증거다). C492 의 over-action 2→8 은 050 `ONLY-PRED:user_discoverable_tools 4건` 이고 050 에 부착된 촉구 6건은 전부 **turn 24~42 의 초반 give/unlock 턴**이었다 — 과행동이 촉구의 부호라는 C492ⓓ 해석은 **이 자리(초반·무목표 촉구)** 에 한정된다 |

---

## 4. 판정 — 전달 결손 확정 / 무죄 / 미확정

**전달 결손 확정(우리 층·[[55]] 순서상 모델 귀속 이전에 수리해야 할 것)**
- (T1) **문면 불일치**: 격리가 잰 D_name 문면은 라이브에 없다. 라이브 처치 = 격리의 N_neg 부류. [행 1]
- (T2) **단일 슬롯 덮어쓰기 + 도달 계기 부재**: 109 발화 중 19~22 부착·64 덮임·사이드카 0건. [행 2]
- (T3) **once-per-sim 재발화 가드**(`_t2_cp2_said` 공유): 결정점(종료 직전)에서 침묵 122회 · 결정점 도달 0/8. [행 3]
- (T4) **CLAIMPROV 과거형 고발의 P1 오발화 88%**(t7296 당시 `_claim_unbacked` 지목-미스 즉시 unbacked): "대라→없다→지금 하라" 가 거짓 위에서 울렸다 — 이 자리는 5189b510 으로 **이미 수리됨**(t7336 halfB DUP 0/20). [행 4]
- (T5) **후속 deny(U2)**: CLAIMPROV regen 의 정답 unlock 을 `T2_UNLOCK_PROV` 가 거부(treat 2 sim). 미수리(T7336 halfB §7 처방 후보 상태). [행 6]

**무죄(격리↔라이브 차이로 판정되지 않는 항목)**
- 위치(끝·recency)는 둘 다 같다 — C578 형(재료 뒤 15,000자 매몰)은 여기 **해당 없음**. `[T2_CP2_APPEND]`(대용량 뒤 이어붙임) 0건.
- `T2_STALE_STRIP`: 촉구 sim 에서 0회. 무관.
- 도구 노출 **방향**: 라이브는 실물 도구를 줬다 — 결손은 라이브가 아니라 **격리가 도구를 안 준 것**(행 5).

**미확정(이 감사로 못 가르는 것)**
- *"결정점에서 D_name 을 실물 도구로 주면 방출하는가"* — 측정된 적이 없다. x459 는 naming(n=5), x330 은 D_name 이 아닌 촉구(11/24)를 쟀다. **이것이 x470 의 질문**이다.
- *"라이브 결정점에 D_name 이 도달하면 reward 가 움직이는가"* — T1~T5 수리 전에는 A/B 가 무효다.

⇒ **결론: 미확정.** *"격리 성공 = 부하 = 경계 아님"* 규칙은 **정보-맞춘 격리**([[18]]·[[62]] §1.4)에만 적용되는데 x459 ⒝ 는 그 조건을 못 채웠고, 라이브 null 은 처치가 결정점에 **도달하지 않은** 런이라 *"결정론기로 안 닫힌다"* 의 근거도 아니다. 사용자의 *"전달 문제 아닌가"* 는 **T1~T5 만큼은 맞다**; 다만 그 수리가 성적을 사는지는 **측정 전**이다([[57]]·C492 *"하나를 사면 하나를 판다"* 그대로).

---

## 5. 수리 후 재측정 설계

### 5.1 x470 — 격리-동형 라이브 재생 프로브 (무료·로컬 8141·LLM 재생만·코드 수정 0)

목적: 행 5 의 계기 결손을 닫는다 — **실물 메시지 + 실물 도구 + 실제 `tool_calls` 방출**로 D_name 을 잰다. x459 ⒜ 의 `replay()`(C584ⓐ 가 교정한 충실 인터페이스)를 ⒝ 결정점에 그대로 적용한다.

- **결정점 풀**(닫힌 술어·x459 `claim_cases` 동일): `mutation_diff.missing ≠ ∅ ∧ 마지막 text-only assistant 발화 직전`. 로컬 영속분 실측 **110개**(t7296 27 · t7297 18 · t7328 28 · t7335 13 · t7336 24) — 태스크별 072 28 · 071 13 · 073 7 · 093 7 · 050 5 · 055 5 · 074 4 … ⇒ 완료-사칭 계열(050/072/073/074)만으로 **44개 ≥ n=24**(C483). 071/093 은 다른 축(등급·APY)이라 제외하되 전이 확인용 보조 블록으로 둔다.
- **팔(한 변수씩·전부 마지막 user 메시지 한 줄·도구 지목 0)**:
  - `A_asis` — 복원 그대로(라이브 재현 기준선·x459 A_asis 1/5 가 재현돼야 계기 정상)
  - `N_neg` — `"Please continue."` (부정통제·[[57]])
  - `L_live` — `"Carry out the next step of this request now."` (t7297 처치 문면 = 격리서 그 문면이 얼마나 사는지)
  - `D_name` — x459 D_name 문면 축자
  - `D_sys` — 같은 문면을 **system 역할**로(C578 위치 축 1변수)
  - `E_early` — 조사 완료 **전** 컷 + `D_name`(x330 `D_EARLY` 동형 부정통제 · 여기서 방출 ≥ 바닥이면 프로브 무효)
- **채점(닫힌·한 함수)**: `replay()` 가 돌려준 `tool_calls` 를 `t2_forensic.nameof/argsof` + `_eff_tool_name` 규약으로 풀어, **MISSING 집합의 도구 이름(접미 제거)이 방출됐는가**(1차) · 변이 도구(`mutating_tools()`)가 하나라도 방출됐는가(2차) · `[]` 인가(text 잔류). naming 은 채점하지 않는다.
- **n·잡음**: 결정점 24 × 팔 6 × 표집 3(온도 0 1회 + 0.7 2회·C483ⒸⒸ "적중은 표집 꼬리") = 432 호출. 사전 고정 판정: `E_early ≥ 8/24` → 무효 · `D_name − N_neg ≥ 5` → D_name 이 **방출 축**에서 산다(그때만 라이브 이관) · `D_name ≈ L_live ≈ N_neg` → 문면 무효 = 잔여는 [[13]] learn 축 · `D_sys − D_name ≥ 5` → 위치 축 추가.
- ⚠[[62]] ③④: 어느 팔도 도구를 지목·순위·정답 문장을 내지 않는다. 엔진은 집합 대조만.

### 5.2 라이브 조립 수리 — 격리 조건을 라이브가 **그대로** 받게 (x470 이 D_name 을 자격시켰을 때만 문면 교체)

| 결손 | 수리 | 근거·주의 |
|---|---|---|
| T1 문면 | `T2_ACT_DEMAND` 의 `_dm` 을 x470 승자 문면으로 교체(또는 `T2_CLAIM_PROV` 창에서 고발문 대신 D_name 문면을 내는 `feedback_name_call` 키를 A2 에 추가 — **출처 = 우리 프로토콜 문면이라 [[23]] 무관**) | 문면 튜닝이 아니라 **격리가 잰 조건의 이식**(C578ⓔ 동형) |
| T2 슬롯 | 촉구는 `_t2_cp2_pending` 과 **별개 슬롯**(`_t2_demand_pending`)으로 두고 소비 지점(`:9785`)에서 재료 **뒤** 마지막 UserMessage 로 부착 — 재료와 합치지 않는다(C578ⓕ *"지시를 재료 뒤에 두지 마라"* 는 **같은 메시지 안**의 매몰 얘기고, 별도 마지막 메시지는 recency 가 산다 — 그래도 `D_sys` 팔로 위치를 잰다) · 사이드카에 `agent="act_demand"` route 행 + `arrived` 기록(`:8650` VIEW_FB 와 같은 형식) | 행 2: 64 덮임·계기 0 |
| T3 재무장 | `_t2_cp2_said` 공유 가드 → 촉구 전용 지문 `(마지막 user turn_idx, A2 action_tools 중 미호출 집합)` — **인자 변화로 재발화**([[57]]) · sim 당 상한은 두지 않되 같은 지문은 1회 | 행 3: 결정점 침묵 122 |
| T4 트리거 | 발화 자리를 `_resign ∧ not _dispatch_since_last_user(msgs,a2)`(U1 이미 구현) 로 좁힌다 — 첫 인사 턴·신원 질문 턴(`_acts` 미호출이 정상)은 `T2_PHASE` 의 verify 국면 술어(`_off_phase`)가 이미 거르므로 그 값을 촉구에도 적용 · CLAIMPROV 쪽은 5189b510(P1) 그대로 | 행 4: 098 발화 16·turn 1 발화 15/15 sim. ⛔의도 분류 금지([[66]]) — 전부 구조·원장 술어 |
| T5 후속 deny | `T2_UNLOCK_PROV`(`:10416`) 술어에 **env 레지스트리 실재**(`_agent_discoverable(env)`·`:2520`)를 출처로 추가: 실재하면 deny 하지 않거나, deny 하되 문면을 *"이 이름은 실재하나 이 대화에서 회수되지 않았다 — 정의 문서를 열어 확인하라"* 로 사실화(T7336 halfB §7 ②) · `t2_stack.admit` 지문에 미호출-집합을 포함해 **최종 턴 억제**를 막는다(T7336 050 §7③) | 행 6: 정답 unlock deny treat 2 sim |

[[05]] 3질문: ⑴ 도메인 어휘 0(문면은 프로토콜·술어는 원장/레지스트리 집합) ⑵ 유동 판단 동결 0(무엇을 부를지는 모델) ⑶ 엔진은 도메인 행동을 수행하지 않는다(촉구·대조만). [[62]] 자기점검: ①격리 측정 = x470 선행 ②격리서 되면 레버는 **전달뿐** = T1~T5 전부 전달 수리 ③결정론 추가 0 ④떠먹이기 0.

### 5.3 라이브 A/B (유료·사용자 승인 후·x470 통과 시에만)

- 편성: 050·072·073·074(결정점 보유 태스크) + **098**(불변 의무) × nt=5 × 2팔 = 50 sim. ctl = T2~T5 수리 ON + 촉구 OFF · treat = + 촉구(x470 승자 문면). 두 팔 차이는 **환경변수 하나**.
- 사전 고정 판정(이 순서):
  - ⓐ 배선: 사이드카 `act_demand arrived=True` 가 **실패 sim 의 x459 형 결정점**(최종 text 턴·MISSING 잔존)에서 ≥1 — t7297 은 여기서 0/8 이었다. 실패면 성적을 읽지 않는다.
  - ⓑ 결정점 전환율: 촉구 도달 결정점에서 변이 도구 방출 sim 비율(treat) ↔ 같은 자리 ctl. **결정점 단위**로 짝짓는다(sim 단위 null 은 행 7 의 함정).
  - ⓒ reward(`db_match`·[[69]]) — 태스크별 부호표 의무([[70]]).
  - ⓓ 부작용: over-action `ONLY-PRED`·DUP(`mutation_diff.dup`)·098 5/5·`T2_UNLOCK_PROV` deny 수·지연 중앙값. 차이 ≥5 만 인용(C483).

---

## 6. 출처·재현

- 격리: `scripts/distill/tau2/x459_dup_and_claim_iso.py`(`render`·`X.ask`·`SYS_NEXT`·D_name 문면) · `x459_dup_claim_iso.json`(claim 행 15개 전부 0/3 또는 3/3) · 렌더 크기 재생 = 로컬 `F.sims('bank_t7328_half{A,B}…', suffix='.results.json.gz')` + `X.claim_cases` + `X.render`.
- 라이브 코드: `t2_gate_patch.py` `:7797`(`_called`=draft) · `:8432-8458`(ACT_DEMAND) · `:8488-8490`(SEARCH_ON_PROCEED `_t2_cp2_said` 공유) · `:8635-8650`(VIEW_FB 슬롯 직접 대입·route 기록) · `:9785-9800`(소비·마지막 UserMessage) · `:10250`(`_resign`) · `:10280-10350`(`_ap_regen`·`admit`) · `:10416-10440`(UNLOCK_PROV followup-regen) · `:11154-11340`(CLAIMPROV 창·`_claim_unbacked(kind_fallback_on_miss=True)`) · `a2/banking_knowledge.gate.json /claim_prov`(question·feedback·event_map 축자) · `go_stack.sh:143-144,221`.
- 라이브 계수: `sim_results/bank_t7297_{ctl,treat}_20260815q.{log,_results.json,fb,trace}.gz`·`bank_t7296_*` 를 scratchpad 에 풀어 sim 별 집계(촉구 발화/같은 문자열/부착 44자/덮임 · CLAIMPROV window hit unbacked·unb_p·regen tool_calls · UNLOCK_PROV deny 의 open-window 귀속 · 사이드카 `channel=claimprov` 전수 파싱 → 현행 event_map 재판정). trace `turn` = `len(state.messages)` 이므로 생성 메시지 `turn_idx = turn+1` 로 정렬했다.
- 원장: C488·C489·C492·C578·C583·C584 · `FALSE_SUCCESS_PRIOR_WORK_2026_08_15.md` §4 · `t7336_tasks/T7336_TASK_050.md` §7 · `T7336_FORENSIC_HALFB_2026_08_22.md` §5 U2.
- ⚠한계: 촉구 부착 턴의 **draft 본문**은 영속되지 않아(regen 이 대체) "사칭 발화 직후였나"는 구조(초반 탐색 턴·read 결과)로만 판정했다. P1 재판정은 claim 의 `tool` 지목값이 로그에 없어 **kind 패턴**으로만 했다 — 지목이 빗나갔는지·event_map 이 비어 있었는지는 못 가르지만, 수리판 술어 아래 결과는 둘 다 "구제"라 비율은 같다.
