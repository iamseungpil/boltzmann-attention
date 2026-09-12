# 설계서 v2 — 레버가 얻는 축과 잃는 축을 레버 안에서 가른다 (2026-09-12, 리뷰 반영)

**기반 = `r153/arm/nc18` = `origin/arm/nc18` = `19b9575c`.** 리모트 팔 트리를 `git fetch ssh://…/repo_lbv2nc18`
로 가져와 `arm/nc9 … nc18` 아홉 ref 를 origin 에 push 했다(2026-09-12). 이 문서가 인용하는 코드 지점은
전부 그 sha 다 — `lb-v5`(설계 v1 이 커밋된 자리)의 같은 파일은 다른 코드다(`lb_runtime.py:121` hand-off
유예 있음 · `:320` `hidden` 이 skip 조건). M 팔은 전부 `arm/nc18` 위에 세운다.

v1 → v2 에서 바뀐 것: ①기반 sha 명시·브랜치 push ②M2 의 손님 도구 출처를 `env_surface.side` 로 정정,
019 진단 정정 ③M3 를 둘로 나누고 되비춤은 표적이 없어 보류 ④채택 규칙에 잡음 임계 ⑤M1 의
인과/상관 분리·`[LEDGER]` 시제 수정 포함 ⑥M5 는 새 키 `met_text` ⑦M4 의 `requires_reads` 통합·036 회귀
감시로 이동 ⑧손실16 정의·8 태스크 측정을 문서에 ⑨비용·병렬 팔.

## 0. 전제와 원칙

사실(전부 실측, 출처 `CLAUDE.md`·`docs/`):
- 한 레버가 얻는 태스크와 잃는 태스크는 **같은 축의 양 끝**이다.
- 손해로 확정된 사례는 전부 ①문구의 지시 ②잘못된 시점 ③개입 총량 ④낱말로 판정한 의미론 ⑤인과 없는
  무조건 호출 중 하나였다. 사실 진술은 손해로 지목된 적이 없다. **재생성**이 손해였다.

원칙(메모리 `94`): 필요한 시점에 최소로 · 문구는 관측만 · 의미론은 LLM 만 · 도구는 사건 뒤에 ·
gold 로 조건 정하지 않음 · **팔 하나에 변경 하나** · 격리 실험은 배선과 같은 함수.

## 1. 축과 판정 묶음

| 축 | 한쪽 | 다른쪽 |
|---|---|---|
| A 요청의 종류 | 행동형 {049 047 045 043} | 조사형 {016 019 081 054 040} |
| B 대화의 종류 | 계좌 다단계 {066 067 062 069 070} | 조회 없음 {001 002 003 006 007 024 025} |
| C 다음 수의 주인 | 어시스턴트 gold {047 045 043 036 048} | **손님 gold** {049 016 019 081 098 023} |
| D 전달 방식 | 쪽지(사실) | 재생성 |
| E 도구 결과 | 사실 | 프레임·지시 |

정의(리뷰 지적 — 어느 문서에도 없었다):
- **손실16** = `PRERUN_FS_2026_09_11.md` 의 손해 15 {007 016 019 023 028 036 040 043 048 049 054 058 070 073
  081 098} 에 **016 을 포함해 센 것**이 아니라, 그 15 에 `004`… 가 아니다 — 정확히는 손해 15 + 음성대조 043
  = 16 이다. 이하 "손실16" 은 이 16 을 뜻한다.
- **손님 gold 6** = 손실16 중 gold 쓰기에 손님 도구가 있는 것: 016(`submit_transaction`) 023
  (`apply_for_credit_card`) 048 049 081(`request_human_agent_transfer`) 098(`submit_referral`). 판정은
  base 승 sim 에서 그 도구를 `role=user` 만 불렀는가(§2 M2 의 `side` 목록과 일치).
- **축 B "조회 없음" 8** = base 승 sim 이 `get_user_information_*` 를 한 번도 부르지 않는 태스크
  {001 002 003 006 007 024 025} + 004(t8 에 조회하나 `log_verification` 없음). 측정: 53 태스크 팔 대조에서
  이 8 은 `verify_identity` 도움 0 / 해 5 · 평균 −13.2%p, 나머지 46 은 +6.4%p. (이 측정은 세션
  `vcond.sh` 결과이며 repo 에 없었다 — 이 절이 기록이다.)

**채택 규칙(잡음 임계).** 같은 sha 재런 차이는 태스크당 0~1 sim(`CLAUDE.md`). 그러므로
- 태스크당 |Δ| ≤ 1 은 잡음으로 본다.
- 묶음 합으로 판정: **이득 쪽 묶음 합 ≥ +2 이고 손해 쪽 묶음 합 ≥ −1** 이면 채택. 손해 쪽 합 ≤ −2 면
  기각. 경계면 Fisher 한쪽꼬리(`CLAUDE.md` 의 방식)로.
- 모든 팔 보고는 손실16 합계 옆에 축 A·B·C 의 양끝 묶음 합을 적는다.

## 2. 조치

구현은 **병렬 팔**(§3). 각각 `arm/nc18` 위에 변경 하나.

---

### M1. advice 는 재생성하지 않는다 — 사실은 다음 턴에 얹는다

**근거 — 층을 나눈다.**
- 상관(원인 확정 아님): 재생성 횟수별 승률 0회 58% → 1회 53% → 2회 45% → 3회 32% (1,336 sim). 어려운
  sim 이 재생성을 더 부를 수 있으므로 인과가 아니다.
- 인과 쪽: 같은 `[ORDER]` 쪽지가 재생성 없이 닿으면 016 +35(7/11 vs 9/31); hand-off **유예**(쪽지를
  싣기 위한 재생성) 41 sim 29% vs 무유예 127 sim 42%; nc17_004 패 3/3 이 유예 형. advice-only 재생성
  359 sim 49% 는 **`lb-regen` 레코드의 `denies==0`** 으로 센 값이다(`lb-advice` 로 센 것이 아님 — 리뷰가
  지적한 오염 경로가 아니다. `say()` 가 `lb-advice` 를 적고 `turn_hook` 이 break 하는 경우는 `lb-regen`
  에 안 남으므로 이 계수에 들어가지 않는다).
- **nc19 가 인과 판정이다.**

**설계.**
- 재생성은 deny 에만. advice 는 `agent._lb_pending_advice` 에 쌓고 **다음 `generate()` 의 view 앞**에
  `ADVICE_MARK` 로 얹는다. 소비 시 `lb-advice-carried`, 대화 종료로 미소비면 `lb-advice-dropped`.
- 마지막 턴의 쪽지는 드롭된다. 받아들이는 근거: base 의 텍스트-전용 턴 2,286 중 마지막인 것 386(17%),
  t4 이전 647 중 646 이 마지막 아님 → 텍스트 턴 대부분은 다음 턴이 있다. **`leaving()` 은 "떠나는 턴"
  에 울리고 M1 은 "안 떠난 경우"에만 닿게 하므로 게이트의 의도와 전달 조건이 뒤집힌다** — 그래서
  드롭률을 `lb-advice-dropped` 로 nc19 에서 실측하고, 높으면 "마지막 턴에 한해 재생성"을 별도 팔로 잰다.
- **`[LEDGER]` 문구는 M1 에 포함**한다(리뷰: 다음 턴에 읽히면 첫 절이 거짓, 끝 절이 049 를 죽인 명령형).
  nc18 A2 는 아직 `…make the call now…` 원문이다(확인함). →
  `[LEDGER] At your previous turn the run record showed no action for: {open}.` 관측만, 시제 과거.
  이것은 문구 변경이지만 M1 의 전달 방식 변경과 **분리할 수 없다**(전달 방식이 시제를 정한다) — 변경 하나로
  본다.

**코드 지점(`arm/nc18`).** `lb_runtime.py:143` `if not d.denies and not (d.advice and not turn.calls):` →
`if not d.denies:`; advice 는 pending 으로. `generate()` 직전 pending 소비. A2 `LB5.open_request.feedback`.

**판정(축 D).** 재생성이 해였던 {004 008 012 014 040 005 037} vs `[ORDER]` 가 득이던 {016 049 036}.
채택 = 앞 묶음 ≥ +2 · 뒤 묶음 ≥ −1.

---

### M5. 통과는 한 낱말, 실패는 전문 — 새 키 `met_text`

**근거.** 016: nc9(도구 없음) 3/4 · nc17(328자, 조회 뒤) 2/4 · nc12(8자) 1/4 · nc11(0자, 재호출 2~4회)
1/4. 0자는 재호출을 부른다. nc17/18 의 328자에는 `you may now call log_verification. Its time_verified
argument must be…` 지시가 있다.

**설계(nc12 `bc165862` 의 코드를 nc18 로 이식).**
- `lb2_decision._match_verdict`: `ctx["_verdict"] = "met"|"unmet"`; `run_tool` 은 `(text, err, ids, verdict)`.
- `execute`: `verdict == "met"` 이고 선언에 **`met_text`** 가 있으면 그것만 보낸다. `unmet` 은 4종 템플릿 전문.
- **`hidden` 의 주석이 nc17/18 에서 이미 거짓이다**(리뷰): `lb_runtime.py:355-357` 축자 *"`hidden` keeps the
  tool out of the model's list while its check goes on running elsewhere"* 인데 `:360` 은 더 이상 `hidden`
  을 보지 않고, A2 는 `hidden:false` + `check_moved_to:"identity_gate"` 가 남아 있으며,
  `tests/test_lb.py:670-682` 의 두 검사는 이 상태에서 공허하게 통과한다. → M5 커밋 하나에서 함께 정리:
  `:355-357` 주석을 현재 동작(`disable` 만 목록에서 뺀다)으로 고치고, `verify_identity` 의 `hidden`·
  `check_moved_to` 키를 지우고, 그 두 테스트를 `met_text` 검사(통과 시 `met_text` 만, 실패 시 전문)로 바꾼다.
- **`hidden` 은 원래 뜻으로 둔다** — 그 키는 `lb_a2.py:219`·`tests/test_lb.py:670`("a hidden verifier stays out
  of the model's tool list") 대로 **목록에서 뺀다**는 뜻으로 둔다. nc17/18 이 `hidden:false` 로 둔 것도
  그 뜻과 일치한다(목록에 있다). "무엇을 말하나"는 새 키 `met_text` 다. 세 뜻이 한 키에 겹치지 않는다.
- A2: `verify_identity.met_text: "VERIFIED"`, `inject_after` 유지, `hidden:false` 유지.
- **의도된 삭제**: `met` 이면 `_match_verdict` 끝의 `advice.unless_ran`(`lb2_decision.py:165-169`,
  `log_verification` 재촉)도 함께 사라진다. 016 의 병이 그 문장이고, base 는 재촉 없이 `log_verification`
  을 부른다.

**판정(축 B).** {016 070} · {066 067 062} · 회귀 감시 {003 007}(도구가 안 나타나야 함).

---

### M2. 손님이 부를 gold 는 재촉하지 않는다

**근거.** 손님 gold 6 태스크(§1). 049: `make the call now` 가 사과 턴을 없애 손님이
`request_human_agent_transfer` 를 부를 턴이 사라짐(손님 호출 1/4 → 하한 뒤 3/4). 004·008·012·014: 유예된
hand-off 텍스트를 손님이 이관으로 읽고 종료.

**정정(리뷰).** 019 에서 "어시스턴트가 손님 도구를 직접 4번 호출" 은 **오독**이었다. 실제 호출은
`give_discoverable_user_tool` 이고 env 는 `Tool given to user: submit_cash_back_dispute_0589 …` 로 정상
응답했다(sw_019 trial 0·2 확인). 손님 시뮬레이터가 *"I don't see any dispute tools"* 라고 한 것은 give 뒤
안내의 문제이지 우리 사실 한 줄의 표적이 아니다. **019 는 M2 의 근거·판정에서 뺀다.**

**손님 도구의 출처(정정).** `list_discoverable_user_tools` 는 손님 측 런타임 도구("List all tools that
have been given to you by the agent")라 정적 목록이 아니고 우리 층이 볼 수도 없다. 맞는 출처는 env 의
등록 측: `scripts/distill/tau2/a2/env_surface.json` `banking_knowledge.tools[*].side == "user_tools"` —
정확히 10개:
`apply_for_credit_card call_discoverable_user_tool deposit_check_3847 get_card_last_4_digits
get_referral_link list_discoverable_user_tools request_human_agent_transfer submit_cash_back_dispute_0589
submit_referral submit_transaction`.

**단, `env_surface.json` 은 스냅샷이지 생성물이 아니다**(리뷰): 커밋 `c801a3e2` 한 번에 들어왔고 저장소
어디에도 그것을 쓰는 스크립트가 없다(전부 읽기 — `_lit_scan` `t2_forensic` `x417` `x953` …). 출처 없는
스냅샷은 저작 데이터와 구별이 안 된다. 그러므로 목록은 **설치 시점에 env 에서 직접 뽑는다**:
`lb_runtime.install()` 에서 `get_environment("alltools")` 가 등록한 user-tool 집합(tau2 Environment 의
user tools API — 구현 전 정확한 속성명 확인)을 읽어 `agent._lb_customer_tools` 로 두고 사이드카
`lb-customer-tools` 에 찍는다. A2 에는 적지 않는다(런타임 사실이지 선언이 아니다). `env_surface.json` 의
`side=="user_tools"` 10개는 **대조본**으로만 쓴다 — 설치 시 집합이 그것과 다르면 `lb-diverge` 로 남긴다.

**설계.**
- `LB5.open_request` 의 답을 nc15 형식(도구 이름 선택, `<name>: yes|no`)으로. 후보 = 쓰기 도구 전부
  **+ 손님 도구 10개**(리뷰: 손님 도구가 후보에 없으면 서브가 가장 가까운 어시스턴트 도구
  `transfer_to_human_agents` 를 골라 004 형 재촉이 된다). `acts` 산문·자유 서술 폐기.
- 엔진: 고른 것 − `ran`. 남은 것이 `agent._lb_customer_tools` 에 있으면 **말하지 않는다**. 이름 정확 일치뿐.
- 어시스턴트가 손님 도구를 **직접** 부르려는 경우는 env 가 스스로 거절/응답하므로 우리 문장은 없다.

**판정(축 C).** {049 016 081 098 023} · {047 045 043}.

---

### M3a. 프레임·지시 문장 셋 제거 (문구만, 변경 하나)

**근거.** `Provisional credit for this dispute` — 비-dispute 태스크에서 부른 17 sim 전멸, 036 0/17.
`Do NOT change any transaction's rewards…` — reward 도구를 부른 4 태스크 전패(018 −42, 019 −30).
`Pass the verdict as eligible_for_provisional_credit.` — 같은 도구의 지시.
네 트리 모두 각 1회 → 제거 후 **남은 개수 0**(`CLAUDE.md` 규칙 충족).

**설계.** 세 문장 제거. 첫째는 `Provisional credit, if a dispute is filed:` 로 사실형. "빈 결과는 사실만"
은 **이미 되어 있다** — 커밋 `04de2125`(2026-09-10, *a catalogue that could not rank says so instead of
ranking*; task_067 이 빈 필드로 불러 카탈로그 순서가 나온 것을 고침, 테스트 포함). M3a 에서 뺀다.

**판정(축 E).** {036 038} · {018 019}. 036 은 여기서는 판정(도구 출력 프레임)이다.

### M3b. 인자 되비춤 — **보류**

070 2×2(12팔 42 sim): fit 호출 9승 9패(50%) vs 미호출 10승 14패(42%). **표적이 없다.** 070 의 −2 는
`AUTOPSY_058_070` 대로 우리 문장에 귀속할 근거가 없고, 패 sim 의 `account_class` 오답은 fit 도구를
불렀든 안 불렀든 같은 비율이다. 되비춤은 어느 태스크에서 "도구를 부른 sim 이 더 진다" 는 2×2 가 나올
때만 다시 올린다. (`040` 도구 답 50건·38,634자에 헤더를 더 얹는 부담도 있다.)

---

### M4. 조건부 주입은 사건으로 — `inject_after`, 그리고 `requires_reads` 통합

**근거.** `inject_after` 가 `verify_identity` 에서 003 2→4, 007 유지. 나머지 도구는 첫 발화 LLM 선택만.

**`requires_reads` 처분(리뷰).** A2 에 이미 선언돼 있으나(`get_correct_savings_apy` 등
`["get_all_user_accounts_by_user_id"]`) `lb_a2.py:220` 이 보존만 하고 **어느 코드도 읽지 않는 죽은
필드**다(nc18 10개 파일 전수에서 소비처는 `lb_a2.py:220` 하나). 같은 개념에 키 둘을 두지 않는다 →
**`requires_reads` 는 `inject_after` 로 흡수하고 지운다**(값 형태 동일: 선행 도구 이름 목록). `lb_a2.py`
keep-list 에서 `requires_reads` 제거, A2 의 세 선언을 `inject_after` 로 옮김.

**코드가 이미 있고, 설계와 한 군데 어긋난다**(리뷰, nc18 `lb_runtime.py:121-128`·`:362-372`):
- `set(d["inject_after"]) & ran` — 목록은 **OR** 다. savings 행의 "A 또는 B" 는 선언만으로 된다.
- `:371 if d.get("select") and d["name"] not in chosen: continue` — "선택 AND 사건" 은 코드와 일치.
- **어긋남**: `ran` 은 어시스턴트 호출의 **도구 이름 집합**이다. dispute 행의 사건
  `unlock_discoverable_agent_tool(file_credit_card_transaction_dispute_*)` 은 unlock 의 인자
  `agent_tool_name` 을 `fam()` 으로 봐야 잡히는데, 지금은 `unlock_discoverable_agent_tool` 이라는 이름
  자체만 걸린다. → **`lb_runtime` 한 줄**: `turn_hook` 에서 `ran` 을 만들 때 디스패처 호출이면
  `dispatch.name_args` 의 인자값을 `fam()` 해서 `ran` 에 함께 넣는다(`Turn.named()` 와 같은 규칙).
  `inject_after` 의 값도 `fam()` 형으로 적는다(`file_credit_card_transaction_dispute`).

**표(env 구조 출처 · 확정 전 격리 측정이 전제).**

| 도구 | inject_after | 출처 | 상태 |
|---|---|---|---|
| `check_credit_dispute_provisional_credit` | `unlock(file_credit_card_transaction_dispute_*)` | 존재 이유가 그 도구의 인자 `eligible_for_provisional_credit` | ✔ |
| `get_debit_dispute_liability_cap` | `unlock(file_debit_card_transaction_dispute_*)` | 같음 | ✔ |
| `get_reward_discrepancies` | `get_credit_card_transactions_by_user` | `op.over: transactions` · `grounded_params.transaction_id.producer_contains: credit_card_transaction_history` | ✔ |
| `get_correct_savings_apy` · `get_interest_correction` | `get_all_user_accounts_by_user_id` | 기존 `requires_reads` | ✔ |
| `check_rebate_qualification` | **없음 — 선택만** | nc18 A2 에 `grounded_params`·`ref_params`·`requires_reads` 가 전혀 없다(확인함). 근거 도구를 먼저 선언(`grounded_params.transactions.producer_contains`)하기 전에는 `inject_after` 를 넣지 않는다 | ✗ |
| fit 도구 4종 | 없음(상담 초반) | | — |

**격리 측정이 표보다 먼저다.** 전 궤적에서 각 도구가 실제로 불린 sim 중 선행 사건이 그 전에 있었던
비율(놓침)과 선행 없이 불린 sim 의 승률을 내고, 놓침이 큰 행은 표에서 뺀다.

**036 은 판정이 아니라 회귀 감시로.** 036 패 sim 은 이미 unlock 했다(`CLAUDE.md` 036 절) — M4 는 036 을
못 고친다.

**판정.** {081} · {018 019 021 028} · {059 063 064}. 회귀 감시 {036}.

---

### M6. 보고 형식

§1 의 묶음과 채택 규칙을 모든 팔 보고에 고정한다.

## 3a. 통합 확인 — 전수런 전에 상쇄를 본다 (리뷰 반영)

단일 팔은 **채택 판정**일 뿐 상쇄는 보여주지 않는다. 최종 스택은 합집합이므로 전수런 전에 한 단계:

- **nc24 = nc18 + 채택된 변경 전부**를 **같은 판정 태스크 8개**에 nt=4 로 돌린다(≈ 3엔진 2h).
- 태스크마다 `nc24 − nc18` 을 그 태스크를 판정한 단일 팔의 Δ 와 비교한다.
  단일 팔 Δ ≥ +2 인데 nc24 Δ ≤ 0 → **상쇄**(관여하는 두 변경을 뺀 팔로 한 쌍만 재확인).
  단일 팔에서 −1 이내인데 nc24 ≤ −2 → **조합 부작용**.
- 같은 자리를 만지는 쌍은 사실상 **M1 × M2**(둘 다 `[LEDGER]`: 전달 vs 내용) 하나다 → 049·016·004 가 직접
  판정, 나머지는 회귀 감시. M1×M5, M5×M3a, M2×M3a 는 다른 파일·다른 도구라 겹치지 않는다.
- 상쇄 없음이 확인된 nc24 로 96 전수.

## 3. 팔 계보 — 병렬

M1(`lb_runtime.py`)·M5(`lb2_decision.py`/`execute`)·M3a(A2 문구)·M2(A2 `open_request`+`lb5`)·M4(A2
`inject_after`+`lb_a2`)는 서로 다른 파일/키를 만진다. **nc18 위에 병렬 팔**로 세우면 귀속이 깨지지 않고,
이긴 것만 마지막에 한 팔로 쌓아 상호작용을 본다.

```
arm/nc19 = nc18 + M1      판정 D  {004 008 012 014 040 005 037} | {016 049 036}       10 태스크
arm/nc20 = nc18 + M5      판정 B  {016 070} | {066 067 062} | 감시 {003 007}          7
arm/nc21 = nc18 + M2      판정 C  {049 016 081 098 023} | {047 045 043}              8
arm/nc22 = nc18 + M3a     판정 E  {036 038} | {018 019}                              4
arm/nc23 = nc18 + M4      판정    {081} | {018 019 021 028} | {059 063 064} | 감시 {036}  9
arm/nc24 = nc18 + 채택된 것 전부   손실16 + 위 판정 합집합                         ~25
```

**비용.** 38 태스크 × 4 sim = 152 sim(병렬 5팔) + 통합 ~100 sim ≈ **250 sim**. 엔진은 8141 하나 —
태스크당 중앙값 43분이면 약 27시간(전수런이 9141·9143 을 쓰는 동안). 직렬 v1 안(≈520 sim)의 절반.
엔진이 하나이므로 "병렬"은 큐 하나에 팔 태그를 섞어 넣는 것이고, 레인 시작 줄의 `sha`/`a2` 지문으로
갈린다.

## 4. 하지 않는 것

태스크별 조건 · 낱말 매칭 · 일괄 문구 변환 · 횟수 상한 · `[ORDER notice]` 제거 · `hidden` 의 뜻 변경 ·
표적 2×2 없는 되비춤.

## 5. 남은 확인

- M4 표의 `check_rebate_qualification` 선행 사건(선언 `ref_params`/`grounded_params` 대조).
- `04de2125` 가 "빈 결과는 사실만"을 이미 했는지.
- M1 드롭률(nc19 `lb-advice-dropped` 실측).
