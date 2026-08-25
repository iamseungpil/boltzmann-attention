# task_074 per-step 포렌식 — bank_t7348 halfB (2026-08-24)

> ⚠이 파일은 trial 별로 **절 단위 추가**된다. 다른 trial 절을 덮어쓰지 말 것.

## §0 종합 — 두 trial 합본 · 세 렌즈 반증 병합 (2026-08-24)

> 이 절은 trial 0(`seed 626729`)·trial 1(`seed 373753`)을 **합쳐** 판정한다. 아래 `## trial 1` 절은 그 trial 의 1차 자료이고 여기서 덮어쓰지 않는다.
> **계기**: 변이 집합 = 정본 `t2_forensic.mutation_diff(sim, F.mutating_tools(), tag)` (손 비교기 0·[[67]]) · 궤적/로그 `gz` 직독 · 코드 인용은 **런 sha `aed30e20` 의 `git show` 판**(worktree 아님·[[77]]).
> **규율**: 측정만. 수리·코드 수정·커밋·SSH **0**([[62]]①). gold(`reward_info`)는 진단용([[23]]).
> **범위**: 074 만. 다른 태스크의 결론은 074 에 대해 가설이고, 인용할 때는 "074 에서 같은 것이 관측되는가"를 먼저 확인해 결과를 함께 적었다.

---

### §0.1 한 줄 결론

**074 는 t7348 에서 gold 13 액션 중 앞 9칸(신원 확인·발견·4계좌 read·크레딧 도구 unlock)을 양 trial 모두 정확히 밟고, 마지막 4칸 — 계좌별 `fee_refund` 금액(`27.00 / 14.50 / 4.75 / 3.70`) — 에서만 갈려 `reward 0.0` 이다. 그 4칸이 `reward_basis=["DB"]` 해시 축의 전부이므로 다른 어떤 성취도 점수가 되지 않는다.**

원인은 **두 층에 각각 하나씩** 있고, **각각 단독으로 실패를 확정**한다:
- **우리 층** — `get_atm_fee_discrepancies` 의 격리 서브(`isolate.mode=fetch_formalize`)가 `transactions` 를 **4계좌 중 3계좌**에서 틀리게 형식화해, 도구가 낸 net 이 `27.00 / 17.00 / 2.50 / 1.45` 로 gold 와 **1/4** 만 맞았다. 모델이 도구에 100% 순종해도 3/4 불일치가 확정이다(천장).
- **모델** — 반환문이 4/4 회 축자로 요구한 *"ONE fee_refund credit for the **net** correction of THIS account"* 를 두 trial 모두 이행하지 못했다. t1 은 음수 델타를 버려 Σ(양수)를 보냈고, t0 은 음수를 절댓값으로 뒤집어 계좌당 4~6건으로 분할한 뒤 총액을 한 번 더 보내 **4계좌 전부 이중 크레딧**했다.

어느 한 층도 단독으로는 pass 를 만들지 못한다. 이것은 반사실이 아니라 산수다.

---

### §0.2 두 trial 은 같은 이유로 죽었나 — **우리 층은 같고, 모델은 다르다**

**우리 층 결손은 두 trial 공통이고 값까지 동형이다.** `[57]/[59]/[61]/[63]` 네 comparator 반환문이 두 trial 에서 **net·coverage·판정 id 까지 동일**하다(직접 대조):

| | Purple `_1` | Light Blue `_2` | Dark Green `_3` | Evergreen `_4` |
|---|---|---|---|---|
| 우리 도구 net | 27.00 | 17.00 | 2.50 | 1.45 |
| `[coverage]` | `18 of 18` | `14 of 14` | `17 of 17` | `17 of 17` |
| 판정 행 n | 5 | 8 | 6 | 6 |

같은 로그 마커도 동형이다 — `T2_SG_BYREF 8 · T2_SG_ARGS 8 · T2_SG_ISOLATE 16 · T2_SG_REQREADS 4 · T2_EPLAN_LISTED_IS_READ 19(t0)/15(t1) · T2_COMPUTE 0`.

**모델 쪽은 다르다.**

| | t0 (626729) | t1 (373753) |
|---|---|---|
| 크레딧 발송 형태 | 계좌당 **4~6건 분할** → 그 뒤 **총액 재발송** | 계좌당 **1건** |
| 음수 델타 7건 처리 | **절댓값으로 부호 반전**해 크레딧 | **버림**(Σ 양수만) |
| 실제 write 건수 | **23**(`mutation_diff.done`) | **5** |
| `mutation_diff` | `matched 1 · missing 4 · wrongarg 22` | `matched 2 · missing 3 · wrongarg 3` |
| 최종 잔액 | `4549.00 / 7277.00 / 2125.00 / 15820.10` | `4527.00 / 7269.50 / 2107.50 / 15806.50` |
| gold 기대 잔액 | `4527.00 / 7264.50 / 2104.75 / 15803.70` | 〃 |
| 규모 | 105 msgs · `duration 2704.5s` | 80 msgs · `duration 2015.7s` |

⇒ **t1 은 우리 층 결손만으로 죽었다**(형태는 gold 와 같은 계좌당 1건이고 Purple 은 `amount 27.0` 으로 `MATCHED`). **t0 은 우리 층 결손 위에 모델의 형태 위반이 겹쳐 죽었다** — t0 에서는 우리 층 결손이 전부 없었어도 pass 가 불가능하다(gold=계좌당 1건 net ↔ 실제 19건 분할 + 재발송 4건).

---

### §0.3 산수 축의 상태 — **닫히지 않았다. 1/4 만 닫혔다**

| 계좌 | **gold** | **우리 도구 net**(양 trial 동일) | **모델 발송** t0 (1차 → 최종 누적) | **모델 발송** t1 | 갈린 이유 |
|---|---|---|---|---|---|
| `chk_ar72c5d8e3_1` Purple | **27.00** | **27.00** ✅ | 24.50 → **49.00** | **27.00** ✅ | 도구는 정확. t0 만 `T2_STALE_STRIP` 이 둘째 `$2.50` 삭제(→24.50) + `[98]` 재발송 |
| `chk_ar72c5d8e3_2` Light Blue | **14.50** | 17.00 (**+2.50**) | 8.00 → **27.00** | 19.50 | 서브가 인출 16행 중 **14행만** 넘김 → `btxn_ar_lb_07f +2.50` 과다 판정 |
| `chk_ar72c5d8e3_3` Dark Green | **4.75** | 2.50 (**−2.25**) | 12.50 → **25.00** | 7.50 | 서브 유령행 `btxn_ar_dg_17 (charged $0.00, documented fee $2.25)` |
| `chk_ar72c5d8e3_4` Evergreen | **3.70** | 1.45 (**−2.25**) | 10.05 → **20.10** | 6.50 | 서브 유령행 `btxn_ar_ev_17` (동형) |

- **우리 도구 ↔ gold = 1/4.** Purple 만 맞다.
- **모델 발송 ↔ gold = t0 0/4 · t1 1/4.**
- **닫힌 것**: Purple 의 rebate 순액화(t7346 의 24.50 → t7348 의 27.00). **안 닫힌 것**: LB·DG·EV 3계좌, 그리고 두 trial 모두의 집계 형태.

**DG·EV 의 −2.25 는 gold 를 안 봐도 우리 자신의 read 와 모순이다.** `sim#49` 축자 `Record ID: btxn_ar_dg_17f | date: 11/17/2025 | amount: -2.25 | type: atm_fee` · `sim#50` 축자 `Record ID: btxn_ar_ev_17f | amount: -2.25 | type: atm_fee` — 수수료는 **부과돼 있었다**. 그 두 줄을 빼면 `4.75 / 3.70` = gold 와 센트까지 일치(확인용 산수·진단 전용·[[23]]).

**산수 op 자체는 무결하다.** 원장 축자에서 기계 구성한 정본 행집합(Purple 18 = 인출 17 + 중복 fee 1 · 나머지 16)을 같은 `scaffold_get_tools[8].op` 에 넣으면 `27.00 / 14.50 / 4.75 / 3.70` = **gold 4/4**(부록 C-1). ⇒ 결손은 계산기가 아니라 **전사 서브의 operand** 하나다.

---

### §0.4 결정 지점과 귀속

턴 번호 = 메시지 인덱스. `[1]`~`[63]` 은 두 trial 이 사실상 동형이고, `[65]` 이후가 갈린다.

| # | 턴 | 축자(발췌) | 귀속 |
|---|---|---|---|
| **DP-1** | `[4]`→`[5..8]` | 우리 deny `Error: [READ-FIRST] … these required reads are missing: get_all_user_accounts_by_user_id, get_bank_account_transactions … an account's display class name like 'Purple Account' is NOT an account id` | **our_layer(양성)** — read 0회 상태의 날조 4연발을 `[12]` unlock 으로 전환. [[64]] 준수 |
| **DP-2** | `[41]`/`[51]`→`[42..55]` | 로그 8쌍: `[T2_SG_BYREF] … 미해석 — isolate(fetch_formalize)가 'transactions' 를 산출하므로 deny 하지 않고 넘긴다` 직후 `[T2_SG_ARGS] … str 잔류(JSON 파싱실패) → 재송신 요구` / 모델에게 간 문면 `Error: [ARGS-FORMAT] … Copy the raw field values exactly as they appear in the records.` | **our_layer 100%** — 우리가 스키마로 광고한 `@last:` 를 우리가 8/8 반려. 왕복 2회전 + 손-전사 날조 유발. **채점 손실 0**(서브가 덮어씀) |
| **DP-3** | `[56]`~`[63]` | `[T2_SG_ISOLATE] operand-size get_atm_fee_discrepancies.transactions: sub=18/14/17/17 rows · source=33/30/29/29 rows ⚠MISMATCH` ↔ 원장 `atm_withdrawal` 실수 17/16/16/16 | **our_layer 100%** — 3/4 계좌 operand 오형식화. **실패의 천장을 만든 자리** |
| **DP-4** | `[65]`→`[68]/[70]/[72..84]` | 반환문 축자 `If corrections are owed, the credit policy requires ONE fee_refund credit for the net correction of THIS account (do not credit the same lines twice).` (4/4 회) ↔ t0 19건 분할 · t1 Σ(양수) | **model 주** — DB 해시 축에서 t0 은 이것만으로 실격. t1 은 형태는 맞고 값만 틀림 |
| **DP-5** | `[68]/[73]/[84]` (t0 만) | `[T2_STALE_STRIP] dropped 1 / 5 / 1 stale/dup call(s)` · 삭제된 `_1 $2.50` 둘째 건이 갚는 원장 라인 = `sim#47` 축자 `btxn_ar_purple_15f_dup … amount: -2.5 … type: atm_fee` | **our_layer** — 은행의 **중복 부과**를 갚는 write 를 인자 바이트 동일이라는 이유로 삭제. 유일하게 gold 정확했던 27.00 → 24.50. **reward 중립**(gold 는 계좌당 1건) |
| **DP-6** | `[91]`→`[98]` (t0 만) | user `[91]` 축자 `Just to confirm, the total credits were: - Purple: $24.50 - Light Blue: $19.00 …` → 모델 `[98]` 이 `24.5 / 19.0 / 12.5 / 10.05` **재발송** → 잔액 `4549.00 / 7277.00 / 2125.00 / 15820.10` | **model 주 · our_layer 종** — 확인 질문을 실행 지시로 읽었다. 우리 층은 침묵했다(규칙②는 `(eff, _call_key)` 완전일치만 잡고 24.5 는 이전 어느 인자와도 다르다 ⇒ **계좌별 누적 환불액 축이 우리 층에 없다**) |
| **DP-7** | `[65]` (t1 만) | user 축자 `for the ones where the difference is negative … I'm only asking for credits back for the overcharges` — 시나리오 `instructions` 에 없는 즉흥 | **user_sim 기여** — [[21]] 대로 면책 아니다. 그 발화의 재료 절반(`dg_17`·`ev_17` 의 허위 음수)은 우리 층이 공급했다. **부정통제**: 그 문장이 없던 t0 도 netting 하지 못했다 ⇒ **필요조건이 아니다** |

**지분** (한 주체로 안 접힌다):

| trial | our_layer | model | user_sim | env |
|---|---|---|---|---|
| t0 (626729) | 45 | **55** | 0 | 0 |
| t1 (373753) | **45** | 45 | 10 | 0 |

`env 0` 의 근거: 모든 도구 결과가 정상 반환됐고(레코드 33/30/29/29 일관 · `Credit applied successfully!` t0 22건/t1 4건), 실패 문면은 `[READ-FIRST]`·`[ARGS-FORMAT]`·`[DUPLICATE-READ]`·`[OPERATOR-SCOPE]` 로 **전부 우리 층 접두**다.

---

### §0.5 반증 병합 — 세 렌즈(execution / alternative / contamination)

병합 규칙 (refutation merge / refuted-if 기준): **둘 이상 REFUTED = 기각** · **하나라도 REFUTED = UNPROVEN 이하** · **CONFIRMED 는 셋 통과만**.

| # | 주장 | exec | alt | cont | **병합** | 남는 것 |
|---|---|---|---|---|---|---|
| 1 | **OL-1** `_iso_owns` 우회가 구조적 死배선(over-str 검사가 isolate 의 **상류**라 주석의 폴백 전제 성립 불가) | C | C | C | **CONFIRMED** | 런 sha 순서 `_byref_resolve:1580` → `_ov`(over-str)`:2020` → `[ARGS-FORMAT]:2026` → isolate 주입 `:2110`. `:1595` 의 `if k not in overs` 때문에 `:1616` 도달 키는 반드시 over 파라미터 ⇒ `_ov=None` 경로 원리상 없음. **양화 정정: 8/8 은 "byref 단계에 도달한 8건" 이다**(총 `@last:` 시도 12건 중 앞 4건은 `[READ-FIRST]` 로 먼저 죽었다) |
| 2 | **OL-2** `dg_17`·`ev_17` 의 `charged $0.00` 이 같은 대화의 read 와 자기모순 | C | C | C | **CONFIRMED** | n=2/4 계좌·양 trial. 원장에 `dg_17f`/`ev_17f` = `-2.25` `atm_fee` 실재. 페어링 정의는 우리 A2 자신(`the adjacent atm_fee record with the same account and date`). **귀속 정밀화: 오류는 comparator 산수가 아니라 서브 operand** ⇒ OL-74a 의 부분집합(이중 계상 금지) |
| 3 | **OL-3** 판정 행수 ↔ 원장 인출 행수 **4/4 불일치**인데 완전 커버리지 주장 | **R** | **R** | **R** | **기각(REJECTED)** | 주장 자신의 반증 조건이 Purple 에서 발화한다 — A2 축자 `if TWO fee lines belong to the SAME withdrawal, include that withdrawal twice`, 원장에 `btxn_ar_purple_15f` / `15f_dup` 가 같은 11/15 인출에 붙은 두 fee 라인 ⇒ **18 이 정답**. **3/4 로 다시 쓰면 성립**(LB 2행 결손 · DG/EV 유령행 +1 · `0 could not be verified` 는 자기참조 분모) |
| 4 | **OL-4** `_stale_call_ids` 규칙①(`if key in seen`)이 정당한 동일-금액 write 삭제 | C | C | C | **CONFIRMED** | 런 sha `t2_gate_patch.py:2024`. `git log -S'if key in seen'` = 커밋 1건(최초 도입) ⇒ 2026-08-22 수리는 규칙②만 좁혔다. `dropped 1/5/1`(t0 만·t1 은 0). **reward 중립 자인**. ⚠"정당한" 은 fee 라인 수준에서만 참(계좌당 1건 정책 아래서는 분할 자체가 위반) |
| 5 | **OL-5** `_STALE_NOTE` 가 다음 턴 본문으로 전사 | C | C | C | **CONFIRMED** | 본문 노트 5개(`[68]'1' [73]'5' [77]'3' [84]'5'+'1'`) ↔ 로그 3줄(`1/5/1`). 노트 붙임과 로그 인쇄가 **같은 `if _stale:` 블록** ⇒ 로그 없는 본문 노트는 **모델 산출**. `[77]` 은 6건 예고·6건 발신인데 노트 '3' = **전사가 아니라 수치 날조**. ⇒ 삭제 건수는 본문이 아니라 로그로 센다 |
| 6 | **OL-6** `scaffold_guard.py` §77 (3)반증조건이 한국어를 못 매칭 | U | U | C | **UNPROVEN** | 기전은 재현된다 — 훅이 `sys.stdin` 만 `reconfigure` 하지 않아 `cp949` 로 UTF-8 을 읽으면 한글 술어가 전멸하고, (3)만 라틴 대안이 `refut` 하나뿐이라 증상이 (3) 단독으로 보인다. **그러나 라이브 훅 payload 를 관측하지 못했다**(합성 재현). 대안 가설(Edit 조각에 '반증'이 빠졌다)도 살아 있다. 074 궤적과 **인과 0**(도구 밖) |
| 7 | **OL-74a** 격리 서브가 3/4 계좌 `transactions` 오형식화 | C | C | C | **CONFIRMED** | 헤드라인 성립. 서브 18/14/17/17 ↔ 정답 18/16/16/16. **⚠마지막 절 정정**: `isolate` 하위에 `ground` 는 없으나 같은 선언에 `grounded_params={"transaction_id":{"producer_contains":["record id:"]}}` 가 **실재한다** ⇒ 결손은 "검산 부재" 가 아니라 **검산이 `transaction_id` 만 덮고 `fee_amount`·행 존재를 안 덮는 것**이다(서브가 쓴 id 는 전부 원장에 실재해 통과한다). 처방을 "검산이 없다" 위에 세우면 표적을 놓친다 |
| 8 | **OL-74b-ⓐ** `operand-size ⚠MISMATCH` 판별력 0 | C | C | C | **CONFIRMED** | 분모가 `_t.count("Record ID:")` = 레코드 전수(33/30/29/29)라 **정답인 Purple(18)에도 붙는다**. 4/4 발화. 엄밀히는 재현율 3/3 · 정밀도 3/4 = 이 sim 안에서 판별력 0 |
| 9 | **OL-74b-ⓑ** `_omitted_rows_note` 사문 | C | C | C | **CONFIRMED** | 런 sha `t2_scaffold_get.py:324` 정의 · `:350` 무조건 `return ""` · 소비자 1곳(`:2495`). LB 는 2행이 빠졌는데 반환문이 `14 of 14 … (0 could not be verified)`. ⚠"버그" 가 아니라 2026-08-14 의 **의도적 무효화**이고, 되살리려면 코드 주석이 지정한 대로 **서브가 자기 후보 수를 선언하는 분모**가 먼저 필요하다 |
| 10 | **OL-74c** `T2_SG_BYREF` ↔ `T2_SG_ARGS` 자기모순 | C | C | C | **CONFIRMED** | 8쌍·양 trial. ⚠**claim 1 과 같은 사건**이다 — 한 결함을 두 항목으로 계상하지 말 것. ⚠모델 앞에 선 것은 deny **하나뿐**이고 BYREF 줄은 stderr 전용이다([[55]] 로그 마크 ≠ 전달) |
| 11 | **OL-74d** `T2_COMPUTE` 0줄 = **이 축의 회수가 실패하고 있다** | C | C | **R** | **UNPROVEN** | 관측은 참이다(halfB 로그 `T2_COMPUTE` **0회**, comparator 는 4회 돌았다). 무너지는 것은 **현재시제 결론**이다 — 성공 마커가 커밋 `bf0f7c59`(2026-08-24 14:45)에서 추가됐고 `git merge-base --is-ancestor bf0f7c59 aed30e20` = **거짓**(런 이후) ⇒ **이미 수리된 것을 미수리로 되살린 것**([[74]] 위반 위험). 단 라이브 계측은 새 런 전까지 미측정 |
| 12 | **OL-74e** `T2_EPLAN_LISTED_IS_READ` 명단이 Dark Green 으로 **고정** | C | C | **R** | **UNPROVEN** | **앞 절 기각**: 인쇄가 `",".join(sorted(_lr))[:120]` 로 **120자 절단**이다(런 sha `t2_gate_patch.py:7253`). 배달 121건 = 33+30+29+29 = **네 계좌 전수**이고 사전순으로 `dg` 가 앞이라 앞 120자에 dg 만 보일 뿐이다(실제로 `btxn_ar_dg_06,bt` 로 토큰 중간에서 끊긴다). **뒷 절은 생존**: `[T2_REDERIVE] raw='Bluest' → Bluest` 1회이고 이 손님 계좌 4건에 Bluest 는 없다(KB 문서 본문에만). 단 모델 본문·write 로 샌 흔적이 없어 손해는 미측정 |
| 13 | **[산수]** 도구 net 1/4 · 모델 t0 0/4 · t1 1/4 · `mutation_diff` 수치 · t0 이중 크레딧 | C | C | C | **CONFIRMED** | 정본 계기로 전 항목 재현(위 §0.2·§0.3). `reward 0.0 · reward_basis ["DB"] · db_match false · user_stop` 양 trial |
| 14 | **[출처]** ⓐ도구 출력 축자 ⓑ모델 산수 **ⓒ손님 발화 숫자 0건** **ⓓ날조 정확히 1건($19.00)** | C | **R** | **R** | **기각(REJECTED)** | **ⓒ 반증**: t0 `[91]` user 축자에 `$24.50 / $19.00 / $12.50 / $10.05` 가 **있다**(t1 `[77]` 도 동형). "손님이 새 숫자를 도입하지 않았다" 로는 참이지만 적힌 대로는 거짓이고, `[98]` 이중 크레딧의 **근접 자극을 한 칸 놓친다**. **ⓓ 반증**: `24.50/12.50/10.05` 도 tool 메시지에는 `New Balance: $4524.50` 같은 **부분문자열로만** 존재하므로 "우리 도구가 준 적 없는 수" 기준으로는 넷 다 해당한다 — 19.00 만 골라낸 것은 문자열 충돌의 산물. **ⓐⓑ 는 생존**(낱개 금액 1:1 대응 · 음수 7건 t0 부호반전/t1 폐기 · t1 값 = Σ(양수) 검산 통과 · `$19.00` 이 `New Balance: $7258.00`(=7250+8.00)과 자기모순인 산수 오류라는 실질 지적도 옳다) |
| 15 | **[판정]** 두 층이 각각 독립으로 실패에 충분 | C | C | C | **CONFIRMED** | ①은 원장·gold 로 독립 확인(위 §0.3) ②는 반환문 축자 위반 ⇒ 둘 다 성립. ⚠판정문 안에 OL-3 의 "4/4" 를 끌어들이면 안 된다(Purple 은 정확했다). ⚠t0 의 부정통제는 **t7348 에서는 신규지만** 직전 런 t7346 t0 에도 같은 구조가 있었으므로 "처음 관측" 은 t7348 에 한정된다 |

**집계: CONFIRMED 10 · UNPROVEN 3 · 기각 2 (n=15).** 상세는 `reports/facet_rft_2026/refute_2026_08_24/refute_074.json`.

---

### §0.6 다음 수 ([[62]] 순서 · ⛔이 문서에서 수리하지 않는다)

**① 격리로 더 재야 하나 — 부분적으로 그렇다. 다만 먼저 필요한 것은 레버가 아니라 계기다.**
서브가 실제로 산출한 행이 **어디에도 영속되지 않는다**(검색한 경로 — grep/ls 한 곳: `bank_t7348_halfB_20260824.log.gz` 전수 · `fb_bank_t7348_halfB_20260824.jsonl.gz` · `trace_bank_t7348_halfB_20260824.jsonl.gz` · `t2_scaffold_get.py:324,946,2495`). 그래서 LB 2행 결손의 기전이 `network` 오라벨인지 `duplicate_of` 오부착인지 **확정 불가**다(부록 C-2 가 해집합을 `{lb_05, lb_06, lb_07}` 로 닫았을 뿐이다). ⇒ **서브 산출 행의 영속이 선행 조건**이고, 그것 없이는 격리 프로브도 같은 자리에서 눈이 먼다.

**② 저작·배선인가 — 저작이다(엔진 아님).**
`get_atm_fee_discrepancies` 의 `isolate` 선언은 이미 `ONE element per atm_withdrawal record of THIS account … Include EVERY atm_withdrawal` 를 축자로 말하고 있는데 서브가 3/4 에서 어겼다. 남은 저작 자리는 두 곳: ⒜`grounded_params` 가 `transaction_id` 만 덮는다(→ `fee_amount`·행 존재로 확장) ⒝`_omitted_rows_note` 를 되살릴 **비교 가능한 분모**(서브가 자기 후보 수를 함께 선언). 둘 다 [[62]] 의 "결정론기는 최소한" 이 아니라 [[72]] 의 "선언은 완결" 쪽이다.

**③ 무료 재생으로 검정 가능한가 — 세 건은 그렇다.**
- **가능**: 정본 행집합 → 같은 `op` 재생(부록 C-1, 이미 실행 · gold 4/4) ⇒ **산수 op 무결이 이미 검정됐다**. `_stale_call_ids` 규칙①에 write 예외를 넣었을 때의 부호표(OL-4·[[70]]), `[ARGS-FORMAT]` 문면 교체(OL-1)도 오프라인 검정 대상.
- **불가능**: 모델의 netting(DP-4)·확인 질문 오독(DP-6)은 재생으로 못 잰다 — 라이브 A/B 가 필요하고, 그 전에 **①의 천장을 먼저 걷어야 측정이 성립한다**(3계좌가 틀린 채로 모델 축을 재면 결손이 섞인다).

**④ 순서 제안**: 계기(서브 행 영속) → 선언 완결(⒜⒝) → 무료 재생으로 부호표 → 그다음에야 모델 축 라이브 대조. **①을 걷기 전에 모델 축 레버를 만들면 [[62]] 위반이다.**

---

### §0.7 못 사는 것 (정직 절)

1. **반증 못 한 것** — OL-6(훅 인코딩)은 합성 payload 재현이지 라이브 훅 관측이 아니다. 이 라운드는 측정-전용 규율이라 그 주장이 지정한 반증 실험(순수 한국어 '반증' 만 담은 `reports/` 쓰기)을 실행할 수 없었다. **UNPROVEN 으로 남긴다.**
2. **측정 못 한 것** — ⒜서브 산출 행(위 §0.6①) ⒝`T2_COMPUTE` 마커 수리(`bf0f7c59`)가 라이브에서 실제로 찍히는지 ⒞OL-74e 의 `Bluest` 재유도가 어떤 손해로 이어지는지(모델 본문·write 로 샌 흔적 0 · `T2_LIMIT_REDUCE` 0회) ⒟`[98]` 재발송의 근인이 손님 `[91]` 복창인지 모델 자기요약 `[77]/[90]` 인지 — **반사실이 없다**.
3. **074 에만 참이고 일반화 안 되는 것** — ⒜Purple 의 18행(= 인출 17 + 중복 fee 1)은 이 계좌에만 있는 `15f`/`15f_dup` 쌍과 A2 의 `duplicate_of` 규칙 때문이다 ⒝`fee_rebate` 5건도 Purple 에만 있다(LB/DG/EV 는 0건·레코드 type 전수 집계) ⒞gold 가 "계좌당 net 1건" 이라는 형태는 이 태스크의 정책 문면에서 온다 ⒟"음수 델타" 라는 실패 형태 자체가 이 comparator 의 출력 형식에 의존한다. **다른 태스크로 옮기지 마라.**
4. **reward 인과의 반사실은 미증명** — "두 층이 각각 충분하다" 는 산수(천장)이지, "어느 한쪽만 고치면 `db_match` 가 참이 된다" 는 증명이 **아니다**. t0 은 두 층을 다 고쳐야 하고, t1 은 우리 층만 고치면 되는 것처럼 보이지만 그것도 미검증이다.
5. **소급 정정 대상(074 자신의 선행 판정)** — ⒜`tasks__20260822/TASK_074.md:215` 가 `⚠MISMATCH` 를 *"분모가 레코드 전수라 생기는 표기일 뿐"* 으로 무해 판정했으나, 이번에 그 계기가 **정답 계좌에도 붙어 진짜 결손 3건을 가렸다** ⇒ 무해 아님 ⒝같은 문서 §5 가 `T2_SG_ISOLATE` 를 *"✅정상"* 으로 적었으나 t7348 에서는 같은 서브가 3/4 계좌에서 틀린 행을 냈다 ⒞`x514_refute_synthesis_2026_08_24.json` 의 `headline_3_amount_axis_is_clean` 과 `axis_ledger` 는 `①금액` 행에 `which:["072","074"]` · `our_defect:"없음"` 으로 적었는데 그 근거는 **072 반증뿐**이다(같은 폴더에 074 파일이 없었음을 `ls reports/facet_rft_2026/refute_2026_08_24/` + `grep -rn 626729 reports/facet_rft_2026/tasks__20260824/` 로 확인) — 본 반증이 074 궤적 단독으로 반대 실물(도구 net 3/4 오답)을 냈으므로 **그 행은 정정 대상**이다.
6. **C215 / x506 caveat 확인** — 축자 *"055·057·072·074·079 는 gold 목록 자체가 런마다 달라진다"* 는 **이 쌍에는 해당하지 않는다**. `t7346 halfB` × 2 seed 와 `t7348 halfB` × 2 seed, **네 sim 의 gold 13행이 `action_id`·도구·인자까지 동일**하다(직접 대조·`IDENTICAL 4/4 True`). ⇒ 런-간 대조는 **유효**하다. 차이는 gold 가 아니라 궤적이다(`t7346 t1` 만 `074_8` 이 `action_match=false`).

---

## trial 1 (seed 373753)

- 런 `bank_t7348_halfB_20260824` (sha `aed30e20` · 조상에 `e78ee2f3`·`3bff2409` = ATM comparator 관련)
- sim `id=46daa849-9143-4000-8288-4b1b2282c4b1` · 사이드카 해시 `bb906ecbfdd7` · 80 msgs · `duration=2015.7s` · `termination_reason=user_stop` · `hallucination_retries_used=0`
- 원자료(전부 로컬·SSH 0·커밋 0):
  - 결과 `C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\bank_t7348_halfB_20260824.results.json.gz`
  - 로그 `...\bank_t7348_halfB_20260824.log.gz` (접두 `[sim=task_074#s373753]` · 396줄)
  - 사이드카 `...\fb_bank_t7348_halfB_20260824.jsonl.gz` (`sidecar=present` · `regen_join=simtag` · 66행)
  - 직전 런 `...\bank_t7346_halfB_20260822.results.json.gz`
- 계기: 변이 집합 = 정본 `t2_forensic.mutation_diff(sim, mutating_tools(), tag)` (손 비교기 0) · 재생 = A2 선언 `scaffold_get_tools[8].op` 를 `t2_compute.apply_op` 로 그대로 돌린 **측정용 재생**(수리·수정 0·[[62]]①)
- **턴 번호 = 메시지 인덱스** (이 sim 은 `turn_idx == index` 로 일치)

---

### 1. 채점 축 (직독 · [[69]])

`sim['reward_info']` 축자:

| 키 | 값 |
|---|---|
| `reward` | **0.0** |
| `reward_basis` | `["DB"]` |
| `reward_breakdown` | `{"DB": 0.0}` |
| `db_check` | `{"db_match": false, "db_reward": 0.0}` |
| `nl_assertions` / `communicate_checks` | `null` / `null` |
| `info.communicate` | `{"note": "No communicate_info to evaluate"}` |

⇒ **DB 해시 축 단독**. 해시에 남는 것 = `mutating_tools()` 가 True 인 도구의 성공 실행뿐이다. 이 궤적에서 상태를 바꾼 것은 `log_verification` 1건 + `apply_checking_account_credit_5829` 4건(모두 `call_discoverable_agent_tool` 경유) = **5건**. `unlock_discoverable_agent_tool`·`KB_search_bm25`·`verify_identity`·`get_current_time`·`get_bank_account_transactions_9173`·`get_all_user_accounts_by_user_id_3847`·`get_atm_fee_discrepancies` 는 해시에 남지 않는다(`mutation_diff` 의 `gold` 5건이 정확히 그 5건이다).

---

### 2. gold (축자) — ⚠t7346 과 **동일**했다

`task['evaluation_criteria']['actions']` 13건 축자:

| aid | 도구 | 인자(축자) |
|---|---|---|
| 074_0 | `log_verification` | `{"name":"Ahmad Razali bin Mohd Yusof","user_id":"ar72c5d8e3","address":"1245 Pioneer Road, Denver, CO 80203","email":"ahmad.razali@gmail.com","phone_number":"303-555-0821","date_of_birth":"06/15/1989","time_verified":"2025-11-14 03:40:00 EST"}` |
| 074_1 | `unlock_discoverable_agent_tool` | `{"agent_tool_name":"get_all_user_accounts_by_user_id_3847"}` |
| 074_2 | `call_discoverable_agent_tool` | `{"agent_tool_name":"get_all_user_accounts_by_user_id_3847","arguments":"{\"user_id\": \"ar72c5d8e3\"}"}` |
| 074_3 | `unlock_discoverable_agent_tool` | `{"agent_tool_name":"get_bank_account_transactions_9173"}` |
| 074_4~7 | `call_discoverable_agent_tool` | `get_bank_account_transactions_9173` × `chk_ar72c5d8e3_1` / `_2` / `_3` / `_4` |
| 074_8 | `unlock_discoverable_agent_tool` | `{"agent_tool_name":"apply_checking_account_credit_5829"}` |
| **074_9** | `call_discoverable_agent_tool` | `apply_checking_account_credit_5829` `{"account_id": "chk_ar72c5d8e3_1", "amount": 27.00, "credit_type": "fee_refund"}` |
| **074_10** | 〃 | `{"account_id": "chk_ar72c5d8e3_2", "amount": 14.50, "credit_type": "fee_refund"}` |
| **074_11** | 〃 | `{"account_id": "chk_ar72c5d8e3_3", "amount": 4.75, "credit_type": "fee_refund"}` |
| **074_12** | 〃 | `{"account_id": "chk_ar72c5d8e3_4", "amount": 3.70, "credit_type": "fee_refund"}` |

**C215 caveat 확인 결과 = 이번 쌍에는 해당 없음.** t7346 halfB 의 두 seed(626729·373753)와 t7348 halfB 의 두 seed — **네 sim 의 gold 13건이 도구·인자까지 동일**하다(직접 대조). ⇒ 이 두 런 사이의 074 대조는 **유효**하다. (t7346 t1 만 `074_8` 의 `action_match=false`, t7348 t1 은 `true` — 이건 gold 가 아니라 **궤적**의 차이다.)

`action_checks` 채점(진단용·[[23]]): t7348 t1 은 **074_0~074_8 = 9/9 match**, **074_9~074_12 = 0/4**. gold 절차의 앞 9칸을 전부 밟고 **마지막 4건의 금액에서만** 갈렸다.

---

### 3. 변이 집합 (정본 `mutation_diff`)

`{gold: 5, done: 5, matched: 2, missing: 3, wrongarg: 3, extra: 0, dup: 0, blocked: 0}` · `sidecar=present` · `regen_join=simtag` · `regen_blocked=26`

| 칸 | 항목 |
|---|---|
| **MATCHED (2)** | ⓐ`log_verification{…, time_verified:"2025-11-14 03:40:00 EST"}` [32] ⓑ`apply_checking_account_credit_5829{account_id:"chk_ar72c5d8e3_1", amount:"27", credit_type:"fee_refund"}` [68] |
| **MISSING (3)** | `apply_checking_account_credit_5829` — `_2 $14.5` · `_3 $4.75` · `_4 $3.7` |
| **WRONGARG (3)** | 같은 도구 — `_2 $19.5` [70] · `_3 $7.5` [72] · `_4 $6.5` [74] |
| **EXTRA / DUP / BLOCKED** | 0 / 0 / 0 |
| **regen_blocked (26)** | 재생성 채널로 지워진 우리 층 반려 — `[PROVENANCE] account_id='account_id_1'`(turn 2) · `[POLICY GATE GB1_VERIFY_BEFORE_ACCOUNT_ACCESS]`(turn 2 ×4, turn 21 ×5) · `[PROVENANCE] customer_name='John Doe'`(turn 21·23) · `[BLOCKED] … another call in the same turn`(turn 2·21·66) · `[OPERATOR-SCOPE] you called 'apply_savings_account_credit_6831'`(turn 66) · `resolve the flagged call(s) first`(turn 28) |

**WRONGARG 필드별 대조** — `account_id`·`credit_type` 은 gold 와 **전부 동일**하고, 틀린 필드는 **`amount` 하나**다. 건수도 계좌당 1건으로 gold 와 같다.

| 계좌(level) | gold `amount` | 보낸 `amount` | 차 | env 반환 |
|---|---|---|---|---|
| `chk_ar72c5d8e3_1` Purple | `27.00` | `27.0` | **0 (일치)** | `Amount: $27.00 · New Balance: $4527.00` |
| `chk_ar72c5d8e3_2` Light Blue | `14.50` | `19.5` | **+5.00** | `Amount: $19.50 · New Balance: $7269.50` |
| `chk_ar72c5d8e3_3` Dark Green | `4.75` | `7.5` | **+2.75** | `Amount: $7.50 · New Balance: $2107.50` |
| `chk_ar72c5d8e3_4` Evergreen | `3.70` | `6.5` | **+2.80** | `Amount: $6.50 · New Balance: $15806.50` |

⇒ **DB 해시는 4계좌 중 3계좌의 크레딧 금액에서 갈렸다.** 절차·도구·계좌 id·건수·`credit_type` 은 전부 맞다.

---

### 4. 턴별 타임라인

#### 4.1 [1]~[22] 발견·READ-FIRST (우리 층 deny 가 경로를 만들었다)

- **[1] 손님**: *"Hi, I've been going through my bank statements and something seems off with my ATM fees. I have four checking accounts with you — a Purple Account, a Light Blue Account, a Dark Green Account, and an Evergreen Account …"*
- **[2]** `KB_search_bm25{query:"retrieving bank account transaction history"}` → **[3]** `doc_bank_accounts_bank_accounts_(general)_018` 회수(도구명 `get_bank_account_transactions_9173` 축자 포함).
- **[4]** 신원 확인 전, read 0회 상태로 comparator 4연발. 인자 축자:
  `get_atm_fee_discrepancies{account_id:"Purple Account", account_class:"Purple Account", transactions:"@last:get_bank_account_transactions_9173"}` ×4 (계좌 **클래스명**을 id 자리에).
- **[5]~[8] 우리 층 deny ×4 (축자)**:
  > *"Error: [READ-FIRST] this audit only judges fee lines that were READ in this conversation, and these required reads are missing: **get_all_user_accounts_by_user_id, get_bank_account_transactions**. … (**an account's display class name like 'Purple Account' is NOT an account id**) … Their exact callable forms are: `unlock_discoverable_agent_tool(agent_tool_name="get_all_user_accounts_by_user_id_3847")` then `call_discoverable_agent_tool` with that name; `unlock_discoverable_agent_tool(agent_tool_name="get_bank_account_transactions_9173")` then …"*

  로그: `[T2_SG_REQREADS] get_atm_fee_discrepancies denied: missing reads ['get_all_user_accounts_by_user_id', 'get_bank_account_transactions']` ×4 (log:51,53,55,56).
- **[9]~[20]** KB 재검색 ×2 → **[12]/[18]** 두 도구 unlock(각 2회) → **[21]** 사용자 id 문서 재검색.
- **[23] 어시스턴트**: *"Could you please provide me with two of the following pieces of information: your date of birth, email address, phone number, or address?"* → **[24] 손님**: *"my date of birth is 06/15/1989, and my email address is ahmad.razali@gmail.com."*

#### 4.2 [25]~[35] 신원 확인 (우리 층 순서 강제가 정확히 작동)

- **[25]** `get_user_information_by_email` + `verify_identity` 를 **같은 배치**로 발사 → **[26]** 우리 층: *"NOT_VERIFIED — the account record has not been fetched yet, so nothing can be matched. Look the customer up with get_user_information_by_name/by_email/by_id, then call this tool again."* / **[27]** 레코드 회수(`ar72c5d8e3`).
- **[28]** 재호출 → **[29]** *"VERIFIED — 2 identity values match the account record (date_of_birth, email); you may now call log_verification. Its time_verified argument must be the CURRENT time: FIRST call get_current_time, wait for its result, then copy that exact timestamp …"*
- **[30]** `get_current_time` → **[31]** `The current time is 2025-11-14 03:40:00 EST.` → **[32]** `log_verification{…, time_verified:"2025-11-14 03:40:00 EST"}` → **[33] 성공**. ⇒ **gold 074_0 MATCHED**(t7336 t1 의 시간 날조 재발 0).
- **[34]/[35]** `get_all_user_accounts_by_user_id_3847{user_id:"ar72c5d8e3"}` → 4계좌(`chk_…_1` Purple / `_2` Light Blue / `_3` Dark Green / `_4` Evergreen · `current_holdings` 4500/7250/2100/15800).

#### 4.3 [36]~[55] 거래 read × 2회전 + `@last` 왕복 낭비

- **[36]/[37]~[40]** `get_bank_account_transactions_9173` ×4 → 레코드 **33 / 30 / 29 / 29**.
- **[41]** 또 `transactions:"@last:get_bank_account_transactions_9173"` ×4 → **[42]~[45]** 우리 deny ×4:
  > *"Error: [ARGS-FORMAT] the 'transactions' argument could not be read as a JSON array — it arrived as a plain string that is not valid JSON. Re-issue this exact call with 'transactions' as a VALID JSON array …"*

  로그 축자(짝으로 붙는다): `[T2_SG_BYREF] get_atm_fee_discrepancies: '@last:get_bank_account_transactions_9173' 미해석 — isolate(fetch_formalize)가 'transactions' 를 산출하므로 deny 하지 않고 넘긴다` → `[T2_SG_ARGS] … 'transactions' 인자 str 잔류(JSON 파싱실패) → 재송신 요구` (log:208~215, 225~232 = **8쌍**).
- **[46]~[50]** 모델이 4계좌 거래를 **다시** read(같은 33/30/29/29) → **[51]** 또 `@last` ×4 → **[52]~[55]** 또 `[ARGS-FORMAT]` ×4. **왕복 2회전 낭비**(우리 층이 `@last` 를 통과시켜 놓고 바로 다음 술어가 같은 인자를 반려하는 자기모순 배선).

#### 4.4 [56]~[63] comparator 4회 — **격리 서브가 operand 를 산출한 자리**

- **[56]/[58]/[60]/[62]** 모델이 손으로 16행씩 타이핑해 넘겼다. 그 행들은 **원장과 다르다**: 네 계좌에 대해 `date`·`fee_amount`·`withdrawal_amount`·`network` 가 **완전히 동일**하고 id 접두만 `purple/lb/dg/ev` 로 갈린다(예: 넷 다 `{"date":"11/17/2025","transaction_id":"btxn_ar_*_18","fee_amount":2.25,"withdrawal_amount":225.0,"network":"non_rho"}`). 원장의 `btxn_ar_purple_18` 은 **11/18 CIMB(foreign·수수료 줄 없음)** 이고 `btxn_ar_dg_18` 은 원장에 실재하지 않는다 ⇒ 메인 인자는 **통짜 날조**다.
- **그 날조는 채점에 닿지 않았다** — `isolate.mode=fetch_formalize` 가 메인 인자를 버리고 격리 서브의 산출로 **덮어썼다**. 로그 축자(4회 반복):
  ```
  [T2_SG_TRACE] get_atm_fee_discrepancies: iso=True mode=fetch_formalize ctx=['account_class','account_id','transactions']
  [T2_SG_ISOLATE] sub-view: record dump kept whole (7675 chars)
  [T2_SG_ISOLATE] operand-size get_atm_fee_discrepancies.transactions: sub=18 rows · source=33 rows  ⚠MISMATCH
  [T2_SG_ISOLATE] fetch get_atm_fee_discrepancies: 2라운드·getter 1회·operand keys=['transactions']
  [T2_SG_ISOLATE] get_atm_fee_discrepancies: fetch-formalize operand 주입 keys=['transactions']
  [T2_SCAFFOLD_GET] get_atm_fee_discrepancies -> 5
  ```
  네 계좌: `sub=18/source=33` · `sub=14/source=30` · `sub=17/source=29` · `sub=17/source=29` (log:239, 251, 263, 274). 메인이 넘긴 16행과 **전부 다르다** ⇒ 주입 확인.
- **[57]/[59]/[61]/[63] 우리 도구 반환문(판정 행만 발췌·축자)**:

| 계좌 | 반환 축자 | coverage | 합(모든 부호) | 양수만 |
|---|---|---|---|---|
| `_1` Purple | `03f_err +2.50; 07f_err +8.00; 09f_err +10.50; 11f +3.50; 15f_dup +2.50` | `18 of 18 rows were checked (0 could not be verified)` | **27.00** | 27.00 |
| `_2` Light Blue | `03f_err +2.50; 05f_err +2.50; 06f +2.50; `**`07f +2.50`**`; 08f_err +4.00; 09f +4.00; 10 −2.50; 13f +1.50` | `14 of 14 … (0 …)` | 17.00 | **19.50** |
| `_3` Dark Green | `02f_err +1.50; 05f −1.00; 07f +2.00; 12f +4.00; 13 −1.75; `**`17 −2.25`** | `17 of 17 … (0 …)` | 2.50 | **7.50** |
| `_4` Evergreen | `03f_err +1.50; 08f +1.50; 09f −1.00; 11f +3.50; 12 −1.80; `**`17 −2.25`** | `17 of 17 … (0 …)` | 1.45 | **6.50** |

  네 반환 모두 꼬리에 같은 우리 문면이 붙었다(축자):
  > *"SCOPE OF THIS CHECK — … a fee that is MISSING where one was due (**it shows as a negative difference**) … If corrections are owed, the credit policy requires **ONE fee_refund credit for the net correction of THIS account** (do not credit the same lines twice)."*

- **[58]/[60]/[62]/[64]** 모델이 그 표를 **부호까지 정확히** 전사해 손님에게 보고했다. 정보 손실 0.

#### 4.5 [65]~[79] write 구간

- **[65] 손님 (⚠시나리오에 없는 즉흥 지시)**:
  > *"Oh wow, so there **ARE** errors? … Yes, please go ahead and apply credits for **all the incorrect charges**. … And just to confirm — for the ones where the **difference is negative** (where I was charged *less* than the documented fee), please don't "correct" those by charging me more. I'm only asking for credits back for the **overcharges**."*

- **[66]** `unlock_discoverable_agent_tool{apply_checking_account_credit_5829}`. 같은 턴에 우리 층이 **savings 도구 오지목을 정확히 차단**했다(재생성으로 궤적에서 삭제·사이드카 축자):
  > *"Error: [OPERATOR-SCOPE] you called 'apply_savings_account_credit_6831'. The declared scope of the candidate tools is: 'apply_savings_account_credit_6831' = … The savings account ID; 'apply_checking_account_credit_5829' = Apply a credit to a customer's checking account …"*

  로그: `[T2_RESOLVE] operator-scope: 지목 대신 범위 표면화 (apply_savings_account_credit_6831, apply_checking_account_credit_5829)` → `[T2_RESOLVE] deny tool=unlock_discoverable_agent_tool arg=agent_tool_name reason=operator-scope` (log:324,325). **[67] unlock 성공**.
- **[68] 결정 지점 A (Purple)** — 본문 축자: 5줄 나열 후 *"**Total Overcharge: $27.00**"* → `amount:27.0` 발사 → **[69]** `Amount: $27.00 · New Balance: $4527.00`. **gold 일치**.
- **[70] 결정 지점 B (Light Blue)** — 본문 축자: `03f_err 2.50 · 05f_err 2.50 · 06f 2.50 · 07f 2.50 · 08f_err 4.00 · 09f 4.00 · 13f 1.50`, *"**Total Overcharge: $19.50**"*. **`btxn_ar_lb_10 −2.50` 을 목록에서 지웠다** → `amount:19.5`.
- **[72] 결정 지점 C (Dark Green)** — 본문에 남긴 것은 `02f_err 1.50 · 07f 2.00 · 12f 4.00` 뿐, *"**Total Overcharge: $7.50**"*. **음수 3건(`05f −1.00` · `13 −1.75` · `17 −2.25`) 전부 삭제** → `amount:7.5`.
- **[74] 결정 지점 D (Evergreen)** — `03f_err 1.50 · 08f 1.50 · 11f 3.50`, *"**Total Overcharge: $6.50**"* → `amount:6.5`.
- **[76]** 요약 보고 `27.00 / 19.50 / 7.50 / 6.50` → **[77]** 손님이 그 네 수를 그대로 복창 → **[78]** 어시스턴트 확인 → **[79]** `###STOP###`.

---

### 5. 실패가 결정된 지점 (축자 · 각 지점의 3문)

#### 결정점 ①  [56]~[63] — 격리 서브의 operand 가 3/4 계좌에서 틀렸다

- **그 시점 문맥에 필요한 값이 실재했나** — 예. [47]~[50] 에 네 계좌 전 레코드가 있었고, 격리 서브는 자기 getter 로 그것을 다시 읽었다(`[T2_SG_ISOLATE] sub-view: record dump kept whole (7675/6752/6708/6701 chars)` · `getter 1회`).
- **우리 층이 그 값을 어떻게 썼나** — 서브가 형식화한 행 수는 **18 / 14 / 17 / 17**. 원장의 `atm_withdrawal` 수는 **17 / 16 / 16 / 16** 이고 Purple 만 중복 fee 라인(`15f`·`15f_dup`)이 있어 선언상 정답이 **18**이다. ⇒ Purple 만 정확, **LB −2행 · DG +1행 · EV +1행**.
- **선언·엔진이 틀린 것인가 서브가 틀린 것인가** — **서브다.** 원장에서 기계적으로 만든 정본 행집합을 A2 선언 `scaffold_get_tools[8].op` 에 그대로 넣어 `t2_compute.apply_op` 로 돌리면 **4/4 gold 정확 일치**한다(부록 C-1). ⇒ 라이브 반환의 **초과 3행**(`lb_07f +2.50` · `dg_17 −2.25` · `ev_17 −2.25`)은 선언·엔진이 만든 것이 아니라 **서브가 넘긴 행 때문**이다.
- **초과 행의 정체(역산)**
  - **DG·EV**: 라이브 초과 행은 `btxn_ar_dg_17 (charged $0.00, documented fee $2.25)` / `btxn_ar_ev_17 (charged $0.00, documented fee $2.25)` 다. 그런데 원장에는 그 인출의 수수료 줄이 **실재**한다 — `btxn_ar_dg_17f  NON-RHO ATM FEE - 1% (MIN $1.50)  amount: -2.25` · `btxn_ar_ev_17f  NON-RHO ATM FEE - 1% (MAX $2.50)  amount: -2.25`. ⇒ 서브가 **11/17 인출 하나를 두 행으로 냈다**: 수수료 줄 id(`…_17f`, fee 2.25 → 기대 2.25 → 불일치 없음)로 한 번, **인출 자신의 id(`…_17`, fee 0)로 또 한 번**. `duplicate_of` 는 붙지 않았다(붙었다면 `t2_compute.py` 의 `if _dupf0 and str(r.get(idf)) in _dup_zero: exp = 0` 으로 기대값이 0 이 되어 `documented fee $2.25` 가 나올 수 없다). 행 수 16→**17** 과 정확히 일치한다.
  - **LB**: 라이브 초과 행은 `btxn_ar_lb_07f (charged $2.50, documented fee $0.00)` 이고 서브 행 수는 **14**(원장 16). "행 2개만 누락"으로는 라이브가 재현되지 않는다 — 예컨대 수수료 없는 rho 인출 `lb_02`·`lb_04` 를 뺀 14행의 재생 결과는 **정답 7행/net 14.50** 이다(부록 C-2). 라이브를 재현하려면 **`lb_05`/`lb_06`/`lb_07` 중 한 행의 기대값이 0** 이어야 하고, 그 조건은 ⓐ그 행의 `network` 가 `rho` 로 잘못 적혔거나 ⓑ그 행에 `duplicate_of` 가 잘못 붙었을 때만 성립한다(두 변형 전수 탐색으로 해집합이 그 셋으로 닫힌다). ⇒ **어느 쪽이든 서브의 행 오류**이며, 회수된 계기에는 서브의 실제 행이 남아 있지 않아 **셋 중 어느 것인지는 확정 불가**(그 자체가 계기 결손·§6·P-74a).

#### 결정점 ②  [70]/[72]/[74] — 모델이 음수 델타를 뺄셈에서 제외했다

- **그 시점 문맥에 필요한 값이 실재했나** — 예. [59]·[61]·[63] 반환문에 부호가 붙어 있었고, 모델은 [60]·[62]·[64] 본문에 *"**Difference:** $-2.50 / $-1.75 / $-2.25 / $-1.00 / $-1.80"* 를 **자기 손으로 적었다**. 같은 반환문 꼬리에 *"ONE fee_refund credit for **the net correction** of THIS account"* 도 있었다.
- **모델이 그 값을 썼나 안 썼나 바꿔 썼나** — **안 썼다.** [70]/[72]/[74] 의 목록에서 음수 항목이 **전부 사라졌고**, 합계 표기가 *"Total **Overcharge**"* 로 바뀌었다. 네 계좌 모두 **Σ(양수 델타)** 를 그대로 크레딧했다(27.00 / 19.50 / 7.50 / 6.50 = 각 계좌 양수합과 소수점까지 일치).
- **우리 층이 그 자리에서 무엇을 발화했나** — 반환문의 net 지시는 발화했다(위 축자). 그러나 **금액을 검산하는 발화는 0**이다: `[T2_WRITE_SUB] 제안 0건 → 근거검산 통과 0건` (log:343,348,353 = 크레딧 3회 턴) — 서브가 제안 자체를 안 냈으므로 `t2_subcall.grounded_calls` 검산도 돌지 않았다. `T2_STALE_STRIP` 0 · `T2_UNAVAIL` 0 · `T2_LIMIT_REDUCE` 0.
- **[65] 손님 발화의 개입** — *"for the ones where the difference is negative … I'm only asking for credits back for the overcharges"*. 이 문장이 결정점 ② 바로 앞에 있다. ⚠[[21]]: **면책 아님**(정책·우리 반환문 둘 다 net 을 말했고, net 크레딧은 손님에게 "더 청구"하는 것이 아니다). 다만 **그 문장이 나온 재료를 우리 층이 절반 공급했다** — [64] 표의 음수 6행 중 **2행(`dg_17 −2.25`·`ev_17 −2.25`)이 결정점 ①의 허위 음수**다.

⇒ **두 결정점은 각각 단독으로 실패를 확정한다.** ①만 있어도(모델이 net 을 완벽히 계산해도) 라이브 net 은 27.00/17.00/2.50/1.45 로 3/4 불일치. ②만 있어도(서브가 완벽해도) 양수합은 27.00/17.00/7.50/6.50 으로 3/4 불일치. 통과는 **둘 다** 고쳐야 나온다.

---

### 6. 레버 발화표 (이 sim 라인 전수 · `T2_` 마커 48종)

| 마커 | 건수 | 판정 |
|---|---|---|
| `T2_SG_REQREADS` | 4 | ✅**발화·순종**. [4] 날조 4연발 → [12]/[18] unlock. `get_all_user_accounts_by_user_id` **단독** 결손 0건(항상 둘 함께 결손) |
| `T2_SG_ISOLATE` (fetch_formalize) | 16 | ⚠**발화했으나 산출이 3/4 오답**. 메인의 통짜 날조를 덮어쓴 것은 ✅, 덮어쓴 값이 틀린 것은 ❌(§5①) |
| ↳ `operand-size … ⚠MISMATCH` | 4/4 | ❌**계기 무효**. 분모가 `Record ID:` 전수(33/30/29/29)라 **정답인 Purple 에도 MISMATCH** — 판별력 0. 표면화 경로도 사문: `t2_scaffold_get.py:324 _omitted_rows_note` 는 본문 첫 줄이 `return ""`(2026-08-14 의도적 무효화) |
| `T2_SG_BYREF` / `T2_SG_ARGS` | 8 / 8 | ⚠**자기모순 오발화**. BYREF 가 *"isolate 가 산출하므로 deny 하지 않고 넘긴다"* 로 통과시킨 같은 인자를 ARGS 가 *"str 잔류 → 재송신 요구"* 로 반려. **왕복 2회전([41]~[55]) 낭비**를 만들고 모델을 손-전사(=[56] 날조)로 몰았다 |
| `T2_SG_TRACE` / `T2_SCAFFOLD_GET` | 6 / 16 | 발화(계기·주입 명단 + 판정 4회) |
| `T2_COMPUTE` | **0** | ⚠**"모른다"**([[67]]: 성공 경로 마커 없음). 실제로는 돌았다 — [57]/[59]/[61]/[63] 반환문이 그 산출물이다. halfB 로그 **전체**에도 `[T2_COMPUTE]` 0줄 = 회수 실패 |
| `T2_RESOLVE` operator-scope | 1 (+deny 1) | ✅**발화·정확**. `apply_savings_account_credit_6831` 오지목 차단 → [67] 정지목 unlock |
| `T2_RESOLVE` action-required | 6 | 발화(`target=call_discoverable_agent_tool` · 전부 정지목) |
| `T2_RESOLVE` **user-action instruct** | **0** | ✅**미발화 = 개선**. t7346 t1 붕괴 원인(`target=submit_transaction`)이 이 궤적에 0건(`grep -c "user-action" = 0`) |
| `T2_WRITE_SUB` | 25 (제안 합 27 · 통과 6) | ⚠**결정점에서 침묵**. 크레딧 3턴(log:343,348,353) 전부 `제안 0건` — 금액 검산 기회 0 |
| `T2_CLAIMPROV` | 31 | ✅발화. `ledger narrowed: 12 failed call(s) excluded [get_atm_fee_discrepancies ×4]` · 마지막 두 창 `unbacked=0 pending=0` — 허위 완료 주장 0 |
| `T2_PROV` | 3 | ✅발화(`account_id='account_id_1'` turn2 · `customer_name='John Doe'` turn21·23) — 전부 재생성으로 교정 |
| `T2_MATERIAL_GATE` / `T2_FORCE_ACTION` | 7 / 7 | 발화 |
| `T2_DECISION_CARRY` | 6 | ⚠발화하되 **39,758자** 부착 1회(turn 66 직전·log:321) |
| `T2_EPLAN_LISTED_IS_READ` | 15 | ⚠**오배송 의심**. 15회 전부 `배달된 121건 … (btxn_ar_dg_01,…)` 로 **Dark Green 명단 고정** — Purple/LB/EV 판정 턴에도 같은 문자열 |
| `T2_REDERIVE` | 1 | ⚠**오발화**. `raw='Bluest' → Bluest` — 이 손님의 4계좌는 Purple/Light Blue/Dark Green/Evergreen 이고 Bluest 는 [35] 계좌 목록에 없다 |
| `T2_LLM_DIAG` `CWE escaped` | 2 | ⚠재생성에서 컨텍스트 초과 탈출 2회(`T2_DYN_MT shrink 8192→6520 / →5956`) |
| `T2_UNCALLED_UNLOCK` / `T2_PAIRFIX` / `T2_KIND` | 1 / 1 / 2 | 발화 |
| `T2_STALE_STRIP` · `T2_UNAVAIL` · `T2_LIMIT_REDUCE` · `T2_ACTION_SUB` · `FAB_STRIP` · `T2_REQUIRE_DOC_DELIVER` · `T2_SG_DOCS` · `T2_ABSTAIN*` · `T2_PIN_READ` · `T2_ARG_PRODUCERS` · `T2_BLOCK_NOTE` · `T2_SG_SCHEMA` · `T2_DEMANDED_STEP` · `T2_FOLLOWUP` | **0** | 미발화(검색 = `grep -c` 로 이 sim 396줄 전수). 그중 **`T2_SG_DOCS` 0** 은 t7346 판정 그대로 재현(이 comparator 선언에 `isolate.docs` 키가 없다) · **`FAB_STRIP` 0** 은 [56] 통짜 날조가 있었는데도 0(fetch_formalize 가 덮어써 결과적으로 무해) |

(전수 집계는 부록 B. 방법 = `grep -o "\[T2_[A-Z0-9_]*\]"` · 396줄 · 48종.)

---

### 7. 선행 대조 (074 자신의 보고서만)

읽은 것: `reports/facet_rft_2026/tasks__20260822/TASK_074.md`(t7346 판) · `reports/facet_rft_2026/t7336_tasks/T7336_TASK_074.md`.

| 축 | t7336 t1 | t7346 t1 | **t7348 t1 (본 조사)** |
|---|---|---|---|
| 감사한 계좌 | 4 | **2** (모델이 재송신에서 2계좌 절단) | **4** |
| 크레딧 실행 | 0 | **0** | **4** |
| 변이 | MISSING 4 | MISSING 4 · WRONGARG 0 | **MATCHED 2 · MISSING 3 · WRONGARG 3** |
| 종료 | 0.0 | `###TRANSFER###`(격노) | `###STOP###`(정상 종료) |
| 확정 원인(선행 축자) | *"우리 `T2_UNAVAIL` 이 궤적에 0회 등장하는 유령 이름 `apply_credits_to_account_1234` 위에서 UNAVAILABLE 판정"* / *"(숨은 축) 금액 4/4 불일치 … comparator 가 rebate 축을 명시 기권 + Light Blue `oon/forx=null`"* | *"**trial 1 = our_layer(주)**. 판정까지는 정상이었고, 손님 승인 직후 우리 지목 채널(OL-D)이 **비존재 도구로 몰아** 실행 자체를 0 으로 만들었다"* | **다르다** — §8 |

**⇒ 같은 원인인가? 아니다. 세 가지가 실제로 바뀌었다.**

1. **OL-D 소멸(확정)** — t7346 t1 의 결정타였던 `[T2_RESOLVE] user-action instruct target=submit_transaction` 이 이 로그에 0건이다. 대신 같은 자리에서 `[T2_RESOLVE] operator-scope` 가 savings 오지목을 막고 checking 도구를 범위로 냈다 → **write 문턱을 처음으로 넘었다**.
2. **OL-A 소멸(확정)** — t7346 판 축자 *"`steps.oon.cases["Light Blue Account"] = null`(L3356) … `[coverage] 1 of 13 rows were checked (12 could not be verified)`"*. t7348 의 같은 키는 `lookup_table{key:"steps.ord", table:[{"<=",2,0},{">",2,2.5}]}` (`a2/banking_knowledge.specific.json:3377` · `forx` 는 :3451)이고 **coverage 는 `14 of 14 … (0 could not be verified)`** 다. LB 판정 보류 12행 → **0행**.
3. **Purple 금액이 처음으로 gold 와 일치** — t7346 comparator net 24.50 → t7348 **27.00**. 새 `op.rebate{field:"rebate_amount", cap:{Purple Account:30.0, Bluest Account:50.0}}` 선언(`_note_rebate` 축자: *"expected_net = 문서 요율 − min(문서 요율, 남은 월 상한) · actual_net = 부과액 − 실제 환급액"*)이 `purple_11f` 를 `+1.00` 에서 `+3.50` 으로 올려 5행 합을 27.00 으로 맞췄다.

**결손이 두 단계 더 하류로 옮겨 갔다**: t7335 *read 를 안 한다* → t7336 *판정하다 ctx 사망/유령 도구* → t7346 *판정은 하는데 지목이 무너져 실행 0* → **t7348 *실행까지 완주, 금액 1/4 정확 — 남은 것은 서브 operand 3건과 모델의 netting***.

⚠**선행 보고와 갈리는 지점** — t7346 판 §5 는 `T2_SG_ISOLATE` 를 *"✅정상. 모델의 행 날조를 원장 재취득으로 무해화"* 로 적었고 `⚠MISMATCH` 를 *"분모가 레코드 전수라 생기는 표기일 뿐"* 이라고 넘겼다. **t7348 에서는 그 판정이 성립하지 않는다** — 같은 서브가 3/4 계좌에서 틀린 행을 냈고, MISMATCH 계기는 정답인 Purple 에도 붙어 그 사실을 가렸다(§6·§8 OL-74b).

---

### 8. 원인 ([[77]] 네 칸 계약)

#### CONFIRMED · 우리 층 — OL-74a. `get_atm_fee_discrepancies` 격리 서브(fetch_formalize)의 `transactions` 형식화가 4계좌 중 3계좌에서 틀렸다

**① 주장+양화** — sim `task_074#s373753`(t7348 halfB) 단일. 격리 서브 호출 **4회 중 3회**(`chk_ar72c5d8e3_2/_3/_4`)에서 산출 행집합이 원장과 다르고, 그 결과 comparator net 이 gold 와 각각 **+2.50 / −2.25 / −2.25** 어긋났다. Purple(`_1`)은 정확했다. *전칭 금지* — 다른 sim·다른 태스크로 확장하지 않는다.

**② 근거(축자 + 파일:줄 / sim#turn)**
- 행 수 축자 — `[T2_SG_ISOLATE] operand-size get_atm_fee_discrepancies.transactions: sub=18 rows · source=33 rows ⚠MISMATCH` / `sub=14 … source=30` / `sub=17 … source=29` / `sub=17 … source=29` (로그 239·251·263·274줄 = `sim#turn` 57·59·61·63 직전). 원장 `atm_withdrawal` 수 = **17 / 16 / 16 / 16**(선언상 정답 = **18 / 16 / 16 / 16**, Purple 만 `15f`/`15f_dup` 중복 1행 가산).
- 반환 축자 — `sim#turn 61`: *"btxn_ar_dg_13 (charged $0.00, documented fee $1.75, difference $-1.75); **btxn_ar_dg_17 (charged $0.00, documented fee $2.25, difference $-2.25)**"*. 그런데 같은 sim `#turn 49` 원장에 *"Record ID: **btxn_ar_dg_17f** … description: NON-RHO ATM FEE - 1% (MIN $1.50) … amount: **-2.25**"* 가 실재한다 ⇒ 그 인출은 부과 0 이 아니다.
- **대안 가설 기각(재생 측정)** — "선언·엔진이 원인" 가설을 기각했다: 원장에서 기계적으로 만든 정본 행집합을 `a2/banking_knowledge.specific.json` `scaffold_get_tools[8].op` 그대로 `t2_compute.apply_op` 에 넣으면 **27.00 / 14.50 / 4.75 / 3.70 = gold 4/4 일치**(부록 C-1). 남는 원인은 operand 뿐이다.
- 선언 위치 — `a2/banking_knowledge.specific.json:3519` `isolate.instructions` 축자 *"Build the transactions array with **ONE element per atm_withdrawal record** of THIS account … Include EVERY atm_withdrawal, also the ones with no fee line"* · `:3520 answer_format` · `:3531 operand_schema`. 이 `isolate` 블록의 키 전수 = `['mode','ref_params','getter_tools','operand_keys','max_rounds','instructions','answer_format','_note','row_fields','_note_row_fields','operand_schema','_schema_note']` — `ground` 키가 목록에 없다(이 배열 operand 에는 축자-인용 검산이 걸려 있지 않다).

**③ 반증 조건 (refutation · 주장과 동시에)** — ⓐ원장 그대로의 `atm_withdrawal` 1행/1건 행집합을 넣었을 때 comparator 가 gold 와 다른 net 을 내면 이 주장은 거짓이 되고 원인은 선언·엔진이다. ⓑ서브의 실제 산출 행이 회수돼 LB=16·DG=16·EV=16 행이었음이 확인되면 거짓이 된다. ⓒ`sub=17` 의 초과 행이 `duplicate_of` 가 붙은 정당한 중복이었다면 거짓이 된다(그 경우 기대값이 0 으로 강제돼 `documented fee $2.25` 가 나올 수 없으므로 현재 관측과 모순).

**④ 선행 확인(검색한 경로 나열)** — `reports/facet_rft_2026/tasks__20260822/TASK_074.md` · `reports/facet_rft_2026/t7336_tasks/T7336_TASK_074.md` · `grep -rln "get_atm_fee_discrepancies" reports/facet_rft_2026 scripts/distill/tau2`(38 파일) · `scripts/distill/tau2/t2_scaffold_get.py:324,956,2122,2507` · `scripts/distill/tau2/t2_compute.py:832-1050` · `scripts/distill/tau2/a2/banking_knowledge.specific.json:3293-3560` · `scripts/distill/tau2/t2_subcall.py` · `scripts/distill/tau2/t2_resolve.py:615-660` · `ls reports/facet_rft_2026/tasks__20260824/`. 선행 보고는 이 서브를 *"✅정상"* 으로 적고 있었다 ⇒ **소급 정정 대상**.

#### CONFIRMED · 우리 층 — OL-74b. 그 오류를 잡을 수 있었던 두 계기가 무효/사문 상태다

**① 주장+양화** — 같은 sim, n=4 호출. ⓐ`operand-size` 마커는 4/4 호출에서 `⚠MISMATCH` 를 찍어 **정답(Purple)과 오답 3건을 구분하지 못한다**(판별력 0) ⓑ그 두 수를 반환문으로 올리는 경로 `_omitted_rows_note` 는 **함수 본문 첫 줄이 `return ""`** 이라 모델에게 아무것도 전달되지 않는다.

**② 근거(축자 + 파일:줄)** — `scripts/distill/tau2/t2_scaffold_get.py:956` `"⚠MISMATCH" if _src_rows and len(_v) != _src_rows else ""` (분모 `_src_rows` = `_t.count("Record ID:")` = 레코드 전수) · 같은 파일 `:324~352` `_omitted_rows_note` 본문 축자 *"⛔**무효화 (2026-08-14 야간·같은 날 출시분의 자기 반증)**: 분모가 틀렸다. … 되살리려면 **서브가 산출해야 할 모집단**과 비교 가능한 분모가 필요하다 … 그 선언이 생기기 전까지는 **말하지 않는다**"* → `return ""`. 궤적 증거: 네 반환문 coverage 가 `18 of 18` / `14 of 14` / `17 of 17` / `17 of 17` 로 전부 "0 could not be verified" — 분모가 서브 자신이 넘긴 수라 **자기를 잰다**.

**③ 반증 조건 (refutation)** — 반환문 어딘가에 "원장 `atm_withdrawal` 수와 넘긴 행 수의 차"가 실려 나간 흔적이 발견되면 ⓑ가 거짓이 된다. `operand-size` 가 정답 Purple 에는 MISMATCH 를 안 찍었음이 확인되면 ⓐ가 거짓이 된다.

**④ 선행 확인** — `grep -n "operand-size|_omitted_rows_note|MISMATCH" scripts/distill/tau2/t2_scaffold_get.py` · `tasks__20260822/TASK_074.md` §5(그 보고는 이 무효화를 *"옳음"* 으로 판정했고 판별력 문제는 그때 제기되지 않았다).

#### CONFIRMED · 모델 — M-74a. 음수 델타를 뺄셈에서 제외하고 Σ(양수)만 크레딧했다

**① 주장+양화** — 같은 sim, 크레딧 **4/4** 에서 보낸 금액이 그 계좌 반환문의 **양수 델타 합**과 소수점까지 일치(27.00 / 19.50 / 7.50 / 6.50). 반환된 음수 델타 7건(`lb_10 −2.50` · `dg_05f −1.00` · `dg_13 −1.75` · `dg_17 −2.25` · `ev_09f −1.00` · `ev_12 −1.80` · `ev_17 −2.25`)이 전부 제외됐다.

**② 근거(축자 + sim#turn)** — `sim#turn 72` 축자: 반환문에 6행이 있었는데 본문은 *"btxn_ar_dg_02f_err — Overcharge: $1.50 / btxn_ar_dg_07f — $2.00 / btxn_ar_dg_12f — $4.00 / **Total Overcharge: $7.50**"* 로 음수 3행이 사라졌다. 그 값들은 바로 앞 `sim#turn 62` 에서 **모델 자신이** *"**Difference:** $-1.00 / $-1.75 / $-2.25"* 로 적었다. 우리 반환문 축자(`#turn 61` 꼬리): *"the credit policy requires ONE fee_refund credit for **the net correction** of THIS account"*.

**③ 반증 조건 (refutation)** — 네 크레딧 중 하나라도 Σ(양수)와 다르면 이 주장은 거짓이 된다. 반환문에 음수 부호가 실리지 않았거나 net 지시가 없었다면 귀속이 "값이 문맥에 실재하지 않았다"로 바뀐다(관측은 반대).

**④ 선행 확인** — `tasks__20260822/TASK_074.md` §7.2②(*"음수 델타 폐기 (양 trial)"*) — **같은 실패가 재발했다** · `t7336_tasks/T7336_TASK_074.md` §6.

#### 기여 · user-sim — U-74a. 시나리오에 없는 "음수는 빼고 과다분만" 지시가 결정점 바로 앞에 들어왔다

**① 주장+양화** — `sim#turn 65` 1건. `task['user_scenario']['instructions']` 5번에는 *"Yes, please go ahead and apply credits for all the incorrect charges. I want to make sure everything is corrected."* 까지만 있고 뒷문장은 그 축자에 포함돼 있지 않다.
**② 근거(축자)** — `#turn 65`: *"for the ones where the **difference is negative** … please don't "correct" those by charging me more. I'm only asking for credits back for the **overcharges**."* · 대조군 = `task` 원문(부록 A).
**③ 반증 조건 (refutation)** — 같은 문장이 `user_scenario.instructions` 축자에 존재하면 "즉흥"이라는 서술이 거짓이 된다. t7346 t1 `#turn 43` 에도 유사 문장이 있으므로 1회성이 아니라 **재현적 즉흥**이다.
**④ 선행 확인** — `tasks__20260822/TASK_074.md` §7.4. **[[21]] 적용**: 종결 카테고리로 쓰지 않는다 — net 크레딧은 "추가 청구"가 아니므로 손님 요구와 gold 는 모순이 아니었고, 흡수 실패는 agent 측이다. ⚠단 그 발화의 **재료 절반은 우리 층이 만들었다**(허위 음수 `dg_17`·`ev_17`).

#### env — 기여 0
반환은 전부 정상(`Credit applied successfully!` ×4 · 레코드 33/30/29/29 일관). deny 는 전부 우리 층 접두(`[READ-FIRST]`·`[ARGS-FORMAT]`·`[PROVENANCE]`·`[POLICY GATE …]`·`[OPERATOR-SCOPE]`). 검색 경로 = 이 sim 80 메시지 전수 정독 + `fb_*.jsonl.gz` 66행.

#### 주 원인 배분
- **our_layer** — OL-74a 가 없었다면 모델이 완벽해도 net 27.00/17.00/2.50/1.45 로 3/4 불일치(천장 존재).
- **model** — M-74a 가 없었다면 서브가 완벽해도 양수합 27.00/17.00/7.50/6.50 으로 3/4 불일치.
- ⇒ **두 원인은 각각 충분조건**이다. 한 주체로 접히지 않는다. 지분: **our_layer 45 · model 45 · user_sim 10 · env 0**.

---

### 9. 처방 후보 (설계만 · 수리·코드 수정 0 · [[62]] 순서 · [[70]] ± 공개)

> ⛔전부 **후보**다. [[62]]① 대로 격리 프로브로 결손을 먼저 재고, 격리에서 되면 레버는 **전달뿐**이다.

1. **P-74a (계기 먼저·비용 0) — 서브의 산출 행을 회수 가능하게 남긴다.**
   지금은 `operand-size` 두 수만 stderr 에 있고 행 자체가 어디에도 남지 않아, LB 오류가 "network 오라벨"인지 "`duplicate_of` 오부착"인지 판정할 방법이 없다(§5①·검색한 곳 = `fb_*.jsonl.gz` 66행 · `trace_*.jsonl.gz` 395행 · 로그 396줄). `t2_scaffold_get.py:956` 자리에서 산출 operand 를 사이드카에 **행 그대로** 남기기만 한다(판정 0·도메인 어휘 0). ± 파는 것: 사이드카 용량(계좌당 ~16행 × 6필드). **이것 없이는 P-74b 설계가 추측이 된다.**
2. **P-74b (OL-74a·측정 선행) — 서브의 모집단을 서브 자신이 선언하게 한다.**
   `answer_format`/`operand_schema` 에 `withdrawal_count`(자기가 센 `atm_withdrawal` 레코드 수)를 **필수 필드**로 넣고, `len(transactions) != withdrawal_count + (duplicate_of 개수)` 면 **서브에게만** 재질의(`max_rounds=3` 안에서·메인 노출 0). 이것이 `_omitted_rows_note` 주석이 축자로 요구한 *"서브가 산출해야 할 모집단과 비교 가능한 분모"* 다.
   ± 파는 것: 서브 라운드 증가 → 지연·토큰. 서브가 count 도 틀리면 재질의 낭비(라운드 상한이 이미 유계).
   부정통제([[57]]): 같은 컷에서 count 요구 없는 팔이 정말 3/4 오답을 재생산하는가 · **정답 팔(Purple)이 count 추가로 깨지지 않는가**.
   ⛔**gold 금지선**: 27.00/14.50/4.75/3.70 을 보고 임계·요율·행 선택을 조정하면 [[23]] 위반. 이 처방은 **행 수의 자기정합성**만 보고 금액을 보지 않는다.
3. **P-74c (OL-74a 대안·[[63]] 빼기) — `transaction_id` 계약을 "인출 id 기준"으로 단순화.**
   현행 계약은 *"수수료 줄의 id, 없으면 인출 자신의 id"* 라 **한 인출이 두 이름을 가질 수 있다** — DG/EV 의 +1행이 정확히 그 틈에서 났다(같은 인출을 `…_17f` 와 `…_17` 로 두 번). `transaction_id` 를 **항상 인출 id**로 두고 수수료 줄 id 를 별도 필드로 빼면 중복 방출이 **구조적으로 불가능**해진다.
   ± 파는 것: `detail_item_template` 이 출력하는 id 가 바뀌어 모델 보고 문면과 원장 대조가 한 칸 멀어진다. Purple 의 `15f_dup` 표기가 사라진다.
4. **P-74d (OL-74b) — `operand-size` 분모를 갈거나 마커를 내린다.**
   현재 분모(`Record ID:` 전수)는 **정답에도 MISMATCH** 를 찍는다. P-74b 가 들어오면 분모를 `withdrawal_count` 로 갈아 끼우고, 안 들어오면 **MISMATCH 표기를 지우는 편이 낫다**(틀린 경보는 침묵보다 나쁘다 — `_omitted_rows_note` 주석이 이미 같은 결론을 적고 있다).
5. **P-74e (M-74a·[[62]]② 전달 축) — 격리 프로브 먼저.**
   같은 컷(4계좌 반환문 확보 직후 + `#turn 65` 손님 문장 포함)에서 ⓐ반환문만 → 모델이 14.50/4.75/3.70 을 내는가 ⓑ반환문 + *"net = 모든 difference 의 합(음수 포함)"* 한 줄 → 내는가 ⓒ손님 문장 제거 통제. **ⓐ가 되면 레버는 전달(문면 강화)뿐이고 엔진 산수는 불필요**하다. ⛔[[62]]: 우리가 net 을 계산해 반환문에 실으면 그것은 **채점되는 인자 그 자체**를 떠먹이는 것이다(t7346 판 §8-6 의 `delta_total` 금지선과 같은 계보) — 하지 말 것.
6. **P-74f (BYREF/ARGS 자기모순·저비용)** — `[T2_SG_BYREF]` 가 *"isolate 가 산출하므로 넘긴다"* 로 통과시킨 인자를 같은 턴 `[T2_SG_ARGS]` 가 반려하지 않게 한다(8쌍 발화·왕복 2회전 낭비·손-전사 유도). ± 파는 것: isolate 가 없는 선언에서 `@last` 가 조용히 통과 → **isolate 보유 선언에 한해서만** 조건화.
7. **관측 등재(무비용)** — `T2_SG_DOCS 0` · `FAB_STRIP 0` · `T2_COMPUTE 0(=모른다·회수 실패)` · `T2_EPLAN_LISTED_IS_READ` 의 DG 명단 고정 · `T2_REDERIVE raw='Bluest'` 오발화를 `TASK_LEVER_MAP_AND_EXCLUSIONS` 에 등재해 다음 런 귀속 혼선을 막는다.

---

### 부록 A — 대조에 쓴 gold/시나리오 축자

- `task['user_scenario']['instructions']` 5번 축자: *"**If agent offers to apply credits:** "Yes, please go ahead and apply credits for all the incorrect charges. I want to make sure everything is corrected. If the agent does not offer, then get angry and request to speak to a real human.""* — `#turn 65` 뒷문장은 이 축자에 포함돼 있지 않다.
- `task['description']['notes']` 축자(발췌): *"**NET CREDITS**: Purple = $27.00, Light Blue = $14.50 ($17.00 overcharges minus $2.50 missing fee), Dark Green = $4.75 ($7.50 overcharges minus $2.75 undercharges/missing), Evergreen = $3.70 ($6.50 overcharges minus $2.80 undercharges/missing)."*
  ⚠이 축자가 M-74a 를 이중으로 확증한다 — 모델이 보낸 **7.50 / 6.50** 은 notes 가 말하는 **"overcharges" 값 그 자체**이고, gold 는 거기서 undercharge 를 뺀 값이다.

### 부록 B — `T2_` 마커 전수 (396줄 · 48종)

`T2_A2_VARIANT 52 · T2_CLAIMPROV 31 · T2_WRITE_SUB 25 · T2_LEVER 22 · T2_STACK 21 · T2_SG_ISOLATE 16 · T2_SCAFFOLD_GET 16 · T2_EPLAN_LISTED_IS_READ 15 · T2_WINDOW 13 · T2_ACTIONREQ 13 · T2_VIEW_COMPACT 12 · T2_SEARCH_AGENT 11 · T2_RESOLVE 9 · T2_GROUPORDER 9 · T2_DOCGROUP 9 · T2_SG_BYREF 8 · T2_SG_ARGS 8 · T2_PHASE_PRECEDE 8 · T2_MATERIAL_GATE 7 · T2_FORCE_ACTION 7 · T2_DISCOVERY_STEP2 7 · T2_SG_TRACE 6 · T2_DECISION_CARRY 6 · T2_AXIS 6 · T2_ARG_DOC_SUB 6 · T2_RESOLVE_CAP 5 · T2_DISCOVERY_NAMES 5 · T2_SG_REQREADS 4 · T2_SELFDECL 4 · T2_PROV 3 · T2_OUR_NAMES 3 · T2_FB_VIEW 3 · T2_SEARCH_REARM 2 · T2_SEARCH_EXHAUST 2 · T2_LLM_DIAG 2 · T2_LEDGER 2 · T2_KIND 2 · T2_EPLAN 2 · T2_DYN_MT 2 · T2_UNCALLED_UNLOCK 1 · T2_SEARCH_ON_PROCEED 1 · T2_REDERIVE 1 · T2_PAIRFIX 1 · T2_OBJ_AXIS 1 · T2_NOW_SELFCALL 1 · T2_KB_NOHIT_SURFACE 1 · T2_DOCDECIDE 1 · T2_CP2_APPEND 1 · T2_ACTION_INDEX 1 · T2_ACTION_HISTORY 1`

### 부록 C — 측정용 재생 (진단 전용 · 수리 0)

**C-1. 정본 행집합 → 선언 그대로 실행.** `a2/banking_knowledge.specific.json` `scaffold_get_tools[8].op` 를 `t2_compute.apply_op` 에 그대로 넣고 `transactions` 만 [47]~[50] 원장에서 기계적으로 구성(1 `atm_withdrawal` = 1행 · 같은 날짜 2번째 `atm_fee` 는 `duplicate_of` · `rebate_amount` = 같은 날짜 `fee_rebate` 합 · `network` = 인출 description 의 `RHO-BANK`/해외 도시 토큰).

| 계좌 | rows | stats | 재생 net | gold | 판정 |
|---|---|---|---|---|---|
| `_1` Purple | 18 | `judged 18 · skipped 0` | 27.00 | 27.00 | **MATCH** |
| `_2` Light Blue | 16 | `judged 16 · skipped 0` | 14.50 | 14.50 | **MATCH** |
| `_3` Dark Green | 16 | `judged 16 · skipped 0` | 4.75 | 4.75 | **MATCH** |
| `_4` Evergreen | 16 | `judged 16 · skipped 0` | 3.70 | 3.70 | **MATCH** |

재생이 낸 판정 행(축자): Purple `03f_err +2.50, 07f_err +8.00, 09f_err +10.50, 11f +3.50, 15f_dup +2.50` · LB `03f_err +2.50, 05f_err +2.50, 06f +2.50, 08f_err +4.00, 09f +4.00, 10 −2.50, 13f +1.50` · DG `02f_err +1.50, 05f −1.00, 07f +2.00, 12f +4.00, 13 −1.75` · EV `03f_err +1.50, 08f +1.50, 09f −1.00, 11f +3.50, 12 −1.80`.

**C-2. LB 역산(해집합 닫기).** 정본 16행에서 **아무 2행이나 빼는** 120가지 중 라이브 8행 출력을 재현하는 것은 **0개**다(예: 수수료 없는 rho 인출 `lb_02`·`lb_04` 를 뺀 14행 → 정답 7행/net 14.50). "2행 누락 + `network` 1건 오라벨"로 넓히면 해가 나오고 **전부 `btxn_ar_lb_07: non_rho→rho`** 다(84 조합·오라벨 대상 유일). "2행 누락 + `duplicate_of` 1건 오부착"으로 넓히면 대상이 `{lb_05, lb_06, lb_07}` 로 닫힌다. ⇒ **어느 경우든 서브의 행 오류**이며, 셋 중 어느 것인지는 회수된 계기로 확정할 수 없다(⇒ P-74a).

**C-3. 스크립트 위치(scratchpad·repo 미커밋)** — `%TEMP%\claude\C--workspace\f479519c-04ec-4837-ae8d-b13afbda47c8\scratchpad\replay074.py` · `lb_search.py` · `t074_s373753.log` · `t074_s373753_traj.txt`.
