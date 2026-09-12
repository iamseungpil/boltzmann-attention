# 설계서 — 레버가 얻는 축과 잃는 축을 레버 안에서 가른다 (2026-09-12)

리뷰 후 구현. 팔 하나에 변경 하나, nt=4, 판정은 축의 양끝을 함께 본다.

## 0. 전제와 원칙

이 문서가 서는 사실(전부 실측, 출처는 `CLAUDE.md` 와 `docs/` 의 포렌식):

- 한 레버가 얻는 태스크와 잃는 태스크는 **같은 축의 양 끝**이다. 태스크별로 맞추면 반대편을 잃는다.
- 손해로 확정된 사례는 전부 ① 문구의 지시 ② 잘못된 시점 ③ 개입 총량 ④ 낱말로 판정한 의미론 ⑤ 인과 없는 무조건 호출 중 하나였다.
- 사실 진술은 손해로 지목된 적이 없다. **재생성**이 손해였다.

지키는 원칙(메모리 `94`, canon):

1. 필요한 시점에 최소로 개입한다. 최소는 0 이 아니다(빈 응답은 재호출을 부른다).
2. 문구는 관측만 담는다. 명령형·규칙·프레임은 넣지 않는다.
3. 의미론은 LLM 만 판단한다. 엔진은 자기가 계산한 값과 이름의 정확 일치로만 분기한다.
4. 도구는 그것이 의미를 갖는 **사건 뒤에** 넣는다. 낱말이 아니라 실행 기록이다.
5. gold 를 보고 조건을 정하지 않는다. 근거는 env 구조·base 행동 분포·우리 사이드카.
6. 팔 하나에 변경 하나. 격리 실험이 배선과 같은 함수를 쓴다.

## 1. 축 — 판정은 이것으로 읽는다

| 축 | 한쪽 | 다른쪽 | 대표 |
|---|---|---|---|
| A 요청의 종류 | 행동형 (`Close …`) | 조사형 (`Check …`) | `[LEDGER]` 049 +46 / 019 −84 |
| B 대화의 종류 | 다단계 계좌 작업 | 즉시 처리 · 카드 쇼핑 | `verify_identity` 066 +75 / 003 −27 |
| C 다음 수의 주인 | 어시스턴트가 부르는 gold | **손님이 부르는 gold** (손실 17 중 6) | 049 016 019 081 098 023 |
| D 전달 방식 | 쪽지 (사실) | **재생성** | `[ORDER]` +35 / 같은 쪽지 유예로 004 0/4 |
| E 도구 결과 | 사실 (적격 여부) | 프레임 · 지시 | `this dispute` 036 0/17 · `you may now call` 016 |

모든 팔의 보고는 손실16 합계 옆에 **축 A·B·C 의 양끝 태스크 묶음**을 따로 적는다. 한쪽만 오르면 그 레버는 축을 못 가른 것이다.

## 2. 조치 — 여섯 개, 각각 팔 하나

구현 순서: **M1 → M5 → M2 → M3 → M4**. M6 은 보고 형식이라 즉시.

---

### M1. advice 는 재생성하지 않는다 — 사실은 다음 턴에 얹는다

**문제.** advice 는 재생성으로만 모델에 닿는다(`lb_runtime.py turn_hook`, `ROUNDS=3`). 재생성 자체가 손해다: 0회 58% → 1회 53% → 2회 45% → 3회 32% (1,336 sim). advice-only 359 sim 49%, `[CLAIM-PROVENANCE]` 226 sim 38%. hand-off 유예는 41 sim 29% vs 무유예 127 sim 42%, 004·008·012·014 패 9/9. 같은 `[ORDER]` 쪽지가 재생성 없이 닿으면 016 +35.

**설계.**

- 재생성은 **deny 에만** 남긴다. 거부된 호출은 실행되면 안 되므로 메시지를 버릴 이유가 있다.
- advice 는 버리지 않는다. 모델의 메시지는 그대로 서고, 쪽지는 **다음 생성의 view 앞에** 얹는다(`ADVICE_MARK` 로 화자 표시 유지). 텍스트-전용 턴이든 hand-off 턴이든 같다.
- 마지막 턴(모델이 떠나는 턴)에는 다음 생성이 없으므로 쪽지가 닿지 않는다. 이것은 **의도된 손실**이다 — 그 턴의 개입이 손해였다(`leaving_after` 실측, 004·049). `[LEDGER]` 의 존재 이유가 "떠나는 순간 한 번"이었으므로 M1 이후 `[LEDGER]` 는 사실상 다음 턴 쪽지가 된다. 이것이 축 C 의 손해(손님 턴 삭제)를 구조적으로 없앤다.

**코드 지점.** `lb_runtime.py`
- `turn_hook`: `if not d.denies and not (d.advice and not turn.calls): break` → `if not d.denies: break` 로. advice 가 있으면 `agent._lb_pending_advice` 에 쌓는다.
- `generate()` 호출 직전: `_lb_pending_advice` 가 있으면 `view += [UserMessage(ADVICE_MARK + text)]` 로 얹고 비운다.
- `REGEN_BUDGET`·`ROUNDS` 는 deny 전용이 되므로 그대로 둔다. nc14 의 게이트는 폐기(M1 이 상위 해법).

**선언.** 없음. 동작 변경이며 A2 는 손대지 않는다.

**엔진이 읽는 것.** `d.denies` 의 존재 여부뿐.

**fail-safe.** 쌓인 advice 는 다음 생성에서 반드시 소비·기록(`lb-advice-carried`). 대화가 끝나 소비되지 않은 것은 `lb-advice-dropped` 로 사이드카에 남긴다.

**검증.**
- 격리: 기존 궤적 재생 — 재생성이 일어났던 562 sim 중 advice-only 였던 것을 세어 "M1 이면 재생성 0" 이 되는 수와, 마지막 턴에 걸려 드롭될 쪽지 수를 미리 낸다.
- 팔: nc18(hand-off 무유예) 위에 M1 을 얹어 **nc19**. 판정 태스크 = 축 D 양끝: 004 008 012 014 040 005 037 (재생성이 해였던 곳) + 016 049 036 (`[ORDER]` 가 득이던 곳). 016·049·036 이 떨어지면 쪽지가 다음 턴에 얹혀서는 부족한 것이고, 그때만 "마지막 턴에 한해 재생성" 을 별도 팔로 잰다.

**되돌림.** 한 줄 조건문과 pending 큐 — `git revert` 로 원복.

---

### M5. 통과는 한 낱말, 실패는 전문 (`verify_identity`)

**문제.** nc17/nc18 은 `hidden:false` 라 328자 전문이 돌아왔다: `VERIFIED — … you may now call log_verification. Its time_verified argument must be …`. 016 이 네 팔 일관 손해(nc9 없음 3/4 · nc17 328자 2/4 · nc12 8자 1/4 · nc11 0자 1/4). 0자는 재호출(2~4회)을 부른다.

**설계.** nc12 의 판정 기반 침묵을 nc18 계열로 옮긴다.
- `lb2_decision._match_verdict`: `ctx["_verdict"] = "met"|"unmet"` (엔진의 산술이 곧 판정).
- `run_tool` 이 `(text, err, ids, verdict)` 를 돌려주고, `execute` 에서 `hidden and verdict=="met"` 이면 `ok_text`(`"VERIFIED"`) 만 보낸다. `unmet` 은 4종 템플릿 전문.
- A2: `verify_identity.hidden: true`, `ok_text: "VERIFIED"`, `inject_after` 유지.
- `inject_tools` 의 skip 줄에서 `hidden` 은 이미 빠져 있다(nc17). `hidden` 은 "무엇을 말하나"의 스위치이지 "있나 없나"가 아니다.

**엔진이 읽는 것.** `_verdict` 값. 문자열 매칭 없음.

**검증.** 격리 = `lb2_decision` self-test 에 met/unmet 케이스. 팔 = nc19 위에 **nc20**, 판정 태스크 = 축 B 양끝: 016 070 (도구가 해였던 계좌 태스크) + 066 067 062 (도구가 득이던 곳) + 003 007 (도구가 안 나타나야 하는 곳, 회귀 감시).

---

### M2. 손님이 부를 gold 는 재촉하지 않는다

**문제.** 손실 17 중 6 태스크의 gold 쓰기가 손님 도구다(`request_human_agent_transfer` 049 081, `submit_transaction` 016, `apply_for_credit_card` 023, `submit_referral` 098, 048). 손님은 어시스턴트의 텍스트에 반응하므로 그 텍스트를 바꾸는 개입(재촉·유예·장황)이 전부 결과를 바꿨다. 019 에서는 `[LEDGER]` 뒤 어시스턴트가 손님 도구를 **직접 4번 호출**했다.

**설계.** 엔진은 "손님 차례"를 모르지만 **어느 도구가 손님 것인지는 env 가 안다**(`list_discoverable_user_tools`, 그리고 base 궤적에서 `role=user` 로만 불린 도구). 
- A2 `LB5.open_request.form` 을 nc15 와 같은 **도구 이름 선택** 형식으로 바꾼다: 서브호출은 "손님이 실행해 달라고 요구한 것"을 **쓰기 도구 이름**으로 고른다(자유 서술 폐기, `acts` 산문 폐기). 엔진은 `ran` 과 대조해 이미 실행된 것을 빼고 남은 것만 말한다.
- 그 남은 것이 **손님 도구**이면 말하지 않는다. 손님 도구 목록은 A2 `LB5.customer_tools` 로 선언하되 **env 도구 목록에서 기계적으로**(`list_discoverable_user_tools` 결과) 만든다. 손님 말은 읽지 않는다.
- 어시스턴트가 손님 도구를 부르려 하면 deny 가 아니라 **사실 한 줄**(`이 도구는 손님이 실행하는 도구다`)을 다음 턴에 얹는다(M1 경로). 019 형.

**엔진이 읽는 것.** 도구 이름의 정확 일치와 `ran`.

**검증.** 격리 = 기존 `lb-ask` 답을 도구 이름으로 다시 받는 실험(97 태스크 × 2, nc15 의 `select_prompt/read_selection` 재사용). 팔 = **nc21**, 판정 = 축 C 양끝: 049 016 019 081 098 023 (손님 도구) + 047 045 043 (어시스턴트가 부르는 Close, `[LEDGER]` 이득 유지).

---

### M3. 도구는 오라클이 아니다 — 인자를 되비추고 프레임을 뺀다

**문제.** 070 에서 `check_business_checking_fit` 가 모든 sim 에서 `Cobalt Blue` 를 1순위로 냈고(정답 `Sky Blue`), 모델이 채운 인자가 sim 마다 달랐다(`can_keep_balance` = "", "0", "9999", "2850", "3000"). 패 sim 은 그 답을 그대로 따랐다. `check_credit_dispute_provisional_credit` 는 `Provisional credit for **this dispute**` 로 dispute 를 전제했고 비-dispute 태스크에서 0/17. `get_reward_discrepancies` 결과 2.7k자에 `Do NOT change any transaction's rewards` 지시가 있고 부른 4 태스크 전부 패.

**설계.**
- 모든 검증 도구 결과의 **머리**에 엔진이 받은 인자를 그대로 되비춘다: `Inputs you gave: can_keep_balance=9999, min_atm_rebates=15, …`. 판단은 없다. 모델이 자기 인자를 보게 하는 것뿐이다.
- `return_template` 들에서 지시·프레임 문장을 뺀다 — 단, 51개 일괄이 아니라 **부작용이 확인된 셋만**: `Do NOT change any transaction's rewards…`(018/019), `Provisional credit for this dispute`→`Provisional credit, if a dispute is filed:`(036), `Pass the verdict as eligible_for_provisional_credit.`.
- 결과가 비었거나(`eligible: []`) 전부 excluded 면 "권고"가 아니라 "조건에 맞는 항목이 없다" 는 사실만.

**검증.** 격리 = `lb2_decision` 렌더 테스트. 팔 = **nc22**, 판정 = 070 062 066 067 (fit 도구) + 036 038 (dispute 도구) + 018 019 (reward 도구).

---

### M4. 조건부 주입은 사건으로 — 도구마다 선행 호출을 선언한다

**문제.** `inject_after` 가 `verify_identity` 에서 003·007 을 살렸다. 나머지 도구는 LLM 선택(첫 발화)만으로 들어가는데, dispute 도구는 dispute 접수 경로에 들어간 뒤에야 의미가 있고(036), reward 도구는 거래 조회 뒤에야 의미가 있다(018/019).

**설계.** A2 각 도구에 `inject_after` 를 **env 구조에서** 적는다(gold 아님):

| 도구 | inject_after |
|---|---|
| `check_credit_dispute_provisional_credit` | `unlock_discoverable_agent_tool(file_credit_card_transaction_dispute_*)` — 접수 도구가 열린 뒤 |
| `get_debit_dispute_liability_cap` | 같은 형태, debit |
| `get_reward_discrepancies` | `get_credit_card_transactions_by_user` |
| `check_rebate_qualification` | `get_credit_card_transactions_by_user` |
| `get_correct_savings_apy` · `get_interest_correction` | `get_all_user_accounts_by_user_id_*` 또는 savings 계좌 조회 |
| fit 도구 4종 | 선행 없음(상담 초반에 쓰임). LLM 선택만 |

LLM 선택(`select`)과 `inject_after` 는 **AND** 다: 선택됐고 사건이 일어났을 때 들어간다. unlock 이름은 `fam()` 으로 접미사를 뗀다.

**검증.** 격리 = 전 궤적에서 "그 도구가 실제로 불린 sim 중 선행 사건이 그 전에 있었던 비율"(놓침) 과 "선행 사건 없이 불린 sim 의 승률". 팔 = **nc23**, 판정 = 036 081 (dispute) + 018 019 021 028 (reward) + 059 063 064 (savings).

---

### M6. 보고 형식 — 축의 양끝을 함께

모든 팔 결과 표에 다음 줄을 고정한다:

```
축A 행동형 {049 047 045 043} / 조사형 {016 019 081 054 040}
축B 계좌 다단계 {066 067 062 069 070} / 쇼핑·즉시 {001 002 003 006 007 024 025}
축C 손님 gold {049 016 019 081 098 023} / 어시스턴트 gold {047 045 043 036 048}
```

각 묶음의 nc/base 합계를 적는다. 한쪽만 오른 레버는 채택하지 않는다.

## 3. 팔 계보와 판정 태스크

```
nc18 (지금)  = nc17 + hand-off 무유예            004 008 012 014 → 손실16 나머지
nc19         = nc18 + M1 advice 무재생성          004 008 012 014 040 005 037 | 016 049 036
nc20         = nc19 + M5 verify 8자               016 070 | 066 067 062 | 003 007
nc21         = nc20 + M2 손님 gold 무재촉          049 016 019 081 098 023 | 047 045 043
nc22         = nc21 + M3 인자 되비춤·프레임 제거     070 062 066 067 | 036 038 | 018 019
nc23         = nc22 + M4 사건 기반 주입             036 081 | 018 019 021 028 | 059 063 064
```

각 팔은 이전 팔의 판정 태스크를 **회귀 감시로 다시 포함**한다(한 변경이 앞 변경을 되돌리지 않았는지). 전수 96 은 nc23 뒤에 한 번.

## 4. 하지 않는 것

- 태스크별 조건. 축은 선언에 들어가지만 태스크 번호는 어디에도 들어가지 않는다.
- 낱말 매칭으로 의미 판정. `inject_when` 류는 복구하지 않는다.
- 51개 문구 일괄 변환. 부작용이 측정된 것만, 하나씩.
- 횟수 상한. 필요성은 사건(deny)과 값(`_verdict`)으로 가른다.
- `[ORDER notice]` 제거. 전체 이득이다(34% vs 25%).

## 5. 리뷰 포인트

1. M1 의 "마지막 턴 쪽지는 드롭된다"를 받아들일 것인가. 대안은 마지막 턴에 한해 재생성 허용(별도 팔).
2. M2 의 손님 도구 목록을 env 에서 기계 추출하는 것이 gold 참조가 아니라는 데 동의하는가.
3. M4 의 `inject_after` 표 — 각 선행 사건이 env 구조에서 정당한가(문서·도구 설명으로 뒷받침).
4. 판정 태스크 묶음이 축을 대표하는가. 빠진 태스크가 있으면 추가.
