# banking 오프라인 ceiling = dispute-proxy 부분모델 발각 (2026-07-15·[[08]] forensic)

> (a) E-PLAN 컨트롤러 오프라인 스모크 구축 중 발각. handoff §0-a·§2의 "COMPUTE 19.2%→LOAD ~60% ceiling"이
> **reward 기준(DB-state)이 아닌 dispute action_checks proxy의 좁은 부분모델**임을 전수 포렌식으로 확정.
> 전부 로컬 무료(`C:/tmp/traj` 17 궤적·[[09]]). 스크립트(repo): `bank_{action_census,coverage_census,proxy_validate,termination_check}.py` + `bank_eplan_controller.py`.

## 0. 한 줄
banking 실패 sim의 reward는 **최종 DB 상태 일치(reward_basis=DB 81%)**로 나고, **dispute는 실패의 ~20%**(나머지는 비-dispute 다단계 액션)다. `bank_operator_replay`의 19.2%(및 파생 ~60% LOAD)는 **dispute action_checks만 검사한 proxy 부분모델**이라 sim-pass ceiling이 아니다.

## 1. 발각 경위 (per-case 정독)
- (a) 스모크의 COMPUTE arm이 19.2% 아닌 9.4% → 분모 불일치(466 vs 1024) 추적.
- raw action_checks 정독(task_087): gold = **20개 다단계 액션 체크**(verify→get_accounts→unfreeze→clear_alert→get_transactions→**file_dispute**→close_card→order_card). dispute는 그 중 1개. dispute를 완벽히 고쳐도 close_card·order_card(met=0)로 sim reward=0.
- 각 gold 액션이 2행(name-only "호출됨?" + args "값 판정")으로 등장 → 초기 dispute 추출이 double-count(golds[0] tid=None 아티팩트).

## 2. 전수 측정 (17 궤적·실패 sim)
### 2.1 dispute vs 비-dispute 미충족 (4632 실패 sim·args행)
| sim 분류 | 수 | % |
|---|---|---|
| 비-dispute만 미충족 (dispute 컨트롤러 무관) | 3583 | **77.4%** |
| dispute + 비-dispute 둘다 | 827 | 17.9% |
| 미충족 0 (assertion/DB 실패) | 136 | 2.9% |
| **DISPUTE-only 미충족 (컨트롤러 사정권)** | **86** | **1.9%** |
→ dispute-only 컨트롤러 최대 사정권 = **1.9% 완전 + 17.9% 부분**.

### 2.2 미충족 gold-액션 분해 (coverage vs args·15,572 미충족)
- **args(호출·오답) 57.7%** · **coverage(미호출) 42.3%**.
- sim 분류: args-only 35.0% · **coverage-only 18.9%**(강제열거 상한) · coverage+args 17.9% · non-arg(DB/assertion) 28.2%.

### 2.3 ★reward 기준 (전 6515 sim)
- **reward_basis: DB 80.9% · ACTION 9.0% · (empty) 9.1% · DB/NL 1.0%.**
- 실패 sim db_check.db_match: **False 4467 · True 165**(=ACTION/NL-basis서 실패) · None 527.
- ⇒ **reward = 최종 DB 상태**(action_checks 아님). dispute action_checks fix는 db_match를 안 바꿈.

### 2.4 종료사유 (실패 sim 5159)
- **user_stop 4632(89.8%)** = under-action(C80 "100% user-턴 종료" 정합·[[07]]).
- too_many_errors 405(7.9%·infra성→능력 ceiling서 제외 대상) · max_steps 122(2.4%).

### 2.5 미충족 액션 도구 Top (비-dispute 지배 실증)
`(no_tool assertion) 6064` · file_credit_dispute 2198 · **get_bank_account_transactions 1845** · file_debit_dispute 1434 · **open_bank_account 1239** · get_user_dispute_history 1234 · get_all_user_accounts 1107 · **get_pending_replacement_orders 1088** · order_debit_card 872 · close_debit_card 843 · unfreeze/freeze_debit_card 812/789 · close_bank_account 511 · submit_interest_discrepancy_report 489 · apply_savings_account_credit 488.

## 3. 함의 (정정)
- **철회 대상**: "COMPUTE 19.2%→LOAD ~60% offline ceiling"을 **sim-pass ceiling**으로 읽는 것. 정확히는 *dispute action_checks 만족율*(reward proxy)의 부분모델. handoff §2 ①·bank_operator_replay 명명(`would_pass_after_COMPUTE`) 오도.
- **생존(강화)**: C80 coverage-지배 · C89 multi-item plan · C92 연산-오분류 이론 — 모두 유효. 단 **scope가 dispute→전 액션집합으로 확대**. coverage(42%)·user_stop(90%)=under-action이 진짜 지배 레버(C80/C90 outer loop·[[14]] E-PLAN 정합).
- **컨트롤러 정정**([[05]] 원래 요구): E-PLAN 아우터 loop은 **도메인일반·전 gold 액션타입**(dispute는 일 인스턴스)에 걸친 {열거+coverage-track+per-action COMPUTE+ASK+H_min}여야. dispute-특화 harness는 폐기.
- **진짜 make-or-break 게이트 = DB-state**. action_checks proxy 오프라인 스모크로는 못 잼. 선택지: (A) 결정론 DB-replay 오프라인 게이트 구축(전 액션 write→DB diff vs gold·tau2 executor 로컬 가용성 의존) (B) 전-액션 coverage proxy로 근사(caveat 명시) 후 (b) (C) 재앵커.

## 3.5 ★proxy tightness 검증 (사용자 결정=재앵커 후·`bank_proxy_validate.py`)
DB-basis sim서 X=전 gold args-row 충족 vs Y=db_match 2x2 (n=3848·args-row 有):
| | db_match=T | db_match=F |
|---|---|---|
| **X=T (전 액션 완결)** | 185 | **139 (over-action: 완결해도 DB불일치)** |
| **X=F** | **261 (checker 엄격: 미완인데 DB일치)** | 3263 |
- **일치도 89.6%** (name+args 엄격판=90.4%). ⇒ 오프라인 coverage proxy = **~90% tight·사용가능·완벽아님**(양방향 ~10% slippage).
- **X=1,Y=0 = 139(3.6%)**: 전 gold 액션 완결해도 DB 실패 = **over-action(추가 write)** — 강제열거 컨트롤러 ceiling<100%의 근본(§1.4 over-action=게이트 금지축).
- **X=0,Y=1 = 261(6.8%)**: action-checker가 DB보다 엄격(비-load-bearing arg 불일치).
- **★사각지대**: args-row 없는 DB-basis sim = **1420건(db_match F 860·T 560)** — action_checks 렌즈가 못 봄. C80/C89 최근작이 이 순수-DB 실패를 미포착. 오프라인 action-proxy ceiling서 구조적 제외(caveat 필수).

## 3.6 ★전-액션 재앵커 ceiling (사용자 결정 실행·`bank_eplan_controller.py` 일반화·DB-basis 실패 4262·infra제외)
dispute-only 부분모델 폐기 → 아우터 loop coverage를 **전 gold 액션타입**에 적용. action-level 판정(met/coverage-miss/args-miss).
- **미충족 gold-액션 분해**: coverage(미호출) **42.9%** · args(호출·오답) 57.1% (그중 ABox-compute 사정권 11%=dispute liability만).
- **sim arm**:
| arm | % | 레버 |
|---|---|---|
| args-only (inner router) | 36.7% | compute/⋈/gather |
| **coverage-only** | **20.4%** | **아우터 loop 강제열거 (상한)** |
| pure-DB 사각지대 | 20.2% | action-check 없음·오프라인 불가 |
| coverage+args 혼합 | 19.5% | 둘 다 |
| over-action(all-met·DB실패) | 3.3% | 게이트 금지축 |
- **★결론**: 아우터 loop(강제열거+coverage-track) 사정권 = **coverage-only 20.4% 완전 + 혼합 19.5% 부분**. inner router(args 36.7%+혼합 일부). **ABox-compute는 args의 11%만**(dispute liability)→fees/eligibility 규칙 확장이 compute 레버. **over-action 3.3%·pure-DB 20.2%는 강제열거로 못 닫음**(전자=suppress-extra·후자=오프라인 blind).
- term 제외: DB-basis 실패엔 too_many_errors 혼입 0(infra 청정).
- 정본 스크립트: `bank_eplan_controller.py`(엔진 selftest PASS·ABox-driven·리터럴0) + `bank_{action_census,coverage_census,proxy_validate}.py`.

## 4. 다음 (사용자 결정 = 전-액션 coverage 재앵커·무료 → 완료)
이 발각은 handoff (a) 전제를 정정하므로 유료 (b) 전 방향 재확정 필요. 권고 = (A) DB-replay 오프라인 게이트 가용성 확인 → 도메인일반 전-액션 컨트롤러로 재구축. dispute-only 유료 검증은 ≤20% 사정권이라 [[09]] 낭비 위험.
