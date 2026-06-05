# 잔여 premature 진단 — login-before-getter 순서 (2026-06-05, DGGATE 후속)

> zero-cost(GPU 0), 전부 **권위 evaluator(`env/evaluator.py`) 재실행**으로 확정. 진입점 핸드오프 `HANDOFF_2026_06_05_PM_argfix_dggate_ladder.md` §0.2 잔여 진단의 결말.

## 0. 결론 (TL;DR)
- **핸드오프 §3 가설("goal-call이 잔액 임계/safety_box 값 등 실제 정책조건 위반") = REFUTED.**
- **진짜 원인 = `login_user`가 value-getter/admin-auth보다 먼저 establish되지 않음** (누락 또는 순서오류).
- **`diag_fix_validate.py`: login_user를 첫 establish로 재배열한 ideal trace → 11/11 premature 전부 cnv&dg&acc&dbm=True full success.**

## 1. census (should_T 48)
| 구분 | n | 정체 |
|---|---|---|
| BOTH (dirgraph∧acc) | 29 | 현 DGGATE |
| premature (acc=T, dg=F) | 11 | get_loan 2·pay_bill 2·pay_loan 2·set_safety_box 5 |
| DENY (acc=F) | 8 | PartA 코드버그(cancel_credit_card 6 + pay_bill_with_credit_card 2) |

## 2. 근본원인 (계측: `diag_cnv_dg_pinpoint.py`)
전 11건의 `constraint_not_violated=False`는 항상 **value-getter(get_account_balance / get_account_owed_balance / internal_get_credit_score) 또는 authenticate_admin_password 호출**에서 strict 시스템이 **False**를 반환해 발생.

이유: dirgraph상 이 getter/admin-auth들은 `login_user`를 **선행 prereq**로 가짐:
```
get_loan ← and{ or{internal_get_database, get_account_owed_balance}, or{internal_check_username, internal_get_database} }
                                            └ get_account_owed_balance ← and{ login_user, ... }   ← login 필수
```
- **login 미호출 케이스**(get_loan ×2, pay_bill 6058, pay_loan ×2, set_safety_box 48e46/3b48): getter가 login 없이 호출 → strict가 getter를 False 반환.
- **login 호출했지만 순서 틀림**(pay_bill fbaa, set_safety_box 1cb22/79c3): getter/admin-auth를 login_user보다 **먼저** 호출 → 그 시점 미로그인 → strict False.

`dirgraph_satisfied=False`도 동일 뿌리: strict-실패한 getter는 `successful_funccalls`에 기록 안 됨 → OR-브랜치(getter vs internal_get_database) 충족 실패.

**왜 DGGATE가 못 고침**: getter가 "호출은 됨"(ungathered 아님)이라 gate가 permit(nfalse=0, nung=0); gate 재구성이 login-before-getter **순서** 의존성을 못 잡음. (transfer는 DGGATE가 고쳤으나 이 패턴은 미해결.)

## 3. FIX 검증 (`diag_fix_validate.py`, 권위 evaluator 재실행)
ideal trace = `internal_check_username_exist → login_user(user_known creds) → authenticate_admin_password(필요시) → getters → goal 1회`:

**11/11 전부 cnv&dg&acc&dbm = True (full success).**

정당성: login creds(identification/admin_password)는 user_known(AUGMENT_CRED)에서 = request params(oracle 아님, ARGFIX 근거와 동일). getter 응답은 login 충족 시 non-strict==strict로 일치(반사실 trace에서 agent content=strict gt로 둔 것이 곧 실제 rollout 값).

## 4. 처방 (설계 대상, 다음 단계)
gate/active-H3가 **login_user를 login-gated getter/admin-auth보다 먼저** deepest-first 구동. flag 후보 `SOPBENCH_LOGINFIRST`.
- 기존 active-H3 deepest-first가 login을 getter의 prereq로 인식하도록 = getter 노드의 하위 `and{login_user,...}`를 먼저 충족.
- ⚠️ offline replay이므로 **live rollout A/B로 확정 필요**(메타규칙). 회귀 0·login-uniform 특별취급 금지(T1T2 BOTH4/4 교훈) 확인.

## 5. ⚠️ 천장 재검토 (사용자 결정 필요)
이 11건은 **cred-부재-불가능(PartB)이 아니라 login-순서로 fixable**(augment creds 가용 시). 현 run은 `SOPBENCH_AUGMENT_CRED=1` 사용 → 실효 천장 = 48 − PartA8 = **40**, 핸드오프 "honest-34"가 아님.
- "honest-34"는 **unaugmented**(login creds 부재) 기준; 현 파이프는 augment 사용 → `29/34` 헤드라인이 두 세팅을 혼용.
- 선택지: (a) augment를 정당 scaffold로 보고 천장 40 채택(BOTH 29→최대 40 목표), (b) augment 끄고 unaugmented 34 기준 유지(이 11은 다시 불가능), (c) 두 세팅 분리 보고.
- 본 진단은 (a)를 지지하나, 최종 프레이밍은 사용자 결정 + live rollout 확정 후 박제.

## 스크립트 (전부 repo `scripts/distill/sopbench/`)
- `diag_residual5.py` — census + offload-log task_sig 조인
- `diag_residual5_v3.py` — canonical bank_tasks.json 매칭 + call trace
- `diag_residual5_v4.py` — 툴 반환값 포함 trace (반복 goal-call 발견)
- `diag_truncate_test.py` — 비종료 가설 반사실(0/11 flip → 비종료는 증상)
- `diag_cnv_dg_pinpoint.py` — strict per-call 불일치 + dirgraph prereq 트리(근본원인 특정)
- `diag_fix_validate.py` — login-first 순서 fix 검증(11/11 full success)
