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

## 5. ★★중대 정정 — cred 제공여부가 불가/가능을 가른다 (bug report 정독 후)
> 초안의 "11건 전부 login-순서 fixable → 천장 40"은 **RETRACTED**. bug report `BUGREPORT_SOPBench_bank_impossible_tasks.md` Part B의 분기 기준(= login cred가 user_known에 제공되는지)을 적용해 재분석.

**`diag_v5_canonical_cred.py`** (canonical bank_tasks.json augment-invariant 매칭): premature 11 =
- **cred-present 4** (pay_bill fbaa · set_safety_box 1cb22/c6454/79c3): canonical user_known에 identification 있음 → **augment 불요·login-first로 진짜 fixable**(canonical cred만으로 4/4 full success).
- **cred-absent 7** (get_loan×2 · pay_bill 6058 · pay_loan 92f3/81ba · set_safety_box 48e46/3b48): canonical cred 없음 + dirgraph 강제-login → **AUGMENT_CRED가 cred 줘야만 통과 = bug report fix #2(벤치 결함 보수)이지 모델 능력 아님.**

**`diag_v7_reconcile.py`** (전 48 should_T 정직분해):
| 분류 | n |
|---|---|
| PartA defect (DENY) | 8 |
| cred-absent · BOTH (**augment-pass only**) | 7 |
| cred-absent · premature | 7 |
| cred-present · BOTH (진짜) | 22 |
| cred-present · premature (**진짜 모델-fixable**) | 4 |

**BOTH 29 = 진짜 22 + augment-pass 7.** 헤드라인 `29/34`는 **내부 불일치**(분자는 augment-pass 포함, 분모 34는 cred-absent 결함 제외). 일관 옵션:
- (A) augment 정당(fix #2): `29 / (48−PartA8 = 40)`.
- (B) unaugmented honest: BOTH = cred-present 22 / (cred-present 26, 또는 bug-report 34).

**진짜 모델-능력 타깃 = cred-present 4 (login-순서).** login-first fix는 augment 무관하게 이 4건을 올린다(22→26 가능).

## 6. 미해결 (reliable 확정 필요, GPU 전 offline 가능)
- pay_loan 92f3/81ba: `diag_payloan_nologin.py` 결과 **no-login 경로 전무**(cred-absent 불가) → bug report가 passable로 제외한 66/67과 **다른 인스턴스**. cred-absent unwinnable 정확수(bug-report 6 vs 본 측정 7+)는 released-model cross-check로 확정.
- cred-absent BOTH 7건: no-login 경로 통과인지 augment-login 통과인지 offline 테스트 → unaugmented honest BOTH(22 vs 그 이상) 확정.

## 스크립트 (전부 repo `scripts/distill/sopbench/`)
- `diag_residual5.py` — census + offload-log task_sig 조인
- `diag_residual5_v3.py` — canonical bank_tasks.json 매칭 + call trace
- `diag_residual5_v4.py` — 툴 반환값 포함 trace (반복 goal-call 발견)
- `diag_truncate_test.py` — 비종료 가설 반사실(0/11 flip → 비종료는 증상)
- `diag_cnv_dg_pinpoint.py` — strict per-call 불일치 + dirgraph prereq 트리(근본원인 특정)
- `diag_fix_validate.py` — login-first 순서 fix 검증(augment cred 사용 → 11/11; §5에서 cred-present 4만 정당으로 정정)
- `diag_v5_canonical_cred.py` — **canonical cred 매칭으로 cred-present 4 / cred-absent 7 분리**
- `diag_payloan_nologin.py` — pay_loan no-login 경로 부재 확인
- `diag_v7_reconcile.py` — **전 48 should_T 정직분해 (cred × status)**
