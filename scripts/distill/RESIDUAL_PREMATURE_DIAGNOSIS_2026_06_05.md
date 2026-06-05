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

## 6. ★★★확정 (released-model cross-check, bug-report 방법 B) — `diag_v9_crosscheck.py`
> §5의 "cred-absent = 불가능" 가정과 §6 초안의 "9 unwinnable"은 **둘 다 부정확**. released 53파일 cross-check로 확정.

**14 cred-absent의 정확한 분해** (released 모델 통과수):
| goal | sig | released pass | 판정 |
|---|---|---|---|
| get_loan ×2 | 1e6a5b, afcfdb | 0/43 | **진짜 defect** |
| pay_bill | 6058 | 0/43 | **진짜 defect** |
| set_safety_box ×2 | 48e46, 3b48 | 0/43 | **진짜 defect** |
| transfer | 047d | 0/43 | **진짜 defect** |
| pay_loan ×2 | 92f3, 81ba | 1/44, 2/43 | 통과가능(defect 아님) |
| transfer | fc3ed | 23/43 | 통과가능 |
| apply_credit_card ×2, deposit, exchange ×2 | | (no-login) | 통과가능 |

→ **진짜 PartB defect = 정확히 6** (get_loan×2, pay_bill, set_safety_box×2, transfer 047d) = **bug-report Part B와 정확히 일치. honest 천장 34 확정.**
→ 나머지 8 cred-absent은 **no-login OR 경로로 통과가능**(cred-absent ≠ 불가능). 그래서 14≠6.

**제 offline no-login 테스트(`diag_v8`/`diag_payloan_nologin`)는 unreliable**: 모델이 실제 쓴 login-필요 getter만 시도→pay_loan/transfer fc3ed의 진짜 no-login 경로 못 찾고 false-"unwinnable". **released cross-check가 권위.** (메타: offline 예측은 run/cross-check로 확정.)

## 7. ★확정된 정직 회계 (이 표가 최종)
- **honest 천장 = 34** (48 − PartA 8 − PartB 6). "40"·"9 defect" 철회.
- **honest BOTH = 28/34** (보고 29 중 transfer 047d 1건은 PartB defect인데 AUGMENT_CRED가 비밀번호 줘서만 통과=released 0/43 → 정직지표서 제외).
- **진짜 fixable 잔여 = 6**: cred-present 4 (pay_bill fbaa·set_safety_box×3 = login-순서) + pay_loan 92f3/81ba (no-login 경로를 모델이 못 찾음). 검산 28+6=34 ✓.
- **AUGMENT_CRED는 계정 DB의 login 비밀번호(identification)를 user_known에 주입**(`apply_two_stage_patch.py` Edit D). cred-absent 태스크가 통과하면 = 유저가 못 받은 비밀번호 사용(= bug-report fix #2, 모델 능력 아님). admin_password는 미주입.
- 처방 분리: cred-present 4 = login-first 순서구동(SOPBENCH_LOGINFIRST 후보). pay_loan ×2 = no-login 경로 라우팅(모델이 login-필요 getter 대신 no-login branch 선택). transfer 047d 등 PartB 6 = defect(천장 밖, 손대지 않음).

## 11. ★★★Fix 3 STOPSUCCESS 결과 (LIVE A/B) — should_T 천장 도달, 공식 50.75%
**B-3 사전검증(`diag_fix3_offline.py`, GPU 전): 12/12 BOTH-but-not-full → full success flip**(첫 성공 goal-call까지 truncate+strict-replay, prefix-identity 신뢰). → 구현·A/B.
**A/B (full-stack incl DGGATE, augment OFF, `offload_stopsuccess.sh`):**
| run | 공식 pass%(134) | should_T full | should_F |
|---|---|---|---|
| S0 (STOPSUCCESS off) | 41.04% (55) | 28/48 | 27/86 |
| **S1 (+STOPSUCCESS)** | **50.75% (68/134)** | **40/48** | 28/86 |
- **should_T 28→40 (+12, B-3 예측 정확)·회귀 0·should_F 무회귀(27→28).** S1의 BOTH-but-not-full = **공백**(goal_calls→1 수렴). 사전등록 §6 4기준 전부 충족.
- **★should_T = 40/48 = 정직천장 도달** (48−PartA8=40; 비-PartA should_T 전부 통과). **잔여 should_T 8 = PartA credit_card 코드버그(불가).**
- **공식 50.75% = 오픈소스 SOTA(Llama3.1-70B 42.54%) 추월·대형 비-reasoning권**(Gemini-2.0-Flash 52.99·Deepseek-R1 54.48 근접). base Qwen2.5-7B 5.22% → **50.75% (≈10×), 7B로 오픈소스 70B 추월.** 리더보드 max=o4-mini-high 76.87%.
- **★다음 레버 = should_F(거부축) 28/86=33%** (상위모델 70–85/86). should_T는 천장이라 전체%의 유일 잔여 레버 = 거부축.
- **누적 공식 사다리(end-to-end, delta합산 아님)**: base 29.85→loginfirst 37.31→logincall 40.30→**stopsuccess 50.75%**.

## 10. ★★Fix 2 결과 + 지표 재구성 (BOTH dg∧acc는 과대계상, full_success가 진짜)
**Fix 2 LOGINCALL** (`SOPBENCH_LOGINCALL`, `offload_logincall.sh`): pay_loan 통과 기전 = login_user를 **호출**(실패해도)하면 evaluator dirgraph가 충족(dfscheck는 call-order만 봄, auth-성공 아님), getter precond는 username뿐. released qwen2.5-7b fc가 이 방식으로 통과(login('password123')→False→get_account_balance→pay_loan→success).

A/B (augment OFF):
| run | full_success(공식) | BOTH(dg∧acc) |
|---|---|---|
| BASE (fix 없음) | **22** | 29 |
| + Fix 1 loginfirst | **25** (+3) | 33 (+4) |
| + Fix 2 logincall | **28** (+3) | 40 (+7) |

- **★지표 경고**: 프로젝트 헤드라인 BOTH=dg∧acc는 **공식 `success`(no_tool_call_error∧cnv∧dbm∧acc∧dg)를 8~12 과대계상**. Fix 2의 BOTH 33→40(+7)은 **대부분 허상**(full_success +3뿐): login-call이 dg∧acc는 통과시키나 login 실패→goal이 잘못된 DB→dbm/cnv 실패.
- **★진짜 잔여 지배 블로커 = goal-call LOOPING** (`diag_loop_check.py`): BOTH-but-not-full 12개 전부 goal 액션 **5-9회 반복 호출**(step cap). 반복이 cnv=False(반복 제약위반)·dbm=False(비-멱등 DB오염)·ntce=False(닫힌계정 재호출 에러) 유발. login-call과 무관(cred-present도 다수).
- **⇒ Fix 3 = STOP-after-success** (goal 1회 성공 후 gate가 STOP/exit 반환). `diag_fix_validate`의 ideal trace(login-first + **goal 1회**)가 11/11 full success였음 → Fix1+Fix3 조합이 full_success를 40 근처로 끌 잠재력. **looping이 +최대12의 진짜 레버.**
- ⚠️ 천장: cred-absent도 login-call로 BOTH 통과 → bug-report "PartB unwinnable"은 BOTH 기준 반증되나, full_success 기준으론 login-call이 cred-absent를 진짜로 풀지 못함(login 실패→dbm/cnv). 정직 full_success 천장·cred-absent 정당성은 Fix 3 후 재판정.

## 9. ★Fix 1 ROLLOUT 결과 (LIVE, `offload_loginfirst.sh`, augment OFF 양쪽, A/B) — **검증 성공**
| | BOTH | premature | deny |
|---|---|---|---|
| BASE (loginfirst off) | 29 | 11 | 8 |
| **FIX1 (SOPBENCH_LOGINFIRST=1)** | **33** | 7 | 8 |
- FLIP not→BOTH = **4** = set_safety_box×3 + pay_bill×1 = **정확히 cred-present 4**. REGRESSION = **0**. (augment-invariant identity 조인, `diag_ab_loginfirst.py`.)
- **Fix 1 = login front-load이 cred-present 4를 BOTH로 flip 확정. BOTH 29→33.**
- **★정정: augment는 BOTH에 영향 없었다.** BASE(augment OFF)도 29 = augmented 29. "augment가 047d 통과시킴→정직28" 주장 **철회**. 실제 = **gate가 구동하는 `internal_get_database`**(login의 OR-대안, DB 통독)가 cred-absent transfer 047d의 login-gated 경로를 충족 → augment 무관. (augment 끄는 건 무해 확인.)
- **새 honest-ceiling 질문**: internal_get_database 구동(DB 비밀 접근, bench상 에이전트 도구 아님)으로 cred-absent defect(047d)가 통과하는 게 정당한가? 부당 시 honest BOTH = 33−1 = 32. → augment와 동류 문제, 사용자 판단 필요.
- FIX1 잔여 premature 7 = 전부 cred-absent: get_loan×2·pay_bill×1·set_safety_box×2 (PartB defect 5) + pay_loan×2 (Fix 2 대상).

## 8. Fix 1 설계 확정 (`diag_fix1_order_test.py`, zero-cost) — **front-load 필수**
evaluator는 func_calls를 **순서대로** 처리하고, getter가 login보다 먼저 호출되면 그 시점 prereq 미충족 → **dirgraph_satisfied 영구 False**(이후 복구 불가).
- (a) front-load(login→getter→goal) = **통과**.
- (b) late-repair(getter→login→getter재구동→goal) = **실패** (dg 영구 False).
⇒ **Fix 1은 login_user(+admin auth)를 모델의 첫 getter 호출 전에 front-load 구동해야 한다.** 현 active-H3의 "ACT 직전 internal_get_database 늦은 구동" 방식으론 안 됨. 구현=gate가 dirgraph상 login 요구 시 **첫 턴에** login_user 구동(user_known identification, cred-present=request param). flag SOPBENCH_LOGINFIRST.
- ⚠️ **AUGMENT_CRED OFF 선결**: genuine 6 중 augment 필요한 것 없음(cred-present 4=비번 보유, pay_loan 2=no-login). augment 켜두면 transfer 047d(PartB defect)가 가짜통과→honest baseline 28 대신 29 보고. 끄면 회귀 없이 정직 28(rollout A/B 실측 확인).
- ✅ **task_sig 충돌 해소**: should_T 48개는 충돌0(핸드오프 유효, `diag_sig_collision.py`). c6454가 fix1테스트서 2번 나온 건 **should_F 엔트리(idx88, acc=False 정상)가 같은 content-sig 공유** — should_T(idx85)는 진짜 cred-present로 front-load 통과. **cred-present 4 온전.** 단 should_T↔should_F 조인 키엔 initial_database 포함해야 함(sig만으론 교차충돌 가능).

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
