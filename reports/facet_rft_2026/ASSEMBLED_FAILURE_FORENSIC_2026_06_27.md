# Assembled-stack failure forensic — per-case, both scales (2026-06-27)

Runs: `asmscale_32b_0626pm` / `asmscale_14b_0626pm` (assembled stack =
present+nested+full-gates+constraints+calc, nt=3, retail, gpt-4.1 user-sim).
Persisted: `sim_results/asmscale_{32b,14b}_0626pm_assembled_retail_t3.results.json.gz`.

Method (per [[08]]): every **clean robust-fail** (fail-all-3, infra trials
excluded) trajectory was read individually — user request, agent tool-call
sequence, gold vs actual write, failed assertion — and the cause confirmed from
the log, not from aggregate field-diff. Reward = `reward_info.reward`; infra =
`termination_reason == "infrastructure_error"` (excluded; nmsg=0, reward=None).

## Headline numbers (robust pass^3; violations=0 → compliant-pass == bench)

| | 14B | 32B |
|---|---|---|
| assembled pass^3 (robust) | 0.313 | 0.457 |
| gate violations (g1–g4) | 0 | 0 |
| floor (base) bench pass^3 | 0.232 | 0.281 |
| floor (base) F4 compliant-pass^3 (full) | 0.152 | — |
| assembled F4 (full) | 0.313 | 0.457 |

- Floor scale ladder (bench pass^3): 7B 0.080 · 14B 0.232 · 32B 0.281.
- **Assembled 14B (0.313) > bare 32B floor (0.281)** — small model + scaffold
  beats larger bare model.
- **F4 gap (14B): floor 0.152 → assembled 0.313 (+16pp, ≈2×)** — the gate removes
  floor violations (g1=38, g2=38) AND lifts pass; the compliance-aware metric
  shows ~2× the value of raw bench pass (+8pp).

## Clean robust-fail counts

- 32B: 114 tasks − 6 pure-infra excluded → **21 clean robust-fail**.
- 14B: 114 tasks − 3 pure-infra excluded → **30 clean robust-fail**.
- Pure-infra excluded: 32B {22,27,28,36,37,71}; 14B {33,36,37}.

## Confirmed cause distribution (per-case read)

| cause | 32B (21) | 14B (30) | scale behavior |
|---|---|---|---|
| ⋈ ORDER (wrong/missed order among several) | 3 (14%) | **7 (23%)** | retires with scale |
| ORCHESTRATION (no-write / incomplete multi / loop) | 4 (19%) | 5 (17%) | loops ↑ at 14B |
| OVER-ACTION (executed disallowed/unrequested write) | **4 (19%)** | 3 (10%) | ↑ at 32B (destructive) |
| CRITERION (wrong variant/item under constraint) | 3 (14%) | 4 (13%) | flat |
| WRONG-OP (cancel/modify/exchange confusion) | 1 (5%) | 3 (10%) | retires with scale |
| PAYMENT (wrong refund-card resolution) | 1 (5%) | 3 (10%) | retires with scale |
| NL/REPORT (tracking#, order-total, payment not told) | 4 (19%) | 3 (10%) | mostly artifact/out-of-scope |
| FORMAT / ADDRESS-value | 1 (5%) | 1 (3%) | |

**No single dominant cause.** Distributed across ⋈ / orchestration / over-action
/ criterion at 10–23% each.

## 32B per-case verdicts

- t10 OVER-ACTION — "refund each order to the *other order's* payment method" = impossible; executed anyway (gold=no write).
- t13 CRITERION — returned an extra gaming item (criterion: non-gaming only).
- t17 FORMAT — rewrote existing "123 Elm Street" as "123 Elm St" (verbatim not preserved).
- t20 CRITERION — "most expensive *but shoes size 9*" → picked size-8 max (joint constraint failed).
- t33 OVER-ACTION — gold = address change only; also cancelled whole order.
- t34 WRONG-OP — "cancel only office items" (impossible) → cancelled whole order (gold=modify).
- t38 ORCHESTRATION — never reached the cancel nor told camera price (reads only).
- t39 ORCHESTRATION — address write never reached (find_user loop).
- t40 NL/REPORT — DB correct; didn't say which payment method was used.
- t41 ⋈ ORDER — fixed address on #W4082615 (gold #W9583042).
- t57 OVER-ACTION — single-item cancel (impossible) + user retraction; cancelled whole order (gold=no write).
- t62 OVER-ACTION (severe) — user only *asked the speaker's price/battery* → agent cancelled the order (destructive spurious write).
- t63 PAYMENT — used gift_card for the modify (gold=paypal).
- t67 NL/REPORT — order total $829.43 not computed/told (wrong-zip friction).
- t68 NL/REPORT — order total $829.43 not told.
- t69 ORCHESTRATION — gold=cancel; returned wrong order ×4 (loop).
- t76 ORCHESTRATION — 2 orders to cancel, did 1 + wrong reason.
- t98 ⋈ ORDER — multi-exchange on wrong order/items.
- t100 CRITERION — wrong skateboard variant (34"+custom not matched).
- t104 NL/REPORT — DB correct; tracking# 286422338955 not provided (out of calc scope).
- t107 ⋈ ORDER — boots & puzzle in *different orders*; only one handled.

## 14B per-case verdicts

- t1 ORCHESTRATION — exchanged keyboard only (thermostat missed); order_id missing '#'.
- t8 CRITERION+PAYMENT — wrong new variant + payment.
- t14 PAYMENT — wrong refund card (paypal vs gold credit_card).
- t19 ORCHESTRATION — write never reached + savings not told.
- t20 ORCH+CRITERION — only 2 of 4 items + size variant wrong.
- t22 OVER-ACTION — address overwrite confusion (101 → reverted 667).
- t27 CRITERION — wrong boots (waterproof) variant.
- t30 REASON — cancel reason 'ordered by mistake' (gold 'no longer needed').
- t31 OVER-ACTION — extra return on #W2692684 not in gold.
- t34 WRONG-OP — cancelled whole order (gold=modify).
- t38 REASON+NL — camera price not told + reason enum.
- t39 ORCHESTRATION — address write never reached (loop).
- t40 NL/REPORT — payment method not told (DB correct).
- t45 OVER-ACTION — exchange executed where gold=no write (likely disallowed exchange).
- t51 PAYMENT — user said "original payment method"; used wrong card + return ×3 loop.
- t53 PAYMENT — wrong card of two + loop.
- t58 CRITERION — both coffee-machine & laptop variants wrong.
- t66 WRONG-OP — "prefer a coat instead" (not a swap) → modify (gold=cancel).
- t69 ORCHESTRATION — write never reached.
- t76 WRONG-OP — 1 modify instead of 2 cancels.
- t83 ⋈+PAYMENT — wrong order (#W3069600 vs #W9571698) + wrong card + loop.
- t85 WRONG-OP+⋈ — exchange-on-delivered instead of modify-pending; wrong order.
- t98 ⋈ ORDER — multi-exchange wrong orders/items.
- t99 ⋈ ORDER — wrong orders + reason.
- t102 ⋈/ORCH — missed exchange on another order (#W3445693).
- t103 NL/REPORT — tracking# not provided.
- t104 NL/REPORT — tracking# not provided.
- t109 ⋈+ADDRESS — wrong order + wrong address value (760 vs 592).
- t110 CRITERION — modify variant wrong.
- t111 ⋈/ORCH — missed item on another order (#W9810810).

## What reading overturned (statistics alone misled — [[08]] evidence)

1. Scripted field-diff said "VARIANT is #1"; reading shows most were actually
   **⋈ (wrong order), WRONG-OP, or PAYMENT** — only visible per-trajectory.
2. **OVER-ACTION is the de-facto dominant 32B mode** (t62 cancels an order the
   user only asked about; t10/t33/t57 execute impossible/unrequested writes). The
   deterministic present-stack induces *over-action* — destructive spurious
   writes are the most dangerous failure.
3. **14B is dominated by ⋈ + loops + wrong-op** — multi-order resolution,
   same-call repetition (t14/t51/t53/t83 return ×3), and operation confusion,
   all retiring with scale (load theory).
4. **Non-model-error slice separated**: NL/REPORT is mostly tracking# (out of
   calc scope) and order-total; REASON enum is under-determined by the dialogue;
   t109/t110 show user-sim address variance — these are benchmark/scope
   artifacts, not capability gaps.

## Conclusion

- **make-or-break NO-GO unchanged**: the learn-candidate slice (non-artifact
  criterion-formalize not closable by present/compute) is ≤3–4 cases per arm.
- **Next deterministic levers (priority)**: ⋈-resolution (esp. 14B) +
  over-action suppression gate (esp. 32B) + calc scope extension (subset-refund;
  tracking# remains out of scope). Consistent with [[13]] (deterministic before
  learning).
- **Infra note**: `sim_results/f3f4_scale_invariant_compliance_2026_06_26.txt.gz`
  is corrupt (a gzipped old shell error, not the table) — regenerate.

---

## Addendum (2026-07-06) — pass^3 분모 exclusion·"infra"의 실체 (raw 재검증·[[08]])

로컬 raw(`sim_results/asmscale_{14b,32b}_0626pm_assembled_retail_t3.results.json.gz`)를
`t2_compliance.py` 로직으로 직접 재처리해 **exclusion 규약과 strict 대안을 확정**.

### 지표 기전 (`t2_compliance.py`)
`reward is None`인 sim은 per-task 리스트에서 **통째 제외**(`continue`), 그리고
`pass_hat_k`는 유효시행 `< k`인 task를 분모서 **skip**. ⇒ pass^3는 "3 clean 시행이
남은 task"에서만 계산 → 크래시가 낀 task는 분모서 빠짐. **floor·assembled 모두 동일
코드**(같은 규약·비대칭 아님).

### ★핸드오프 §0.2 "infra=게이트-예산 소진 오분류" — 헤드라인 런에는 **거짓**(정정)
헤드라인 asmscale 런의 dropped(`reward=None`) sim을 전수 검사:
- **14B: infrastructure_error 26건 = 전부 `nmsg=0`**(메시지 0·`POLICY GATE` 0·
  context-window 마커 0) = **진짜 크래시**(시뮬 미생성). too_many_errors는 **별개**
  6건이고 **reward 보유 → fail로 집계·드롭 안 됨**.
- **32B: infrastructure_error 39건 = 전부 `nmsg=0`**·too_many_errors 0.
- ⇒ **드롭된 것은 게이트-예산 소진이 아니라 nmsg=0 launch 크래시.** 게이트-예산
  소진(too_many_errors)은 드롭되지 않고 fail로 이미 반영됨. 핸드오프 §0.2의
  "게이트-abort가 분모서 오분류-제외" 서사는 **이 헤드라인 런에 성립 안 함**
  (retry 오버나이트 배치 g15retry에는 적용될 수 있으나 **미검증** — g15retry raw
  로컬 부재. 별도 확인 필요).

### 크래시율·strict 대안 (raw 계산)
| arm | total | crash(nmsg=0) | 크래시율 | pass^3 CURRENT(크래시 task 드롭) | pass^3 STRICT(크래시=fail) |
|---|---|---|---|---|---|
| assembled 14B | 342 | 26 | 7.6% | **0.313** (n_task=99) | **0.272** (n_task=114) |
| assembled 32B | 342 | 39 | 11.4% | **0.457** (n_task=94) | **0.377** (n_task=114) |

### ★★remote 정밀 검증 (2026-07-06·floor+assembled 동일 로직·`data/simulations`)
`on_n{7b,14b,32int8}_floor_retail` + assembled를 원격서 재처리(무료·기존결과):

| arm | total | 크래시(nmsg=0) | 크래시율 | pass^3 CURRENT(드롭) | **pass^3 STRICT(크래시=fail)** |
|---|---|---|---|---|---|
| floor 7B | 342 | 1 | 0.3% | 0.0796 | **0.0789** |
| floor 14B | 342 | 3 | 0.9% | 0.2321 | **0.2281** |
| floor 32B | 342 | **0** | **0%** | 0.2807 | **0.2807** |
| assembled 14B | 342 | 26 | 7.6% | 0.3131 | **0.2719** |
| assembled 32B | 342 | 39 | **11.4%** | 0.4574 | **0.3772** |

**★크래시 비대칭 = 인과적(coincidence 아님).** floor 크래시율 ~0%(32B는 정확히 0),
assembled 8~11%. 크래시 레코드 `info.error` 전수: **게이트가 차단한 write 호출을 모델이
재시도**(`[POLICY GATE G5_STATUS_PRECONDITION] ... do NOT retry`)하자 실제 반환(base 도구
에러)이 기대 게이트-궤적과 divergence → 하네스 `infrastructure_error` abort. floor는 게이트
부재로 이 divergence **구조적 불가** → **assembled-전용 실패모드**(게이트-write-재시도).
= 크래시=random-infra 아님·**방법 결부**. ⇒ 드롭은 부당(assembled에만 유리), STRICT가 공정.

**★공식 tau2-bench crash 규약 — 두 경로(정밀·정정).**
- `scripts/get_experiment_results.py`(요약 스크립트): infra를 `fail_terms`에 포함(fail 카운트 보고용) — **pass^k 아님.**
- **`metrics/agent_metrics.py::compute_metrics`(실제 리더보드 pass^k)**: `df = df[df.termination_reason
  != INFRASTRUCTURE_ERROR]` = **infra 드롭**(docstring "simulations that never ran"). ⇒ **공식
  리더보드 pass^k는 infra 드롭 = 우리 CURRENT와 기계적으로 동일.**
- **★그러나 함정**: 공식 드롭의 전제 = infra는 "아예 안 돈 시뮬"(API 끊김·frontier ~0건). 우리 infra는
  **평가 시점 방법-유발 크래시**(`environment.py:389` set_state replay assertion). ⇒ 공식 코드를
  기계적으로 적용하면 **우리 방법-유발 실패 11%가 frontier가 못 받는 "무료 드롭"으로 처리되는 역설**.
  실질 공정 = 무료 드롭 제거 = **STRICT**.

**★크래시 근본원인 (`environment.py:357-392` set_state).** 평가 시 저장된 mutating tool call을
pristine env에 **재실행**해 기록 응답과 대조·불일치면 `ValueError`→infra. 우리 게이트=agent-side
패치(`t2_gate_patch`)라 **live엔 적용·eval-replay엔 미적용** → replay가 게이트 없는 base 도구
재실행 → 기록된 `POLICY GATE` 메시지와 content 불일치 → 크래시. (게이트-차단·base-에러 **둘 다 DB
무변경 no-op**·순수 content mismatch.) floor는 게이트無로 구조적 불가 → assembled 전용.
**크래시 궤적은 미저장**(results.json messages=0·`info.error_traceback`만) → **무료 재채점 불가·재런 필요**.

**★replay는 표준·회피 불가 (`evaluator/evaluator_env.py`).** tau2 reward는 live env 최종상태가
아니라 **fresh predicted env + fresh gold env를 각각 message-history로 set_state(replay) 재구성 후
DB 비교**로 산출. **모든 리더보드 모델이 이 동일 replay로 채점**됨 — "실시간 측정" 경로는 tau2에 없고,
evaluator를 바꾸면 리더보드 비교 가능성이 깨짐. 정상 에이전트는 히스토리가 자기-일관(기록 tool 응답=실제
재실행 결과)이라 무크래시. 우리만 크래시하는 건 **agent-side 게이트가 합성 응답을 주입해 히스토리를
non-replayable하게** 만든 탓 = replay 문제 아니라 게이트 구현 문제.

**★크래시 분류 (info.error 전수·게이트명·divergence 종류)**:
| | G2_CONFIRM | G5_PRECOND | G7_CONSTRAINT | 미식별 | content-only(no-op) | **state-div** |
|---|---|---|---|---|---|---|
| 32B(39) | 10 | 10 | 12 | 7 | 26 | **13** |
| 14B(26) | 2 | 6 | 7 | 11 | 13 | **13** |
= 게이트 3종(G2/G5/G7) 전반. **절반이 state-divergence**(게이트 차단 write를 replay base env가 실행·성공
→ DB 갈라짐). ⇒ assertion 완화식 얕은 fix는 이 13개에 **틀린 reward** → **유일 정답=히스토리 replay-clean**.

**★해결법 (다음 런·generation-level·replay-safe).** 근본원인=게이트가 `_execute_tool_calls`를 패치해
차단 시 합성 `ToolMessage(error)`를 **대화 히스토리에 커밋** → replay가 그 mutating 호출을 재실행→크래시/
divergence. 정답=코드에 **이미 있는 replay-safe 패턴**(`apply_provenance_regen`: 거부 피드백을 *작업
버퍼*에서만 주고 재생성, 유효 호출만 `state.messages` 커밋)을 **게이트에도 적용** — 즉 게이트를
`_execute_tool_calls`(post-hoc·히스토리 오염)서 **agent 생성 레벨**(`LLMAgent._generate_next_message`)로
이동: 차단→작업버퍼 피드백→재생성(최대 K)→**compliant 호출 또는 ask/transfer만 커밋**. 히스토리가
executed/valid 호출만 담음 → **표준 evaluator가 frontier와 동일 채점**. 게이밍 아님: (a) 차단은 여전히
`num_errors++`로 예산압박 유지(런타임-게이트 semantics 보존), (b) db_check 보상=실제 궤적 상태 그대로.
벤치 코드 불변(우리 패치만 이동). 단 semantics 미세변화(거부턴이 히스토리서 사라짐)라 **재측정 필요**.
스모크(10 task·공식 evaluator)로 크래시0·정상채점 검증 후 → 승인 시 nt=4 공식 프로토콜 full 재런서 진짜값.

**parity 감사 (results.json info)**: max_errors 10=10 ✓·max_steps 200=200 ✓·temp 0.0/0.0 ✓·
user-sim gpt-4.1 ✓·taskset retail 114=114 ✓ / **num_trials 우리 3 vs 리더보드 4 ✗**(k 다름·
리더보드 제출용은 nt=4 재런 필요). base-vs-assembled 내부비교는 둘 다 nt=3 → 정합.

### ★결론 (leaderboard-consistent 재판정)
> **★★2026-07-07 SUPERSEDED — 아래 strict 추정(bound)은 클린 nt=4 재런 실측으로 대체됨.**
> replay-safe 게이트 구현·`asmregen{14b,32b}_regen_retail_t4`(infra=0·위반=0) 확정값 →
> `REPLAY_SAFE_GATE_DESIGN_2026_07_06.md §12`. 요지: **진짜 pass^3=32B 0.423/14B 0.336**(옛 strict
> 0.377/0.272와 drop 0.457/0.313 *사이*). **same-scale 우위 더 큼**(32B +14.2pp·14B +10.8pp). **★crossover
> 부활·재확립**(14B-asm 0.336 > 32B-floor 0.281 — strict서 flip됐던 게 클린서 되살아남·strict가 방법-유발
> 크래시 과벌한 탓). **frontier는 모든 k서 ~9pp 아래**(pass 미달·moat=compliance). 아래는 fix 前 bound 기록.

- **같은-scale 우위 = 유지**: 32B 0.3772 > 0.2807(**+9.7pp**)·14B 0.2719 > 0.2281(**+4.4pp**).
  scaffold는 동일 규모서 base를 이긴다(strict서도). [클린: +14.2/+10.8pp]
- **★cross-scale crossover = FLIP(철회)** [★클린서 재확립·철회 취소]: "작은모델+scaffold > 큰모델 bare"
  (14B-asm > 32B-floor)는 CURRENT 0.3131 > 0.2807이나 STRICT 0.2719 < 0.2807 = 뒤집힘(strict의 과벌).
  **클린 nt=4: 14B-asm 0.336 > 32B-floor 0.281 = 성립**(§12).
- **★frontier 병치 = k-불일치·철회**(공식 참조로 확정): `data/tau2/results/final/o4-mini-2025-04-16_
  retail_default_gpt-4.1-2025-04-14_4trials.json`을 공식 `compute_metrics`로 채점 → **o4-mini retail
  pass^1 0.715·pass^2 0.594·pass^3 0.5175·pass^4 0.4561**(n=456·**infra 0**·config nt4/steps200/
  err10/gpt-4.1=우리와 동일). ⇒ 헤드라인 "0.457≈o4-mini 0.468"은 **우리 pass^3를 o4-mini pass^4(0.456)와
  비교한 k-불일치**. **같은 k(pass^3)**: 우리 32B current 0.457/strict 0.377 vs **o4-mini 0.5175** = 같은
  k서 이미 frontier 아래(strict ~14pt). **"frontier 진입/근접" 철회.** frontier 크래시=0(무료-드롭 비대칭
  논거 정량 확증). **살아있는 moat = pass 병치 아니라 compliance**(게이트=위반0·scale-불변 / frontier=
  confirm 위반 존재·[[46]]).
- **부수 발견(신규 레버 후보)**: assembled scaffold가 게이트-write-재시도서 하네스 크래시
  ~11%(32B). 이는 (a) 지표 부풀림 원인이자 (b) **실제 신뢰성 결함** — 게이트 deny 후 모델
  재시도를 결정론적으로 흡수(재시도 차단·즉시 fallback)하면 크래시 제거+strict 점수 상승 여지.
  단 [[13]] 게이트 확장은 static-ceiling·over-block 측정 하에.

### retry/provenance (오버나이트 A/B·별도 런·핸드오프 §0.2)
- retry_controller·provenance = **해로움**(treat 0.077 < control 0.154). 결론 유지.
  기전이 "게이트-abort 오분류-제외"라는 §0.2 서사는 헤드라인 런선 반증(위) → retry 배치
  raw로 재확인 필요(미검증).

⇒ 위 Conclusion "deterministic levers largely exhausted"는 유지. **헤드라인 정정: 리더보드-
일관 지표(STRICT)에서 같은-scale scaffold 우위는 견고(32B +9.7pp·14B +4.4pp)하나
cross-scale crossover는 철회, frontier는 "진입"→"근접"(0.377).**
