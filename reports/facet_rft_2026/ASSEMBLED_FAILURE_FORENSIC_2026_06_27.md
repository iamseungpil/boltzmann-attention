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
