# Next deterministic levers — design (2026-06-27)

Derived from the per-case assembled-failure forensic
(`ASSEMBLED_FAILURE_FORENSIC_2026_06_27.md`). Discipline: only A2 changes; the
scaffold engine stays domain-general (gate_spec-driven, no retail hardcoding);
no gate proliferation without a verified deterministic basis and a static ceiling
([[05]], [[06]]). Lever-type ≠ solution — g15 precedent showed net-pass ≈ 0 when
suppression trades off against another layer; every lever below is a **hypothesis
with a static ceiling**, to be measured with per-task double-confirm
(robust-fail → robust-pass ∧ write gold-correct), watching for over-block.

## Forensic residual → candidate levers

The 51 clean robust-fails (32B 21, 14B 30) split into a deterministically
addressable part and an LLM/benchmark-residual part. **Verification (below)
moved two candidates.**

### Lever A — refund-target gate  [CONDITIONAL — verify first]
**Claim:** a return/exchange refund `payment_method_id` must equal the specific
order's original payment method (from that order's `payment_history`), or a gift
card. Agent errors: wrong card / wrong order's card / looped wrong card.
**Verification (t14/t51/t53/t10):**
- t14: gold refunds each of two orders to *its own* original PM
  (credit_card_3124723 for #W5490111, paypal for #W7387996); the agent used
  paypal for both → the rule holds, agent violated it per-order.
- t51/t53: gold PM ≠ the card the agent used; needs the order's true
  `payment_history` to adjudicate (crude scan insufficient).
- **t10: gold refund is EMPTY** — the "refund to the *other* order's payment"
  request must be *refused*; this is **over-action, not payment** (moved to C).
**Design:** gate checks refund PM ∈ `fetch_record(order).payment_history`
originals ∪ user gift cards. Uses `fetch_record` (allowed resolver, gate1 test
still passes) — NOT a new join-resolver. Domain-general via gate_spec
(`kind: refund_target`, applies_to = return/exchange tools, source field from A2).
**Static ceiling:** PAYMENT cases = 14B {t14,t51,t53} + 32B {t63} + partial
{t8,t83} ≈ **4–6**. **Risks:** (i) multi-order refunds (must resolve per order);
(ii) legitimate gift-card refunds must not be blocked; (iii) many are multi-cause
(t83 = ⋈+payment) so pass-flip < ceiling.
**Verdict:** BUILD after confirming per-order `payment_history` resolution on the
5 cases (GPU-free, existing trajectories).

**VERIFIED 2026-06-27 (DB ground truth):** for all 5 gold orders, the gold refund
`payment_method_id` == the order's original payment in `db.json` payment_history:
#W5490111→credit_card_3124723, #W4689314/#W3916020→credit_card_8105988,
#W6390527→paypal_7644869, #W7387996→paypal_9497703 — **5/5 MATCH**. The rule is
deterministic: refund PM ∈ {order's payment_history payment} ∪ {gift cards}. Gate
confirmed BUILDABLE. **But coupling:** the trajectory extraction returned the gold
order's payment as `None` in the wrong-order cases — the agent used the wrong card
*because it operated on the wrong order* (⋈). So the PAYMENT residual is largely a
**downstream symptom of ⋈**; the refund_target gate enforces payment-consistency
*given* the order but cannot fix order-choice. **⇒ do Lever B (⋈) first** — it may
absorb most of the PAYMENT residual, shrinking Lever A's marginal ceiling to the
right-order-wrong-card slice.

### Lever B — ⋈ present-quality  [NOT a resolver]
**Claim:** when the user has several orders, the agent operates on the wrong or a
missed order. The fix is **present quality, not a resolver** — gate1 forbids a
join-resolver that pre-selects the order (that would rig make-or-break; the model
must still choose). Make-or-break already found genuine-⋈ is present-addressable
(7/13 when present enumerates).
**Design:** extend `present_fields` (A2) so the order enumeration carries
disambiguating fields — which items each order contains, status, date — so the
model can pick correctly. No engine change; A2 present_fields only. gate1 test
(enumerate-not-resolve) unaffected.
**Static ceiling:** pure-⋈ = 32B {t41,t107} + 14B {t102,t109,t111,t85} ≈ **6**
(excludes ⋈ tangled with wrong-op/payment). **Risk:** present already enumerates
orders; if the model still mis-picks with disambiguators present, the residual is
operand-formalize (LLM), not present — bound honestly by measurement.
**Verdict:** BUILD-lite (cheap, A2-only); measure whether disambiguated present
closes ⋈ or leaves an LLM residual.

**VERIFIED 2026-06-27 → NO-GO (already maximal).** The `G6_SELECT_CONFIRM`
`present_fields` are already `[status, address, items]`, and `candidate_summary`
(read-aug, T2_PRESENT_READS) **already fires** in all ⋈-failure cases (32B
t41/t98/t107, 14B t83/t85/t102/t109/t111 = all True; 248/342 and 259/342 sims
overall). The DISAMBIGUATION NOTE dumps every order with **item names + prices +
options + address + status** — e.g. t83 ("return the pricier of two tablets")
shows both tablets with prices ($938.92 etc.). The model sees the full
disambiguated list and *still* picks the wrong order. So the ⋈ residual is
**LLM operand-formalize** (reading + comparing + matching over the enumerated
candidates), NOT a present-content gap — Lever B as designed is a no-op. This is
the [[06]] "lever-type ≠ solution" precedent (like eligibility-steer=0), caught
by measurement *before* a paid re-run. Adding a "compare-across-orders" compute
would be a new speculative mechanism (make-or-break already offloads
criterion-comparison to the LLM) — do not build. **⋈ stays in the make-or-break
LLM region; it retires with scale (forensic 14B 23% → 32B 14%).**

### Lever C — over-action  [split: gateable vs LLM-residual]
The forensic's biggest surprise: over-action (executing disallowed / unrequested
/ impossible writes) is the de-facto dominant 32B mode (t62 cancels an order the
user only asked about).
- **Deterministically gateable slice:**
  - *Duplicate-write / idempotence gate* — block a 2nd identical (tool,args)
    write after the first (the return×3 loops: t69, t14, t51, t53, t83). New
    domain-general gate `kind: idempotent`. **But** the first write is usually
    already wrong → this cuts cost/over-action, **low pass-flip**. DEFER (bound,
    don't build for pass).
  - *Refuse-impossible* — t10 (cross-order refund routing) has empty gold; the
    request is structurally disallowed. Partial coverage via Lever A (refund PM
    not resolvable → block). 
- **NOT deterministically gateable (LLM scope-inference residual):** t62, t33,
  t57, t45, t31, t22 — the agent writes something the user never requested. Scope
  inference is the LLM's job (thesis: LLM = boundary translator), not a
  deterministic gate. → present/capability, **~6 cases stay open** as the
  irreducible make-or-break slice. Do NOT add a scope-guessing gate ([[06]]:
  lever-type ≠ solution; would over-block).

## Not addressed (make-or-break NO-GO region + artifacts)
- Criterion-formalize under joint constraint (t20 size-9∧max, t27, t58, t100) =
  LLM/compute; some present/calc-addressable, none SFT (make-or-break settled).
- Wrong-op (cancel vs modify vs exchange: t34, t66, t85) = intent = LLM.
- Benchmark/scope artifacts: REASON enum (dialogue under-determined), tracking#
  (out of calc scope), address format ("St"/"Street"), user-sim address variance
  (t109/t110). Not model errors — exclude from any lever's denominator.

## Aggregate static ceiling (measure-before-build, [[06]])
Deterministic levers (A refund-target + B ⋈-present) address a static ceiling of
~**10–12** of 51 fails; realistic pass-flip is lower (multi-cause overlap,
over-block risk). The remaining ~40 are LLM-residual (make-or-break region) or
artifacts. This bounds the levers as *incremental*, not a ceiling-breaker —
consistent with the headline (deterministic scaffold + base innate skill + TCO;
no learn wing on τ²).

## Recommended order (updated 2026-06-27 after verification)
- **Lever B — NO-GO (verified).** present_fields already `[status,address,items]`
  and the DISAMBIGUATION NOTE already fires with full item names/prices in every
  ⋈ case; the model still mis-picks → LLM operand residual, not present. No build.
- **Lever A — the one buildable deterministic lever**, rule verified 5/5 (DB). But
  since wrong-order (⋈) is now LLM-residual, Lever A only helps the
  *right-order-wrong-card* slice — small (≈ t63; most PAYMENT fails are ⋈-driven).
  Marginal ceiling ~2–4, with gift-card over-block risk.
- **Lever C — DEFER** (idempotence bounds cost, not pass); unrequested over-action
  = LLM scope residual (do NOT gate, [[06]]).
- **High-scale sweep (72B/235B) — abandoned** (coworker unavailable; models not
  local).

**Net:** after verification, the deterministic levers are largely **exhausted** —
the assembled stack (present+nested+gates+constraints+calc) already fires
maximally; the dominant residuals (⋈ comparison, criterion-formalize, over-action
scope, wrong-op intent) are **LLM operand/scope-inference** (make-or-break region,
retiring with scale) plus **benchmark artifacts** (reason enum, tracking#, format,
user-sim variance). This **strengthens** the headline: deterministic scaffold +
gates buy compliance and close the mechanical layers; the remaining task-pass gap
is base-model reasoning that scale (not scaffold, not SFT on τ²) buys.
**Decision point:** build Lever A for the small right-order-wrong-card slice
(low ROI, one clean deterministic policy) vs. declare the deterministic levers
exhausted and pivot to writing up the forensic+scale result. Recommend the latter
unless the paper needs the refund_target gate as a completeness point.
