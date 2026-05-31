# [DRAFT] GitHub issue for Leezekun/SOPBench — bank: oracle trajectory fails its own evaluator

> Repo: **https://github.com/Leezekun/SOPBench** (NOT `zli12321/SOPBench` — our memory/docs had
> the wrong handle; corrected 2026-06-01). Issue tracker active, only 1 unrelated issue (#1
> "license term") → this is a **novel report, not a duplicate**.
> Status: **DRAFT — blanks (⟦…⟧) get filled by `mre_bank_impossible.py` on the remote clone.**
> Tone: collaborative, evidence-first. Do NOT post until the MRE table is filled + verified.

---

## Title
`bank`: some `should_succeed=true` tasks appear structurally unsolvable — the ground-truth `directed_action_graph` does not pass `evaluator_function_directed_graph`

## Body

Thanks for releasing SOPBench — we've been using the `bank` domain (Qwen via vLLM) to study
structured planning. While reproducing the leaderboard we found a subset of
`action_should_succeed=true` bank tasks that **no agent can pass, because the task's own
ground-truth answer doesn't pass the evaluator.**

### What we did (no agent, no LLM — pure oracle replay)
For every `bank` task with `action_should_succeed=true`, we **replay the task's own
`directed_action_graph`** (the oracle call sequence, with each node's `{param→slot}` arg
binding resolved from `user_known` + the acting account row) on the **strict** domain system,
then score it with the **unchanged** `evaluator_function_directed_graph`. If the oracle answer
fails, the task is unpassable by construction.

Repro (clone root, py≥3.10):
```bash
python scripts/mre_bank_impossible.py --domain bank --crosscheck --out mre_bank.json
```
(script attached below; it only imports your `env.*` and replays the GT graph.)

### Result
- should_succeed=true tasks: **⟦N_should_true⟧**
- oracle-replay **FAILS**: **⟦N_impossible⟧ tasks / ⟦K⟧ unique goals**
- ⇒ effective ceiling on should_succeed=true ≈ **⟦ceiling%⟧** (not 100%)

| goal | oracle replay | dominant failing sub-check | passed by ANY author model? |
|---|---|---|---|
| `cancel_credit_card` | FAIL | ⟦database_match⟧ | ⟦none⟧ |
| `pay_bill_with_credit_card` | FAIL | ⟦database_match⟧ | ⟦none⟧ |
| ⟦…⟧ | | | |

Cross-check: in `output/bank/*.json`, these goals are passed by **⟦0⟧** of the released models
— consistent with structural unsolvability rather than task difficulty.

### Root cause (our hypothesis — please verify)
The `bank` account schema stores `credit_cards` as a **list**:
```
credit_cards observed: type=⟦list⟧, sample=⟦{...}⟧
```
but the domain method(s) handling credit cards index/compare it as if it were a **dict** keyed
by card id, so the card lookup `⟦card_num (dict) == card_number (str)⟧` can never match →
the cancel/pay path always fails its post-condition `database_match`.

Suspected site(s):
```
⟦env/.../bank...py:LINE⟧: ⟦offending line⟧
```

### Why it matters
These tasks lower the reported ceiling for **all** evaluated models equally, so they bias
absolute pass-rates downward on `bank`. Flagging or fixing them would make the `bank` numbers
cleaner for everyone.

### Suggested resolutions (any one)
1. Fix the credit-card lookup to match the list schema (or store `credit_cards` as a dict).
2. Mark the affected task instances as `action_should_succeed=false` (refusal) if that's the
   intended semantics.
3. Document them as known-unsolvable so reproductions can exclude them.

Happy to send a PR for (1) if you point us at the intended schema. MRE script + full
`mre_bank.json` artifact attached.

---

## Pre-post checklist (must all be ✅ before filing)
- [ ] `mre_bank_impossible.py` ran on a **clean** Leezekun/SOPBench clone (no our patches) in py≥3.10.
- [ ] N_impossible ≥ 1 with the oracle replay genuinely failing (not an arg-sourcing artifact —
      confirm by also trying `--toposort` and by spot-checking one task's replayed `content`).
- [ ] The failing sub-check is **database_match** (or constraint), NOT `dirgraph_satisfied`
      (a dirgraph mismatch could be a replay-order artifact, not a benchmark bug — investigate
      before claiming).
- [ ] Root-cause line(s) located in source and the dict-vs-list mismatch confirmed by reading
      the method (not just grep).
- [ ] Cross-check shows 0 author-model passes for the impossible goals.
- [ ] Re-searched Leezekun/SOPBench issues at post time for any new duplicate.
- [ ] One task fully narrated end-to-end (inputs → calls → evaluator dict) in an appendix.

## Honest caveats / failure modes to rule out first
- **Arg-sourcing**: our replay resolves args from `user_known`+account row. If the *intended*
  oracle expects an arg we don't supply, the failure could be ours, not the benchmark's. The
  `--toposort` cross-run + reading the replayed `content` per call guards this.
- **Multi-task goals**: a goal key holds many task instances; report per-instance counts, not
  just unique goals, so "26%" is unambiguous.
- **Refusal semantics**: confirm these really are labelled `should_succeed=true` (impossible)
  and not mislabeled refusals.
