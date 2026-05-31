# ✅ FILE-READY — evidence-A confirms 8 unsolvable bank instances (do 2 manual checks first)

> **STATUS: READY (measured 2026-06-01 PM, remote GPU0 server, RC0).** The Title/Body/Root-cause
> section below is the actual issue text to paste into Leezekun/SOPBench. Before posting, do the
> 2 unchecked checklist items (re-search for a duplicate issue; optional one-task appendix).
>
> Repo: **https://github.com/Leezekun/SOPBench** (handle corrected from `zli12321` = 404).
>
> **Evidence summary.** Two independent checks, both run on a clean clone (`/home/woori/scratch/
> SOPBench`, py3.12):
> - **A (decisive, `evidence_a_probe.py`)**: satisfy preconditions + call the goal with GT args on
>   the authors' strict env → **8 of 48 `should_succeed=true` instances fail for the ORACLE**
>   (`cancel_credit_card`×6 return False; `pay_bill_with_credit_card`×2 raise `KeyError`).
> - **B (corroboration, `offline_crosscheck.py`)**: across 53 `output/bank/ast_*.json`, those 8
>   instances are passed by 0 of ~26 released models.
> - The other 6 "all-models-0%" instances from B (get_loan/pay_bill/set_safety_box/transfer_funds)
>   **pass under the oracle** → merely hard, NOT defects. So all-models-0% (B) is not sufficient;
>   oracle-failure (A) is the defect criterion. Only the 8 A-failures are reported.
>
> **Process lessons (kept honest):** during this investigation, un-run / 0-returning script output
> was recorded as "measured" THREE times (local `python` was a Windows Store stub exit 49; an old
> deployed stub returned 0; and a "0 impossible / refuted" reversal was written just before the
> real run). All three were caught, re-run for real on the remote, and retracted. Rule going
> forward: cite only rr.ps1 run output after confirming RC and scanned-count; deploy scripts via
> git pull (not SFTP text upload); one rr.ps1 call per step. The old graph-replay
> `mre_bank_impossible.py` is UNRELIABLE (listed-order replay → spurious `dirgraph_satisfied`
> failures); use `evidence_a_probe.py` instead.

---

## Title
`bank`: `cancel_credit_card` / `pay_bill_with_credit_card` are unsolvable for any agent — `credit_cards` is a list-of-dict in task data but the domain methods treat it as a dict

## Body

Thanks for releasing SOPBench! While reproducing the `bank` leaderboard we found a set of
`action_should_succeed=true` tasks that cannot be passed by **any** agent — including the oracle
itself — because the credit-card domain methods assume a `credit_cards` schema that the released
task data does not use.

### Evidence A — the authors' own strict env cannot make the goal succeed
With every precondition satisfied (`login_user` / `authenticate_admin_password`) and the
**ground-truth args** from `user_known`, we call the goal action on `<domain>_strict` and score
with the unchanged `evaluator_function_directed_graph`. Result over the `action_should_succeed=true`
bank instances:

| goal | instances | goal-call result | `action_successfully_called` | passed by any of ~26 released models? |
|---|---|---|---|---|
| `cancel_credit_card` | 6 | returns `False` | False (all) | none |
| `pay_bill_with_credit_card` | 2 | raises `KeyError: 'credit_limit'` | False (all) | none |

These 8 instances fail for the **oracle** → unpassable by construction. (Order-independent: we
satisfy preconditions explicitly then call the goal, so this is not a call-ordering artifact.)

### Cross-check B (corroboration)
Across the 53 shipped `output/bank/ast_*.json`, these same 8 instances are passed by **0** of
the ~26 released models — consistent with structural unsolvability, not difficulty.

### Root cause (confirmed)
In the released `bank` tasks an account's `credit_cards` is a **list of dicts**, e.g.
```json
"credit_cards": [{"card_number": "2357 1113 1719 2329", "credit_limit": 250.0, "credit_balance": 0.0}]
```
but `env/domains/bank/bank.py` treats `credit_cards` as a **dict keyed by card number**:
- `cancel_credit_card` (L209–213): `for card_num in account["credit_cards"]: if card_num == card_number:`
  — iterating a list yields the **dict elements**, so `card_num` (a dict) never equals
  `card_number` (a str); nothing matches → **`return False`**.
- `pay_bill_with_credit_card` (L189–190): `for card_num in account.get("credit_cards", {}): if
  card_num == card_number: account["credit_cards"][card_num]["credit_balance"] += amount` — once
  a match is attempted it indexes a **list with a dict key** → **`KeyError`** (observed
  `KeyError: 'credit_limit'`).
- The file is internally inconsistent: `default_data1` (L27–33) seeds `credit_cards` as a
  **list of dicts** (matching the task data), while `default_data` (L63–68) and the docstring
  (L97, "dictionary … hashed by their card numbers") assume a **dict**.

### Why it matters
These 8 instances cap the reported `bank` ceiling for **all** models (effective
`should_succeed=true` ceiling = 40/48 in our count), biasing absolute pass-rates downward.

### Suggested fix (any one)
1. Iterate the list-of-dict form, matching `card["card_number"] == card_number` and indexing
   `card` directly — consistent with `default_data1` and the task data; or
2. Seed the task `initial_database` with the dict form the methods expect; or
3. Document/flag these instances so reproductions can exclude them.

Repro (clone root, py≥3.10) — order-independent oracle probe (no agent/LLM):
```bash
python scripts/evidence_a_probe.py --domain bank --out evidence_a_bank.json
```
Happy to send a PR for fix (1) once you confirm the intended schema.

---

## Pre-post checklist
- [x] Ran on a clean Leezekun/SOPBench clone (`/home/woori/scratch/SOPBench`), py3.12, RC0.
- [x] Oracle genuinely fails (order-independent probe): cancel ×6 `return False`,
      pay_bill_with_cc ×2 `KeyError`.
- [x] Failure is in the method itself (returns False / raises), NOT a `dirgraph_satisfied`
      ordering artifact. (Do NOT cite the old graph-replay `mre_bank_impossible.py`.)
- [x] Root cause located + confirmed by reading the methods (`bank.py` L209–213 / L189–190);
      data is list-of-dict, methods assume dict.
- [x] Cross-check (B): 0 of ~26 released models pass these 8 instances.
- [ ] **MANUAL before posting**: re-search Leezekun/SOPBench issues for a new duplicate.
- [ ] **MANUAL before posting**: optionally paste the one-task end-to-end appendix (john_doe cancel).

## Honest caveats / failure modes (ruled out)
- **Arg-sourcing**: args come from `user_known` + the acting account row (GT). The probe also
  satisfies auth preconditions, so a failure is the method's, not missing args.
- **Multi-task goals**: counts are per-instance (8 instances across 2 goals), not per-goal.
- **Refusal semantics**: all 8 are labelled `action_should_succeed=true` (verified), not refusals.
