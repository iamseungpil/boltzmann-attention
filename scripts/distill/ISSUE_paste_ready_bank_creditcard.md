bank: cancel_credit_card / pay_bill_with_credit_card are unsolvable for any agent — credit_cards is a list-of-dict in task data but the domain methods treat it as a dict

---

Thanks for releasing SOPBench! While reproducing the `bank` leaderboard we found a set of
`action_should_succeed=true` tasks that cannot be passed by **any** agent — including the oracle
itself — because the credit-card domain methods assume a `credit_cards` schema that the released
task data does not use.

### Evidence A — the strict env cannot make the goal succeed even with correct args

For each `action_should_succeed=true` bank instance, we satisfy the preconditions
(`login_user` / `authenticate_admin_password`) and call the goal action on `bank_strict` with the
**ground-truth args** from `user_known`, then score with the unchanged
`evaluator_function_directed_graph`:

| goal | instances | observed goal-call result | `action_successfully_called` | passed by any of ~26 released models |
|---|---|---|---|---|
| `cancel_credit_card` | 6 | returns `False` | False (all) | none |
| `pay_bill_with_credit_card` | 2 | raises `KeyError: 'credit_limit'` | False (all) | none |

These 8 instances fail for the **oracle**, so they are unpassable by construction. (This is
order-independent: we satisfy preconditions explicitly and then call the goal, so it is not a
call-ordering artifact.)

### Cross-check B

Across the 53 shipped `output/bank/ast_*.json` result files, these same 8 instances are passed
by **0** of the ~26 released models — consistent with structural unsolvability rather than
difficulty.

### Root cause

In the released `bank` tasks, an account's `credit_cards` is a **list of dicts**, e.g.

```json
"credit_cards": [{"card_number": "2357 1113 1719 2329", "credit_limit": 250.0, "credit_balance": 0.0}]
```

but `env/domains/bank/bank.py` treats `credit_cards` as a **dict keyed by card number**:

- `cancel_credit_card` (around L209–213):
  ```python
  for card_num in account["credit_cards"]:
      if card_num == card_number:
          account["credit_cards"].pop(card_num, None)
          return True
  return False
  ```
  Iterating a list yields the **dict elements**, so `card_num` (a dict) never equals
  `card_number` (a str); the loop matches nothing and the method `return False`.

- `pay_bill_with_credit_card` (around L189–190):
  ```python
  for card_num in account.get("credit_cards", {}):
      if card_num == card_number: account["credit_cards"][card_num]["credit_balance"] += amount
  ```
  same list-vs-dict assumption; in our run this path ends in `KeyError: 'credit_limit'` (the
  credit-card lookup / credit-limit check operates on the dict-keyed assumption).

- `get_credit_card_info` (around L254–255) and `internal_check_credit_card_exist` (around
  L261–265) have the same `for card_num in account["credit_cards"]: ... account["credit_cards"][card_number]`
  pattern, so any constraint that depends on them is affected too.

The file is also internally inconsistent: `default_data1` seeds `credit_cards` as a **list of
dicts** (matching the task data), while `default_data` and the docstring ("dictionary of credit
cards, hashed by their credit card numbers") assume a **dict**.

### Why it matters

These instances cap the reported `bank` ceiling for every model equally (in our count the
effective `should_succeed=true` ceiling is 40/48), biasing absolute pass-rates downward.

### Suggested fix (any one)

1. Make the credit-card methods iterate the list-of-dict form, matching
   `card["card_number"] == card_number` and operating on `card` directly — consistent with
   `default_data1` and the released task data. (Happy to send a PR.)
2. Or seed the task `initial_database` with the dict form the methods expect.
3. Or document/flag these instances so reproductions can exclude them.

### Repro (no agent / no LLM)

```bash
# from the clone root, py>=3.10
python scripts/evidence_a_probe.py --domain bank --out evidence_a_bank.json
```
The probe satisfies each goal's preconditions and then calls the goal with ground-truth args on
`bank_strict`, scoring with the unchanged evaluator.
