"""fix_bank_creditcard.py — patch env/domains/bank/bank.py so the credit-card methods handle
the list-of-dict `credit_cards` schema actually used by the released task data.

Idempotent, anchor-based (asserts each anchor occurs exactly once), backs up to bank.py.bak,
and py_compile-checks the result. This is the candidate PR for the bug reported in
ISSUE_paste_ready_bank_creditcard.md (Leezekun/SOPBench `bank`).

Methods fixed (all assumed `credit_cards` was a dict keyed by number; data is a list of dicts):
  - cancel_credit_card             : list iterate -> match card["card_number"], remove(card)
  - pay_bill_with_credit_card      : list iterate -> match card["card_number"], mutate card
  - get_credit_card_info           : list iterate -> return the matching card dict
  - internal_check_credit_card_exist: rewrite the broken loop (uses card["card_number"])
NOTE: apply_credit_card also has a separate inverted-condition + dict-shape bug; left ALONE here
(it returns True and is not in the unsolvable set) — flag separately if desired.

RUN:  python fix_bank_creditcard.py /path/to/SOPBench
"""
import os, sys, py_compile

EDITS = [
    # --- cancel_credit_card ---
    ('        account = self.accounts.get(username)        \n'
     '        for card_num in account["credit_cards"]:\n'
     '            if card_num == card_number:\n'
     '                account["credit_cards"].pop(card_num, None)\n'
     '                return True\n'
     '        return False',
     '        account = self.accounts.get(username)\n'
     '        for card in account["credit_cards"]:\n'
     '            if card["card_number"] == card_number:\n'
     '                account["credit_cards"].remove(card)\n'
     '                return True\n'
     '        return False'),
    # --- pay_bill_with_credit_card ---
    ('        for card_num in account.get("credit_cards", {}):\n'
     '            if card_num == card_number: account["credit_cards"][card_num]["credit_balance"] += amount  \n'
     '        return True',
     '        for card in account.get("credit_cards", []):\n'
     '            if card["card_number"] == card_number: card["credit_balance"] += amount\n'
     '        return True'),
    # --- get_credit_card_info ---
    ('        for card_num in account["credit_cards"]:\n'
     '            if card_num == card_number: return True, account["credit_cards"][card_number]\n'
     '        return False, {}',
     '        for card in account["credit_cards"]:\n'
     '            if card["card_number"] == card_number: return True, card\n'
     '        return False, {}'),
    # --- internal_check_credit_card_exist ---
    ('        cc_number_found:bool = False\n'
     '        for card_num in account["credit_cards"]:\n'
     '            if not cc_number_found and card_num == card_number: cc_number_found = True\n'
     '        return True, cc_number_found ',
     '        cc_number_found:bool = False\n'
     '        for card in account["credit_cards"]:\n'
     '            if card["card_number"] == card_number: cc_number_found = True\n'
     '        return True, cc_number_found '),
]

MARKER = 'card["card_number"] == card_number'


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python fix_bank_creditcard.py <SOPBench_clone_dir>")
    clone = os.path.abspath(sys.argv[1])
    path = os.path.join(clone, "env", "domains", "bank", "bank.py")
    src = open(path, encoding="utf-8").read()
    if MARKER in src:
        print(f"  {path}: ALREADY PATCHED — skip")
        return
    open(path + ".bak", "w", encoding="utf-8").write(src)
    for old, new in EDITS:
        n = src.count(old)
        assert n == 1, f"anchor found {n}x (expected 1):\n{old[:90]}..."
        src = src.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(src)
    py_compile.compile(path, doraise=True)
    print(f"  {path}: PATCHED + compiles OK")
    print("  verify: python scripts/evidence_a_probe.py --domain bank  "
          "(cancel_credit_card / pay_bill_with_credit_card should now succeed)")


if __name__ == "__main__":
    main()
