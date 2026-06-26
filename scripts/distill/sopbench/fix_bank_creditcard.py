"""fix_bank_creditcard.py — patch env/domains/bank/bank.py so the credit-card methods handle
the list-of-dict `credit_cards` schema actually used by the released task data.

Idempotent, anchor-based (asserts each anchor occurs exactly once), backs up to bank.py.bak,
and py_compile-checks the result. Candidate PR for the bug in
ISSUE_paste_ready_bank_creditcard.md (Leezekun/SOPBench `bank`).

Methods fixed (all assumed `credit_cards` was a dict keyed by number; data is a list of dicts):
  - cancel_credit_card              : iterate list, match card["card_number"], remove(card)
  - pay_bill_with_credit_card       : iterate list, match card["card_number"], mutate card
  - get_credit_card_info            : iterate list, return the matching card dict
  - internal_check_credit_card_exist: iterate list, match card["card_number"]
NOTE: apply_credit_card also has a separate dict-shape assumption; left ALONE (it returns True
and is not in the unsolvable set). Flag separately if desired.

Anchors use the distinctive buggy lines and tolerate trailing whitespace via .rstrip-free exact
match — verify with --check (applies to a /tmp copy, does not touch the clone).

RUN:  python fix_bank_creditcard.py /path/to/SOPBench
"""
import os, sys, py_compile

# (anchor, replacement). Anchors copied verbatim from the released source (incl. trailing ws).
EDITS = [
    # cancel_credit_card  (L209-213)
    ('        for card_num in account["credit_cards"]:\n'
     '            if card_num == card_number:\n'
     '                account["credit_cards"].pop(card_num, None)\n'
     '                return True\n'
     '        return False',
     '        for card in account["credit_cards"]:\n'
     '            if card["card_number"] == card_number:\n'
     '                account["credit_cards"].remove(card)\n'
     '                return True\n'
     '        return False'),
    # pay_bill_with_credit_card  (L189-190; note 2 trailing spaces after amount)
    ('        for card_num in account.get("credit_cards", {}):\n'
     '            if card_num == card_number: account["credit_cards"][card_num]["credit_balance"] += amount  \n',
     '        for card in account.get("credit_cards", []):\n'
     '            if card["card_number"] == card_number: card["credit_balance"] += amount\n'),
    # get_credit_card_info  (L254-256)
    ('        for card_num in account["credit_cards"]:\n'
     '            if card_num == card_number: return True, account["credit_cards"][card_number]\n'
     '        return False, {}',
     '        for card in account["credit_cards"]:\n'
     '            if card["card_number"] == card_number: return True, card\n'
     '        return False, {}'),
    # internal_check_credit_card_exist  (L276-277)
    ('        for card_num in account["credit_cards"]:\n'
     '            if not cc_number_found and card_num == card_number: cc_number_found = True\n',
     '        for card in account["credit_cards"]:\n'
     '            if card["card_number"] == card_number: cc_number_found = True\n'),
]

MARKER = 'card["card_number"] == card_number'


def patch_text(src):
    for old, new in EDITS:
        n = src.count(old)
        assert n == 1, f"anchor found {n}x (expected 1):\n{old[:90]!r}"
        src = src.replace(old, new, 1)
    return src


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python fix_bank_creditcard.py <SOPBench_clone_dir> [--check]")
    clone = os.path.abspath(sys.argv[1])
    check = "--check" in sys.argv[2:]
    path = os.path.join(clone, "env", "domains", "bank", "bank.py")
    src = open(path, encoding="utf-8").read()
    if MARKER in src:
        print(f"  {path}: ALREADY PATCHED — skip")
        return
    new = patch_text(src)
    if check:
        tmp = "/tmp/bank_patched_check.py"
        open(tmp, "w", encoding="utf-8").write(new)
        py_compile.compile(tmp, doraise=True)
        print(f"  CHECK OK: all 4 anchors matched, patched copy compiles ({tmp}). Clone untouched.")
        return
    open(path + ".bak", "w", encoding="utf-8").write(src)
    open(path, "w", encoding="utf-8").write(new)
    py_compile.compile(path, doraise=True)
    print(f"  {path}: PATCHED + compiles OK (backup: bank.py.bak)")
    print("  verify: python scripts/evidence_a_probe.py --domain bank")


if __name__ == "__main__":
    main()
