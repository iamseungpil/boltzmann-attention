# -*- coding: utf-8 -*-
r"""x199 — A3 가 **문서에 실재하는데 놓친** 예치 문턱을 채운다 (유료 0·엔진 변경 0).

## 왜 이것이 gold 맞추기가 아닌가 ([[23]] · ⛔0)

- 출처는 **정책 문서**뿐이다. 각 행은 `source.doc` + **축자 인용**을 들고 오고, 인용이 문서
  본문의 부분 문자열인지 원격에서 대조해 확인했다(4/4 EXACT).
- 빌더 누락임이 **자기 증거로 확정**된다: `Gold Years` 의 `deposit_window_days` 행이 이미
  *"They must deposit $1,000 within 90 days of account opening"* 를 인용하고 있다 — 같은 문장에서
  90 은 뽑고 **$1,000 은 버렸다**.
- **선택적으로 채우지 않는다**: 개인 체킹 제품 중 문서가 예치 문턱을 말하는 세 곳을 **전부**
  채운다. 그중 `Light Blue`·`Light Green` 은 채우면 오히려 표에 **남는다**(문턱이 낮아 통과).
- ⛔**카드는 채우지 않는다.** 카드는 `qualifying_spend_usd` 를 갖는다 — 예치는 빈칸이 아니라
  **없는 개념**이고 채우면 날조다([[25]]).

엔진·`eligible` 기준은 **손대지 않는다**. 기준에는 이미
`{"axis": "qualifying_deposit_usd", "operand": "stated", "compare": "ge"}` 가 있다 — 값이 없어서
안 걸린 것뿐이다.

두 층([[24]])에 **같은 객체**를 넣고, 넣은 뒤 두 층의 `rows` 가 동일한지 검사한다.

실행: py -3 x199_fill_a3_deposits.py [--apply]
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"]

# 다른 `qualifying_deposit_usd` 행과 **같은** 판정 성격이다 — 도구 경계에서 걸리는 게이트가
# 아니라 권고를 만들 때 쓰는 피연산자. basis 문구는 기존 행에서 축자로 가져온다.
BASIS = ("**도구 경계에서 판정되지 않는다** — 권고를 만들 때 쓰는 피연산자다. 억지로 도구에 "
         "붙이면 없는 게이트를 지어내는 것이라 비워 둔다(엔진은 표면화만 한다).")

NEW = [
    ("Gold Years", "qualifying_deposit_usd", 1000,
     "doc_checking_accounts_gold_years_account_010",
     "They must deposit $1,000 within 90 days of account opening"),
    ("Light Blue", "qualifying_deposit_usd", 500,
     "doc_checking_accounts_light_blue_account_007",
     "They deposit at least $500 within 60 days"),
    ("Light Green", "qualifying_deposit_usd", 100,
     "doc_checking_accounts_light_green_account_002",
     "The person you refer must deposit at least $100 within 90 days of opening their account."),
    ("Light Green", "deposit_window_days", 90,
     "doc_checking_accounts_light_green_account_002",
     "The person you refer must deposit at least $100 within 90 days of opening their account."),
]


def row(subject, axis, value, doc, quote):
    return {"applies_to": {"consumers": [], "basis": BASIS},
            "subject": subject, "axis": axis, "value": value,
            "against": None, "compare": None, "when": [],
            "source": {"doc": doc, "quote": quote, "quote_match": "exact"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    made = {}
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        txt = io.open(p, encoding="utf-8").read()
        doc = json.loads(txt)
        # 재직렬화가 바이트 동일한지 **먼저** 확인한다 — 아니면 손대지 않는다.
        back = json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "")
        if back != txt:
            print("  중단: %s 는 재직렬화가 바이트 동일하지 않다" % rel)
            return 1
        rows = doc["policy_ontology"]["rows"]
        have = {(r.get("subject"), r.get("axis")) for r in rows}
        added = 0
        for subj, ax, val, dc, q in NEW:
            if (subj, ax) in have:
                print("  건너뜀 (이미 있다): %s / %s" % (subj, ax))
                continue
            # 같은 주어의 마지막 행 뒤에 넣어 묶음을 유지한다
            idx = max([i for i, r in enumerate(rows) if r.get("subject") == subj] or [len(rows) - 1])
            rows.insert(idx + 1, row(subj, ax, val, dc, q))
            added += 1
        made[rel] = (doc, added, txt.endswith("\n"))
        print("  %-38s +%d행 → %d행" % (rel, added, len(rows)))

    a_rows = made[LAYERS[0]][0]["policy_ontology"]["rows"]
    b_rows = made[LAYERS[1]][0]["policy_ontology"]["rows"]
    same = json.dumps(a_rows, ensure_ascii=False, sort_keys=True) == \
        json.dumps(b_rows, ensure_ascii=False, sort_keys=True)
    print("  두 층 rows 동일: %s" % ("OK" if same else "**불일치**"))
    if not same:
        return 1
    if not a.apply:
        print("\n(미적용 — 쓰려면 --apply)")
        return 0
    for rel, (doc, _n, nl) in made.items():
        p = os.path.join(HERE, rel)
        out = json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if nl else "")
        io.open(p, "w", encoding="utf-8", newline="").write(out)
        print("  wrote %s" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
