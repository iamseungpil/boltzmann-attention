# -*- coding: utf-8 -*-
r"""x430 — **계좌 클래스 사실표를 정책 축자로 저작** (사용자 지시 2026-08-20 · C462 전범)

## 왜
x428·x429 로 *"상한은 formalize 없이 못 잰다"* 까지 왔다. C462(`get_atm_fee_discrepancies`)가 밟은
길이 유일하게 정직한 길이다 — **정책 축자 추출(에이전트·gold 미접촉) → 그 위에서 판정**.

여기서는 **checking 계좌 클래스 전체 × 속성 전체**를 문서에서 뽑는다. 표적 태스크가 무엇을 필요로
하는지에 따라 속성을 고르지 않는다(고르면 gold-fit 이 된다·[[23]]). 뽑는 규칙은 하나 —
**문서가 값을 명시한 속성이면 전부 싣는다.**

## 규율
★값마다 **문서 id + 축자 문장**을 함께 싣는다. 못 대면 그 칸은 비운다(추정 금지).
★한 속성에 **서로 다른 값**이 여러 문서에서 나오면 충돌로 표시하고 둘 다 남긴다(우리가 고르지 않는다).
★gold 를 보지 않는다. 이 스크립트는 `sim_results` 를 읽지 않는다 — 문서 디렉터리만 읽는다.

사용: py -3 x430_account_facts.py [--docdir ...] [--family checking_accounts]
"""
import argparse
import collections
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DOCDIR = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

# 속성 = (이름, 그 속성을 부르는 축자 표현들). 값은 문장 안의 첫 통화·퍼센트·정수.
ATTRS = [
    ("monthly_maintenance_fee", ["monthly maintenance fee", "monthly maintenance"]),
    ("waiver_min_daily_balance", ["minimum daily balance", "waiver requirement", "minimum balance to waive"]),
    ("overdraft_fee", ["overdraft fee", "overdraft fees", "overdrafts"]),
    ("overdraft_protection_transfer_fee", ["overdraft protection transfer"]),
    ("oon_atm_fee", ["out-of-network atm", "out of network atm", "foreign atm"]),
    ("free_oon_atm_per_month", ["free out-of-network atm", "free out of network atm"]),
    ("apy", ["apy"]),
    ("daily_atm_withdrawal_limit", ["daily atm withdrawal limit", "atm withdrawal limit"]),
    ("min_age", ["minimum primary holder age"]),
    ("max_age", ["maximum primary holder age"]),
    ("monthly_deposit_limit", ["direct deposits up to", "mobile check deposit daily limit"]),
]
RE_VAL = re.compile(r"(\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s?%|\bnone\b|\bno\b|\bunlimited\b|\b\d+\b)", re.I)


def sentences(txt):
    return [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+|\n", txt) if s.strip()]


def norm_val(v):
    v = v.strip()
    if v.lower() in ("none", "no"):
        return "0"
    return v.replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--family", default="checking_accounts")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pre = "doc_%s_" % a.family
    files = sorted(f for f in os.listdir(a.docdir) if f.startswith(pre) and f.endswith(".json"))
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        cls = re.sub(r"_\d+\.json$", "", f[len(pre):])
        with io.open(os.path.join(a.docdir, f), encoding="utf-8") as fh:
            d = json.load(fh)
        txt = (d.get("title") or "") + ". " + (d.get("content") or "")
        for s in sentences(txt):
            low = s.lower()
            for attr, keys in ATTRS:
                if not any(k in low for k in keys):
                    continue
                # 속성어 뒤쪽에서 첫 값
                pos = min(low.find(k) for k in keys if k in low)
                m = RE_VAL.search(s, pos)
                if not m:
                    continue
                table[cls][attr].append({"value": norm_val(m.group(1)),
                                         "doc": f.replace(".json", ""), "quote": s[:220]})
    print("=" * 108)
    print("x430 · %s · 클래스 %d · 문서 %d" % (a.family, len(table), len(files)))
    print("=" * 108)
    attrs = [x[0] for x in ATTRS]
    print("%-26s %s" % ("class", " ".join("%-12s" % x[:12] for x in attrs[:6])))
    out = {}
    for cls in sorted(table):
        row, cells = {}, []
        for attr in attrs:
            vals = table[cls].get(attr) or []
            uniq = sorted({v["value"] for v in vals})
            row[attr] = {"values": uniq, "conflict": len(uniq) > 1,
                         "evidence": vals[:4]}
            cells.append(("|".join(uniq)[:12]) if uniq else "-")
        out[cls] = row
        print("%-26s %s" % (cls[:26], " ".join("%-12s" % c for c in cells[:6])))
    conf = [(c, k) for c, r in out.items() for k, v in r.items() if v["conflict"]]
    print("\n충돌(한 속성에 값 2개 이상) %d건 — 우리가 고르지 않고 둘 다 남긴다" % len(conf))
    for c, k in conf[:12]:
        print("   %-24s %-34s %s" % (c[:24], k, "|".join(out[c][k]["values"])[:40]))
    miss = [(c, k) for c, r in out.items() for k, v in r.items() if not v["values"]]
    print("\n미기재(문서가 값을 안 준 칸) %d" % len(miss))
    p = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                              "reports", "facet_rft_2026", "x430_account_facts.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
