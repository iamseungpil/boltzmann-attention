# -*- coding: utf-8 -*-
r"""x415 - task_050 의 역사: 과거 런 전수에서 pass 여부와 `get_payment_history_6183.months` 를 비교

사용자 지적 2026-08-19: "050 은 이전에 많이 pass 했던 태스크다 비교해보라."
"""
import glob
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SIM = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))
PAT = re.compile(r'"months"\s*:\s*"?(\d+)"?')


def rows():
    for p in sorted(glob.glob(os.path.join(SIM, "*.results.json.gz")), key=os.path.getmtime):
        tag = os.path.basename(p).split(".results")[0]
        try:
            raw = gzip.open(p, "rb").read().decode("utf-8", "replace")
        except Exception:
            continue
        if "task_050" not in raw:
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        for s in (d.get("simulations") or d.get("results") or []):
            if str(s.get("task_id")) != "task_050":
                continue
            rw = (s.get("reward_info") or {}).get("reward")
            months, gm = [], None
            for ck in ((s.get("reward_info") or {}).get("action_checks") or []):
                a = ck.get("action") or {}
                ar = json.dumps(a.get("arguments") or {}, ensure_ascii=False)
                if "get_payment_history" in ar or "get_payment_history" in str(a.get("name")):
                    m = PAT.search(ar)
                    if m:
                        gm = m.group(1)
            for m_ in (s.get("messages") or []):
                for tc in (m_.get("tool_calls") or []):
                    ar = json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
                    if "get_payment_history" in ar:
                        mm = PAT.search(ar)
                        months.append(mm.group(1) if mm else "∅")
            ac = ((s.get("reward_info") or {}).get("action_checks") or [])
            yield {"tag": tag, "trial": s.get("trial"), "rw": rw,
                   "gold_months": gm, "months": months,
                   "n_gold": len(ac), "n_match": sum(1 for c in ac if c.get("action_match")),
                   "mtime": os.path.getmtime(p)}


def main():
    R = list(rows())
    print("=" * 112)
    print("x415 · task_050 역사 — 런 %d개 시행" % len(R))
    print("=" * 112)
    print("%-44s %-3s %-6s %-7s %-9s %s" % ("tag", "tr", "reward", "gold월", "매치", "실제 months"))
    for r in R:
        print("%-44s %-3s %-6s %-7s %-9s %s"
              % (r["tag"][:44], r["trial"], r["rw"], r["gold_months"],
                 "%d/%d" % (r["n_match"], r["n_gold"]), ",".join(r["months"]) or "(호출0)"))
    ok = [r for r in R if (r["rw"] or 0) >= 1.0]
    print("\n통과 %d / %d" % (len(ok), len(R)))
    import collections
    print("\n## 실제 months 값 분포 × 통과여부")
    c = collections.Counter()
    for r in R:
        key = (",".join(r["months"]) or "(호출0)", "PASS" if (r["rw"] or 0) >= 1.0 else "FAIL")
        c[key] += 1
    for k, v in c.most_common():
        print("   %-22s %-5s %d" % (k[0][:22], k[1], v))
    return 0


sys.exit(main())
