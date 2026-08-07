# -*- coding: utf-8 -*-
"""결정론만으로 gold 상품에 도달하는가 — 유료 런 전에, 코퍼스에서 직접 [[09]].

레버를 더 얹기 전에 답해야 하는 물음이 있다: **원장·계좌·정책 상수만으로 gold가 지목한 상품이
유일하게 결정되는가.** 되면 남은 실패는 집행 문제이고, 안 되면 레버를 아무리 고쳐도 못 맞힌다 —
그리고 그 판정은 궤적 없이, 돈 없이 할 수 있다.

읽는 것: 각 상품 문서에서 (연간 상한 · 추천인 보너스 · 피추천인 보너스 · 최소 관계기간 ·
최소 예치금). 코퍼스가 같은 상수를 여러 문형으로 쓰므로 전부 받는다.
푸는 것: 태스크의 시딩 원장/계좌 + 그 상수 → 후보 집합 → gold와 대조.

⚠이 스크립트는 **분석 도구**다(엔진 아님). 코퍼스 텍스트를 읽는 것이 이 도구의 일이고,
[[59]]의 금지 대상(엔진이 도메인 텍스트에서 사실을 뽑는 것)이 아니다 — 여기서 뽑은 값은
런타임에 쓰이지 않고, 사람이 판정을 내리기 위한 것이다.
"""

import collections
import datetime
import glob
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
DOM = os.path.join(TAU2, "data", "tau2", "domains", "banking_knowledge")
TODAY = datetime.date(2025, 11, 14)

CAP = [r"Annual limit\**:?\**\s*(\d+)\s*referral", r"up to (\d+) referral bonuses per calendar year",
       r"Annual (?:maximum|cap)\**:?\**\s*(\d+)\s*referral", r"Maximum referrals(?: per year)?\s*[:|]\s*(\d+)",
       r"Maximum per year\**:?\**\s*(\d+)\s*referral", r"\|\s*Annual cap\s*\|\s*(\d+)\s*referrals?\s*\|",
       r"\|\s*Maximum referrals per year\s*\|\s*(\d+)\s*\|"]
MINE = [r"Your bonus\**:?\**\s*\$([\d,]+)", r"\$([\d,]+) for each (?:qualifying|successful) referral"]
THEIRS = [r"Their bonus\**:?\**\s*\$([\d,]+)", r"\$([\d,]+) welcome bonus"]
TENURE = [r"minimum relationship duration of (\d+) days", r"(\d+)[- ]day tenure",
          r"tenure threshold[^.]{0,40}?(\d+) days", r"requires? (\d+) days of tenure"]
DEPOSIT = [r"deposit at least \$([\d,]+)", r"must deposit \$([\d,]+)",
           r"qualifying deposit[^.]{0,40}?\$([\d,]+)"]


def _num(s):
    return int(str(s).replace(",", ""))


def _first(txt, pats):
    for p in pats:
        m = re.search(p, txt, re.I)
        if m:
            return _num(m.group(1))
    return None


def products():
    """문서 파일명 → 상품, 그 상품의 모든 문서를 합쳐 상수 추출."""
    by = collections.defaultdict(list)
    for p in glob.glob(os.path.join(DOM, "documents", "*")):
        fn = os.path.basename(p).lower()
        m = re.match(r"doc_(business_)?(checking_accounts|savings_accounts|business_checking_accounts)"
                     r"_(.+?)_\d+\.json$", fn)
        if not m:
            continue
        slug = re.sub(r"_account$", "", m.group(3))
        cls = "business" if (m.group(1) or "business" in m.group(2)) else "personal"
        try:
            by[(cls, slug)].append(io.open(p, encoding="utf-8", errors="replace").read())
        except Exception:
            pass
    out = {}
    for (cls, slug), texts in by.items():
        blob = "\n".join(texts)
        name = " ".join(w.capitalize() for w in slug.split("_")) + " Account"
        out[name] = {"class": cls, "cap": _first(blob, CAP), "mine": _first(blob, MINE),
                     "theirs": _first(blob, THEIRS), "tenure": _first(blob, TENURE),
                     "deposit": _first(blob, DEPOSIT), "docs": len(texts)}
    return out


def task(tid):
    return json.load(io.open(os.path.join(DOM, "tasks", tid + ".json"), encoding="utf-8"))


def seeded(t, table):
    return ((((t.get("initial_state") or {}).get("initialization_data") or {})
             .get("agent_data") or {}).get(table) or {}).get("data") or {}


def used_counts(t):
    c = collections.Counter()
    for r in seeded(t, "referrals").values():
        if r.get("referred_account_type"):
            c[r["referred_account_type"]] += 1
    return c


def tenure_days(t):
    ds = []
    for a in seeded(t, "accounts").values():
        try:
            ds.append(datetime.datetime.strptime(a.get("date_opened", ""), "%m/%d/%Y").date())
        except Exception:
            pass
    return (TODAY - min(ds)).days if ds else None


def gold_types(t):
    return [a.get("arguments", {}).get("account_type")
            for a in ((t.get("evaluation_criteria") or {}).get("actions") or [])
            if a.get("name") == "submit_referral"]


def main():
    P = products()
    print("코퍼스에서 뽑은 상품 %d종 (상한/내보너스/상대보너스/최소기간/최소예치)" % len(P))
    for n, v in sorted(P.items()):
        print("  %-34s %-9s cap=%-4s mine=%-6s theirs=%-6s tenure=%-5s dep=%s"
              % (n, v["class"], v["cap"], v["mine"], v["theirs"], v["tenure"], v["deposit"]))
    for tid in (sys.argv[1:] or ["task_100", "task_101"]):
        t = task(tid)
        used, ten, gold = used_counts(t), tenure_days(t), gold_types(t)
        print("\n" + "=" * 96)
        print("== %s ==  gold 제출 = %s" % (tid, gold))
        print("   관계기간 %s일 · 시딩 원장 %d행" % (ten, sum(used.values())))
        rows = []
        for n, v in sorted(P.items()):
            cap, u = v["cap"], used.get(n, 0)
            rem = (cap - u) if cap is not None else None
            ok_ten = None if (v["tenure"] is None or ten is None) else (ten >= v["tenure"])
            rows.append((n, v["class"], cap, u, rem, v["tenure"], ok_ten, v["mine"], v["theirs"]))
        print("   %-34s %-9s %-5s %-4s %-5s %-7s %-6s %s" %
              ("상품", "class", "cap", "쓴", "잔여", "tenure", "충족", "보너스(나/상대)"))
        for n, cls, cap, u, rem, tn, ok, mine, th in rows:
            mark = "★" if n in gold else " "
            print("  %s%-34s %-9s %-5s %-4s %-5s %-7s %-6s %s/%s"
                  % (mark, n, cls, cap, u, rem, tn, ok, mine, th))
        # 결정론이 gold를 유일하게 뽑는가
        for g in gold:
            v = P.get(g)
            print("   ⇒ gold '%s' : 코퍼스에 %s" % (g, "있음" if v else "**없음**"))


if __name__ == "__main__":
    main()
