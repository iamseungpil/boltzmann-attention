# -*- coding: utf-8 -*-
"""DB에 무엇이 쓰이는가를 정하는 것은 도구 호출이 아니라 **직전 답변의 산문**인가.

`submit_referral`은 손님 도구다. 우리 게이트는 에이전트의 호출만 가로막을 수 있고, 손님이 제 도구로
쓰는 write는 보지도 못한다. 그런데 DB 채점은 그 write로 결정된다. 그래서 이 물음이 남는다 —
손님은 무엇을 근거로 그 인자를 고르는가.

이 도구는 각 `submit_referral`(손님 호출) 앞의 **가장 가까운 에이전트 답변**을 찾아, 그 답변이
그 유형 이름을 담고 있었는지 본다. 담고 있다면 그 write의 결정권은 산문에 있었던 것이고, 우리
층이 통제하는 지점(도구 호출)에는 애초에 없던 것이다.

또 그 답변이 **몇 개의 유형을 나열했는지**와 실제 제출 수를 나란히 놓는다. 둘이 같으면
"나열 = 제출"이 기전이고, 다르면 아니다. 세지 않고 단언하면 안 되는 종류의 주장이다.

usage: x122_prose_is_the_write.py --dirs a,b --tasks task_101,task_102 [--show]
"""

import collections
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
DOMAIN = os.path.join(TAU2, "data", "tau2", "domains", "banking_knowledge")
SIMBASES = [os.path.join(TAU2, "data", "simulations"),
            os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")]


def arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


DIRS = [d for d in (arg("--dirs") or "").split(",") if d]
TASKS = [t for t in (arg("--tasks") or "task_101,task_102").split(",") if t]
SHOW = "--show" in sys.argv

# 코퍼스에 실재하는 상품 이름 — 답변이 몇 개를 나열했는지 세려면 후보 목록이 필요하다.
# 문서 파일명에서 뽑는다(태스크에서 뽑으면 gold 경유가 된다).
def product_names():
    names = set()
    for p in glob.glob(os.path.join(DOMAIN, "documents", "*")):
        fn = os.path.basename(p).lower()
        m = re.match(r"doc_(?:business_)?(?:checking_accounts|savings_accounts|bank_accounts|"
                     r"credit_cards|business_credit_cards)_(.+?)_\d+\.json$", fn)
        if not m:
            continue
        slug = re.sub(r"_account$", "", m.group(1))
        if slug.endswith("_(general)") or "general" in slug:
            continue
        names.add(" ".join(w.capitalize() for w in slug.split("_")))
    return sorted(names)


def jopen(p):
    with io.open(p, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def load_sims():
    out = []
    for base in SIMBASES:
        for d in DIRS:
            for p in glob.glob(os.path.join(base, d, "results.json")):
                for s in (jopen(p).get("simulations") or []):
                    s["_src"] = d
                    out.append(s)
    return out


def args_of(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def main():
    prods = product_names()
    sims = load_sims()
    for tid in TASKS:
        mine = [s for s in sims if s.get("task_id") == tid]
        if not mine:
            continue
        tot = collections.Counter()
        print("=" * 100)
        print("== %s == (상품 후보 %d종)" % (tid, len(prods)))
        for s in sorted(mine, key=lambda x: (x["_src"], x.get("trial") or 0)):
            msgs = s.get("messages") or []
            per = collections.Counter()
            rows = []
            for i, m in enumerate(msgs):
                if m.get("role") != "user":
                    continue
                for tc in (m.get("tool_calls") or []):
                    nm = tc.get("name") or (tc.get("function") or {}).get("name")
                    if nm != "submit_referral":
                        continue
                    at = args_of(tc).get("account_type") or ""
                    prev = ""
                    for j in range(i - 1, -1, -1):
                        if msgs[j].get("role") == "assistant" and msgs[j].get("content"):
                            prev = str(msgs[j]["content"])
                            break
                    stem = at.replace(" Account", "").strip()
                    named = bool(stem) and re.search(re.escape(stem), prev, re.I) is not None
                    listed = [p for p in prods if re.search(r"\b" + re.escape(p) + r"\b", prev, re.I)]
                    per["제출"] += 1
                    per["직전답변에_있음" if named else "직전답변에_없음"] += 1
                    rows.append((at, named, len(listed), listed[:6]))
            if not rows:
                tot["제출0인_trial"] += 1
                continue
            nlisted = rows[-1][2]
            print("  [%-22s t%s] 제출 %d건 · 직전답변 일치 %d/%d · 그 답변이 나열한 상품 %d종"
                  % (s["_src"], s.get("trial"), per["제출"], per["직전답변에_있음"], per["제출"], nlisted))
            if SHOW:
                for at, named, n, ls in rows:
                    print("       %-34s 직전답변에 %s (그 답변 나열 %d종: %s)"
                          % (at, "있음" if named else "없음", n, ", ".join(ls)))
            tot["trial"] += 1
            tot["제출"] += per["제출"]
            tot["일치"] += per["직전답변에_있음"]
            tot["불일치"] += per["직전답변에_없음"]
        print("  ── 합계 %s" % dict(tot))


if __name__ == "__main__":
    main()
