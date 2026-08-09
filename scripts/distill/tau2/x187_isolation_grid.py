# -*- coding: utf-8 -*-
r"""x187 — **격리 사다리 × 질문 난이도 × 후보 × 정렬** 격자 (유료 0·사용자 지시).

설계서: `reports/facet_rft_2026/ISOLATION_GRID_DESIGN_2026_08_09.md`(§0b [[05]] 3질문 포함).

  격리  L0 full · L2 facts_full · L3 facts_operand · L4 bare
  난이도 Q1 순수argmax(축명시·타입없음) · Q2 +타입제약 · Q3 +축해석(현행)
  후보  all · chk(계좌 축이 있는 행만)      정렬 name_asc · name_desc      태스크 099·100

판정 규칙은 설계서 §4 에 사전 등록돼 있다. ⚠교락(양자화·모델 2점)은 §5.

실행: python x187_isolation_grid.py [N]
"""
import collections
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCOUNT_AXIS = "referrer_tenure_days"
TYPE_PHRASE = "business checking account"       # x149.QUESTION 축자 — 신규 저작 아님
LEAD = "Here is a customer-service conversation so far.\n\n"


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def drop_named_sentences(text, names):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return " ".join(s for s in sents if not any(nm in s for nm in names))


def drop_tail_assistant(ms):
    out = list(ms)
    while out and out[-1].get("role") == "assistant":
        out.pop()
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    bax = next(a for a in axes if "bonus" in a.lower() and "referrer" in a.lower())
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    def bval(nm):
        v = (maps.get(bax) or {}).get(nm)
        try:
            return float(str(v[0]).replace(",", ""))
        except Exception:
            return -1.0

    # 난이도 사다리 — 현행 문장을 축자 재사용하고 A2 축 이름만 끼운다
    Q3 = X.QUESTION
    Q2 = Q3.replace("if the customer wants to maximise the bonus THEY receive? ",
                    "with the highest %s? " % bax)
    Q1 = Q2.replace(TYPE_PHRASE, "product")
    assert Q2 != Q3 and Q1 != Q2, "질문 사다리 치환 실패 — 원문이 바뀌었다"

    print("model=%s · n=%d" % (MODEL, n))
    print("Q1 %s\nQ2 %s\nQ3 %s\n" % (Q1, Q2, Q3))
    out = []
    for task, case in CASE.items():
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]
        # 정답은 엔진이 계산한다 (Q1 은 동점 집합 허용)
        m_all = max(bval(s) for s in ALL)
        GOLD1 = {s for s in ALL if bval(s) == m_all}
        GOLD23 = {X.GOLD[task]}
        assert X.GOLD[task] == max(CHK, key=bval), "Q2/Q3 정답이 엔진 argmax 와 불일치"

        ms = Y.msgs_of(TAG, task)
        full_txt = Y.render(ms)
        facts_full = X.FACTS[task]
        facts_oper = drop_named_sentences(facts_full, ALL)
        isos = [("L0 full     ", LEAD + full_txt + "\n\n", facts_full),
                ("L2 facts    ", "", facts_full),
                ("L3 operand  ", "", facts_oper),
                ("L4 bare     ", "", "")]
        qs = [("Q1 argmax ", Q1, GOLD1, ("all",)),
              ("Q2 +type  ", Q2, GOLD23, ("all", "chk")),
              ("Q3 +interp", Q3, GOLD23, ("all", "chk"))]

        print("\n" + "=" * 104)
        print("%s  Q1정답=%s · Q2/Q3정답=%r · 표 %d행(chk %d)"
              % (task, sorted(GOLD1), X.GOLD[task], len(ALL), len(CHK)))
        print("=" * 104)
        hdr = "  %-12s |" % "격리"
        for ql, _q, _g, chs in qs:
            for ch in chs:
                hdr += " %-13s |" % ("%s/%s" % (ql.strip()[:2], ch))
        print(hdr + "  (asc / desc)")
        for ilabel, pre, ftxt in isos:
            cells = []
            for ql, q, goldset, chs in qs:
                for ch in chs:
                    choices = ALL if ch == "all" else CHK
                    hits = []
                    for rev in (False, True):
                        order = sorted(body, key=name, reverse=rev)
                        tbl = "\n".join(head[:1] + order + head[1:]).strip()
                        mid = ("\n\n" + ftxt) if ftxt else ""
                        prompt = pre + tbl + mid + "\n\n" + q
                        c = collections.Counter()
                        for i in range(n):
                            try:
                                c[guided_full(prompt, choices, 0.0 if i == 0 else 0.7)] += 1
                            except Exception as e:
                                c["ERR %s" % type(e).__name__] += 1
                        g = sum(v for k, v in c.items() if k in goldset)
                        hits.append(g)
                        out.append({"task": task, "iso": ilabel.strip(), "q": ql.strip(),
                                    "choices": ch, "sort": "desc" if rev else "asc",
                                    "gold_hit": g, "n": n, "dist": dict(c)})
                    cells.append("%d/%d %d/%d" % (hits[0], n, hits[1], n))
            print("  %-12s | %s" % (ilabel, " | ".join("%-13s" % c for c in cells)))

    json.dump(out, open(os.environ.get("T2_X187_OUT", "x187_out.json"), "w"), indent=1)
    print("\n  판정 규칙 = 설계서 §4 · 교락(양자화·모델 2점) = §5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
