# -*- coding: utf-8 -*-
"""Is the wrong account class a capability limit, or the load of the conversation it sits in?

In 24 of 47 failures the document naming gold's class had already been retrieved and the
agent still chose another. That is the shape that gets called a boundary, and [[18]] says
not to call it one before running an information-matched isolation probe: the same model,
the same facts, two context conditions.

  A_minimal   the customer's own sentences + the documents the run retrieved, nothing else
  B_fullctx   the whole conversation up to the choice, exactly as the agent saw it
  C_decomposed  list the candidates from the documents, judge each one against the criteria
                separately, and let the engine keep the ones judged to qualify — the same
                shape the card-fit tool already has, and the load-minimal form of the
                category question ("is this account in that category?") the user asked about

Both arms are asked for one class name and scored against gold. A ≫ B means the decision
was reachable and the surrounding conversation is what broke it — a load problem a
scaffold can fix by isolating the decision, which is what the card-fit tool already does
on the card side. A ≈ B, both low, means the decision itself is out of reach here and the
residual belongs to the learn axis ([[11]]), not to another gate.

Free: local vLLM only, no user simulator ([[09]]).

  usage: x85_choice_isolation_probe.py --base http://localhost:8140/v1 [--limit 12]
"""

import argparse
import collections
import glob
import gzip
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

from x50_says_not_does import ARMS, SIM  # noqa: E402

TOOL, ARG = "open_bank_account_4821", "account_class"


def inner(a):
    a = a if isinstance(a, dict) else {}
    nm = a.get("agent_tool_name") or a.get("discoverable_tool_name") or a.get("user_tool_name")
    sub = a.get("arguments")
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except Exception:
            sub = None
    return (nm, sub if isinstance(sub, dict) else {}) if nm else (None, a)


def cases(limit):
    """Sims where gold's class was retrieved and a different one was chosen."""
    out = []
    for p in sorted(glob.glob(os.path.join(SIM, ARMS["N97B"] + "*.results.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        for s in (d.get("simulations") if isinstance(d, dict) else d):
            golds = [ga.get(ARG) for c in ((s.get("reward_info") or {}).get("action_checks") or [])
                     for nm, ga in [inner((c.get("action") or {}).get("arguments"))]
                     if nm == TOOL and ga.get(ARG) and not c.get("action_match")]
            if not golds:
                continue
            user_txt, docs, chosen, cut = [], [], None, len(s.get("messages") or [])
            for i, m in enumerate(s.get("messages") or []):
                if m.get("role") == "user" and m.get("content"):
                    user_txt.append(m["content"])
                if m.get("role") == "tool" and m.get("content"):
                    docs.append(str(m["content"]))
                for tc in m.get("tool_calls") or []:
                    nm, a = inner(tc.get("arguments"))
                    if (nm or tc.get("name")) == TOOL and a.get(ARG) and chosen is None:
                        chosen, cut = a[ARG], i
            gold = golds[0]
            if chosen is None or chosen == gold:
                continue
            if gold not in " ".join(docs):
                continue                       # 회수된 적 없으면 이 실험의 대상이 아니다
            out.append({"task": s["task_id"], "trial": s.get("trial"), "gold": gold,
                        "chosen": chosen, "user": user_txt, "docs": docs,
                        "msgs": (s.get("messages") or [])[:cut]})
            if len(out) >= limit:
                return out
    return out


ASK = ("Answer with the single official account class name, exactly as a document writes it. "
       "Reply with the name only — no explanation.")


def prompt_A(c):
    docs = "\n\n".join(d[:1500] for d in c["docs"][:12])
    return ("The customer said:\n%s\n\n=== POLICY / ACCOUNT DOCUMENTS RETRIEVED ===\n%s\n\n"
            "Which account class should be opened for this customer? %s"
            % ("\n".join(c["user"])[:2500], docs, ASK))


def prompt_B(c):
    lines = []
    for m in c["msgs"]:
        r = m.get("role")
        t = " ".join(str(m.get("content") or "").split())[:600]
        if t:
            lines.append("%s: %s" % (r, t))
        for tc in m.get("tool_calls") or []:
            lines.append("assistant_call: %s %s" % (tc.get("name"),
                                                    json.dumps(tc.get("arguments"), ensure_ascii=False)[:200]))
    return ("=== CONVERSATION SO FAR ===\n%s\n\nWhich account class should be opened for this "
            "customer? %s" % ("\n".join(lines)[:24000], ASK))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--limit", type=int, default=12)
    A = ap.parse_args()
    import requests

    def ask(text):
        r = requests.post(A.base.rstrip("/") + "/chat/completions", timeout=600, json={
            "model": A.model, "temperature": 0.0, "max_tokens": 40,
            "messages": [{"role": "user", "content": text}]})
        return r.json()["choices"][0]["message"]["content"].strip()

    def decomposed(c):
        """List candidates, judge each alone, engine keeps the qualifying ones."""
        docs = "\n\n".join(d[:1500] for d in c["docs"][:12])
        names = ask("=== DOCUMENTS ===\n%s\n\nList every official account class name these "
                    "documents describe, one per line, names only." % docs)
        cands = [x.strip("-* \t") for x in names.splitlines() if x.strip()][:12]
        keep = []
        for nm in cands:
            v = ask("Customer said:\n%s\n\n=== DOCUMENTS ===\n%s\n\nDoes the account class "
                    "'%s' satisfy everything the customer asked for? Answer YES or NO only."
                    % ("\n".join(c["user"])[:2000], docs, nm))
            if v.strip().upper().startswith("Y"):
                keep.append(nm)
        return keep, cands

    cs = cases(A.limit)
    print("대상 %d건 (gold를 회수했는데 다른 걸 고른 경우)\n" % len(cs))
    tally = collections.Counter()
    for c in cs:
        a = ask(prompt_A(c))
        b = ask(prompt_B(c))
        norm = lambda x: re.sub(r"[^a-z0-9 ]", "", str(x).lower()).strip()
        ok_a, ok_b = norm(c["gold"]) in norm(a), norm(c["gold"]) in norm(b)
        keep, cands = decomposed(c)
        ok_c = any(norm(c["gold"]) in norm(k) for k in keep)
        tally["A"] += ok_a
        tally["B"] += ok_b
        tally["C"] += ok_c
        tally["cand_ok"] += any(norm(c["gold"]) in norm(k) for k in cands)
        tally["n"] += 1
        print("  %-10s gold=%-24s A=%s B=%s C=%s (후보 %d개·유지 %d개)"
              % (c["task"], c["gold"][:24], "✓" if ok_a else "✗", "✓" if ok_b else "✗",
                 "✓" if ok_c else "✗", len(cands), len(keep)))
    n = max(1, tally["n"])
    print("\n  A_minimal(격리)  %d/%d = %.0f%%" % (tally["A"], n, 100 * tally["A"] / n))
    print("  B_fullctx(전체)  %d/%d = %.0f%%" % (tally["B"], n, 100 * tally["B"] / n))
    print("  C_decomposed     %d/%d = %.0f%%  (후보 나열에 gold 포함 %d/%d)"
          % (tally["C"], n, 100 * tally["C"] / n, tally["cand_ok"], n))
    print("\n  라이브 실측(같은 결정) = 0%% — 이 표본은 전부 라이브에서 틀린 것이다.")


if __name__ == "__main__":
    main()
