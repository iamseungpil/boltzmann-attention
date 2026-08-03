"""Is the card choice a boundary, or is it load?

Six failures die on which card the customer ends up applying for, and the right card
was in context before the decision every time. That is either a semantic limit — the
model cannot tell that a work truck is not operations spend — or it is load: it can
tell, but not while carrying a full trajectory. The two have opposite prescriptions,
so [[18]] requires the isolation probe before either label is used (C124: a wrong-pick
called a boundary turned out to be load plus self-anchoring once it was run).

Three arms, same question, information matched from weakest to strongest context:

  BARE   the customer's own words + the eligible-card table the fit tool returned
  MIN    the same, plus the documents defining what each card's bonus rate covers
  FULL   the actual trajectory up to the decision step, as the run really had it

MIN right and FULL wrong is load. BARE right and MIN right and FULL right means the
run's own context is what broke it. All three wrong is the boundary reading, and only
then is "no lever" the honest answer.

Nothing gold-derived enters any prompt: the customer's utterances and the fit table
are what the agent held, and the documents are ones it could have retrieved.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
import urllib.request

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
ARMS = {"A": "bank_ax33n_gpu*_20260803g", "B4": "bank_b4_gpu*_20260803h"}

ASK = ("Question: which ONE card should this customer apply for?\n"
       "Answer in exactly two lines:\n"
       "CARD: <exact card name>\n"
       "WHY: <one sentence>")


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def chat(prompt, port, temp, model):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": 220,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def card_docs(names, cap=1400):
    """Documents for the candidate cards — retrievable, not gold-derived."""
    out = []
    for p in sorted(glob.glob(os.path.join(DOCS, "*"))):
        base = os.path.basename(p)
        slug = base[4:].rsplit("_", 1)[0]
        if not any(n.lower().replace(" ", "_") in slug for n in names):
            continue
        try:
            d = json.load(open(p, encoding="utf-8", errors="replace"))
        except Exception:
            continue
        out.append(f"### {d.get('title')}\n{str(d.get('content'))[:cap]}")
    return out


def build(sim):
    """(customer words, fit table, candidate names, full prefix, decision step)."""
    msgs = sim.get("messages") or []
    fit_out = fit_step = None
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and isinstance(m.get("content"), str) \
                and "'eligible'" in m["content"] and fit_out is None:
            fit_out, fit_step = m["content"], i
    if fit_out is None:
        return None

    # The decision is the first place the customer could act on the advice: their
    # own call, or failing that the end of the run.
    dec = next((i for i, m in enumerate(msgs)
                if m.get("role") == "user" and m.get("tool_calls")), len(msgs))
    said = [m["content"] for m in msgs[:dec]
            if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip()]
    names = re.findall(r"'card': '([^']+)'", fit_out)

    prefix = []
    for m in msgs[:dec]:
        role, c = m.get("role"), m.get("content")
        if role == "user" and isinstance(c, str) and c.strip():
            prefix.append(f"CUSTOMER: {c}")
        elif role == "assistant" and isinstance(c, str) and c.strip():
            prefix.append(f"AGENT: {c}")
        elif role == "tool" and isinstance(c, str):
            prefix.append(f"TOOL RESULT: {c[:2500]}")
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            prefix.append(f"AGENT CALLS: {n}({json.dumps(norm(tc.get('arguments')))[:300]})")
    return said, fit_out, names, "\n\n".join(prefix), dec


def prompts(said, fit_out, names, full):
    who = "\n".join(f"- {' '.join(s.split())}" for s in said)
    bare = (f"A bank customer said:\n{who}\n\n"
            f"An eligibility tool returned these cards and their documented facts:\n{fit_out}\n\n{ASK}")
    docs = card_docs(names)
    mind = (f"A bank customer said:\n{who}\n\n"
            f"An eligibility tool returned these cards and their documented facts:\n{fit_out}\n\n"
            f"The knowledge base documents for these cards:\n" + "\n\n".join(docs) + f"\n\n{ASK}")
    fullp = (f"Here is a support conversation so far.\n\n{full}\n\n{ASK}")
    return {"BARE": bare, "MIN": mind, "FULL": fullp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B4", choices=sorted(ARMS))
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    sims = []
    for p in sorted(glob.glob(f"{SIM}/{ARMS[args.arm]}.results.json.gz")):
        sims.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])

    targets = []
    for s in sims:
        ri = s.get("reward_info") or {}
        if (ri.get("reward") or 0.0) == 1.0:
            continue
        for c in ri.get("action_checks") or []:
            a = c.get("action") or {}
            if not c.get("action_match") and a.get("name") == "apply_for_credit_card":
                g = norm(a.get("arguments")).get("card_type")
                if g:
                    targets.append((s, g))
                break

    print(f"[{args.arm}] card_type 실패 {len(targets)}건 · n={args.n} · temp={args.temp}\n")
    tally = collections.Counter()
    for s, gold in sorted(targets, key=lambda x: (x[0]["task_id"], x[0].get("trial") or 0)):
        built = build(s)
        key = f"{s['task_id']}/t{s.get('trial')}"
        if not built:
            print(f"  {key}: fit 출력 없음 — 제외")
            continue
        said, fit_out, names, full, dec = built
        ps = prompts(said, fit_out, names, full)
        print(f"  {key}  gold={gold}  후보 {len(names)}  결정 step {dec}")
        for arm, prompt in ps.items():
            hits, picks = 0, collections.Counter()
            for _ in range(args.n):
                try:
                    txt = chat(prompt, args.port, args.temp, args.model)
                except Exception as e:
                    picks[f"ERR {type(e).__name__}"] += 1
                    continue
                m = re.search(r"CARD:\s*(.+)", txt)
                pick = (m.group(1) if m else txt.splitlines()[0]).strip().strip("*` ")
                picks[pick] += 1
                if gold.lower() in pick.lower():
                    hits += 1
            tally[f"{arm} {hits > 0}"] += 0
            tally[arm] += hits
            tally[f"{arm}_n"] += args.n
            print(f"    {arm:5s} {hits}/{args.n}  {dict(picks)}")
        print()

    print("=== 합계 ===")
    for arm in ("BARE", "MIN", "FULL"):
        n = tally[f"{arm}_n"]
        print(f"  {arm:5s} {tally[arm]}/{n}" + (f" = {tally[arm] / n:.0%}" if n else ""))
    print("\n  MIN 높고 FULL 낮으면 = 부하 · 셋 다 낮으면 = 경계")


if __name__ == "__main__":
    main()
