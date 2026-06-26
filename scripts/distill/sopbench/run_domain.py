#!/usr/bin/env python3
"""run_domain.py — TBox(fixed executor) + ABox(per-domain ontology) evaluation.

TBox  = workflow_executor.CallGraphExecutor (domain-INVARIANT; never changes).
ABox  = abox/ontology_<domain>.json  (+ optional abox/<domain>_functions.py for any
        compute/decide step the domain does NOT expose as a tool).

The experiment: swap ONLY the ABox per domain, keep the TBox executor fixed, and check
each domain hits 100% TSR.

Usage: run_domain.py <domain> [max_tasks]
       run_domain.py --all            # run every domain with an ABox, print a table
"""
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABOX = os.path.join(HERE, "abox")
sys.path.insert(0, HERE)
sys.path.insert(0, ABOX)

import workflow_executor as wfx
from workflow_executor import WorkflowAgent
from amazon_sop_bench import evaluate


def load_abox(domain):
    """Register the domain's wrapped functions (if any) and return its ontology dict."""
    wfx.WRAPPED.clear()
    fn_mod = os.path.join(ABOX, f"{domain}_functions.py")
    if os.path.exists(fn_mod):
        mod = importlib.import_module(f"{domain}_functions")
        importlib.reload(mod)
        mod.register(wfx.WRAPPED)
    return json.load(open(os.path.join(ABOX, f"ontology_{domain}.json")))


def run_one(domain, max_tasks=100000):
    ontology = load_abox(domain)
    agent = WorkflowAgent(ontology)
    res = evaluate(domain, agent=agent, max_tasks=max_tasks, max_workers=1)
    return res.get("task_success_rate"), res.get("num_correct"), res.get("num_tasks")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        domains = sorted(f[len("ontology_"):-len(".json")]
                         for f in os.listdir(ABOX) if f.startswith("ontology_") and f.endswith(".json"))
        print(f"{'domain':32} {'TSR':>7}  correct/total")
        rows = []
        for d in domains:
            try:
                tsr, c, n = run_one(d)
                rows.append((d, tsr, c, n))
                print(f"{d:32} {tsr*100:6.1f}%  {c}/{n}")
            except Exception as e:
                print(f"{d:32}  ERROR: {e}")
        done = [r for r in rows if r[1] is not None]
        if done:
            avg = sum(r[1] for r in done) / len(done)
            full = sum(1 for r in done if r[1] >= 0.999)
            print(f"\n  mean TSR = {avg*100:.1f}%   domains@100% = {full}/{len(done)}")
        return 0

    domain = sys.argv[1] if len(sys.argv) > 1 else "customer_service"
    max_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    tsr, c, n = run_one(domain, max_tasks)
    print(f"[{domain}] TSR = {tsr*100:.1f}%  ({c}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
