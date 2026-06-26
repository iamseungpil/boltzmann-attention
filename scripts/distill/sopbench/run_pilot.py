#!/usr/bin/env python3
"""run_pilot.py — run the deterministic WorkflowAgent on a SOP-Bench domain.

Usage: run_pilot.py <domain> <ontology.json> [max_tasks]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workflow_executor as wfx
from workflow_executor import WorkflowAgent

# register domain wrapped functions
import cs_functions
cs_functions.register(wfx.WRAPPED)

from amazon_sop_bench import evaluate


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "customer_service"
    ont_path = sys.argv[2] if len(sys.argv) > 2 else "ontology_customer_service.json"
    max_tasks = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    ontology = json.load(open(ont_path))
    agent = WorkflowAgent(ontology)
    print(f"[pilot] domain={domain} ontology={ont_path} max_tasks={max_tasks}")
    results = evaluate(domain, agent=agent, max_tasks=max_tasks, max_workers=1)
    # print key metrics
    keys = ["task_success_rate", "execution_completion_rate", "conditional_task_success_rate",
            "tool_accuracy", "TSR", "ECR", "C-TSR"]
    print("=== metrics ===")
    if isinstance(results, dict):
        for k, v in results.items():
            if not isinstance(v, (list, dict)):
                print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
