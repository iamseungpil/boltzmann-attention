#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""db_match 진단: gold DB vs agent DB 직접 diff (2026-07-24·C149).
db_match = strict full-DB 해시(`get_dict_hash(db.model_dump())`)이라 어떤 필드 차이도 False.
이 도구는 gold(golden_actions 리플레이) DB와 agent(궤적 리플레이) DB의 model_dump()를 재귀
diff해 *정확히 어떤 DB 필드가 다른지* 출력 — banking은 `agent_discoverable_tools`(호출된
discoverable 도구 CALLED 기록·reads+writes)가 해시에 포함돼 read-coverage도 db_match에 영향.
Run(remote): seka python dbdiff_task.py <SAVE_TAG> [task_id]  (예: bank_reg043fix_treat_20260724 task_043)
필수: PYTHONPATH=tau2-bench/src·no_knowledge variant로 KB 임베딩(OpenAI) 우회."""
import sys
from pathlib import Path

from tau2.registry import registry
from tau2.data_model.simulation import Results

TAG = sys.argv[1] if len(sys.argv) > 1 else "bank_reg043fix_treat_20260724"
TASK_ID = sys.argv[2] if len(sys.argv) > 2 else "task_043"
DOMAIN = sys.argv[3] if len(sys.argv) > 3 else "banking_knowledge"
SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"

env_ctor = registry.get_env_constructor(DOMAIN)
tasks = registry.get_tasks_loader(DOMAIN)()
task = [t for t in tasks if t.id == TASK_ID][0]
results = Results.load(Path("%s/%s/results.json" % (SIMROOT, TAG)))
sim = [s for s in results.simulations if s.task_id == TASK_ID][0]
istate = task.initial_state

# no_knowledge = KB 파이프라인(임베딩) 없이 DB만 — db_match는 KB 무관.
gold = env_ctor(retrieval_variant="no_knowledge")
gold.set_state(istate.initialization_data, istate.initialization_actions,
               list(istate.message_history or []))
for a in (task.evaluation_criteria.actions or []):
    try:
        gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
    except Exception as e:
        print("gold ERR %s: %r" % (a.name, e))

pred = env_ctor(retrieval_variant="no_knowledge")
pred.set_state(istate.initialization_data, istate.initialization_actions, list(sim.messages))

print("agent_db_match:", gold.tools.get_db_hash() == pred.tools.get_db_hash(),
      "| user_db_match:",
      (gold.user_tools.get_db_hash() == pred.user_tools.get_db_hash())
      if gold.user_tools else "n/a")


def diff(g, p, path=""):
    if type(g) != type(p):
        print("  TYPE %s: %r vs %r" % (path, g, p)); return
    if isinstance(g, dict):
        for k in set(g) | set(p):
            if k not in g:
                print("  ONLY-PRED %s.%s = %r" % (path, k, p[k]))
            elif k not in p:
                print("  ONLY-GOLD %s.%s = %r" % (path, k, g[k]))
            else:
                diff(g[k], p[k], path + "." + str(k))
    elif isinstance(g, list):
        if len(g) != len(p):
            print("  LEN %s: gold=%d pred=%d" % (path, len(g), len(p)))
        for i in range(min(len(g), len(p))):
            diff(g[i], p[i], "%s[%d]" % (path, i))
    elif g != p:
        print("  DIFF %s: gold=%r pred=%r" % (path, g, p))


print("=== AGENT DB diff (gold vs pred) ===")
diff(gold.tools.db.model_dump(), pred.tools.db.model_dump())
if gold.user_tools:
    print("=== USER DB diff ===")
    diff(gold.user_tools.db.model_dump(), pred.user_tools.db.model_dump())
