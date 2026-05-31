"""
run_two_stage.py — arm-3 (L1) runner for SOPBench (Zekun Li, 2503.08669).

Same task-loop as run_simulation.py but injects TwoStageClient (planner+resolver) as the
assistant's client. Evaluates with the native rule oracle and reports pass@1 + coverage%.

DEPLOY: copy this + two_stage_client.py into the SOPBench clone `scripts/`.
Run (with a pre-served vLLM endpoint):
  SOPBENCH_VLLM_BASE_URL=http://localhost:9100/v1 python scripts/run_two_stage.py \
    --domain bank --model Qwen/Qwen2.5-7B-Instruct --tool_list full \
    --num_tasks 5 --output_dir ./output_two_stage

NOTE: import is same-dir style (`from two_stage_client import ...`) so it works whether
scripts/ is a package or not — run from the clone root.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this file's dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse, json
from tqdm import tqdm
from two_stage_client import TwoStageClient            # same-dir import
from swarm.core import Swarm
from swarm.types import Agent
from swarm.util import function_to_json
from env.task import task_initializer, task_default_dep_full
from env.evaluator import evaluator_function_directed_graph


def exit_conversation():
    """End the conversation."""
    return "Conversation ended."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="bank")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tool_list", default="full", choices=["full", "oracle"])
    p.add_argument("--num_tasks", type=int, default=None)
    p.add_argument("--max_num_turns", type=int, default=20)
    p.add_argument("--max_num_actions", type=int, default=10)
    p.add_argument("--output_dir", default="./output_two_stage")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--default_constraint_option", default="full")
    p.add_argument("--constraint_descr_format", default="structured")
    return p.parse_args()


def main():
    args = parse_args()
    base_url = os.environ.get("SOPBENCH_VLLM_BASE_URL", "http://localhost:9100/v1")

    raw = json.load(open(f"{args.data_dir}/{args.domain}_tasks.json"))
    tasks = [t for v in raw.values() for t in v]
    if args.num_tasks:
        tasks = tasks[:args.num_tasks]

    dep_innate, dep_full, dep_descr = task_default_dep_full(
        args.domain, args.default_constraint_option, args.constraint_descr_format,
        dependency_verb_dep_orig=True)

    os.makedirs(f"{args.output_dir}/{args.domain}", exist_ok=True)
    out_file = (f"{args.output_dir}/{args.domain}/"
                f"two_stage_{args.model.replace('/','_')}-tool_{args.tool_list}.json")

    client = TwoStageClient(base_url=base_url, model_name=args.model)
    assistant_agent = Agent(name="assistant", client=client, instructions="",
                            functions=[], tool_call_mode="fc",
                            temperature=0.0, top_p=0.01, max_tokens=512)
    user_agent = Agent(name="user", client=None, default_response="", response_repeat=True)

    results = []
    for task in tqdm(tasks, desc="Running 2-stage"):
        client.reset()
        included = ([n[0] for n in task["directed_action_graph"]["nodes"]
                     if isinstance(n, list)] if args.tool_list == "oracle" else None)

        domain_system, user_info, assistant_info, task_info = task_initializer(
            args.domain, task, dep_innate, dep_full, dep_descr,
            included, "prompt", False, args.constraint_descr_format)

        fns = assistant_info["tools"] + [function_to_json(exit_conversation)]
        assistant_agent.instructions = assistant_info["instructions"]
        assistant_agent.functions = fns
        assistant_agent.name = f"{args.domain} assistant"

        default_msg = ("Here is all the information I can provide:\n" +
                       json.dumps(user_info["known"], indent=4) +
                       f"\n\nIf you have completed my request or cannot assist me, "
                       f"please use the `{exit_conversation.__name__}` action.")
        user_agent.default_response = default_msg

        swarm = Swarm(system=domain_system, max_turns=args.max_num_turns,
                      max_actions=args.max_num_actions)
        interaction = swarm.run_user_assistant_interaction(
            user_agent=user_agent, assistant_agent=assistant_agent,
            messages=[{"role": "user", "content": default_msg, "sender": "user"}],
            debug=False, execute_tools=True,
            start_agent="assistant", finished_action=exit_conversation)

        func_calls = [(m["tool_name"], json.loads(m.get("content", "{}") or "{}"))
                      for m in interaction.messages if m.get("role") == "tool"]
        log_flat = [tc for m in interaction.messages if m.get("role") == "assistant"
                    for tc in (m.get("tool_calls", []) or [])]
        ev = evaluator_function_directed_graph(
            args.domain, task, log_flat, func_calls,
            {"final_database": domain_system.data}, args.default_constraint_option)

        results.append({"domain": args.domain, "task": task,
                        "interactions": [{"messages": interaction.messages}],
                        "evaluations": [ev], "coverage": client.coverage()})
        json.dump(results, open(out_file, "w"), indent=2)

    n = len(results)
    passed = sum(r["evaluations"][0].get("success", False) for r in results)
    cov = sum(r["coverage"]["deterministic"] for r in results)
    turns = sum(r["coverage"]["turns"] for r in results)
    print(f"\n=== arm-3 L1 TwoStage ({args.domain}, {args.tool_list}, {args.model}) ===")
    print(f"  pass@1 : {passed/n:.4f}  ({passed}/{n})")
    print(f"  coverage% (deterministic resolver): {cov/turns:.3f}" if turns else "  coverage%: n/a")
    print(f"  results: {out_file}")


if __name__ == "__main__":
    main()
