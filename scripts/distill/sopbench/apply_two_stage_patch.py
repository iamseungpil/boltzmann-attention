"""
apply_two_stage_patch.py — deploy arm-3 (L1 TwoStageClient) into a SOPBench clone.

Why this exists (2026-06-01 BLOCKING-fix): the original run_two_stage.py did its OWN
inline evaluation with a tuple/format that did NOT match the author's
evaluator_function_directed_graph (which expects dict func_calls paired
interaction[i]<->[i+1] and the run_simulation save schema). That made any arm-3 number
untrustworthy. The fix (handoff project_handoff_2026_06_01.md §4.5) is to swap ONLY the
assistant client into the author's standard run_simulation.py and reuse the standard
run_evaluation.py UNCHANGED, so output + scoring are 100% the author's pipeline.

This script applies, idempotently (with .bak backups), to a SOPBench clone:
  1. cp two_stage_client.py            -> <clone>/scripts/two_stage_client.py
  2. run_simulation.py  : add --two_stage / --two_stage_det flags, per-task client
                          reset, and conditional TwoStageClient construction.
  3. swarm/types.py     : relax Agent.client type so a TwoStageClient is accepted
                          (the original Union[OpenAIHandler, None] rejects it — this
                          pydantic error is the bug the first smoke surfaced, because
                          arm-3 had never actually been run before).
  4. swarm/llm_handler.py : _init_vllm honours $SOPBENCH_VLLM_BASE_URL (use a PRE-SERVED
                          vLLM endpoint instead of spawning one). Needed for arm-1 baseline
                          to share the same endpoint as arm-3. (arm-3 client connects to the
                          endpoint directly and does not need this.)
  5. swarm/constants.py : register the open-source models in FUNCTION_CALLING_MODELS["vllm"]
                          (original is []), so arm-1 fc mode passes the tool-calling assert
                          for qwen2.5-{3b,7b,14b,32b,72b}-instruct + llama3.x. (arm-3 bypasses
                          OpenAIHandler entirely, so it works regardless.)

Usage (on the remote, from this file's dir or anywhere):
  python apply_two_stage_patch.py /home/woori/scratch/SOPBench

Then run (vLLM pre-served on $SOPBENCH_VLLM_BASE_URL, default localhost:9100/v1):
  SOPBENCH_VLLM_BASE_URL=http://localhost:9100/v1 python run_simulation.py \
    --domain bank --assistant_model Qwen/Qwen2.5-7B-Instruct \
    --tool_call_mode fc --tool_list full --two_stage \
    --output_dir ./output_two_stage --env_mode prompt
  python run_evaluation.py --domain bank --assistant_model Qwen/Qwen2.5-7B-Instruct \
    --tool_call_mode fc --tool_list full --output_dir ./output_two_stage
"""
import os
import shutil
import sys
import py_compile


def _patch(path, edits, marker):
    """Apply (old -> new) edits to `path` once; back up to .bak. Idempotent on marker."""
    src = open(path, encoding="utf-8").read()
    if marker in src:
        print(f"  {path}: ALREADY PATCHED (marker present) — skip")
        return
    open(path + ".bak", "w", encoding="utf-8").write(src)
    for old, new in edits:
        n = src.count(old)
        assert n == 1, f"{path}: anchor found {n}x (expected 1):\n{old[:80]}..."
        src = src.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(src)
    py_compile.compile(path, doraise=True)
    print(f"  {path}: PATCHED + compiles OK")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python apply_two_stage_patch.py <SOPBench_clone_dir>")
    clone = os.path.abspath(sys.argv[1])
    here = os.path.dirname(os.path.abspath(__file__))

    # 1) deploy the client into the clone's scripts/
    os.makedirs(os.path.join(clone, "scripts"), exist_ok=True)
    shutil.copy(os.path.join(here, "two_stage_client.py"),
                os.path.join(clone, "scripts", "two_stage_client.py"))
    print(f"  copied two_stage_client.py -> {clone}/scripts/")

    # 2) run_simulation.py
    rs = os.path.join(clone, "run_simulation.py")
    _patch(rs, [
        # Edit A: CLI flags
        ("    args = parser.parse_args()",
         '    parser.add_argument("--two_stage", action="store_true",\n'
         '                       help="arm-3 L1: use TwoStageClient (planner+resolver) as assistant client")\n'
         '    parser.add_argument("--two_stage_det", action="store_true",\n'
         '                       help="arm-3: enable deterministic slot-state resolver shortcut (rung a)")\n'
         '    parser.add_argument("--two_stage_v2", action="store_true",\n'
         '                       help="arm-3v2: structured planner (ABox precond/produces + READY/BLOCKED gate + STOP)")\n'
         '    parser.add_argument("--ont_dir", default="./induced",\n'
         '                       help="arm-3v2: induced ontology dir (ontology_<domain>.json)")\n'
         "\n"
         "    args = parser.parse_args()"),
        # Edit D: H3 credential-augment (env SOPBENCH_AUGMENT_CRED, lock #5) — inject ONLY the login
        # credential (identification) into user_known (model-visible) from the account, so login can
        # SUCCEED and the deterministic gate reads logged_in=True. NO balance/condition/account field
        # is touched => NO oracle leak (the model must still CALL getters to gather those facts).
        ('            task["user_goal"] = task_key\n'
         '            tasks.append(task)',
         '            task["user_goal"] = task_key\n'
         '            if os.environ.get("SOPBENCH_AUGMENT_CRED"):\n'
         '                _uk = task.get("user_known") or {}\n'
         '                _un = _uk.get("username")\n'
         '                _acct = (task.get("initial_database") or {}).get("accounts", {}).get(_un, {})\n'
         '                if isinstance(_acct, dict) and "identification" in _acct and "identification" not in _uk:\n'
         '                    _uk["identification"] = _acct["identification"]   # login credential ONLY\n'
         '                    task["user_known"] = _uk\n'
         '            tasks.append(task)'),
        # Edit B: per-interaction slot-state reset
        ('    # Get the included functions in the oracle trajectory\n'
         '    if args.tool_list == "oracle":',
         "    # arm-3: reset TwoStageClient per-interaction slot state (no-op for OpenAIHandler).\n"
         "    # Pass task constraints + goal so mechanism A (TASK_CONSTRAINT_DESIGN, env SOPBENCH_LIGHTEN)\n"
         "    # can render the goal-status line from THIS task's constraint instead of the default precond.\n"
         '    if hasattr(assistant_agent.client, "reset"):\n'
         "        try:\n"
         '            assistant_agent.client.reset(task_constraints=task.get("constraints"), goal=task.get("user_goal"),\n'
         '                                         task_db=task.get("initial_database"),\n'
         '                                         constraint_params=task.get("constraint_parameters"),\n'
         '                                         constraints_original=task.get("constraints_original"),\n'
         '                                         domain=args.domain, user_known=task.get("user_known"))\n'
         "        except TypeError:\n"
         "            assistant_agent.client.reset()\n"
         '    # Get the included functions in the oracle trajectory\n'
         '    if args.tool_list == "oracle":'),
        # Edit C: conditional client construction
        ("        # Initialize OpenAI handler once (can be reused for both agents)\n"
         "        openai_handler = OpenAIHandler(\n"
         "            model_name=args.assistant_model,\n"
         "            num_gpus=args.num_gpus,\n"
         "            gpu_memory_utilization=args.gpu_memory_utilization,\n"
         "            tool_calling=True,\n"
         "        )\n",
         "        # Initialize the assistant client once (can be reused for both agents)\n"
         "        if args.two_stage:\n"
         "            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), \"scripts\"))\n"
         "            from two_stage_client import TwoStageClient\n"
         "            _ts_base_url = os.environ.get(\"SOPBENCH_VLLM_BASE_URL\", \"http://localhost:9100/v1\")\n"
         "            _abox = (os.path.join(args.ont_dir, f\"ontology_{args.domain}.json\") if args.two_stage_v2 else None)\n"
         "            print(f\"[arm-3 two_stage] base_url={_ts_base_url} det={args.two_stage_det} v2={args.two_stage_v2}\")\n"
         "            openai_handler = TwoStageClient(\n"
         "                base_url=_ts_base_url, model_name=args.assistant_model,\n"
         "                use_deterministic_shortcut=args.two_stage_det,\n"
         "                planner=(\"v2\" if args.two_stage_v2 else \"naive\"), abox=_abox)\n"
         "        else:\n"
         "            openai_handler = OpenAIHandler(\n"
         "                model_name=args.assistant_model,\n"
         "                num_gpus=args.num_gpus,\n"
         "                gpu_memory_utilization=args.gpu_memory_utilization,\n"
         "                tool_calling=True,\n"
         "            )\n"),
    ], marker="args.two_stage")

    # 3) swarm/types.py — relax Agent.client type
    tp = os.path.join(clone, "swarm", "types.py")
    _patch(tp, [
        ("    client: Union[OpenAIHandler, None] = None",
         "    client: Optional[object] = None  # arm-3: allow TwoStageClient (duck-typed .inference/.model_name_huggingface)"),
    ], marker="arm-3: allow TwoStageClient")

    # 4) swarm/llm_handler.py — _init_vllm honours $SOPBENCH_VLLM_BASE_URL (pre-served endpoint)
    lh = os.path.join(clone, "swarm", "llm_handler.py")
    _patch(lh, [
        ('        """Initialize VLLM backend."""\n'
         "        self.VLLM_PORT = random.randint(3000, 8000)",
         '        """Initialize VLLM backend."""\n'
         "        import os as _os  # arm-3 infra: reuse a pre-served vLLM endpoint if provided\n"
         '        _ext = _os.getenv("SOPBENCH_VLLM_BASE_URL")\n'
         "        if _ext:\n"
         '            self.client = OpenAI(base_url=_ext, api_key="EMPTY")\n'
         "            self.process = None\n"
         '            print(f"[patch] using pre-served vLLM endpoint {_ext}")\n'
         "            return\n"
         "        self.VLLM_PORT = random.randint(3000, 8000)"),
    ], marker="SOPBENCH_VLLM_BASE_URL")

    # 5) swarm/constants.py — register OSS models for fc tool-calling (original list is [])
    cs = os.path.join(clone, "swarm", "constants.py")
    _patch(cs, [
        ('    "vllm": [],',
         '    "vllm": [  # arm-3 infra: original was [] (OSS models blocked from fc assert)\n'
         '        "qwen2.5-3b-instruct", "qwen2.5-7b-instruct", "qwen2.5-14b-instruct",\n'
         '        "qwen2.5-32b-instruct", "qwen2.5-72b-instruct",\n'
         '        "llama3.1-8b-instruct", "llama3.1-70b-instruct", "llama3.3-70b-instruct",\n'
         '    ],'),
    ], marker="arm-3 infra: original was []")

    # 6) swarm/core.py — Cause-2 fix (RESIDUAL11_FIX_DESIGN): upstream strips the success-bool from
    # tuple-return tool results (raw_result = raw_result[1]) -> goal-call content="93.0" (value only)
    # -> evaluator action_successfully_called (tuple[0] check) can't recognize success for tuple-return
    # goals (exchange_foreign_currency, get_account_owed_balance). Released runs keep the full tuple
    # (content "(True, 93.0)", asc 44/86) => preserving the tuple RESTORES the official measurement
    # (not gaming). Guard behind SOPBENCH_KEEPTUPLE for clean A/B.
    cc = os.path.join(clone, "swarm", "core.py")
    _patch(cc, [
        ("                if isinstance(raw_result, tuple):\n"
         "                    raw_result = raw_result[1]",
         "                if isinstance(raw_result, tuple) and not __import__(\"os\").environ.get(\"SOPBENCH_KEEPTUPLE\"):\n"
         "                    raw_result = raw_result[1]"),
    ], marker="SOPBENCH_KEEPTUPLE")

    print("DONE.")


if __name__ == "__main__":
    main()
