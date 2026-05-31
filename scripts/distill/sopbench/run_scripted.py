"""run_scripted.py — §8.1 harness verification with a DETERMINISTIC (no-LLM) scripted-gather
agent run through the REAL SOPBench harness + evaluator.

Goal: settle, reliably (real evaluator_function_directed_graph, not a hand-replay), whether
under-verification is the should_T binding and whether an A+B+C gather reconstructs full success:
  - mode=oracle : emit exactly the directed_action_graph tool nodes (args-aware, dups kept) + goal.
                  SANITY: should reproduce the goal-callable ceiling (~40/48). If it does, the
                  harness+eval wiring is trustworthy and the abc/ab numbers below can be believed.
  - mode=abc    : NON-oracle gather from task["constraints"] (A: callable checks, args-aware)
                  + condition->getter (B) + innate_dep login/auth (C) + goal.
  - mode=ab     : same as abc but WITHOUT C (no innate login/auth unless in the task constraint).
                  abc>ab  => C (innate-dep) is strictly required.

The scripted client emits ONE synthetic tool-call completion per turn (reusing the same
ChatCompletion synthesis as TwoStageClient's deterministic path) so NO model/vLLM/OpenRouter is
needed. func_calls are built as DICTS {tool_name,arguments,content} (evidence_a_probe-proven
format) — NOT run_two_stage's deprecated tuple format.

RUN (clone root, seka_env python):
  /home/woori/venvs/seka_env/bin/python scripts/run_scripted.py --domain bank --mode oracle
"""
import sys, os, json, argparse, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from swarm.core import Swarm
from swarm.types import Agent
from swarm.util import function_to_json
from env.task import task_initializer, task_default_dep_full
from env.variables import domain_assistant_keys
from env.evaluator import evaluator_function_directed_graph

# B: condition predicate -> verifying getter tool (domain-constant HOW; from lever_decomp co-occurrence)
GETTER = {"minimal_elgibile_credit_score": "internal_get_credit_score",
          "sufficient_account_balance": "get_account_balance",
          "no_credit_card_balance_on_card": "get_credit_card_info",
          "not_over_credit_limit": "get_credit_card_info",
          "internal_check_credit_card_exist": "get_credit_card_info",
          "get_loan_owed_balance_restr": "get_account_owed_balance",
          "pay_loan_account_balance_restr": "get_account_balance",
          "pay_loan_amount_restr": "get_account_balance",
          "safety_box_eligible": "get_account_balance",
          "maximum_exchange_amount": "internal_get_database",
          "maximum_deposit_limit": "internal_get_database"}
CALLABLE_CHECK = {"internal_check_username_exist", "internal_check_foreign_currency_available"}
AUTH = ("login_user", "authenticate_admin_password")
AUTH_PRED = {"logged_in_user": "login_user", "authenticated_admin_password": "authenticate_admin_password"}
# preferred call order: existence checks -> login -> auth -> (post-auth) getters -> goal
PRE = ["internal_get_database", "internal_check_username_exist", "internal_check_foreign_currency_available",
       "login_user", "authenticate_admin_password",
       "internal_get_credit_score", "get_credit_cards", "get_credit_card_info",
       "get_account_balance", "get_account_owed_balance", "get_loan"]


def exit_conversation():
    """End the conversation."""
    return "Conversation ended."


def walk(tree, acc):
    if not tree or not isinstance(tree, (list, tuple)) or not tree:
        return
    h = tree[0]
    if h == "single":
        acc.append((tree[1][4:] if tree[1].startswith("not ") else tree[1], tree[2] if len(tree) > 2 else {}))
    elif h in ("and", "or", "chain", "gate"):
        for s in tree[1]:
            walk(s, acc)


class ScriptedGatherClient:
    """OpenAIHandler-compatible deterministic client: emits a precomputed call plan, no LLM."""
    def __init__(self, mode, ont, dep_innate):
        self.mode = mode; self.ont = ont; self.dep_innate = dep_innate
        self.model_name_huggingface = f"scripted_{mode}"
        self.plan = []; self._i = 0; self._tid = 0

    def _slots(self, task):
        s = dict(task.get("user_known", {}))
        accts = task["initial_database"].get("accounts", {}); un = s.get("username")
        if un in accts and isinstance(accts[un], dict):
            for k, v in accts[un].items():
                s.setdefault(k, v)
        return s

    def _argmaps(self, task):
        return {n[0]: (n[1] or {}) for n in task.get("directed_action_graph", {}).get("nodes", [])
                if isinstance(n, (list, tuple)) and len(n) == 2 and isinstance(n[0], str)}

    def reset(self, task=None, **kw):
        self._i = 0; self._tid = 0; self.plan = []
        if task is None:
            return
        goal = task["user_goal"]; slots = self._slots(task)
        am = self._argmaps(task)
        ops = self.ont.get("operators", {}); preds = self.ont.get("predicates", {})

        def resolve(name, pmap):
            return {p: slots[s] for p, s in pmap.items() if s in slots}

        calls = []  # list of (name, args), order-sorted later (goal always last)
        if self.mode == "oracle":
            # every dirgraph tool node (keep dups: e.g. check(source)+check(dest)), goal last
            for n in task.get("directed_action_graph", {}).get("nodes", []):
                if isinstance(n, (list, tuple)) and len(n) == 2 and isinstance(n[0], str) and n[0] != goal:
                    calls.append((n[0], resolve(n[0], n[1] or {})))
        else:  # abc / ab : NON-oracle gather from task constraints + ontology
            leaves = []; walk(task.get("constraints"), leaves)
            for nm, pm in leaves:
                if nm in AUTH_PRED:                       # establishable login/auth (in constraint)
                    by = AUTH_PRED[nm]
                    calls.append((by, resolve(by, ops.get(by, {}).get("args", {}))))
                elif nm in CALLABLE_CHECK:                # A: callable check, args-aware
                    calls.append((nm, resolve(nm, pm)))
                elif preds.get(nm, {}).get("kind") == "condition" and nm in GETTER:  # B: condition->getter
                    g = GETTER[nm]; calls.append((g, resolve(g, ops.get(g, {}).get("args", pm))))
            if self.mode == "abc":                        # C: innate login/auth not in constraint
                inn = json.dumps(self.dep_innate.get(goal))
                for pred, by in AUTH_PRED.items():
                    if pred in inn and not any(c[0] == by for c in calls):
                        calls.append((by, resolve(by, ops.get(by, {}).get("args", {}))))
        # order: PRE first (stable), then goal
        rank = {n: i for i, n in enumerate(PRE)}
        calls.sort(key=lambda c: rank.get(c[0], 999))
        # de-dup identical (name,args) keep order
        seen = set(); plan = []
        for nm, ar in calls:
            k = (nm, json.dumps(ar, sort_keys=True))
            if k in seen:
                continue
            seen.add(k); plan.append((nm, ar))
        plan.append((goal, resolve(goal, am.get(goal, ops.get(goal, {}).get("args", {})))))
        plan.append(("exit_conversation", {}))
        self.plan = plan

    def _completion(self, name, args):
        self._tid += 1
        tc = ChatCompletionMessageToolCall(id=f"call_{name}_{self._tid}", type="function",
                                           function=Function(name=name, arguments=json.dumps(args)))
        msg = ChatCompletionMessage(role="assistant", content=None, tool_calls=[tc])
        ch = Choice(index=0, message=msg, finish_reason="tool_calls", logprobs=None)
        return ChatCompletion(id=f"s_{self._tid}", object="chat.completion", created=0,
                              model=self.model_name_huggingface, choices=[ch], usage=None)

    def inference(self, create_params, debug=False, mode="chat", tool_call_mode="fc"):
        if self._i < len(self.plan):
            name, args = self.plan[self._i]; self._i += 1
        else:
            name, args = "exit_conversation", {}
        return {"idx": self._i, "completion": self._completion(name, args)}

    def kill_process(self):
        pass


def build_func_calls(messages):
    """Pair each tool result with its preceding assistant tool-call -> dict func_calls."""
    out = []; pend = []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                nm = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                ar = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
                try:
                    ar = json.loads(ar) if isinstance(ar, str) else (ar or {})
                except Exception:
                    ar = {}
                pend.append((nm, ar))
        if m.get("role") == "tool":
            nm = m.get("tool_name"); content = m.get("content")
            # harness stringifies tool returns; restore python objects (True/False/tuples) so the
            # evaluator's value checks (action_successfully_called etc.) see real bools, not "True".
            if isinstance(content, str):
                import ast as _ast
                try:
                    content = _ast.literal_eval(content)
                except Exception:
                    pass
            args = {}
            for j in range(len(pend) - 1, -1, -1):
                if pend[j][0] == nm:
                    args = pend[j][1]; pend.pop(j); break
            out.append({"tool_name": nm, "arguments": args, "content": content})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bank")
    ap.add_argument("--mode", default="oracle", choices=["oracle", "abc", "ab"])
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--ont_dir", default="./induced")
    ap.add_argument("--option", default="full")
    ap.add_argument("--fmt", default="structured")
    ap.add_argument("--tool_list", default="full", choices=["full", "oracle"])
    ap.add_argument("--debug", default=None, help="goal name to dump once (e.g. transfer_funds)")
    args = ap.parse_args()
    _dumped = [False]

    ont = json.load(open(f"{args.ont_dir}/ontology_{args.domain}.json"))
    dep_innate_v = domain_assistant_keys[args.domain].action_innate_dependencies
    dep_innate, dep_full, dep_descr = task_default_dep_full(
        args.domain, args.option, args.fmt, dependency_verb_dep_orig=True)
    raw = json.load(open(f"{args.data_dir}/{args.domain}_tasks.json"))
    tasks = [dict(t, user_goal=g) for g, v in raw.items() for t in v]

    client = ScriptedGatherClient(args.mode, ont, dep_innate_v)
    assistant = Agent(name="assistant", client=client, instructions="", functions=[],
                      tool_call_mode="fc", temperature=0.0, top_p=0.01, max_tokens=512)
    user = Agent(name="user", client=None, default_response="", response_repeat=True)

    st = sf = stp = sfp = 0
    fails = []
    for task in tasks:
        should = bool(task.get("action_should_succeed"))
        client.reset(task=task)
        included = ([n[0] for n in task["directed_action_graph"]["nodes"] if isinstance(n, list)]
                    if args.tool_list == "oracle" else None)
        dsys, uinfo, ainfo, tinfo = task_initializer(
            args.domain, task, dep_innate, dep_full, dep_descr, included, "prompt", False, args.fmt)
        assistant.instructions = ainfo["instructions"]
        assistant.functions = ainfo["tools"] + [function_to_json(exit_conversation)]
        assistant.name = f"{args.domain} assistant"
        # filter the scripted plan to tools that actually exist in the agent's tool list
        avail = {t["function"]["name"] for t in assistant.functions}
        if not _dumped[0]:
            _dumped[0] = True
            need = set(GETTER.values()) | CALLABLE_CHECK | set(AUTH)
            print(f"[tool_list={args.tool_list}] full-avail B-getters/checks present: "
                  f"{sorted(n for n in need if n in avail)}  | MISSING: {sorted(n for n in need if n not in avail)}")
        client.plan = [(n, a) for (n, a) in client.plan if n in avail]
        if not client.plan or client.plan[-1][0] != "exit_conversation":
            client.plan.append(("exit_conversation", {}))
        dm = ("Here is all the information I can provide:\n" + json.dumps(uinfo["known"], indent=4) +
              f"\n\nIf you have completed my request or cannot assist me, please use the "
              f"`exit_conversation` action.")
        user.default_response = dm
        sw = Swarm(system=dsys, max_turns=30, max_actions=20)
        inter = sw.run_user_assistant_interaction(
            user_agent=user, assistant_agent=assistant,
            messages=[{"role": "user", "content": dm, "sender": "user"}],
            debug=False, execute_tools=True, start_agent="assistant",
            finished_action=exit_conversation)
        fc = build_func_calls(inter.messages)
        db = dsys.evaluation_get_database() if hasattr(dsys, "evaluation_get_database") else dsys.data
        ev = evaluator_function_directed_graph(args.domain, task, [], fc,
                                               {"final_database": db}, args.option)
        ok = bool(ev.get("success"))
        if args.debug and should and task["user_goal"] == args.debug:
            print(f"\n--- DEBUG {task['user_goal']} ---")
            print("  plan:", [(n, a) for n, a in client.plan])
            print("  func_calls:", json.dumps(fc)[:600])
            print("  avail_has_internal_get_db:", "internal_get_database" in avail)
            print("  ev:", {k: ev.get(k) for k in ("success", "action_successfully_called",
                  "dirgraph_satisfied", "action_called_correctly", "constraint_not_violated",
                  "database_match", "no_tool_call_error")})
            args.debug = None  # once
        if should:
            st += 1; stp += ok
            if not ok:
                fails.append((task["user_goal"], ev.get("action_successfully_called"),
                              ev.get("dirgraph_satisfied"), ev.get("constraint_not_violated")))
        else:
            sf += 1; sfp += ok
    print(f"=== run_scripted mode={args.mode} domain={args.domain} ===")
    print(f"  should_T success: {stp}/{st}")
    print(f"  should_F success: {sfp}/{sf}")
    print(f"  (should_T fails: asc/dirgraph/constraint gate breakdown)")
    from collections import Counter
    c = Counter((a, d, cn) for _, a, d, cn in fails)
    for (a, d, cn), n in c.most_common():
        print(f"    {n:3d}  action_called={a} dirgraph={d} constraint_ok={cn}")


if __name__ == "__main__":
    main()
