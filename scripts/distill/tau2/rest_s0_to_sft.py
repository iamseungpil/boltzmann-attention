#!/usr/bin/env python
"""ReST S0: 32B 생성 keep(db_match) 궤적 → SFT jsonl (실도구·프롬프트제거·discipline 타깃).

설계=REST_INTERNALIZE_DESIGN §3 타깃화:
- keep = reward≥1 (DB-match) 궤적만.
- system = base retail 정책(SYSTEM_PROMPT.format)·**NOFAB rules 제거**(프롬프트 없이 discipline 발현).
- tools = 실 retail 16도구(openai_schema)·#9 교정(추상 alias 금지).
- tool_calls = concrete 실값(빈arg/$ref 아님·생성궤적이 이미 그러함).
- supervise="assistant"(discipline 시퀀스=producer 호출·복사·순서 내재화).
포맷=fc_build/sft_solo_*.jsonl 호환 {tools, messages, _meta, supervise}.

+ §4b-b coverage 진단: keep가 *하드*(base7B floor 실패=discipline-critical) 대표하나 vs 쉬운쪽 쏠림.
Usage: rest_s0_to_sft.py <results.json> <out.jsonl> [max_per_task]
"""
import json, sys, os, collections

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/home/woori/scratch/tau2-bench/data/simulations/rest_s0_gen32b/results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/woori/scratch/fc_build/sft_rest_s0_retail.jsonl"
MAX_PER_TASK = int(sys.argv[3]) if len(sys.argv) > 3 else 4   # 쏠림 완화·다양성 유지

sys.path.insert(0, "src")
sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")

# --- clean base retail system prompt + tools (NO NOFAB) ---
from tau2.registry import registry
from tau2.agent.llm_agent import LLMAgent
env = registry.get_env_constructor("retail")()
tools = env.get_tools()
tools_openai = [t.openai_schema for t in tools]
policy = env.get_policy() if hasattr(env, "get_policy") else None
agent = LLMAgent(tools=tools, domain_policy=policy, llm="x", llm_args={})
SYS_CLEAN = agent.system_prompt
assert "NEVER FABRICATE" not in SYS_CLEAN, "NOFAB leaked into base system prompt!"
print(f"[sys] clean retail system prompt = {len(SYS_CLEAN)} chars · tools={len(tools_openai)}")


def norm_tool_calls(tcs):
    out = []
    for i, tc in enumerate(tcs or []):
        fn = tc.get("function") or {}
        name = tc.get("name") or fn.get("name")
        args = tc.get("arguments")
        if args is None:
            args = fn.get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        elif args is None:
            args = "{}"
        out.append({"id": tc.get("id") or f"call_{i+1}", "type": "function",
                    "function": {"name": name, "arguments": args}})
    return out


def clean_messages(msgs):
    out = []
    for m in msgs:
        r = m.get("role")
        if r == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        elif r == "assistant":
            mm = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = norm_tool_calls(m["tool_calls"])
            out.append(mm)
        elif r == "tool":
            out.append({"role": "tool", "content": m.get("content") or "",
                        "tool_call_id": m.get("tool_call_id") or m.get("id") or ""})
    return out


d = json.load(open(RESULTS))
sims = d.get("simulations") or []
floor_path = "/home/woori/scratch/tau2-bench/data/simulations/on_n7b_floor_retail/results.json"
floor_pass = set()
if os.path.exists(floor_path):
    fd = json.load(open(floor_path))
    for s in fd.get("simulations") or []:
        db = ((s.get("reward_info") or {}).get("db_check") or {})
        if db.get("db_match") or (((s.get("reward_info") or {}).get("reward", 0)) or 0) >= 0.999:
            floor_pass.add(str(s.get("task_id")))

by_task = collections.defaultdict(list)
for s in sims:
    db = ((s.get("reward_info") or {}).get("db_check") or {})
    ok = bool(db.get("db_match")) or (((s.get("reward_info") or {}).get("reward", 0)) or 0) >= 0.999
    if not ok:
        continue
    msgs = [m for m in (s.get("messages") or []) if isinstance(m, dict)]
    # must contain >=1 assistant tool_call (discipline content)
    if not any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs):
        continue
    by_task[str(s.get("task_id"))].append(s)

written = 0
kept_tasks = sorted(by_task, key=lambda x: int(x) if x.isdigit() else 1e9)
ncalls_hist = collections.Counter()
with open(OUT, "w") as f:
    for tid in kept_tasks:
        for s in by_task[tid][:MAX_PER_TASK]:
            msgs = [m for m in (s.get("messages") or []) if isinstance(m, dict)]
            conv = clean_messages(msgs)
            ncalls = sum(len(m.get("tool_calls", [])) for m in conv if m["role"] == "assistant")
            ncalls_hist[ncalls] += 1
            rec = {"tools": tools_openai,
                   "messages": [{"role": "system", "content": SYS_CLEAN}] + conv,
                   "_meta": {"task_id": tid, "trial": s.get("trial"), "source": "rest_s0_gen32b",
                             "domain": "retail", "bench": "tau2_retail", "db_match": True,
                             "floor_fail": tid not in floor_pass},
                   "supervise": "assistant"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

# ---- coverage diagnostic (§4b-b) ----
covered = set(by_task)
floor_fail_covered = [t for t in covered if t not in floor_pass]
floor_pass_covered = [t for t in covered if t in floor_pass]
print(f"\n[OUT] {OUT}  written={written} trajectories  (max_per_task={MAX_PER_TASK})")
print(f"[coverage] tasks covered = {len(covered)}/114")
print(f"  EASY (base7B floor passes)        = {len(floor_pass_covered)}")
print(f"  ★HARD/discipline-critical (floor 실패하나 keep 확보) = {len(floor_fail_covered)}")
print(f"  → ReST가 배울 *하드* gold 궤적 보유 태스크 = {len(floor_fail_covered)}개")
percnt = {t: len(v) for t, v in sorted(by_task.items(), key=lambda kv: -len(kv[1]))}
print(f"[balance] keep/task 분포 (cap전): max={max(percnt.values())} min={min(percnt.values())} "
      f"·5개꽉={sum(1 for v in percnt.values() if v>=5)} ·1개뿐={sum(1 for v in percnt.values() if v==1)}")
print(f"[depth] assistant tool_call 수/궤적 분포: {dict(sorted(ncalls_hist.items()))}")
miss = sorted(set(str(i) for i in range(114)) - covered, key=int)
print(f"[hole] 정답0 잔여(72B/gpt-4.1 top-up 대상) = {miss}")
