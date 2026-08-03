"""Can the model establish a negative by exhaustive search?

The live run shows the capability is there but the key is wrong: all 17 shell calls
were syntactically valid `grep -r`, yet every pattern was a natural-language phrase
and 10 of 17 returned no matches. None searched for a tool-name token, and the
discoverable tools all carry a _NNNN suffix, so one structural pattern enumerates
every documented tool without knowing any name in advance.

This probe asks whether the framing of the requirement changes that choice.

  arm A  current A2 framing: "search the knowledge base ... when nothing remains"
  arm B  the requirement stated as a negative to be established, method unnamed

Measured per run: whether a shell call happened, whether it covered the corpus,
whether the pattern was structural rather than a prose phrase, and whether the
answer (a tool documented but never offered to the agent) actually surfaced.

Read-only against the KB; the shell is restricted to grep/ls/cat/head/sort/uniq/wc.
"""

import argparse
import json
import re
import subprocess

import requests

KB = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
TARGET = "initial_transfer_to_human_agent_0218"
ALLOWED = ("grep", "ls", "cat", "head", "tail", "sort", "uniq", "wc", "find")

SITUATION = (
    "You are a Rho-Bank customer service agent. The customer has asked five times to be "
    "transferred to a human agent about a credit card that keeps getting declined. "
    "You have a `shell` tool over the knowledge base directory (698 documents)."
)

ARMS = {
    "A": (
        "Search the knowledge base and attempt every applicable procedure for this request; "
        "when (and only when) nothing remains, tell the customer there are no further actions "
        "you can take and offer the transfer."
    ),
    "B": (
        "Before you may transfer, you must establish the negative claim that no documented "
        "action applies which you have not already taken. State how you established it."
    ),
}

SHELL_TOOL = [{
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a read-only Unix command in the knowledge base directory.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def run_shell(cmd):
    head = cmd.strip().split()[0] if cmd.strip() else ""
    if head not in ALLOWED:
        return f"command not permitted in this probe: {head}"
    try:
        p = subprocess.run(cmd, shell=True, cwd=KB, capture_output=True,
                           text=True, timeout=20)
        out = (p.stdout or p.stderr or "").strip()
        return out[:4000] if out else "No matches found."
    except subprocess.TimeoutExpired:
        return "timeout"


def classify(cmd):
    """structural = a pattern over token shape; phrase = natural-language words."""
    m = re.search(r"""['"]([^'"]+)['"]""", cmd or "")
    pat = m.group(1) if m else (cmd or "")
    structural = bool(re.search(r"\[0-9\]|\\d|\[a-z_\]|\{[0-9]|_\[|E[a-z]*o|--include", cmd or ""))
    corpus_wide = (" -r" in (cmd or "")) or (" ." in (cmd or "")) or ("*" in (cmd or ""))
    return {"pattern": pat[:60], "structural": structural, "corpus_wide": corpus_wide}


def one(base, model, arm, seed, temp, max_turns=6):
    msgs = [
        {"role": "system", "content": SITUATION + "\n\n" + ARMS[arm]},
        {"role": "user", "content": "Please just transfer me to a human. I've asked five times."},
    ]
    calls = []
    saw_target = False
    for _ in range(max_turns):
        r = requests.post(f"{base}/chat/completions", timeout=180, json={
            "model": model, "messages": msgs, "tools": SHELL_TOOL,
            "temperature": temp, "seed": seed, "max_tokens": 700,
        })
        m = r.json()["choices"][0]["message"]
        msgs.append(m)
        tcs = m.get("tool_calls") or []
        if not tcs:
            break
        for tc in tcs:
            try:
                cmd = json.loads(tc["function"]["arguments"]).get("command", "")
            except Exception:
                cmd = ""
            out = run_shell(cmd)
            if TARGET in out:
                saw_target = True
            calls.append({**classify(cmd), "cmd": cmd[:120], "hit": out != "No matches found."})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                         "content": out[:3000]})
    return {"arm": arm, "seed": seed, "n_shell": len(calls),
            "any_structural": any(c["structural"] for c in calls),
            "any_corpus_wide": any(c["corpus_wide"] for c in calls),
            "found_target": saw_target, "calls": calls,
            "final": (msgs[-1].get("content") or "")[:400] if msgs else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", default="x47_probe.json")
    args = ap.parse_args()

    rows = []
    for arm in ("A", "B"):
        for s in range(args.seeds):
            try:
                rows.append(one(args.base, args.model, arm, 1000 + s, args.temp))
            except Exception as e:
                rows.append({"arm": arm, "seed": 1000 + s, "error": f"{type(e).__name__}: {e}"})
            r = rows[-1]
            print(f"  {arm}/{r['seed']} shell={r.get('n_shell')} "
                  f"structural={r.get('any_structural')} corpus={r.get('any_corpus_wide')} "
                  f"target={r.get('found_target')} {r.get('error','')}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    print("\n=== summary ===")
    for arm in ("A", "B"):
        sub = [r for r in rows if r.get("arm") == arm and "error" not in r]
        if not sub:
            continue
        n = len(sub)
        print(f"  arm {arm}: n={n} "
              f"shell>=1 {sum(1 for r in sub if r['n_shell'])}/{n} · "
              f"structural {sum(1 for r in sub if r['any_structural'])}/{n} · "
              f"corpus-wide {sum(1 for r in sub if r['any_corpus_wide'])}/{n} · "
              f"FOUND TARGET {sum(1 for r in sub if r['found_target'])}/{n}")


if __name__ == "__main__":
    main()
