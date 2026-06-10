#!/usr/bin/env python
"""Outcome-RFT rollout (RAFT round, Exp-A stage-2; FIELD_GAP §18.1, reward unblocked by A-0).

Samples K completions per training prompt from the served SFT policy, scores each
against gold with node/edge-F1 (exact match — A-0: real-error 73%, convention 27%
documented caveat), and keeps the best sample above --min_reward as a new SFT
target (winners.jsonl, trainer-compatible messages format).

Usage (remote, tbeval venv):
  python tb_rft_rollout.py --sft_jsonl /home/woori/scratch/tb_sft/train_lodo_mm.jsonl \
      --api http://localhost:8000/v1/chat/completions --model tb_lodo_mm \
      --k 8 --temp 1.0 --min_reward 0.8 --out /home/woori/scratch/tb_rft/winners_lodo_mm.jsonl
"""
import argparse, asyncio, json, os, re

import aiohttp


def norm(s):
    return str(s).replace("_", " ")


def tag_links(nodes):
    names = [norm(n.get("task", "")) for n in nodes]
    links = set()
    for inx, node in enumerate(nodes):
        args_ = node.get("arguments", [])
        if not isinstance(args_, list):
            continue
        for argument in args_:
            try:
                if isinstance(argument, dict):
                    argument = list(argument.values())[0]
                if isinstance(argument, list):
                    argument = " ".join(str(a) for a in argument)
                if isinstance(argument, str) and "<node-" in argument:
                    j = int(argument[argument.index("<node-") + 6:argument.index(">")])
                    if j != inx and 0 <= j < len(names):
                        links.add((names[j], names[inx]))
            except Exception:
                pass
    return links


def f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    p = inter / len(pred)
    r = inter / len(gold)
    return 2 * p * r / (p + r) if p + r else 0.0


def parse_result(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        d = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except Exception:
            return None
    return d if isinstance(d, dict) else None


def reward(sample_text, gold_result, dep):
    d = parse_result(sample_text)
    if d is None:
        return 0.0, False
    pn = d.get("task_nodes")
    if not isinstance(pn, list) or any(not isinstance(x, dict) for x in pn):
        return 0.0, False
    gn = gold_result["task_nodes"]
    node_f1 = f1({norm(x.get("task", "")) for x in pn}, {norm(x["task"]) for x in gn})
    if dep == "temporal":
        gl = {(norm(l["source"]), norm(l["target"])) for l in gold_result.get("task_links", [])}
        pl_raw = d.get("task_links")
        pl = set()
        if isinstance(pl_raw, list):
            pl = {(norm(l["source"]), norm(l["target"])) for l in pl_raw
                  if isinstance(l, dict) and "source" in l and "target" in l}
        edge_f1 = f1(pl, gl)
    else:
        edge_f1 = f1(tag_links(pn), tag_links(gn))
    return 0.3 * node_f1 + 0.7 * edge_f1, True


async def roll_one(session, sem, rec, args):
    user = rec["messages"][0]["content"]
    gold = json.loads(rec["messages"][1]["content"])
    dep = "temporal" if rec.get("meta", {}).get("domain", "").find("dailylife") >= 0 else "resource"
    payload = {"model": args.model,
               "messages": [{"role": "user", "content": user}],
               "n": args.k, "temperature": args.temp, "top_p": 0.95,
               "max_tokens": 2000}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(args.api, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=600)) as r:
                    resp = await r.json()
                break
            except Exception:
                if attempt == 2:
                    return None
                await asyncio.sleep(5)
    best_r, best_t = -1.0, None
    parsed_ok = 0
    for ch in resp.get("choices", []):
        t = ch["message"]["content"] or ""
        rw, ok = reward(t, gold, dep)
        parsed_ok += ok
        if rw > best_r:
            best_r, best_t = rw, t
    return {"id": rec.get("meta", {}).get("id"), "domain": rec.get("meta", {}).get("domain"),
            "best_reward": best_r, "parsed": parsed_ok,
            "winner": {"messages": [{"role": "user", "content": user},
                                    {"role": "assistant", "content": best_t}],
                       "meta": {**rec.get("meta", {}), "rft_reward": best_r}}}


async def main_async(args):
    recs = [json.loads(l) for l in open(args.sft_jsonl)]
    if args.max_prompts:
        recs = recs[:args.max_prompts]
    sem = asyncio.Semaphore(args.concurrency)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kept = 0
    rewards = []
    async with aiohttp.ClientSession() as session:
        tasks = [roll_one(session, sem, r, args) for r in recs]
        with open(args.out, "w") as wf, open(args.out + ".stats", "w") as sf:
            done = 0
            for fut in asyncio.as_completed(tasks):
                res = await fut
                done += 1
                if res is None:
                    continue
                rewards.append(res["best_reward"])
                sf.write(json.dumps({k: res[k] for k in ("id", "domain", "best_reward", "parsed")}) + "\n")
                if res["best_reward"] >= args.min_reward:
                    wf.write(json.dumps(res["winner"]) + "\n")
                    kept += 1
                if done % 200 == 0:
                    print(f"[rft] {done}/{len(recs)} kept={kept} "
                          f"mean_best={sum(rewards)/len(rewards):.3f}", flush=True)
    print(f"[rft] DONE prompts={len(recs)} kept={kept} ({100*kept/max(len(recs),1):.1f}%) "
          f"mean_best_reward={sum(rewards)/max(len(rewards),1):.3f} -> {args.out}", flush=True)
    print("ROLLOUT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--api", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="tb_lodo_mm")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--min_reward", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_prompts", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
