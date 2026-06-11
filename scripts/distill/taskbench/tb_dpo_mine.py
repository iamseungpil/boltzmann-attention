#!/usr/bin/env python
"""L2: mine early-closure DPO pairs from RFT rollout .all dumps (census 누락축 처방).

Pair per prompt: chosen = best-reward COMPLETE sample (n_nodes >= gold_n, reward >= min_r);
rejected = an EARLY-CLOSED sample (n_nodes < gold_n) with reward <= chosen - margin.
Output = {"prompt", "chosen", "rejected"} jsonl for scripts/distill/sopbench/dpo_train.py.

Usage:
  python tb_dpo_mine.py --all winners_rft2_mm.jsonl.all --sft_jsonl train_lodo_mm.jsonl \
      --out dpo_earlyclose.jsonl --min_r 0.8 --margin 0.05
"""
import argparse, json, re


def parse_nodes(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        d = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except Exception:
            return None
    n = d.get("task_nodes") if isinstance(d, dict) else None
    return len(n) if isinstance(n, list) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True)
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_r", type=float, default=0.8)
    ap.add_argument("--margin", type=float, default=0.05)
    a = ap.parse_args()

    meta = {}  # id -> (prompt, gold_n)
    for l in open(a.sft_jsonl):
        d = json.loads(l)
        g = json.loads(d["messages"][1]["content"])
        meta[d["meta"]["id"]] = (d["messages"][0]["content"], len(g["task_nodes"]))

    n_pairs = 0
    stats = {"no_meta": 0, "no_complete": 0, "no_short": 0}
    with open(a.out, "w") as wf:
        for l in open(a.all):
            d = json.loads(l)
            if d["id"] not in meta:
                stats["no_meta"] += 1
                continue
            prompt, gold_n = meta[d["id"]]
            best_c = None  # (reward, text)
            worst_s = None
            for s in d["samples"]:
                n = parse_nodes(s["text"])
                if n is None:
                    continue
                if n >= gold_n and s["reward"] >= a.min_r:
                    if best_c is None or s["reward"] > best_c[0]:
                        best_c = (s["reward"], s["text"])
                elif n < gold_n:
                    if worst_s is None or s["reward"] < worst_s[0]:
                        worst_s = (s["reward"], s["text"])
            if best_c is None:
                stats["no_complete"] += 1
                continue
            if worst_s is None:
                stats["no_short"] += 1
                continue
            if best_c[0] - worst_s[0] < a.margin:
                continue
            wf.write(json.dumps({"prompt": prompt, "chosen": best_c[1],
                                 "rejected": worst_s[1]}) + "\n")
            n_pairs += 1
    print(f"[dpo-mine] pairs={n_pairs} skip={stats}")


if __name__ == "__main__":
    main()
