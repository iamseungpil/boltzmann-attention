#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-P2 - NESTful v2 분해: tier2 몫-안정성 검증 (P_OC_MAPPING §4).

각 샘플을 로컬 vLLM으로 생성 -> gold(중첩 var-참조 시퀀스)와 정렬 대조 ->
관찰 O 분류 {MISSING_STEP, EXTRA_STEP, WRONG_FUNC, ARG_LITERAL, ARG_REF, PARSE_FAIL}
-> C 배정은 per-case 정독으로 확정([[08]]·P_OC §2e 다대다).
검증 물음: 기존 11-C 셀 밖의 *새 non-empty 류*가 나오는가 (§0-2 몫 안정성).

크래시 안전: jsonl append·재실행 이어감.
Run: python3 ep2_nestful.py --n 200 --port 8141 --model Qwen/QwQ-32B-AWQ
"""
import argparse
import json
import os
import re
import urllib.request

DATA = "/home/woori/scratch/NESTFUL/data_v2/nestful_data.jsonl"
OUT = "/home/woori/scratch/ep2_nestful_results.jsonl"

SYS = ("You are a function-composition assistant. Given TOOLS and a QUESTION, output ONLY a JSON "
       "array of call steps that answers the question, in execution order. Each step: "
       '{"name": "<tool name>", "label": "$var_N", "arguments": {<param>: <value or reference>}}. '
       'To use the output of an earlier step as an argument, write the string "$var_N.<output_key>$" '
       '(e.g. "$var_1.result$"). Use only the provided tools. No prose, no markdown fence — JSON array only.')


def chat(port, model, msgs, max_tokens=1600):
    p = {"model": model, "messages": msgs, "temperature": 0.0, "max_tokens": max_tokens}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"].get("content") or ""


def parse_pred(txt):
    t = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    t = re.sub(r"```(?:json)?", "", t)
    m = re.search(r"\[.*\]", t, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def ref_of(v):
    m = re.match(r"^\$(var_\d+)\.([\w]+)\$$", str(v).strip())
    return m.groups() if m else None


def label_index(seq):
    return {str(s.get("label") or f"$var_{i+1}").lstrip("$"): i for i, s in enumerate(seq)}


def norm_args(seq):
    """각 step의 arguments를 (리터럴 정규화 · 참조는 참조-step의 시퀀스 인덱스로) 변환."""
    lidx = label_index(seq)
    out = []
    for s in seq:
        d = {}
        for k, v in (s.get("arguments") or {}).items():
            r = ref_of(v)
            if r:
                d[k] = ("REF", lidx.get(r[0], -1), r[1])
            else:
                try:
                    d[k] = ("LIT", round(float(v), 6))
                except Exception:
                    d[k] = ("LIT", str(v).strip().lower())
        out.append((str(s.get("name")), d))
    return out


def classify(gold, pred):
    """관찰 O 다중라벨 + 정오."""
    obs = set()
    G, P = norm_args(gold), norm_args(pred)
    used = set()
    align = []
    for gi, (gn, ga) in enumerate(G):
        pj = next((j for j, (pn, _) in enumerate(P) if j not in used and pn == gn), None)
        if pj is None:
            obs.add("MISSING_STEP" if all(pn != gn for pn, _ in P) else "MISSING_STEP")
            align.append((gi, None))
        else:
            used.add(pj)
            align.append((gi, pj))
    if len(P) > len(used):
        obs.add("EXTRA_STEP")
    gnames = [n for n, _ in G]
    pnames = [n for n, _ in P]
    if len(G) == len(P) and sorted(gnames) != sorted(pnames) and any(a[1] is None for a in align):
        obs.add("WRONG_FUNC")
    exact = len(G) == len(P)
    gi2pj = {gi: pj for gi, pj in align}
    for gi, pj in align:
        if pj is None:
            exact = False
            continue
        ga, pa = G[gi][1], P[pj][1]
        for k, gv in ga.items():
            pv = pa.get(k)
            if pv is None:
                obs.add("ARG_MISSING")
                exact = False
            elif gv[0] == "REF":
                ok = (pv[0] == "REF" and gi2pj.get(gv[1]) == pv[1] and gv[2] == pv[2])
                if not ok:
                    obs.add("ARG_REF")
                    exact = False
            else:
                if pv != gv:
                    obs.add("ARG_LITERAL")
                    exact = False
        if set(pa) - set(ga):
            obs.add("ARG_EXTRA")
            exact = False
    return sorted(obs), exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/QwQ-32B-AWQ")
    a = ap.parse_args()
    samples = [json.loads(l) for l in open(DATA, encoding="utf-8")][: a.n]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(l)["sample_id"])
            except Exception:
                pass
    print("samples %d · done %d" % (len(samples), len(done)), flush=True)
    fp = open(OUT, "a", encoding="utf-8")
    for i, s in enumerate(samples):
        if s["sample_id"] in done:
            continue
        tools = json.dumps(s.get("tools") or [], ensure_ascii=False)
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": "TOOLS:\n" + tools + "\n\nQUESTION: " + s["input"]}]
        try:
            txt = chat(a.port, a.model, msgs)
        except Exception as e:
            txt = "ERR:" + type(e).__name__
        pred = parse_pred(txt)
        if pred is None:
            rec = {"sample_id": s["sample_id"], "obs": ["PARSE_FAIL"], "exact": False,
                   "n_gold": len(s["output"]), "raw": (txt or "")[:200]}
        else:
            try:
                obs, exact = classify(s["output"], pred)
            except Exception as e:
                obs, exact = ["CLASSIFY_ERR:" + type(e).__name__], False
            rec = {"sample_id": s["sample_id"], "obs": obs, "exact": exact,
                   "n_gold": len(s["output"]), "n_pred": len(pred), "pred": pred[:8]}
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fp.flush()
        if (i + 1) % 10 == 0:
            print("  ..%d/%d" % (i + 1, len(samples)), flush=True)
    fp.close()
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    from collections import Counter
    print("\n=== E-P2 NESTful (n=%d) ===" % len(rows))
    print("exact:", sum(1 for r in rows if r.get("exact")), "/", len(rows))
    c = Counter(o for r in rows for o in r.get("obs") or [])
    for k, v in c.most_common():
        print("  %-14s %d" % (k, v))


if __name__ == "__main__":
    main()
