#!/usr/bin/env python
"""native-FC TaskBench plan 스코어러 (M3.6 LODO format-등방화 검증, 2026-06-15).

가설(M3.6): native-FC가 daily(named-arg)·HF/MM(positional)을 모두 named-dict로 표준화 = format 등방화
→ gold-SFT의 daily LODO 붕괴(-8.5)를 뒤집고 held-out daily 전이가 in-domain 근접하면 step-2 충분.

방법: 각 daily native-FC 예제를 multi-turn replay(gold tool-result 주입)하며 모델의 tool_call 수집
→ 예측 node집합(도구명)·edge집합(arg threading res_ref) vs gold → node-F1·edge-F1 micro.

Usage: tbnfc_score.py --eval daily_nfc.jsonl --base URL --model NAME [--n 300]
"""
import argparse, json, re
import requests

RES = re.compile(r"res_[0-9a-f]+|res_\d+|call_\d+")


def gold_plan(ex):
    """gold messages서 node(도구명)·edge(threading) 추출."""
    nodes, edges, callid2name = [], set(), {}
    for m in ex.get("messages", []):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                nm = fn.get("name")
                cid = tc.get("id")
                nodes.append(nm)
                callid2name[cid] = nm
                args = fn.get("arguments")
                # threading: arg값에 이전 call의 출력 ref(res_/call_) → edge
                for ref in RES.findall(str(args)):
                    edges.add((ref, nm))
    return nodes, edges


def ask_plan(base, model, ex, max_turns=12):
    """replay: system+user → tool_call → gold tool-result 주입 → 반복. 예측 node/edge 수집."""
    msgs = []
    gold_msgs = ex.get("messages", [])
    tools = ex.get("tools")
    # system + 첫 user 까지
    i = 0
    while i < len(gold_msgs) and gold_msgs[i].get("role") in ("system", "user"):
        msgs.append({k: gold_msgs[i][k] for k in ("role", "content") if k in gold_msgs[i]})
        i += 1
    # gold의 tool 결과를 순서대로 주입할 큐
    gold_tool_results = [m for m in gold_msgs if m.get("role") == "tool"]
    pred_nodes, pred_edges = [], set()
    gi = 0
    for _ in range(max_turns):
        try:
            r = requests.post(base + "/chat/completions", json={
                "model": model, "messages": msgs, "tools": tools,
                "tool_choice": "auto", "temperature": 0.0, "max_tokens": 256}, timeout=60)
            choice = r.json()["choices"][0]["message"]
        except Exception:
            break
        tcs = choice.get("tool_calls")
        if not tcs:
            break
        msgs.append(choice)
        for tc in tcs:
            fn = tc.get("function", tc)
            nm = fn.get("name")
            pred_nodes.append(nm)
            for ref in RES.findall(str(fn.get("arguments"))):
                pred_edges.add((ref, nm))
        # gold tool 결과 주입 (없으면 합성 ok)
        for tc in tcs:
            if gi < len(gold_tool_results):
                tr = gold_tool_results[gi]; gi += 1
                content = tr.get("content", "ok")
            else:
                content = "ok"
            msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": str(content)})
    return pred_nodes, pred_edges


def f1(pred, gold):
    ps, gs = set(pred), set(gold)
    if not ps and not gs:
        return 1.0, 1.0, 1.0
    tp = len(ps & gs)
    p = tp / len(ps) if ps else 0.0
    r = tp / len(gs) if gs else 0.0
    return (2 * p * r / (p + r) if (p + r) else 0.0), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8")][:a.n]

    nf, ef = [], []
    nP = nR = 0
    for k, ex in enumerate(rows):
        g_nodes, g_edges = gold_plan(ex)
        if not g_nodes:
            continue
        p_nodes, p_edges = ask_plan(base, a.model, ex)
        nfv, np_, nr_ = f1(p_nodes, g_nodes)
        efv, _, _ = f1(p_edges, g_edges)
        nf.append(nfv); ef.append(efv); nP += np_; nR += nr_
        if (k + 1) % 50 == 0:
            print("  [%d] node-F1=%.3f edge-F1=%.3f" % (k + 1, sum(nf) / len(nf), sum(ef) / len(ef)), flush=True)
    print("\n=== M3.6 native-FC daily LODO 전이 ===")
    print("n=%d  node-F1=%.4f  edge-F1=%.4f" % (len(nf), sum(nf) / max(1, len(nf)), sum(ef) / max(1, len(ef))))


if __name__ == "__main__":
    main()
