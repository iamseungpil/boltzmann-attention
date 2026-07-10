#!/usr/bin/env python3
"""C51 - DISAMBIGUATE 경계 대규모 프로브: 후보 2+개 지점에서 옳은 값을 고르는가.

물음: 날조를 0으로 닫은 뒤 남는 잔여(retail write 인자의 45.8% = 후보 2+개·C49 §3.4)가
      (i) 진짜 F3 ⋈ 경계인가 (어떤 arm으로도 안 열림) 아니면 (ii) 후보 열거로 열리는가.
C46은 "안 닫힘"이라 했으나 n=3. 여기서 대규모(target ~600·전수 근접)로 확정.

각 결정점(원 궤적서 후보 2+개인 write 인자) x 3 arm, gold와 대조:
  A full     : 원 궤적 접두 전체 + "이 인자 값을 골라라"
  B enumerate: 접두 + [후보 명시 열거] + "사용자 요구에 맞는 것을 골라라"  (DISAMBIGUATE 처방)
  C short    : 정보-맞춤 짧은 컨텍스트(사용자 요구 요약 + 후보 목록) + 선택

★크래시 안전: 매 결정점 결과를 jsonl append. 재실행 시 이어서.
Run: python3 c51_disambig_boundary.py --n 600 --arm A,B,C
"""
import argparse
import gzip
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, "/home/woori/scratch")
import c47_dprime as D  # noqa: E402

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
OUT = "/home/woori/scratch/c51_results.jsonl"
POLICY = open("/home/woori/scratch/tau2-bench/data/tau2/domains/retail/policy.md").read()
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"

WRITE = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
         "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
         "modify_user_address", "place_order"}
ARGS = ("new_item_ids", "payment_method_id", "address1", "item_ids", "order_id")

CAND_PAT = {
    "payment_method_id": r"(?:credit_card|gift_card|paypal)_\d+",
    "address1": r"\d+ [A-Z][a-z]+ (?:Street|Avenue|Drive|Lane|Road|Boulevard|Way|Court)",
    "new_item_ids": r"\b\d{10}\b",
    "item_ids": r"\b\d{10}\b",
    "order_id": r"#W\d{7}",
}

AI = ("You are a customer service agent. Choose the correct value for a tool argument. "
      "Reply with JSON only: {\"value\": \"<exact value>\"}")


def norm(x):
    return D.norm(x)


def candidates(sim, idx, key):
    tool_txt = " ".join(str(m.get("content")) for m in sim["messages"][:idx] if m.get("role") == "tool")
    pat = CAND_PAT.get(key)
    if not pat:
        return []
    seen, out = set(), []
    for m in re.findall(pat, tool_txt):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def user_request(sim, idx):
    return " ".join((m.get("content") or "") for m in sim["messages"][:idx] if m.get("role") == "user")


def collect_points(sims, n):
    """원 궤적서 write 인자이면서 후보 2+개인 결정점 (gold=FIND 즉 값이 문맥에 실재)."""
    pts = []
    for sim in sims:
        if sim.get("termination_reason") != "user_stop":
            continue
        for i, m in enumerate(sim.get("messages", [])):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") not in WRITE:
                    continue
                a = tc.get("arguments") or {}
                for key in ARGS:
                    if key not in a:
                        continue
                    gold, gv = D.gold_label(sim, tc, key, i)
                    if gold != "FIND" or gv is None:
                        continue
                    cands = candidates(sim, i, key)
                    if len(cands) < 2:
                        continue
                    if norm(gv) not in " ".join(norm(c) for c in cands):
                        continue                       # gold가 후보집합에 없으면 스킵(패턴 한계)
                    pts.append((str(sim.get("task_id")), sim.get("trial"), i, key, str(gv), cands, tc.get("name")))
                    break
    # 결정적 순서 (seed 없이) : task,trial,idx 정렬
    pts.sort(key=lambda r: (int(r[0]), r[1], r[2]))
    return pts[:n]


def build(sim, idx, key, tcname, cands, arm):
    if arm == "C":
        req = user_request(sim, idx)[-1200:]
        body = ("Customer request so far:\n" + req + "\n\n"
                "Candidate values for `" + key + "` (from earlier tool output):\n"
                + "\n".join("- " + c for c in cands) + "\n\n"
                "Pick the ONE that matches the customer's request. " + AI)
        return [{"role": "system", "content": AI}, {"role": "user", "content": body}]
    msgs = [{"role": "system", "content": "<policy>\n" + POLICY + "\n</policy>\n" + AI}]
    for m in sim["messages"][:idx]:
        r = m.get("role")
        if r == "user":
            msgs.append({"role": "user", "content": m.get("content") or ""})
        elif r == "assistant":
            am = {"role": "assistant", "content": m.get("content") or ""}
            tcs = m.get("tool_calls") or []
            if tcs:
                am["tool_calls"] = [{"id": t["id"], "type": "function",
                                     "function": {"name": t["name"], "arguments": json.dumps(t.get("arguments") or {})}} for t in tcs]
            msgs.append(am)
        elif r == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("id") or m.get("tool_call_id"), "content": str(m.get("content"))})
    tail = "You are about to call `" + tcname + "`. Choose the value for `" + key + "`."
    if arm == "B":
        tail += "\nCandidate values from earlier tool output:\n" + "\n".join("- " + c for c in cands)
    tail += "\n" + AI
    msgs.append({"role": "user", "content": tail})
    return msgs


def chat(msgs):
    p = {"model": MODEL, "messages": msgs, "temperature": 0.0, "max_tokens": 80}
    req = urllib.request.Request("http://localhost:8140/v1/chat/completions",
                                 data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"].get("content") or ""


def pick(txt):
    m = re.search(r"\{.*?\}", txt or "", re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("value", "")).strip()
        except Exception:
            pass
    return (txt or "").strip()[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--arm", default="A,B,C")
    a = ap.parse_args()
    arms = a.arm.split(",")

    sims = json.load(gzip.open(SIM + "fl32b_floor_retail_t4.results.json.gz"))["simulations"]
    pts = collect_points(sims, a.n)

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["task"], r["trial"], r["idx"], r["arg"]))
            except Exception:
                pass
    print("DISAMBIGUATE 결정점 %d · 이미완료 %d" % (len(pts), len(done)), flush=True)

    fp = open(OUT, "a", encoding="utf-8")
    for j, (t, tr, idx, key, gv, cands, tcname) in enumerate(pts):
        if (t, tr, idx, key) in done:
            continue
        sim = next(s for s in sims if str(s.get("task_id")) == t and s.get("trial") == tr)
        rec = {"task": t, "trial": tr, "idx": idx, "arg": key, "gold": gv, "ncand": len(cands)}
        for arm in arms:
            try:
                v = pick(chat(build(sim, idx, key, tcname, cands, arm)))
            except Exception as e:
                v = "ERR:" + type(e).__name__
            rec[arm] = v
            rec[arm + "_ok"] = int(norm(v) == norm(gv))
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fp.flush()
        if (j + 1) % 20 == 0:
            print("  ..%d/%d" % (j + 1, len(pts)), flush=True)
    fp.close()

    # 집계
    from collections import Counter
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    print("\n=== C51 DISAMBIGUATE 경계 (n=%d) ===" % len(rows))
    for arm in arms:
        oks = [r[arm + "_ok"] for r in rows if arm + "_ok" in r]
        print("  arm %s: 정확도 %.3f (n=%d)" % (arm, sum(oks) / max(len(oks), 1), len(oks)))
    print("\n  인자별 arm B 정확도:")
    byarg = {}
    for r in rows:
        byarg.setdefault(r["arg"], []).append(r.get("B_ok", 0))
    for k, v in byarg.items():
        print("    %-18s %.3f (n=%d)" % (k, sum(v) / max(len(v), 1), len(v)))
    print("\n  후보수별 arm B 정확도:")
    bync = {}
    for r in rows:
        bync.setdefault(r["ncand"], []).append(r.get("B_ok", 0))
    for k in sorted(bync):
        print("    ncand=%d: %.3f (n=%d)" % (k, sum(bync[k]) / max(len(bync[k]), 1), len(bync[k])))


if __name__ == "__main__":
    main()
