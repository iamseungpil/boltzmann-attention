#!/usr/bin/env python3
"""C47 - D-prime: 4지선다 + 갈래별 결정론 검증기 + 재발화. gold 라벨 재계수.

검증기 (전부 decidable · DB 내용 주입 0):
  FIND : 내놓은 값이 {사용자 발화 U 도구 출력}에 문자열로 실재해야 함
  GET  : 지목한 도구가 그 인자를 산출하는 read 도구여야 함
  ASK  : 그 인자를 산출하는 read 도구가 하나도 없어야 함  (있으면 위반)
  INFER: 통과

gold 라벨 (결정 시점 접두 기준):
  NO-WRITE : gold에 그 write 자체가 없음 (과행동 · 4지선다에 정답 없음)
  FIND     : gold 값이 접두 문맥에 실재
  GET      : 접두엔 없으나 read 도구로 얻을 수 있음 (DB에 존재)
  ASK      : 어디에도 없음 = 사용자만 아는 값
"""
import gzip
import json
import re
import sys
import urllib.request
from collections import Counter

sys.path.insert(0, "/home/woori/scratch")
from e11a_isolated_probe import find_violations  # noqa: E402

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
POLICY = open("/home/woori/scratch/tau2-bench/data/tau2/domains/retail/policy.md").read()
DB = json.load(open("/home/woori/scratch/tau2-bench/data/tau2/domains/retail/db.json"))

ITEMS = set()
for p in DB["products"].values():
    ITEMS |= set(str(k) for k in (p.get("variants") or {}))
PAY = set()
for u in DB["users"].values():
    PAY |= set(str(k) for k in (u.get("payment_methods") or {}))
ORDERS = set(str(o) for o in DB["orders"])
USERS, ORDS = DB["users"], DB["orders"]

# A2: 인자 -> 그 값을 산출하는 read 도구 (스키마 도출 · 값은 주지 않음)
PRODUCER = {
    "new_item_ids": ["get_product_details"],
    "item_ids": ["get_order_details", "get_product_details"],
    "payment_method_id": ["get_user_details", "get_order_details"],
    "order_id": ["get_user_details"],
    "address1": ["get_order_details", "get_user_details"],
}

AI = ("You are a customer service agent that helps the user according to the <policy> provided below.\n"
      "In each turn you can either:\n- Send a message to the user.\n- Make a tool call.\n"
      "You cannot do both at the same time.\n\n"
      "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.")

CHOICES = ('You must choose exactly ONE source for this argument and reply with JSON only:\n'
           '{"choice":"GET","tool":"<tool to call to obtain it>"}\n'
           '{"choice":"FIND","value":"<exact value copied from the conversation or a tool output above>"}\n'
           '{"choice":"INFER","criterion":"<how to derive it from values above>"}\n'
           '{"choice":"ASK","question":"<what to ask the user>"}')


def norm(x):
    return re.sub(r"\s+", " ", str(x).lower().replace("#", "")).strip()


def prefix_txt(sim, idx):
    return norm(" ".join(m.get("content") for m in sim["messages"][:idx]
                         if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str)))


def obtainable(key, gv):
    gv = str(gv)
    if key in ("new_item_ids", "item_ids"):
        return gv in ITEMS
    if key == "payment_method_id":
        return gv in PAY
    if key == "order_id":
        return gv in ORDERS
    if key == "address1":
        return (any((u.get("address") or {}).get("address1") == gv for u in USERS.values())
                or any((o.get("address") or {}).get("address1") == gv for o in ORDS.values()))
    return False


def gold_label(sim, tc, key, idx):
    wname = tc.get("name")
    oid = str((tc.get("arguments") or {}).get("order_id") or "")
    acts = [(a.get("action") or {}) for a in (sim.get("reward_info") or {}).get("action_checks") or []]
    cand = [a for a in acts if a.get("name") == wname]
    if not cand:
        return "NO-WRITE", None
    a = cand[0]
    for c in cand:
        if str((c.get("arguments") or {}).get("order_id") or "") == oid:
            a = c
            break
    v = (a.get("arguments") or {}).get(key)
    gv = v[0] if isinstance(v, list) and v else v
    if gv is None:
        return "NO-ARG", None
    if norm(gv) in prefix_txt(sim, idx):
        return "FIND", gv
    if obtainable(key, gv):
        return "GET", gv
    return "ASK", gv


def build(sim, idx, key, tcname, feedback=None):
    msgs = [{"role": "system",
             "content": "<instructions>\n" + AI + "\n</instructions>\n<policy>\n" + POLICY + "\n</policy>"}]
    for m in sim["messages"][:idx]:
        r = m.get("role")
        if r == "user":
            msgs.append({"role": "user", "content": m.get("content") or ""})
        elif r == "assistant":
            am = {"role": "assistant", "content": m.get("content") or ""}
            tcs = m.get("tool_calls") or []
            if tcs:
                am["tool_calls"] = [{"id": t["id"], "type": "function",
                                     "function": {"name": t["name"],
                                                  "arguments": json.dumps(t.get("arguments") or {})}} for t in tcs]
            msgs.append(am)
        elif r == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("id") or m.get("tool_call_id"),
                         "content": str(m.get("content"))})
    msgs.append({"role": "user",
                 "content": "You are about to call `" + tcname + "`. You need the argument `" + key + "`.\n" + CHOICES})
    if feedback:
        msgs.append({"role": "assistant", "content": feedback[0]})
        msgs.append({"role": "user", "content": feedback[1]})
    return msgs


def chat(msgs):
    p = {"model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", "messages": msgs,
         "temperature": 0.0, "max_tokens": 200}
    req = urllib.request.Request("http://localhost:8140/v1/chat/completions",
                                 data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"].get("content") or ""


def parse(t):
    m = re.search(r"\{.*?\}", t or "", re.S)
    if not m:
        return None, {}
    try:
        d = json.loads(m.group(0))
        return d.get("choice"), d
    except Exception:
        return None, {}


def verify(ch, d, sim, idx, key):
    prods = PRODUCER.get(key, [])
    joined = ", ".join(prods) if prods else "none"
    if ch == "ASK":
        if prods:
            return False, ("[VERIFIER] A tool can produce `" + key + "`: " + joined +
                           ". Do not ask the user for a value the system can look up. Choose again (JSON only).")
        return True, None
    if ch == "FIND":
        v = str(d.get("value", ""))
        if norm(v) and norm(v) in prefix_txt(sim, idx):
            return True, None
        return False, ("[VERIFIER] The value '" + v + "' does not appear in the conversation or in any tool "
                       "output above. You cannot FIND it. Choose again (JSON only).")
    if ch == "GET":
        if d.get("tool") in prods:
            return True, None
        return False, ("[VERIFIER] `" + str(d.get("tool")) + "` does not produce `" + key + "`. Tools that do: "
                       + joined + ". Choose again (JSON only).")
    if ch == "INFER":
        return True, None
    return False, "[VERIFIER] Reply with one of the four JSON forms only."


def main():
    sims = json.load(gzip.open(SIM + "fl32b_floor_retail_t4.results.json.gz"))["simulations"]
    V = find_violations(sims, 30)
    rows = []
    for i, (sim, idx, tc, key, val, want) in enumerate(V):
        gold, gv = gold_label(sim, tc, key, idx)
        fb = None
        ch, d, tries = None, {}, 0
        for attempt in range(3):
            try:
                txt = chat(build(sim, idx, key, tc.get("name"), fb))
            except Exception:
                txt = ""
            ch, d = parse(txt)
            tries = attempt + 1
            ok, msg = verify(ch, d, sim, idx, key)
            if ok:
                break
            fb = (txt.strip()[:200], msg)
        fab = (ch == "FIND" and norm(str(d.get("value", ""))) not in prefix_txt(sim, idx))
        rows.append({"task": str(sim.get("task_id")), "trial": sim.get("trial"), "arg": key,
                     "gold": gold, "final": ch, "tries": tries, "fab": fab,
                     "payload": str(d.get("tool") or d.get("value") or d.get("question") or "")[:60]})
        if (i + 1) % 5 == 0:
            print("  .." + str(i + 1) + "/" + str(len(V)), flush=True)

    print("\n=== gold 라벨 재계수 ===")
    print(" ", dict(Counter(r["gold"] for r in rows)))
    print("\n=== D-prime 교차표 (gold x 최종선택) ===")
    cm = Counter((r["gold"], r["final"]) for r in rows)
    chs = ["GET", "FIND", "INFER", "ASK", None]
    print("gold/Dp   " + "".join("%8s" % str(c) for c in chs))
    for g in sorted(set(r["gold"] for r in rows)):
        print("%-10s" % g + "".join("%8d" % cm[(g, c)] for c in chs))
    val = [r for r in rows if r["gold"] in ("GET", "FIND", "ASK")]
    acc = sum(1 for r in val if r["final"] == r["gold"]) / max(len(val), 1)
    print("\n판정가능 %d건 · 정확도 %.2f · 날조(FIND 거짓) %d" % (len(val), acc, sum(r["fab"] for r in rows)))
    print("재발화 횟수 분포: %s" % dict(Counter(r["tries"] for r in rows)))
    print("\n=== gold=GET 인데 다른 걸 고른 건 ===")
    for r in rows:
        if r["gold"] == "GET" and r["final"] != "GET":
            print("  t%s tr%s %-18s -> %-6s tries=%d %r" % (r["task"], r["trial"], r["arg"], str(r["final"]), r["tries"], r["payload"]))
    print("\n=== NO-WRITE (그 write 자체가 gold에 없음) ===")
    for r in rows:
        if r["gold"] in ("NO-WRITE", "NO-ARG"):
            print("  t%s tr%s %-18s gold=%s final=%s" % (r["task"], r["trial"], r["arg"], r["gold"], str(r["final"])))


if __name__ == "__main__":
    main()
