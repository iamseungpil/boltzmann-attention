#!/usr/bin/env python3
"""E11-a 격리 프로브 — provenance 피드백이 gather를 유도하는가 (무료·단일턴·user-sim 불필요).

정본 doc: reports/facet_rft_2026/SCAFFOLD_AUDIT_RULE0_2026_07_08.md §6
선행: 7B retail서 deny만으로는 no_gather 36→36 (무변화·C35). 32B서도 무효면 scaffold 경로 종결 → learn 확정.

설계 (짝지은 대조·같은 접두 문맥):
  기존 floor 궤적에서 '날조 operand write' 지점을 찾는다.
  그 지점까지의 메시지를 그대로 주고, **그 write에 대한 도구 응답만** 두 가지로 바꿔 다음 assistant 턴을 1회 생성:
    ARM env  : 환경이 실제로 낸 에러 (예: "Error: Variant not found")        ← 현재 상태
    ARM e11a : REGEN_FEEDBACK ("지어낸 값이다. getter를 불러 실값을 읽어라")   ← T2_PROV_REGEN
  측정: 다음 호출이 **올바른 getter**인가 = no_gather 탈출률.

user-sim이 개입하지 않으므로 [[08]]의 user-sim 편차 오염이 없다. 유료 0.

Run:
  python3 e11a_isolated_probe.py --base http://localhost:8140/v1 \
      --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --arm both --n 50
"""

import argparse
import gzip
import json
import re
import sys
import urllib.request

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
DBP = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/db.json"
POLICY = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/policy.md"

WRITE = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
         "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
         "modify_user_address", "place_order"}
ITEMW = {"exchange_delivered_order_items", "modify_pending_order_items"}
ADDR = {"modify_pending_order_address", "modify_user_address"}
GETTERS = {"get_product_details", "get_order_details", "get_user_details", "get_item_details",
           "find_user_id_by_email", "find_user_id_by_name_zip", "list_all_product_types"}
TOOLS_JSON = "/home/woori/scratch/retail_tools_openai.json"  # tau2 Tool.openai_schema 실물

# tau2 src/tau2/agent/llm_agent.py 와 동일
AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n- Send a message to the user.\n- Make a tool call.\n"
    "You cannot do both at the same time.\n\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."
)

# t2_gate_patch.py:373 REGEN_FEEDBACK 과 동일 문안 (정보 주입 0)
REGEN_FEEDBACK = (
    "Error: [PROVENANCE] argument '{k}'='{v}' was not provided by the user nor returned by any tool "
    "— it looks invented (e.g. a schema example value). Do NOT use placeholder/example values and do NOT "
    "ask the user. Instead call a lookup/getter tool that produces this value (e.g. a getter to "
    "retrieve the user's records, payment methods, or addresses) and read the real value from its output. "
    "Now emit a corrected tool call."
)


def _norm(t):
    return re.sub(r"\s+", " ", str(t).lower().replace("#", "")).strip()


def load_db():
    db = json.load(open(DBP, encoding="utf-8"))
    item2prod = {}
    for pid, p in db["products"].items():
        for v in (p.get("variants") or {}):
            item2prod[str(v)] = str(pid)
    pay = set()
    for u in db["users"].values():
        pay |= {str(k) for k in (u.get("payment_methods") or {})}
    return item2prod, {str(o) for o in db["orders"]}, pay


ITEM2PROD, ORDERS, PAY = load_db()


def find_violations(sims, limit=None):
    """(sim, msg_idx, tool_call, 위반인자, 위반값, 기대 getter) 목록."""
    out = []
    for s in sims:
        if s.get("termination_reason") != "user_stop":
            continue
        prov, fetched = [], set()
        for i, m in enumerate(s.get("messages", [])):
            role = m.get("role")
            if role in ("tool", "user"):
                c = m.get("content")
                if isinstance(c, str):
                    prov.append(_norm(c))
                continue
            if role != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                nm, a = tc.get("name"), tc.get("arguments") or {}
                if nm == "get_product_details":
                    fetched.add(str(a.get("product_id")))
                    continue
                if nm not in WRITE:
                    continue
                blob = " ".join(prov)
                hit = None
                if nm in ITEMW:
                    olds = [str(x) for x in (a.get("item_ids") or [])]
                    for j, nv in enumerate([str(x) for x in (a.get("new_item_ids") or [])]):
                        if _norm(nv) in blob:
                            continue
                        pid = ITEM2PROD.get(olds[j]) if j < len(olds) else None
                        if pid and pid not in fetched:
                            hit = ("new_item_ids", nv, "get_product_details")
                            break
                if hit is None and nm in ADDR:
                    a1 = (a.get("address1") or "").strip()
                    if a1 and _norm(a1) not in blob:
                        hit = ("address1", a1, "get_order_details|get_user_details")
                if hit is None:
                    pm = a.get("payment_method_id")
                    if pm and _norm(pm) not in blob:
                        hit = ("payment_method_id", str(pm), "get_user_details")
                if hit:
                    out.append((s, i, tc, hit[0], hit[1], hit[2]))
                    break
            else:
                continue
            break
    return out[:limit] if limit else out


def build_prefix(sim, upto_idx):
    """upto_idx 이전의 공식 대화 + 그 assistant 턴(날조 호출)까지. system=tau2 SYSTEM_PROMPT 동형."""
    msgs = []
    sysmsg = (f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n"
              f"<policy>\n{open(POLICY, encoding='utf-8').read()}\n</policy>")
    msgs.append({"role": "system", "content": sysmsg})
    for m in sim["messages"][:upto_idx + 1]:
        r = m.get("role")
        if r == "user":
            msgs.append({"role": "user", "content": m.get("content") or ""})
        elif r == "assistant":
            am = {"role": "assistant", "content": m.get("content") or ""}
            tcs = m.get("tool_calls") or []
            if tcs:
                am["tool_calls"] = [{"id": t["id"], "type": "function",
                                     "function": {"name": t["name"],
                                                  "arguments": json.dumps(t.get("arguments") or {})}}
                                    for t in tcs]
            msgs.append(am)
        elif r == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("id") or m.get("tool_call_id"),
                         "content": str(m.get("content"))})
    return msgs


def env_error_for(sim, tc):
    """그 write에 환경이 실제로 낸 응답(없으면 generic)."""
    for m in sim["messages"]:
        if m.get("role") == "tool" and (m.get("id") or m.get("tool_call_id")) == tc.get("id"):
            return str(m.get("content"))
    return "Error: invalid argument"


def chat(base, model, messages, tools, timeout=180):
    body = json.dumps({"model": model, "messages": messages, "tools": tools,
                       "tool_choice": "auto", "temperature": 0.0, "max_tokens": 256}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]


def tools_schema():
    """tau2 실물 도구 스키마 (Tool.openai_schema 16개). 최소본 대체 금지 — 충실도 문제."""
    return json.load(open(TOOLS_JSON, encoding="utf-8"))


def orig_next(sim, idx):
    """원 궤적에서 그 도구 응답 뒤 에이전트가 실제로 한 것 (프로브 충실도 대조군)."""
    for m in sim["messages"][idx + 1:]:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            return "TEXT"
        nm = tcs[0]["name"]
        return ("GETTER:" + nm) if nm in GETTERS else ("WRITE:" + nm)
    return "END"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--arm", choices=["env", "e11a", "both"], default="both")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tag", default="fl32b_floor_retail_t4")
    args = ap.parse_args()

    sims = json.load(gzip.open(SIM + args.tag + ".results.json.gz"))["simulations"]
    viol = find_violations(sims, args.n)
    print(f"위반 지점 {len(viol)}개 (arm={args.arm})", flush=True)
    tools = tools_schema()

    stats = {"env": {"getter": 0, "write": 0, "text": 0, "err": 0},
             "e11a": {"getter": 0, "write": 0, "text": 0, "err": 0}}
    rows = []
    for k, (sim, idx, tc, key, val, want) in enumerate(viol):
        prefix = build_prefix(sim, idx)
        tcid = tc.get("id")
        arms = ["env", "e11a"] if args.arm == "both" else [args.arm]
        res = {}
        for arm in arms:
            content = (env_error_for(sim, tc) if arm == "env"
                       else REGEN_FEEDBACK.format(k=key, v=val))
            msgs = prefix + [{"role": "tool", "tool_call_id": tcid, "content": content}]
            try:
                out = chat(args.base, args.model, msgs, tools)
            except Exception as e:
                stats[arm]["err"] += 1
                res[arm] = f"ERR:{type(e).__name__}"
                continue
            tcs = out.get("tool_calls") or []
            if not tcs:
                stats[arm]["text"] += 1
                res[arm] = "TEXT"
            else:
                nm = tcs[0]["function"]["name"]
                if nm in GETTERS:
                    stats[arm]["getter"] += 1
                    res[arm] = f"GETTER:{nm}"
                else:
                    stats[arm]["write"] += 1
                    res[arm] = f"WRITE:{nm}"
        rows.append((str(sim.get("task_id")), sim.get("trial"), key, val[:24],
                     orig_next(sim, idx), res.get("env", "-"), res.get("e11a", "-")))
        if (k + 1) % 5 == 0:
            print(f"  ..{k+1}/{len(viol)}", flush=True)

    print(f"\n{'task':>6}{'tr':>3}  {'인자':<18}{'날조값':<26}{'ORIG(원궤적)':<26}{'ARM env':<26}{'ARM e11a':<26}")
    for r in rows:
        print(f"{r[0]:>6}{r[1]:>3}  {r[2]:<18}{r[3]:<26}{r[4]:<26}{r[5]:<26}{r[6]:<26}")

    # ★충실도 검정: ARM env 가 ORIG 를 재현하는가 (재현 못 하면 Δ 해석 금지)
    if args.arm == "both":
        def cls(x):
            return x.split(":")[0]
        agree = sum(1 for r in rows if cls(r[4]) == cls(r[5]))
        print(f"\n=== 충실도: ARM env ≡ ORIG 일치 {agree}/{len(rows)} ({100*agree/max(len(rows),1):.0f}%) ===")

    print("\n=== 집계: 다음 호출이 무엇인가 ===")
    from collections import Counter
    print(f"  ORIG  {dict(Counter(r[4].split(':')[0] for r in rows))}")
    for arm in (["env", "e11a"] if args.arm == "both" else [args.arm]):
        s = stats[arm]
        n = sum(s.values()) or 1
        print(f"  {arm:<5} getter {s['getter']:>3} ({100*s['getter']/n:.0f}%) · "
              f"write(재시도) {s['write']:>3} · 텍스트 {s['text']:>3} · 에러 {s['err']:>3}")
    if args.arm == "both":
        d = stats["e11a"]["getter"] - stats["env"]["getter"]
        print(f"\n★Δgetter = {d:+d}  (7B 선행: deny는 no_gather 36→36 = Δ0)")
        flips = [r for r in rows if r[5].startswith(("WRITE", "TEXT")) and r[6].startswith("GETTER")]
        back = [r for r in rows if r[5].startswith("GETTER") and r[6].startswith(("WRITE", "TEXT"))]
        print(f"  e11a가 살린 케이스 {len(flips)}: {[(r[0],r[2]) for r in flips]}")
        print(f"  되레 망친 케이스 {len(back)}: {[(r[0],r[2]) for r in back]}")
        # 올바른 getter인가 (기대 getter와 대조)
        right = sum(1 for r in rows if r[6].startswith("GETTER") and r[6].split(":")[1] in
                    [x for x in ("get_product_details", "get_order_details", "get_user_details", "get_item_details")])
        print(f"  e11a getter 중 조회계열 {right}")


if __name__ == "__main__":
    sys.exit(main())
