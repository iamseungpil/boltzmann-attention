#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""plan_probe.py — Phase-0: plan-in-isolation 측정 (gpt-4.1 0·로컬 32B).

질문: orchestration 실패가 *planning 자체*냐(격리서도 plan 틀림=learn 후보)
      *실행 부하*냐(격리선 plan 맞음=C1/C2 결정론 scaffold).
방법: open NL 목표 + reads-done 주문맥락만 주고 **plan-spec(실행 0)** 요청 →
      추상 구조(action·order·item-grouping/batching·include/omit)만 채점.
      concrete operand(어느 변형)는 *안 봄*(이미 GIVEN-SPEC 100%·누설 0).
부하 0(한 샷)·turn-confound 면역. (i)SELECT/(ii)GENERATE 구분은 출력만으론 불가→안 함.

사용: python plan_probe.py --tasks 20,36,37,99,17,92,71 [--agent_base ... --agent_model ...]
"""
import json, urllib.request, argparse, re
from collections import defaultdict

DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"
db = json.load(open(DOM + "/db.json"))
TASKS = {str(t["id"]): t for t in json.load(open(DOM + "/tasks.json"))}
orders, users, prods = db["orders"], db["users"], db["products"]

WRITES = {"modify_pending_order_items", "exchange_delivered_order_items",
          "return_delivered_order_items", "modify_pending_order_address",
          "cancel_pending_order", "modify_pending_order_payment"}
TOOLDOC = """Available WRITE actions (you plan which to call, on which order, with which items):
- modify_pending_order_items(order_id, item_ids[], new_item_ids[])  # ALL item changes for one order MUST be ONE call (order becomes non-pending after first modify)
- exchange_delivered_order_items(order_id, item_ids[], new_item_ids[])  # all exchanges for one order in ONE call
- return_delivered_order_items(order_id, item_ids[])  # all returns for one order in ONE call
- modify_pending_order_address(order_id, ...address)
- cancel_pending_order(order_id)
- modify_pending_order_payment(order_id, payment_method_id)"""


def ask(prompt, model, base, mx=900):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": mx}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"]["content"]


def norm_oid(o):
    """주문 id 정규화 (#W123 / W123 / w123 → W123). 계측기 # 불일치 버그 방지."""
    return str(o).strip().lstrip("#").upper()


def gold_structure(task):
    """gold actions → {(action_name, order_id): [frozenset(item_ids) per call]}. operand(new_item_ids) 무시."""
    g = defaultdict(list)
    for a in (task.get("evaluation_criteria", {}) or {}).get("actions", []):
        n = a.get("name")
        if n not in WRITES:
            continue
        ar = a.get("arguments") or {}
        oid = norm_oid(ar.get("order_id", "?"))
        items = frozenset(ar.get("item_ids") or [])
        g[(n, oid)].append(items)
    return g


def user_orders_context(task):
    """gold가 건드리는 주문들의 user(들)의 전 주문 = reads-done 맥락 (planning grounding·operand 누설 0)."""
    uids = set()
    for a in (task.get("evaluation_criteria", {}) or {}).get("actions", []):
        oid = (a.get("arguments") or {}).get("order_id")
        if oid in orders:
            uids.add(orders[oid].get("user_id"))
    lines = []
    seen = set()
    for uid in uids:
        for oid, o in orders.items():
            if o.get("user_id") != uid or oid in seen:
                continue
            seen.add(oid)
            its = "; ".join(f"{it.get('item_id')}={it.get('name')}{it.get('options')}" for it in o.get("items", []))
            lines.append(f"  {oid} [status={o.get('status')}] items: {its}")
    return "\n".join(lines) if lines else "  (no orders found)"


def parse_plan(txt):
    """모델 출력서 JSON 배열 추출 → [(action, order_id, frozenset(items))]."""
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    out = []
    for e in arr:
        if not isinstance(e, dict):
            continue
        n = str(e.get("action") or e.get("name") or "").strip()
        n = next((w for w in WRITES if w in n), n)  # normalize to canonical write name
        oid = norm_oid(e.get("order_id") or e.get("order") or "?")
        raw_items = e.get("items") or e.get("item_ids") or []
        if isinstance(raw_items, str):
            raw_items = [raw_items]
        items = frozenset(str(x) for x in raw_items)
        out.append((n, oid, items))
    return out


def grade(gold, plan):
    """구조 비교 → (label, issues). missing/extra/batch-split/wrong-action."""
    if plan is None:
        return "PARSE_FAIL", ["plan JSON 파싱 실패"]
    gold_keys = set(gold.keys())
    plan_by_key = defaultdict(list)
    for (n, oid, items) in plan:
        plan_by_key[(n, oid)].append(items)
    plan_keys = set(plan_by_key.keys())
    issues = []
    missing = gold_keys - plan_keys
    extra = plan_keys - gold_keys
    for k in sorted(missing):
        issues.append(f"MISSING {k}")
    for k in sorted(extra):
        issues.append(f"EXTRA {k}")
    # batching: gold가 (action,order)당 1 call인데 plan이 split → BATCH_SPLIT
    for k in gold_keys & plan_keys:
        gcalls, pcalls = gold[k], plan_by_key[k]
        if len(gcalls) == 1 and len(pcalls) > 1:
            issues.append(f"BATCH_SPLIT {k} (gold 1 call / plan {len(pcalls)})")
    label = "STRUCT_OK" if not issues else ("/".join(sorted({i.split()[0] for i in issues})))
    return label, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="20,36,37,99,17,92,71")
    ap.add_argument("--agent_base", default="http://localhost:8360/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    tids = [t.strip() for t in a.tasks.split(",") if t.strip()]
    summary = []
    for tid in tids:
        task = TASKS.get(tid)
        if not task:
            print(f"\n### t{tid}: NO TASK"); continue
        reason = str(task.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", ""))
        ctx = user_orders_context(task)
        gold = gold_structure(task)
        prompt = (
            "You are PLANNING (not executing) a retail customer-service request. "
            "Do NOT call tools or pick specific replacement variants — only lay out the WRITE actions you WOULD take.\n\n"
            f"{TOOLDOC}\n\nCustomer's request:\n{reason[:800]}\n\n"
            f"Customer's orders (already looked up for you):\n{ctx}\n\n"
            "Output ONLY a JSON array. One element per tool call you would make, in order. "
            'Each element: {"action": "<write action name>", "order_id": "<#W...>", "items": ["<item_id>", ...], "why": "<short>"}. '
            "For replacement variants use \"items\" = the item_ids being changed (NOT the new variant). "
            "Group correctly: all item changes for one order go in ONE element. Omit actions that are not possible/needed."
        )
        try:
            out = ask(prompt, a.agent_model, a.agent_base)
        except Exception as e:
            print(f"\n### t{tid}: ASK_ERR {type(e).__name__}: {str(e)[:80]}"); continue
        plan = parse_plan(out)
        label, issues = grade(gold, plan)
        summary.append((tid, label))
        print(f"\n### t{tid}: {label}")
        print(f"  GOLD struct: { {f'{n}|{o}': [sorted(s) for s in v] for (n,o),v in gold.items()} }")
        print(f"  PLAN parsed: {[(n, o, sorted(it)) for (n,o,it) in plan] if plan else 'PARSE_FAIL'}")
        if issues:
            print(f"  ISSUES: {issues}")
        print(f"  RAW: {out.strip()[:600]}")
    print("\n\n=== SUMMARY (plan-in-isolation·구조채점·gpt-4.1 0) ===")
    ok = sum(1 for _, l in summary if l == "STRUCT_OK")
    for tid, l in summary:
        print(f"  t{tid}: {l}")
    print(f"  STRUCT_OK {ok}/{len(summary)}  (OK=planning OK→실행부하 / 그외=planning이 블로커→learn 후보)")


if __name__ == "__main__":
    main()
