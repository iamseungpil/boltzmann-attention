#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eplan_iso_probe.py — E-PLAN 격리검증 ④/⑤ (`E_PLAN_LIVE_WIRING_DESIGN` §5(d)·무료).

④ (--mode predicate·오프라인): 실 궤적(gz)에서 PlanLedger를 재구성해 L1/L2 술어의
   발화/침묵을 trial별로 판정 — t95 ⓐ형(미검토 sibling·발화 기대) vs ⓑ형(전부 검토·침묵 기대).
⑤ (--mode replan·32B 서버 필요): 실패 trial의 전 대화를 stop-time plan-추출 프롬프트로
   격리 서브콜 → 산출 write 집합이 gold(intent×entity)를 커버하는지 = C14 정보-맞춤 재검증.

usage:
  py -3 eplan_iso_probe.py --mode predicate --results <gz> --task 95
  (remote) --mode replan --results <gz> --task 95 --gold "exchange:#W2905754,exchange:#W4073673" \
           --base http://localhost:8141/v1 --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
"""
import argparse, gzip, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_eplan_patch import PlanLedger, load_eplan_spec, discovery_L1, discovery_L2  # noqa: E402

WR = ("modify", "exchange", "return", "cancel")


def load_sims(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)["simulations"]


def args_of(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def replay_ledger(sim, spec):
    """궤적 → (ledger, write_events). write_events=[(ledger 스냅샷 판정용 시점, tool, ok)]."""
    led = PlanLedger(spec)
    res_by_id = {m.get("id"): m for m in (sim.get("messages") or []) if m.get("role") == "tool"}
    fire_log = []
    for m in sim.get("messages") or []:
        role, c = m.get("role"), m.get("content")
        if role == "user" and isinstance(c, str):
            led.accumulate_qty(c)
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                nm = tc.get("name") or ""
                ar = args_of(tc.get("arguments"))
                tm = res_by_id.get(tc.get("id"))
                ok = tm is not None and not tm.get("error")
                if any(w in nm for w in WR):
                    # write 시도 시점의 L1/L2 판정 (deny 시뮬레이션·개입은 안 함=관측만)
                    l1 = discovery_L1(led)
                    l2 = discovery_L2(led, intent_class=nm)
                    fire_log.append({"tool": nm, "ok": ok, "L1": l1, "L2": list(l2),
                                     "qty": led.qty_mentioned,
                                     "listed": len(led.listed), "examined": len(led.examined)})
                    if ok:
                        led.note_write(nm, ar.get(spec.get("entity_key")), ar.get("item_ids") or ())
                else:
                    out = None
                    if tm is not None and isinstance(tm.get("content"), str) and not tm.get("error"):
                        out = tm["content"]
                    led.note_read(nm, ar, out)
    return led, fire_log


def mode_predicate(sims, task_id, spec):
    print(f"== ④ L2 술어 특이도 (task {task_id}·설계 §5(d)) ==")
    for s in sorted([x for x in sims if str(x.get("task_id")) == str(task_id)],
                    key=lambda x: x.get("trial") or 0):
        led, log = replay_ledger(s, spec)
        r = (s.get("reward_info") or {}).get("reward")
        print(f"\n-- trial {s.get('trial')} (reward={r}) qty_final={led.qty_mentioned} "
              f"listed={sorted(led.listed)} examined={sorted(led.examined)}")
        for ev in log:
            fired = "L1" if ev["L1"] else ("L2->%s" % ",".join(ev["L2"]) if ev["L2"] else "-")
            print(f"   write {ev['tool']}({'ok' if ev['ok'] else 'ERR'}) "
                  f"qty={ev['qty']} listed={ev['listed']} examined={ev['examined']} | fire: {fired}")
        # 종결 시점 L2 (walk 직전 관점)
        l2_end = discovery_L2(led, intent_class=None)
        print(f"   [end] L2(any-class): {l2_end if l2_end else '침묵'}")


PLAN_PROMPT = (
    "You are auditing a finished customer-service conversation. Read the FULL transcript below. "
    "List every WRITE action (modify/exchange/return/cancel) the customer asked for in this conversation "
    "— including ones the agent failed to perform. Output ONLY a JSON array; each element: "
    '{"action": "<tool name>", "order_id": "<#W...>", "items": ["..."]}. '
    "Use the order ids as they appear in the transcript. Do not invent ids.\n\n--- TRANSCRIPT ---\n"
)


def mode_replan(sims, task_id, gold_pairs, base, model, max_chars=24000):
    import urllib.request
    print(f"== ⑤ stop-time 재-plan (task {task_id}·C14 재검증) gold={gold_pairs} ==")
    hits = 0, 0
    tot = 0
    full = 0
    for s in sorted([x for x in sims if str(x.get("task_id")) == str(task_id)],
                    key=lambda x: x.get("trial") or 0):
        lines = []
        for m in s.get("messages") or []:
            role, c = m.get("role"), m.get("content")
            if isinstance(c, str) and c.strip():
                lines.append(f"[{role}] {c.strip()}")
            for tc in (m.get("tool_calls") or []) if role == "assistant" else []:
                lines.append(f"[assistant->tool] {tc.get('name')} {json.dumps(args_of(tc.get('arguments')), ensure_ascii=False)}")
        txt = "\n".join(lines)[-max_chars:]
        body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 600,
                           "messages": [{"role": "user", "content": PLAN_PROMPT + txt}]}).encode()
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            out = json.load(urllib.request.urlopen(req, timeout=180))
            ans = out["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"-- trial {s.get('trial')}: SUBCALL_ERR {type(e).__name__}")
            continue
        try:
            arr = json.loads(ans[ans.index("["):ans.rindex("]") + 1])
        except Exception:
            print(f"-- trial {s.get('trial')}: PARSE_FAIL {ans[:120]!r}")
            continue
        got = {(str(w.get("action", "")).strip(), str(w.get("order_id", "")).strip()) for w in arr}
        want = set(gold_pairs)
        cov = want & {(a, o) for (a, o) in got}
        tot += 1
        full += (cov == want)
        print(f"-- trial {s.get('trial')}: replan={sorted(got)} | gold-cover {len(cov)}/{len(want)}"
              f" {'FULL' if cov == want else ''}")
    if tot:
        print(f"\n⑤ 판정: full-cover {full}/{tot} (C14 기대=높음. 낮으면 CP5 주축 재고)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["predicate", "replan"], required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--gold", default="", help="replan용: 'action:order,action:order'")
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    sims = load_sims(a.results)
    spec = load_eplan_spec(a.domain) or {}
    if a.mode == "predicate":
        mode_predicate(sims, a.task, spec)
    else:
        gold = [tuple(x.split(":", 1)) for x in a.gold.split(",") if ":" in x]
        mode_replan(sims, a.task, gold, a.base, a.model)


if __name__ == "__main__":
    main()
