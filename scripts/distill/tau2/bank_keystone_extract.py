# -*- coding: utf-8 -*-
"""⋈ 케이스 컴팩트 추출 (2026-07-14·formalize half 실측용·리모트 32B e2e 입력).
로컬 궤적(C:/tmp/traj·1.8G·커밋금지)서 reference-filter formalize→filter e2e에 필요한 최소만 추출:
케이스별 {user 발화·수집 record·gold_id·gold_record·chosen}. → gz(작음·커밋)→리모트서 formalize.
모집단·추출로직 = bank_keystone_replay와 동일(비교가능)."""
import json, glob, gzip
import bank_filter_repro as B

OUT = "../../../reports/facet_rft_2026/sim_results/bank_xmatch_cases.jsonl.gz"
RFIELDS = ["transaction_id", "date", "amount", "type", "description"]


def user_texts_before(msgs, gt, chosen):
    """★라이브 formalize_reference_criteria 재현: dispute tool-call *시점까지*의 user 발화 last-8.
    (대화 전체 last-8이 아님 — 다중-dispute서 '지금 어느 거래' 앵커 보존.)"""
    cut = len(msgs)
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            if tc.get("name") == "call_discoverable_agent_tool":
                ar = B.nd(tc.get("arguments"))
                if str(ar.get("agent_tool_name")) == str(gt) \
                        and str(B.nd(ar.get("arguments")).get("transaction_id") or "") == str(chosen):
                    cut = i; break
        if cut != len(msgs):
            break
    out = [m["content"] for m in msgs[:cut]
           if m.get("role") == "user" and isinstance(m.get("content"), str)]
    return out[-8:] if out else \
        [m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)][-8:]


def main():
    per = {}; data = {}
    for f in glob.glob("C:/tmp/traj/*_banking.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        data[f] = d
        for s in d["simulations"]:
            r = (s.get("reward_info") or {}).get("reward")
            if r is None:
                continue
            t = str(s["task_id"]); per.setdefault(t, [0, 0]); per[t][1] += 1
            if r == 1.0:
                per[t][0] += 1
    hard = {t for t, p in per.items() if p[1] >= 10 and p[0] / p[1] <= 0.10}

    rows = []
    for f, d in data.items():
        model = f.split("\\")[-1].split("/")[-1].replace("_banking.json", "")
        for s in d["simulations"]:
            if str(s["task_id"]) not in hard:
                continue
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            msgs = s.get("messages") or []
            recs = B.gathered_records(msgs)
            if len(recs) < 2:
                continue
            byid = {r.get("transaction_id"): r for r in recs}
            cl = [(tc.get("name"), B.nd(tc.get("arguments")))
                  for m in msgs for tc in (m.get("tool_calls") or [])]
            for ac in (ri.get("action_checks") or []):
                if ac.get("action_match"):
                    continue
                a = ac.get("action") or {}
                if a.get("name") != "call_discoverable_agent_tool":
                    continue
                g = B.nd(a.get("arguments")); gt = g.get("agent_tool_name"); gn = B.nd(g.get("arguments"))
                same = [B.nd(ar.get("arguments")) for nm, ar in cl
                        if nm == "call_discoverable_agent_tool" and str(ar.get("agent_tool_name")) == str(gt)]
                if not same:
                    continue
                an = same[0]
                gid = str(gn.get("transaction_id") or ""); chosen = str(an.get("transaction_id") or "")
                if not gid or gid == chosen:
                    continue
                grec = byid.get(gid)
                if not grec:
                    continue
                rows.append({
                    "model": model, "tid": str(s["task_id"]),
                    "gold_id": gid, "chosen_id": chosen,
                    "gold": {k: grec.get(k) for k in RFIELDS},
                    "records": [{k: r.get(k) for k in RFIELDS} for r in recs],
                    "users": [u[:1200] for u in user_texts_before(msgs, gt, chosen)],
                })
    with gzip.open(OUT, "wt", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d ⋈ cases → %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
