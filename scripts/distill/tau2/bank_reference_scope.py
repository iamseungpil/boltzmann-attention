# -*- coding: utf-8 -*-
"""참조/reach 축 재설계 STEP 0 (2026-07-13·[[08]]): banking hard-core 참조-ID 실패 3분.
날조(provenance·값이 수집record에 없음) / ⋈오선택(gold∈수집·agent 다른 in-record값) / reach(gold∉수집).
= retail operand-해소 동형. 닫히는 슬라이스(prov+reach) vs 경계(⋈) 정량 → 레버 설계 근거."""
import json, glob, re
from collections import Counter, defaultdict

REF_PARAMS = ["transaction_id", "account_id", "card_id", "card_last_4_digits"]


def nd(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            r = json.loads(v)
            return r if isinstance(r, dict) else {}
        except Exception:
            return {}
    return {}


def gathered_text(msgs):
    """성공 tool-result 내용 전부(수집된 사실) — 값 실재 판정용."""
    out = []
    for m in msgs:
        if m.get("role") == "tool":
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
    return "\n".join(out)


def main():
    per = defaultdict(lambda: [0, 0]); data = {}
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
            t = str(s["task_id"]); per[t][1] += 1
            if r == 1.0:
                per[t][0] += 1
    hard = {t for t, p in per.items() if p[1] >= 10 and p[0] / p[1] <= 0.10}

    cls = Counter(); byparam = defaultdict(Counter)
    for f, d in data.items():
        for s in d["simulations"]:
            if str(s["task_id"]) not in hard:
                continue
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            msgs = s.get("messages") or []
            gtxt = gathered_text(msgs)
            cl = [(tc.get("name"), nd(tc.get("arguments")))
                  for m in msgs for tc in (m.get("tool_calls") or [])]
            for ac in (ri.get("action_checks") or []):
                if ac.get("action_match"):
                    continue
                a = ac.get("action") or {}
                if a.get("name") != "call_discoverable_agent_tool":
                    continue
                g = nd(a.get("arguments")); gt = g.get("agent_tool_name"); gn = nd(g.get("arguments"))
                same = [nd(ar.get("arguments")) for n, ar in cl
                        if n == "call_discoverable_agent_tool" and str(ar.get("agent_tool_name")) == str(gt)]
                if not same:
                    continue
                an = same[0]
                for k in REF_PARAMS:
                    if k not in gn:
                        continue
                    goldv = str(gn.get(k)); agentv = str(an.get(k))
                    if goldv == agentv:
                        continue                              # 이 param은 맞음
                    gold_in = goldv and goldv in gtxt
                    agent_in = agentv and agentv in gtxt and agentv != "None"
                    if not agent_in:
                        c = "날조(provenance·agent값 수집外)"
                    elif gold_in:
                        c = "⋈오선택(gold∈수집·다른 in-record값)"
                    else:
                        c = "reach(gold∉수집)"
                    cls[c] += 1; byparam[k][c] += 1
    print("=== hard-core 참조-ID 실패 3분 (닫히는 슬라이스 판정) ===")
    tot = sum(cls.values())
    for c, k in cls.most_common():
        print("  %-40s %6d (%.1f%%)" % (c, k, 100 * k / max(tot, 1)))
    print("\n=== param별 ===")
    for p in REF_PARAMS:
        if byparam[p]:
            print("  %-20s %s" % (p, dict(byparam[p])))
    prov = cls.get("날조(provenance·agent값 수집外)", 0)
    reach = cls.get("reach(gold∉수집)", 0)
    xj = cls.get("⋈오선택(gold∈수집·다른 in-record값)", 0)
    print("\n★닫히는 슬라이스(prov+reach)=%.1f%% · 경계(⋈)=%.1f%%"
          % (100 * (prov + reach) / max(tot, 1), 100 * xj / max(tot, 1)))


if __name__ == "__main__":
    main()
