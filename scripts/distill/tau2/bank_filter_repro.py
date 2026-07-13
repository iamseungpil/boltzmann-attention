# -*- coding: utf-8 -*-
"""reference-filter 오프라인 재현 게이트 (2026-07-13·[[08]]·R4/R3).
수집 record 파싱 → gold record가 식별기준(date·type·amount·merchant)으로 유일 필터되나.
= 결정론 필터 half의 gold-id 재현 상한(formalize half=LLM·별도). gold-blind(고정 기준·역산 아님)."""
import json, glob, re
from collections import Counter, defaultdict

REC_FIELD = re.compile(r"^\s*(transaction_id|account_id|card_id|date|amount|type|description|status)\s*:\s*(.+?)\s*$")


def parse_records(text):
    """tool-result 텍스트서 record 블록 파싱 → [{field:val}]. 'transaction_id:' 시작마다 새 record."""
    recs = []; cur = None
    for line in text.split("\n"):
        m = REC_FIELD.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k == "transaction_id":
            if cur:
                recs.append(cur)
            cur = {}
        if cur is not None:
            cur[k] = v
    if cur:
        recs.append(cur)
    return recs


def gathered_records(msgs):
    allrec = []
    for m in msgs:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            allrec += parse_records(m["content"])
    # dedup by transaction_id (기록 반복)
    seen = {}
    for r in allrec:
        tid = r.get("transaction_id")
        if tid:
            seen[tid] = r
    return list(seen.values())


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

    crit_unique = Counter(); n = 0; no_goldrec = 0
    for f, d in data.items():
        for s in d["simulations"]:
            if str(s["task_id"]) not in hard:
                continue
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            msgs = s.get("messages") or []
            recs = gathered_records(msgs)
            if len(recs) < 2:
                continue
            byid = {r.get("transaction_id"): r for r in recs}
            cl = [(tc.get("name"), nd(tc.get("arguments")))
                  for m in msgs for tc in (m.get("tool_calls") or [])]
            for ac in (ri.get("action_checks") or []):
                if ac.get("action_match"):
                    continue
                a = ac.get("action") or {}
                if a.get("name") != "call_discoverable_agent_tool":
                    continue
                g = nd(a.get("arguments")); gt = g.get("agent_tool_name"); gn = nd(g.get("arguments"))
                same = [nd(ar.get("arguments")) for nm, ar in cl
                        if nm == "call_discoverable_agent_tool" and str(ar.get("agent_tool_name")) == str(gt)]
                if not same:
                    continue
                an = same[0]
                gid = str(gn.get("transaction_id") or "")
                if not gid or gid == str(an.get("transaction_id") or ""):
                    continue                                   # transaction_id ⋈ 아님
                grec = byid.get(gid)
                if not grec:
                    no_goldrec += 1; continue                  # gold record 파싱 실패/미수집
                n += 1
                # 여러 기준별 유일성: gold record와 같은 값 가진 record 수
                def uniq(fields):
                    c = sum(1 for r in recs if all(r.get(fl) == grec.get(fl) for fl in fields))
                    return c == 1
                if uniq(["date"]):
                    crit_unique["date 단독 유일"] += 1
                elif uniq(["date", "amount"]):
                    crit_unique["date+amount 유일"] += 1
                elif uniq(["date", "type"]):
                    crit_unique["date+type 유일"] += 1
                elif uniq(["amount", "type"]):
                    crit_unique["amount+type 유일"] += 1
                elif uniq(["description"]):
                    crit_unique["description 유일"] += 1
                else:
                    crit_unique["유일식별 실패(진짜 ⋈·동일record)"] += 1
    print("=== transaction_id ⋈ 케이스 필터-유일성 (파싱된 n=%d·gold_rec 미파싱 %d) ===" % (n, no_goldrec))
    tot = sum(crit_unique.values())
    filterable = tot - crit_unique.get("유일식별 실패(진짜 ⋈·동일record)", 0)
    for k, v in crit_unique.most_common():
        print("  %-40s %6d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
    print("\n★결정론-필터 유일식별 가능(=재현 상한): %.1f%%" % (100 * filterable / max(tot, 1)))


if __name__ == "__main__":
    main()
