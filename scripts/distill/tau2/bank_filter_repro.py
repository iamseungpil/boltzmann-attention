# -*- coding: utf-8 -*-
"""reference-filter 오프라인 재현 게이트 (2026-07-13·[[08]]·R4/R3).
수집 record 파싱 → gold record가 식별기준(date·type·amount·merchant)으로 유일 필터되나.
= 결정론 필터 half의 gold-id 재현 상한(formalize half=LLM·별도). gold-blind(고정 기준·역산 아님)."""
import json, glob, re
from collections import Counter, defaultdict

REC_FIELD = re.compile(r"^\s*(transaction_id|account_id|card_id|date|amount|type|description|status)\s*:\s*(.+?)\s*$")


def parse_records(text):
    """tool-result 텍스트서 record 블록 파싱 → [{field:val}]. 'transaction_id:' 시작마다 새 record.
    ★강건(Step1): 완전 record(transaction_id+date+amount 다 있음)만 반환 = 산발적 id 언급 오염 제거."""
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
    return [r for r in recs if r.get("transaction_id") and r.get("date") and r.get("amount")]


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
                allf = ["date", "amount", "type", "description"]
                if uniq(["date"]) or uniq(["date", "amount"]) or uniq(["date", "type"]) \
                        or uniq(["amount", "type"]) or uniq(["description"]):
                    crit_unique["①결정론필터 유일식별"] += 1
                else:
                    # 실패 → 진짜중복(gold와 全필드 동일 record ≥2) vs ASK가능(어떤 속성 다름)
                    ident = sum(1 for r in recs if all(r.get(fl) == grec.get(fl) for fl in allf))
                    if ident >= 2:
                        crit_unique["③진짜중복(全필드동일·ASK불가·벤치아티팩트?)"] += 1
                    else:
                        crit_unique["②ASK가능(속성상이·DISAMBIGUATE)"] += 1
    print("=== transaction_id ⋈ 케이스 (파싱된 n=%d·gold_rec 미파싱 %d) ===" % (n, no_goldrec))
    tot = sum(crit_unique.values())
    for k, v in sorted(crit_unique.items()):
        print("  %-44s %6d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
    f = crit_unique.get("①결정론필터 유일식별", 0)
    ask = crit_unique.get("②ASK가능(속성상이·DISAMBIGUATE)", 0)
    dup = crit_unique.get("③진짜중복(全필드동일·ASK불가·벤치아티팩트?)", 0)
    print("\n★필터+ASK 잠재 사정권 = %.1f%% (필터 %.1f + ASK %.1f) · 진짜중복 %.1f%%"
          % (100 * (f + ask) / max(tot, 1), 100 * f / max(tot, 1), 100 * ask / max(tot, 1),
             100 * dup / max(tot, 1)))


if __name__ == "__main__":
    main()
