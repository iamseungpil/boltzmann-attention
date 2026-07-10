#!/usr/bin/env python3
"""DB-only 기준 hard-core 재선별 + SAME-count 버킷의 클래스별 전수 분해.

정본 doc: reports/facet_rft_2026/DB_ONLY_HARDCORE_FORENSIC_2026_07_08.md

기준은 reward가 아니라 reward_info.db_check.db_match (C19: 유일한 arm-공통 채점축).
gold write열 ↔ 실제 성공 write열은 **최소-diff 순열 매칭**으로 짝짓는다.
인덱스 순서로 짝지으면 같은 집합을 다른 순서로 실행한 sim이 "⋈ 틀린 주문"으로 오분류된다(doc §2).

사용:
    python3 dbonly_forensic.py --stage core       # hard-core 재선별
    python3 dbonly_forensic.py --stage buckets    # ZERO/FEWER/MORE/EXTRA/SAME 버킷
    python3 dbonly_forensic.py --stage classes    # SAME 버킷 클래스별 3-arm Δ 표
    python3 dbonly_forensic.py --stage fabricate  # free-text 주소 날조 census (doc §7)
    python3 dbonly_forensic.py --stage reason     # reason enum 노이즈 census (doc §8)
    python3 dbonly_forensic.py --stage validity   # termination_reason / infra 선검사
"""

import argparse
import gzip
import itertools
import json
from collections import Counter, defaultdict

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
FR = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench/data/tau2/results/final/"

WRITE = {
    "return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
    "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
    "modify_user_address", "place_order",
}
ADDR_WRITE = {"modify_pending_order_address", "modify_user_address"}


def _gz(tag):
    return json.load(gzip.open(SIM + tag + ".results.json.gz"))["simulations"]


def _plain(fname):
    return json.load(open(FR + fname))["simulations"]


def arms():
    return [
        ("ours", _gz("asmregen32b_regen_retail_t4")),
        ("o4-mini", _plain("o4-mini-2025-04-16_retail_default_gpt-4.1-2025-04-14_4trials.json")),
        ("gpt-4.1", _plain("gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json")),
    ]


def ri(s):
    return s.get("reward_info") or {}


def db_match(s):
    return (ri(s).get("db_check") or {}).get("db_match")


def task_id(s):
    return str(s.get("task_id"))


def gold_writes(s):
    out = []
    for a in ri(s).get("action_checks") or []:
        act = a.get("action") or {}
        if act.get("name") in WRITE:
            out.append((act["name"], act.get("arguments") or {}))
    return out


def executed_writes(s, include_errors=False):
    """성공한(=tool이 error를 내지 않은) write 호출열. include_errors면 (…, err) 동봉."""
    by_id = {m.get("id") or m.get("tool_call_id"): m
             for m in s.get("messages", []) if m.get("role") == "tool"}
    out = []
    for m in s.get("messages", []):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if tc.get("name") not in WRITE:
                continue
            tm = by_id.get(tc.get("id"))
            err = tm is not None and bool(tm.get("error"))
            if include_errors:
                out.append((tc["name"], tc.get("arguments") or {}, err))
            elif tm is not None and not err:
                out.append((tc["name"], tc.get("arguments") or {}))
    return out


def _norm(v):
    return sorted(str(x) for x in v) if isinstance(v, list) else v


def diff_keys(gold_args, our_args):
    keys = (set(gold_args) | set(our_args)) - {"user_id"}
    return [k for k in sorted(keys) if _norm(gold_args.get(k)) != _norm(our_args.get(k))]


def _cost(g, o):
    return 100 if g[0] != o[0] else len(diff_keys(g[1], o[1]))


def best_pairing(gold, ours):
    """최소 총 diff 비용 순열. write 수가 작아(≤6) 완전탐색으로 충분."""
    best = None
    for perm in itertools.permutations(range(len(ours))):
        c = sum(_cost(gold[i], ours[perm[i]]) for i in range(len(gold)))
        if best is None or c < best[0]:
            best = (c, perm)
    return best[1]


def classify(keys, gold_args, our_args):
    if "order_id" in keys:
        return "⋈ order_id (틀린 주문)"
    if "item_ids" in keys and "new_item_ids" in keys:
        return "item_ids+new_item_ids 혼합"
    if "new_item_ids" in keys:
        return "F2 변형선택(new_item_ids)"
    if "item_ids" in keys:
        g = _norm(gold_args.get("item_ids") or [])
        o = _norm(our_args.get("item_ids") or [])
        if len(o) > len(g):
            return "item 과포함"
        if len(o) < len(g):
            return "item 누락"
        return "item 오선택(동수)"
    if "payment_method_id" in keys:
        return "payment_method"
    if any(k.startswith("address") or k in ("city", "state", "zip", "country") for k in keys):
        return "주소 필드"
    if "reason" in keys:
        return "reason enum"
    if "amount" in keys:
        return "amount"
    return "기타:" + ",".join(keys)


def bucket(s):
    need, did = len(gold_writes(s)), len(executed_writes(s))
    if need == 0 and did > 0:
        return "EXTRA (gold=0 write)"
    if did == 0 and need > 0:
        return "ZERO write"
    if did < need:
        return "FEWER (미완)"
    if did > need:
        return "MORE (과잉)"
    return "SAME count, wrong content"


# ---------------------------------------------------------------- stages

def stage_validity():
    for name, sims in arms():
        term = Counter(str(s.get("termination_reason")) for s in sims)
        fails = [s for s in sims if db_match(s) is False]
        print(f"{name}: sim={len(sims)} term={dict(term)} "
              f"db_fail={len(fails)} infra={term.get('infrastructure_error', 0)}")


def stage_core():
    tabs = {}
    for name, sims in arms():
        d = defaultdict(list)
        for s in sims:
            d[task_id(s)].append(db_match(s))
        tabs[name] = d
    o, m, g = tabs["ours"], tabs["o4-mini"], tabs["gpt-4.1"]
    tasks = sorted(set(o) & set(m) & set(g), key=int)
    npass = lambda d, t: sum(1 for v in d[t] if v is True)  # noqa: E731

    core = [t for t in tasks if npass(o, t) <= 1 and npass(m, t) >= 3 and npass(g, t) >= 3]
    rev = [t for t in tasks if npass(o, t) >= 3 and npass(m, t) <= 1 and npass(g, t) <= 1]
    print(f"DB-only HARD CORE = {len(core)}: {core}")
    for t in core:
        print(f"  t{t:>4}: ours {npass(o,t)}/4  o4 {npass(m,t)}/4  g41 {npass(g,t)}/4")
    print(f"역방향 = {len(rev)}: {rev}")


def stage_buckets():
    res = {}
    for name, sims in arms():
        c = Counter(bucket(s) for s in sims if db_match(s) is False)
        res[name] = c
        print(f"\n{name}: db_fail={sum(c.values())}")
        for k, v in c.most_common():
            print(f"   {v:4d} ({100*v/sum(c.values()):>4.1f}%)  {k}")
    print("\nΔ vs o4-mini")
    for k in set(res["ours"]) | set(res["o4-mini"]):
        print(f"   {res['ours'][k]:4d} vs {res['o4-mini'][k]:4d}  Δ{res['ours'][k]-res['o4-mini'][k]:+4d}  {k}")


def stage_classes():
    res = {}
    for name, sims in arms():
        agg, nsim = Counter(), 0
        for s in sims:
            if db_match(s) is not False:
                continue
            G, O = gold_writes(s), executed_writes(s)
            if len(G) != len(O) or not G or len(G) > 6:
                continue
            nsim += 1
            perm = best_pairing(G, O)
            rows = []
            for i in range(len(G)):
                gn, ga = G[i]
                on, oa = O[perm[i]]
                if gn != on:
                    rows.append("★op(연산자) 불일치")
                    continue
                k = diff_keys(ga, oa)
                rows.append("MATCH" if not k else classify(k, ga, oa))
            if all(r == "MATCH" for r in rows):
                agg["전부 일치인데 DB fail(variant-leak 버그)"] += 1
                continue
            for r in rows:
                if r != "MATCH":
                    agg[r] += 1
        res[name] = agg
        print(f"{name}: SAME-count db_fail sim = {nsim}")

    keys = sorted(set().union(*(set(a) for a in res.values())), key=lambda k: -res["ours"][k])
    print(f"\n{'클래스':<34}{'ours':>7}{'o4-mini':>9}{'gpt-4.1':>9}{'Δo4':>7}{'Δg41':>7}")
    for k in keys:
        a, b, c = res["ours"][k], res["o4-mini"][k], res["gpt-4.1"][k]
        print(f"{k:<34}{a:>7}{b:>9}{c:>9}{a-b:>+7}{a-c:>+7}")


def _ctx_before(s, idx):
    return " ".join(m.get("content") for m in s.get("messages", [])[:idx]
                    if isinstance(m.get("content"), str))


def stage_fabricate():
    """write 시점 이전 문맥에 address1 문자열이 존재했는가 (doc §7)."""
    for name, sims in arms():
        tot = fab = fab_fail = 0
        rows = []
        for s in sims:
            by_id = {m.get("id") or m.get("tool_call_id"): m
                     for m in s.get("messages", []) if m.get("role") == "tool"}
            for i, m in enumerate(s.get("messages", [])):
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    if tc.get("name") not in ADDR_WRITE:
                        continue
                    tm = by_id.get(tc.get("id"))
                    if tm is None or tm.get("error"):
                        continue
                    a1 = ((tc.get("arguments") or {}).get("address1") or "").strip()
                    if not a1:
                        continue
                    tot += 1
                    if a1 not in _ctx_before(s, i):
                        fab += 1
                        if db_match(s) is False:
                            fab_fail += 1
                        rows.append((task_id(s), s.get("trial"), a1))
        print(f"{name}: 주소 write {tot} · 문맥에 없던 address1 {fab} ({100*fab/max(tot,1):.0f}%) · db_fail {fab_fail}")
        for r in rows[:10]:
            print(f"    t{r[0]:>4} tr{r[1]}  {r[2]!r}")


def stage_reason():
    """gold reason vs 실행 reason, 그리고 사용자 발화가 우리 값을 지지하는가 (doc §8)."""
    for name, sims in arms():
        tot = mism = supported = 0
        rows = []
        for s in sims:
            gold = {a[1].get("order_id"): a[1].get("reason")
                    for a in gold_writes(s) if a[0] == "cancel_pending_order"}
            if not gold:
                continue
            by_id = {m.get("id") or m.get("tool_call_id"): m
                     for m in s.get("messages", []) if m.get("role") == "tool"}
            for i, m in enumerate(s.get("messages", [])):
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    if tc.get("name") != "cancel_pending_order":
                        continue
                    tm = by_id.get(tc.get("id"))
                    if tm is None or tm.get("error"):
                        continue
                    args = tc.get("arguments") or {}
                    oid = args.get("order_id")
                    if oid not in gold:
                        continue
                    tot += 1
                    if args.get("reason") == gold[oid]:
                        continue
                    mism += 1
                    utter = " ".join(x.get("content") or "" for x in s.get("messages", [])[:i]
                                     if x.get("role") == "user").lower()
                    mine = (args.get("reason") or "").lower()
                    if "mistake" in mine:
                        sup = "mistake" in utter or "accident" in utter
                    else:
                        sup = any(p in utter for p in ("no longer need", "don't need", "dont need", "not need"))
                    supported += bool(sup)
                    rows.append((task_id(s), s.get("trial"), gold[oid], args.get("reason"), sup))
        print(f"{name}: cancel {tot} · reason 불일치 {mism} · 사용자 발화가 우리 값 지지 {supported}")
        for r in rows[:12]:
            print(f"    t{r[0]:>4} tr{r[1]}  gold={r[2]!r} exec={r[3]!r} user지지={r[4]}")


STAGES = {
    "validity": stage_validity, "core": stage_core, "buckets": stage_buckets,
    "classes": stage_classes, "fabricate": stage_fabricate, "reason": stage_reason,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    STAGES[ap.parse_args().stage]()
