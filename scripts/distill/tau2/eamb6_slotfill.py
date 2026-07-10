#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-AMB-6 (T6) - 면적 계산기 = slot-filling 대조기 (반사실 오라클 평가·무료).

정본: E_AMB_MEASUREMENT_PLAN_2026_07_10.md §5b
각 write 결정점의 슬롯마다: 후보(DB-유효 ∩ 소유) ∩ 발화-제약(A2-자동 어휘 정확일치)
  -> 0=ASK / 1=FILL / n=ENUM. gold 대조로 이 아키텍처의 정확도 상한을 측정.

[[05]]: 엔진=일반 루프(후보->제약->셈). 도메인 지식은 SLOT_SPEC/CUES 데이터 블록만(DB 스키마 기계 도출).
Run (remote): python3 eamb6_slotfill.py [--sim fl32b_floor_retail_t4]
"""
import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict

REPO = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
DBP = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/db.json"

# ---------------- A2-계열 데이터 (DB 스키마 도출·도메인 데이터 블록) ----------------
WRITE_STATUS = {  # 도구 -> 요구 주문 상태 (A2 G5 preconditions 계열)
    "cancel_pending_order": "pending",
    "modify_pending_order_items": "pending",
    "modify_pending_order_address": "pending",
    "modify_pending_order_payment": "pending",
    "return_delivered_order_items": "delivered",
    "exchange_delivered_order_items": "delivered",
}
PAY_TYPE_CUES = {"paypal": "paypal", "gift card": "gift_card", "giftcard": "gift_card",
                 "credit card": "credit_card", "store credit": "gift_card"}
ORIGINAL_CUES = ("original payment", "original method", "same payment", "same method",
                 "method i used", "how i paid", "original form of payment")
RECENT_CUES = ("most recent", "latest order", "last order", "newest order")
# ------------------------------------------------------------------------------


def norm(x):
    return re.sub(r"\s+", " ", str(x).lower().replace("#", "")).strip()


def load_db():
    db = json.load(open(DBP, encoding="utf-8"))
    item2prod, prod_variants, prod_name = {}, {}, {}
    for pid, p in db["products"].items():
        vs = p.get("variants") or {}
        prod_variants[str(pid)] = vs
        prod_name[str(pid)] = str(p.get("name") or "")
        for v in vs:
            item2prod[str(v)] = str(pid)
    return db, item2prod, prod_variants, prod_name


def user_text(sim, idx):
    return norm(" ".join(m.get("content") or "" for m in sim["messages"][:idx]
                         if m.get("role") == "user" and isinstance(m.get("content"), str)))


def find_user_id(sim, idx, db):
    for m in sim["messages"][:idx]:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            c = m["content"].strip().strip('"')
            if c in db["users"]:
                return c
    return None


def asserted_attrs(utts, variants, old_opts):
    """A2-자동 어휘(이 상품의 option 값들)의 발화 정확일치 · 속성별 최후-단언(recency)."""
    vocab = defaultdict(set)
    for v in variants.values():
        for a, val in (v.get("options") or {}).items():
            vocab[a].add(norm(val))
    out, conflict_old = {}, 0
    for a, vals in vocab.items():
        best = None
        for val in vals:
            p = utts.rfind(val)
            if p >= 0 and (best is None or p > best[1]):
                best = (val, p)
        if best:
            out[a] = best[0]
            if old_opts and norm(old_opts.get(a, "")) == best[0]:
                conflict_old += 1
    return out, conflict_old


def variant_slot(sim, idx, old_item, db, item2prod, prod_variants):
    """new_item_ids 슬롯: (verdict, value_set, |C|, extras)"""
    pid = item2prod.get(str(old_item))
    if not pid:
        return None
    variants = prod_variants[pid]
    old_opts = (variants.get(str(old_item)) or {}).get("options") or {}
    utts = user_text(sim, idx)
    cons, conflict_old = asserted_attrs(utts, variants, old_opts)
    cand = []
    for vid, v in variants.items():
        if v.get("available") is False:
            continue
        opts = {a: norm(val) for a, val in (v.get("options") or {}).items()}
        if all(opts.get(a) == val for a, val in cons.items()):
            cand.append(str(vid))
    # 독립근사 (T6e): |C|_ind = N * prod(p_attr)
    N = sum(1 for v in variants.values() if v.get("available") is not False)
    ind = float(N)
    for a, val in cons.items():
        p = sum(1 for v in variants.values()
                if v.get("available") is not False and norm((v.get("options") or {}).get(a, "")) == val) / max(N, 1)
        ind *= p
    # ★모드 B (T6g): 디폴트 = 기존 속성 유지 ⊕ 단언된 변경만 오버라이드
    consB = {a: norm(val) for a, val in old_opts.items()}
    label = "KEEP"
    for a, val in cons.items():
        if consB.get(a) != val:
            consB[a] = val
            label = "OVERRIDE"
    candB = []
    for vid, v in variants.items():
        if v.get("available") is False:
            continue
        opts = {a: norm(val) for a, val in (v.get("options") or {}).items()}
        if all(opts.get(a) == val for a, val in consB.items()):
            candB.append(str(vid))
    return {"cand": cand, "C": len(cand), "n_cons": len(cons), "conflict_old": conflict_old,
            "C_ind": ind, "pid": pid, "candB": candB, "CB": len(candB), "labelB": label}


def payment_slot(sim, idx, call_oid, db, uid):
    pms = list((db["users"].get(uid) or {}).get("payment_methods") or {}) if uid else []
    utts = user_text(sim, idx)
    # 명시 id
    m = re.findall(r"(?:credit_card|gift_card|paypal)_\d+", utts)
    if m:
        return {"cand": [m[-1]], "C": 1, "cue": "explicit"}
    # original-cue: 주문의 원결제 = 결정론 조회 (calc-계열)
    if any(c in utts for c in ORIGINAL_CUES):
        o = db["orders"].get("#" + call_oid.upper().lstrip("#").replace("W", "W")) or db["orders"].get(call_oid)
        if o is None:
            for k, v in db["orders"].items():
                if norm(k) == norm(call_oid):
                    o = v
                    break
        if o:
            ph = o.get("payment_history") or []
            if ph and ph[0].get("payment_method_id"):
                return {"cand": [str(ph[0]["payment_method_id"])], "C": 1, "cue": "original"}
    # 타입 cue (최후 단언)
    best = None
    for cue, pref in PAY_TYPE_CUES.items():
        p = utts.rfind(cue)
        if p >= 0 and (best is None or p > best[1]):
            best = (pref, p)
    cand = [p for p in pms if best is None or p.startswith(best[0])]
    out = {"cand": cand, "C": len(cand), "cue": best[0] if best else None}
    # ★모드 B (T6g): 디폴트 = 원결제(정책-도출·cue 불요), 타입-cue가 다르면 OVERRIDE
    o = None
    for k, v in db["orders"].items():
        if norm(k) == norm(call_oid):
            o = v
            break
    dflt = None
    if o:
        ph = o.get("payment_history") or []
        if ph and ph[0].get("payment_method_id"):
            dflt = str(ph[0]["payment_method_id"])
    if dflt:
        if best and not dflt.startswith(best[0]):
            cB = [p for p in pms if p.startswith(best[0])]
            out.update(candB=cB, CB=len(cB), labelB="OVERRIDE")
        else:
            out.update(candB=[dflt], CB=1, labelB="KEEP")
    else:
        out.update(candB=cand, CB=len(cand), labelB="NO-DEFAULT")
    return out


def order_slot(sim, idx, tool, db, uid, prod_name, item2prod):
    orders = {k: v for k, v in db["orders"].items() if uid and v.get("user_id") == uid}
    utts = user_text(sim, idx)
    need = WRITE_STATUS.get(tool)
    cand = {k: v for k, v in orders.items() if not need or str(v.get("status", "")).lower() == need}
    # 상품명 언급 필터 (판별적일 때만)
    def mentions(o):
        for it in (o.get("items") or []):
            nm = norm(it.get("name") or prod_name.get(item2prod.get(str(it.get("item_id")), ""), ""))
            if nm and nm in utts:
                return True
        return False
    f2 = {k: v for k, v in cand.items() if mentions(v)}
    if f2:
        cand = f2
    # recency cue
    if any(c in utts for c in RECENT_CUES) and len(cand) > 1:
        ts_key = None
        for probe in ("timestamp", "created_at", "order_date", "date"):
            if any(probe in v for v in cand.values()):
                ts_key = probe
                break
        if ts_key:
            k = max(cand, key=lambda x: str(cand[x].get(ts_key, "")))
            cand = {k: cand[k]}
    return {"cand": [norm(k) for k in cand], "C": len(cand)}


def gold_variant_stats(sims, item2prod):
    """LOTO용: product -> task -> set(gold new variants). 통계-디폴트 arm의 정직한 산정."""
    st = defaultdict(lambda: defaultdict(set))
    for sim in sims:
        tid = str(sim.get("task_id"))
        for x in (sim.get("reward_info") or {}).get("action_checks") or []:
            act = x.get("action") or {}
            for gv in ((act.get("arguments") or {}).get("new_item_ids") or []):
                pid = item2prod.get(norm(gv))
                if pid:
                    st[pid][tid].add(norm(gv))
    return st


def _pick(seq, seedstr):
    """결정론적 의사-랜덤 선택 (Date/random 비의존·재현가능)."""
    if not seq:
        return None
    h = 0
    for ch in seedstr:
        h = (h * 131 + ord(ch)) % 100003
    return sorted(seq)[h % len(seq)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="fl32b_floor_retail_t4")
    a = ap.parse_args()
    db, item2prod, prod_variants, prod_name = load_db()
    data = json.load(gzip.open(f"{REPO}/{a.sim}.results.json.gz", "rt", encoding="utf-8"))
    sims = data["simulations"]
    print("종료사유:", dict(Counter(s.get("termination_reason") for s in sims)))
    gstats = gold_variant_stats(sims, item2prod)

    WRITE = set(WRITE_STATUS) | {"modify_user_address", "place_order"}
    res = defaultdict(list)   # slot_type -> records
    for sim in sims:
        msgs = sim["messages"]
        uid = None
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") not in WRITE:
                    continue
                args = tc.get("arguments") or {}
                acts = [(x.get("action") or {}) for x in (sim.get("reward_info") or {}).get("action_checks") or []]
                gacts = [g for g in acts if g.get("name") == tc.get("name")]
                if not gacts:
                    continue
                oid = norm(args.get("order_id") or "")
                gbest = gacts[0]
                for g in gacts:
                    if norm((g.get("arguments") or {}).get("order_id") or "") == oid:
                        gbest = g
                        break
                uid = uid or find_user_id(sim, i, db)
                # --- order_id 슬롯 ---
                if "order_id" in args and uid:
                    v = order_slot(sim, i, tc.get("name"), db, uid, prod_name, item2prod)
                    gold_set = {norm((g.get("arguments") or {}).get("order_id") or "") for g in gacts}
                    v.update(task=str(sim.get("task_id")), trial=sim.get("trial"),
                             gold=sorted(gold_set), gold_in=bool(gold_set & set(v["cand"])),
                             fill_ok=(v["C"] == 1 and v["cand"][0] in gold_set))
                    res["order_id"].append(v)
                # --- payment 슬롯 ---
                if "payment_method_id" in args and uid:
                    gv = (gbest.get("arguments") or {}).get("payment_method_id")
                    if gv:
                        v = payment_slot(sim, i, oid, db, uid)
                        v.update(task=str(sim.get("task_id")), trial=sim.get("trial"), gold=norm(gv),
                                 gold_in=norm(gv) in [norm(c) for c in v["cand"]],
                                 fill_ok=(v["C"] == 1 and norm(v["cand"][0]) == norm(gv)))
                        res["payment"].append(v)
                # --- variant 슬롯 (old item별) ---
                golds = (gbest.get("arguments") or {})
                gold_old = [norm(x) for x in (golds.get("item_ids") or [])]
                gold_new = [norm(x) for x in (golds.get("new_item_ids") or [])]
                for j, old in enumerate([norm(x) for x in (args.get("item_ids") or [])]):
                    if "new_item_ids" not in args:
                        continue
                    gv = None
                    if old in gold_old:
                        k = gold_old.index(old)
                        gv = gold_new[k] if k < len(gold_new) else None
                    if gv is None:
                        continue
                    v = variant_slot(sim, i, old, db, item2prod, prod_variants)
                    if v is None:
                        continue
                    v.update(task=str(sim.get("task_id")), trial=sim.get("trial"), gold=gv,
                             gold_in=gv in v["cand"],
                             fill_ok=(v["C"] == 1 and v["cand"][0] == gv))
                    # ★T6h: 디폴트 3-arm (같은 오버라이드 기계 위 디폴트만 교체)
                    pid = v["pid"]
                    tid = str(sim.get("task_id"))
                    V = [norm(k) for k, vv in prod_variants[pid].items() if vv.get("available") is not False]
                    F = v["cand"]  # 증거(단언 속성) 필터 결과
                    loto = Counter()
                    for t2k, gs in gstats.get(pid, {}).items():
                        if t2k != tid:
                            loto.update(gs)
                    d_rand = _pick(V, f"{tid}|{sim.get('trial')}|{old}")
                    d_freq = (loto.most_common(1)[0][0] if loto else d_rand)
                    d_prin = (v["candB"][0] if v.get("labelB") == "KEEP" and v.get("CB") == 1 else
                              (norm(old) if norm(old) in V else d_rand))
                    arms = {}
                    for nm, d in (("rand", d_rand), ("freq", d_freq), ("prin", d_prin)):
                        if v["n_cons"] == 0:
                            ans, mode = d, "default"          # 무증거 → 디폴트
                        elif len(F) == 1:
                            ans, mode = F[0], "evidence"      # 증거가 확정 → 디폴트 무관
                        elif len(F) >= 2:
                            ans, mode = (d if d in F else _pick(F, f"{tid}|{old}|{nm}")), "enum"
                        else:
                            ans, mode = d, "default"          # 증거 모순(F=0) → 디폴트
                        arms[nm] = (ans, mode, ans == gv)
                    v["t6h"] = {nm: {"mode": m, "ok": ok} for nm, (ans, m, ok) in arms.items()}
                    res["variant"].append(v)

    print("\n=== T6 slot-filling 대조 (arm=%s) ===" % a.sim)
    for st, rows in res.items():
        n = len(rows)
        fill = [r for r in rows if r["C"] == 1]
        enum = [r for r in rows if r["C"] >= 2]
        ask = [r for r in rows if r["C"] == 0]
        print("\n--- %s (n=%d) ---" % (st, n))
        print("  T6a 판정 분포: FILL %d (%.0f%%) · ENUM %d (%.0f%%) · ASK %d (%.0f%%)" % (
            len(fill), 100 * len(fill) / max(n, 1), len(enum), 100 * len(enum) / max(n, 1),
            len(ask), 100 * len(ask) / max(n, 1)))
        if fill:
            ok = sum(1 for r in fill if r["fill_ok"])
            print("  T6b FILL 정확도: %d/%d = %.3f" % (ok, len(fill), ok / len(fill)))
        if enum:
            gin = sum(1 for r in enum if r["gold_in"])
            szs = Counter(min(r["C"], 5) for r in enum)
            print("  T6c ENUM gold-포함률: %d/%d = %.3f · 크기분포 %s" % (
                gin, len(enum), gin / len(enum), dict(szs)))
        if ask:
            print("  T6d ASK 건수: %d (per-case로 정밀도 판정 필요)" % len(ask))
        bad_fill = [r for r in fill if not r["fill_ok"]][:5]
        for r in bad_fill:
            print("   [FILL-오답] t%s tr%s cand=%s gold=%s %s" % (
                r["task"], r["trial"], r["cand"][:2], r["gold"],
                ("cons=%d" % r.get("n_cons")) if "n_cons" in r else r.get("cue", "")))
        # ★T6g 모드 B (명시적 디폴트 ⊕ 오버라이드)
        wb = [r for r in rows if "CB" in r]
        if wb:
            fb = [r for r in wb if r["CB"] == 1]
            g = lambda r: (r["gold"] if isinstance(r["gold"], str) else None)
            okb = sum(1 for r in fb if g(r) and norm(r["candB"][0]) == g(r))
            keep = [r for r in wb if r.get("labelB") == "KEEP"]
            okk = sum(1 for r in keep if r["CB"] == 1 and g(r) and norm(r["candB"][0]) == g(r))
            ov = [r for r in wb if r.get("labelB") == "OVERRIDE"]
            oko = sum(1 for r in ov if r["CB"] == 1 and g(r) and norm(r["candB"][0]) == g(r))
            print("  ★T6g 모드B: FILL(CB=1) %d/%d · 정확도 %.3f | KEEP %d(정확 %.3f) · OVERRIDE %d(정확 %.3f)" % (
                len(fb), len(wb), okb / max(len(fb), 1),
                len(keep), okk / max(len(keep), 1), len(ov), oko / max(len(ov), 1)))
            badb = [r for r in fb if g(r) and norm(r["candB"][0]) != g(r)][:5]
            for r in badb:
                print("   [B-오답·%s] t%s tr%s candB=%s gold=%s" % (
                    r.get("labelB"), r["task"], r["trial"], r["candB"][:2], r["gold"]))
    # ★T6h: 디폴트 불변성 검정 (variant)
    th = [r for r in res["variant"] if "t6h" in r]
    if th:
        print("\n=== T6h 디폴트 불변성 (variant·n=%d) ===" % len(th))
        modes = Counter(r["t6h"]["rand"]["mode"] for r in th)
        print("  결정 경로: evidence(디폴트 무관) %d · enum %d · default(무증거/모순) %d" % (
            modes.get("evidence", 0), modes.get("enum", 0), modes.get("default", 0)))
        for nm in ("rand", "freq", "prin"):
            tot = sum(1 for r in th if r["t6h"][nm]["ok"])
            dd = [r for r in th if r["t6h"][nm]["mode"] == "default"]
            dok = sum(1 for r in dd if r["t6h"][nm]["ok"])
            print("  arm %-4s 최종 %.3f (%d/%d) · default-결정 구간만 %.3f (%d/%d)" % (
                nm, tot / len(th), tot, len(th), dok / max(len(dd), 1), dok, len(dd)))
        ev = [r for r in th if r["t6h"]["rand"]["mode"] == "evidence"]
        evok = sum(1 for r in ev if r["t6h"]["rand"]["ok"])
        print("  evidence-결정 구간 정확도(모든 arm 동일): %.3f (%d/%d)" % (
            evok / max(len(ev), 1), evok, len(ev)))

    # T6e: 독립근사 vs 정확 (variant)
    vr = [r for r in res["variant"] if r.get("n_cons", 0) >= 1]
    if vr:
        errs = [abs(math.log2(max(r["C_ind"], .25)) - math.log2(max(r["C"], .25))) for r in vr]
        print("\n  T6e 독립근사 |log2(C_ind/C)| 평균 %.2f 비트 (n=%d·제약>=1) — 속성 얽힘 보정 크기" % (
            sum(errs) / len(errs), len(errs)))
    out = f"{REPO}/eamb6_{a.sim}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for st, rows in res.items():
            for r in rows:
                r["slot"] = st
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
