#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-AMB-7 - 격리→e2e 희석의 per-step 전수 포렌식 (prov·DISAMB).

물음: prov(단일턴 67→0%)·DISAMB(격리 +31pp)가 e2e pass^1을 거의 못 사는 이유를
      3-arm(floor/prov/routerv1) 전수 궤적의 step-이벤트로 원인 분해.

가설: H-A 표적 발생률 낮음 / H-B 다중결함(교정해도 타 결함) / H-C 개입 탈선(무-write)
      / H-D 환경이 이미 차단·회복시키던 실수 / H-E 부작용 상쇄(분산).

[[08]]: 종료사유·step 분류·arm 교차표·per-case 후보 출력까지. CPU-only.
Run: python3 eamb7_dilution_census.py
"""
import gzip
import json
import re
from collections import Counter, defaultdict

R = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
ARMS = {"floor": "fl32b_floor_retail_t4", "prov": "prov_e2e_retail_t4", "router": "routerv1_retail_t4"}

WRITE = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
         "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
         "modify_user_address", "place_order"}
ID_ARGS = {"order_id": r"#?W\d{7}", "item_ids": r"\b\d{10}\b", "new_item_ids": r"\b\d{10}\b",
           "payment_method_id": r"(?:credit_card|gift_card|paypal)_\d+"}
FMT = dict(ID_ARGS, address1=r"\d+ [A-Z][a-z]+ (?:Street|Avenue|Drive|Lane|Road|Boulevard|Way|Court)")


def norm(x):
    return re.sub(r"\s+", " ", str(x).lower().replace("#", "")).strip()


def as_set(v):
    vals = v if isinstance(v, list) else [v]
    return frozenset(norm(x) for x in vals if x is not None)


def sim_steps(sim):
    """step 이벤트: write-call마다 (idx, tool, args, env_ok, ctx이전텍스트)."""
    msgs = sim["messages"]
    ctx_parts, out = [], []
    tool_resp = {}
    for m in msgs:
        if m.get("role") == "tool":
            tool_resp[m.get("id") or m.get("tool_call_id")] = str(m.get("content"))
    for i, m in enumerate(msgs):
        r = m.get("role")
        if r in ("user", "tool") and isinstance(m.get("content"), str):
            ctx_parts.append(m["content"])
        if r != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            if tc.get("name") not in WRITE:
                continue
            resp = tool_resp.get(tc.get("id"), "")
            env_ok = not resp.strip().lower().startswith("error")
            out.append({"idx": i, "tool": tc.get("name"), "args": tc.get("arguments") or {},
                        "env_ok": env_ok, "ctx": norm(" ".join(ctx_parts))})
    return out


def fab_events(steps):
    """날조 이벤트: id-형 인자값 ∉ 이전 문맥."""
    ev = []
    for s in steps:
        for k, pat in ID_ARGS.items():
            if k not in s["args"] or s["args"][k] in (None, [], ""):
                continue
            for v in (s["args"][k] if isinstance(s["args"][k], list) else [s["args"][k]]):
                nv = norm(v)
                if len(nv) >= 4 and nv not in s["ctx"]:
                    ev.append({"tool": s["tool"], "arg": k, "val": nv, "env_ok": s["env_ok"], "idx": s["idx"]})
    return ev


def gold_map(sim):
    g = {}
    for a in (sim.get("reward_info") or {}).get("action_checks") or []:
        act = a.get("action") or {}
        if act.get("name") in WRITE:
            g.setdefault(act.get("name"), []).append(act.get("arguments") or {})
    return g


def wrongpick_events(sim, steps):
    """오선택: env-수락 write 인자가 gold와 불일치 ∧ 같은-형식 후보 2+ ∧ gold∈문맥."""
    gm = gold_map(sim)
    ev = []
    for s in steps:
        if not s["env_ok"]:
            continue
        cands_g = gm.get(s["tool"]) or []
        if not cands_g:
            continue
        oid = norm(s["args"].get("order_id") or "")
        best = cands_g[0]
        for c in cands_g:
            if norm(c.get("order_id") or "") == oid:
                best = c
                break
        for k, pat in FMT.items():
            if k not in s["args"] or best.get(k) in (None, [], ""):
                continue
            gset = as_set(best.get(k))
            vset = as_set(s["args"].get(k))
            if vset == gset:
                continue
            nc = len(set(re.findall(pat, s["ctx"], re.I)))
            gin = all(g in s["ctx"] for g in gset)
            if nc >= 2 and gin:
                ev.append({"tool": s["tool"], "arg": k, "nc": nc})
    return ev


def load(arm):
    d = json.load(gzip.open(R + ARMS[arm] + ".results.json.gz", "rt", encoding="utf-8"))
    per = {}
    for sim in d["simulations"]:
        key = (str(sim.get("task_id")), sim.get("trial"))
        steps = sim_steps(sim)
        r = (sim.get("reward_info") or {}).get("reward")
        gm = gold_map(sim)
        n_gold_w = sum(len(v) for v in gm.values())
        n_ok_w = sum(1 for s in steps if s["env_ok"])
        per[key] = {"reward": r, "steps": steps, "fab": fab_events(steps),
                    "wp": wrongpick_events(sim, steps), "n_gold_w": n_gold_w, "n_ok_w": n_ok_w,
                    "term": sim.get("termination_reason")}
    return per


def main():
    data = {a: load(a) for a in ARMS}
    for a in ARMS:
        print("[%s] term:" % a, dict(Counter(v["term"] for v in data[a].values())))

    F, P, RT = data["floor"], data["prov"], data["router"]
    tasks = sorted({k[0] for k in F}, key=int)

    # ── A. prov 희석 분해 (표적 = floor의 날조) ──
    print("\n=== A. prov 희석 분해 (floor 날조 표적의 운명) ===")
    fab_tt = [k for k, v in F.items() if v["fab"]]
    fab_fail = [k for k in fab_tt if (F[k]["reward"] or 0) < 0.999]
    print("floor: 날조≥1 (task,trial) %d/456 · 그중 실패 %d (표적 상한 = %.1fpp)" % (
        len(fab_tt), len(fab_fail), 100 * len(fab_fail) / 456))
    # H-D: floor 안에서 날조가 env에 막히고 이후 회복됐나
    blocked = corrected = 0
    for k in fab_tt:
        v = F[k]
        for e in v["fab"]:
            if not e["env_ok"]:
                blocked += 1
                if any(s["env_ok"] and s["tool"] == e["tool"] and s["idx"] > e["idx"] for s in v["steps"]):
                    corrected += 1
    tot_fab = sum(len(v["fab"]) for v in F.values())
    accepted = tot_fab - blocked
    print("H-D: floor 날조 이벤트 %d = env-차단 %d(그중 이후 같은도구 env-수락 재시도 %d = %.0f%%) + env-수락(free-text류) %d" % (
        tot_fab, blocked, corrected, 100 * corrected / max(blocked, 1), accepted))
    # 표적 (task,trial) 위 prov의 결과 — task 단위 pass율 비교
    fab_tasks = sorted({k[0] for k in fab_fail}, key=int)
    def pr(arm_d, ts):
        rs = [arm_d[(t, tr)]["reward"] or 0 for t in ts for tr in range(4) if (t, tr) in arm_d]
        return sum(1 for r in rs if r >= 0.999) / max(len(rs), 1)
    print("표적 task %d개: floor pass1 %.3f -> prov %.3f (Δ%.1fpp) | 비표적: floor %.3f -> prov %.3f" % (
        len(fab_tasks), pr(F, fab_tasks), pr(P, fab_tasks), 100 * (pr(P, fab_tasks) - pr(F, fab_tasks)),
        pr(F, [t for t in tasks if t not in fab_tasks]), pr(P, [t for t in tasks if t not in fab_tasks])))
    # H-B/H-C: 표적 (task,trial)에서 prov도 실패한 건의 잔여 원인
    cause = Counter()
    hb_examples = []
    for (t, tr) in fab_fail:
        pk = (t, tr)
        if pk not in P:
            continue
        pv = P[pk]
        if (pv["reward"] or 0) >= 0.999:
            cause["prov가 살림(pass 전환)"] += 1
            continue
        if pv["fab"]:
            cause["날조 잔존(prov 뚫림)"] += 1
        elif pv["n_ok_w"] == 0 and pv["n_gold_w"] > 0:
            cause["무-write(H-C 탈선/미도달)"] += 1
        elif pv["wp"]:
            cause["⋈ 오선택 잔존(H-B)"] += 1
        elif pv["n_ok_w"] < pv["n_gold_w"]:
            cause["미완/coverage(H-B)"] += 1
        else:
            cause["기타(scope·NL 등·H-B)"] += 1
            if len(hb_examples) < 4:
                hb_examples.append(pk)
    print("표적 실패 %d건의 prov-arm 운명: %s" % (len(fab_fail), dict(cause)))
    print("  기타-분류 per-case 후보:", hb_examples)

    # ── B. DISAMB 희석 분해 (표적 = prov의 ⋈ 오선택) ──
    print("\n=== B. DISAMB 희석 분해 (prov ⋈-오선택 표적의 운명) ===")
    wp_tt = [k for k, v in P.items() if v["wp"]]
    wp_fail = [k for k in wp_tt if (P[k]["reward"] or 0) < 0.999]
    print("prov: 오선택≥1 (task,trial) %d/456 · 그중 실패 %d (표적 상한 = %.1fpp)" % (
        len(wp_tt), len(wp_fail), 100 * len(wp_fail) / 456))
    cause2 = Counter()
    for (t, tr) in wp_fail:
        rk = (t, tr)
        if rk not in RT:
            continue
        rv = RT[rk]
        if (rv["reward"] or 0) >= 0.999:
            cause2["router가 살림"] += 1
        elif rv["wp"]:
            cause2["오선택 잔존(재확인 무효)"] += 1
        elif rv["n_ok_w"] == 0 and rv["n_gold_w"] > 0:
            cause2["무-write 탈선(H-C)"] += 1
        elif rv["n_ok_w"] < rv["n_gold_w"]:
            cause2["미완(H-B)"] += 1
        else:
            cause2["기타(H-B)"] += 1
    print("표적 실패 %d건의 router-arm 운명: %s" % (len(wp_fail), dict(cause2)))

    # ── C. 부작용(H-E): prov→router 무-write 탈선 신규 발생 ──
    print("\n=== C. H-E 부작용: write-소멸 교차표 (prov write-유 → router write-무) ===")
    lost = [(t, tr) for (t, tr) in P
            if P[(t, tr)]["n_ok_w"] > 0 and (t, tr) in RT and RT[(t, tr)]["n_ok_w"] == 0 and RT[(t, tr)]["n_gold_w"] > 0]
    lost_fail = [k for k in lost if (RT[k]["reward"] or 0) < 0.999]
    print("prov write-성공 → router write-소멸: %d건 (그중 router 실패 %d)" % (len(lost), len(lost_fail)))
    print("  task 분포:", sorted(Counter(t for t, _ in lost_fail).items(), key=lambda x: -x[1])[:8])

    # ── D. 회계: pass^1 Δ의 성분 합 ──
    print("\n=== D. 회계 (pass^1·456 시행 기준) ===")
    for a, b in (("floor", "prov"), ("prov", "router")):
        A, B = data[a], data[b]
        common = [k for k in A if k in B]
        da = sum(1 for k in common if (B[k]["reward"] or 0) >= 0.999 and (A[k]["reward"] or 0) < 0.999)
        db = sum(1 for k in common if (B[k]["reward"] or 0) < 0.999 and (A[k]["reward"] or 0) >= 0.999)
        print("%s→%s: 살림 %d − 죽임 %d = 순 %+d 시행 (%.1fpp)" % (a, b, da, db, da - db, 100 * (da - db) / len(common)))


if __name__ == "__main__":
    main()
