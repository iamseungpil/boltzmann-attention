"""The 64/64 close-out numbers for the A vs B4 pair, both arms from persisted gz.

The handoff carried these as 48-sim in-progress observations. Judgement rests on
them, so they are recomputed over the completed runs, from the persisted archives
rather than the scratch simulation dirs (which the next run overwrites).

Two families, both from `TRANSFER_INSTRUCTION_FIDELITY_DESIGN` §5:

  Δspurious   transfers the gold never asked for. Baseline O8 = 36 sims (21.2%).
              ① loosened the toll, so the risk is that the gate now opens too easily.

  ④ axis      the matches line rewrites EVERY KB_search return, so it needs a
              two-sided harmlessness check, not just the transfer metrics:
                "185 not shown" -> more searching   : KB_search count, context growth
                "all 4 shown"   -> premature closing: under-search
                global change                       : non-transfer pass regression

Every predicate here is decidable from the trajectory — no semantic judgement.
Pass is `reward == 1.0`; sims are paired by (task_id, trial) across arms.
"""

import argparse
import collections
import glob
import gzip
import json
import os

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")

ARMS = {
    "A":  "bank_ax33n_gpu*_20260803g",
    "B4": "bank_b4_gpu*_20260803h",
}

TRANSFER = "transfer_to_human_agents"


def load(pattern):
    files = sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz"))
    if not files:
        raise SystemExit(f"no runs matched {SIM}/{pattern}.results.json.gz")
    out = {}
    for p in files:
        print(f"  read {os.path.basename(p)}")
        for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
            out[(s.get("task_id"), s.get("trial"))] = s
    return out


def calls(sim):
    """(name, index of the message the call sits in), in trajectory order."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            out.append(((tc.get("name") or (tc.get("function") or {}).get("name") or ""), i))
    return out


def gold_names(sim):
    return {(c.get("action") or {}).get("name")
            for c in (sim.get("reward_info") or {}).get("action_checks") or []}


def chars(sim):
    """Characters the model had to carry: every message body plus its call arguments."""
    n = 0
    for m in sim.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        for tc in m.get("tool_calls") or []:
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            n += len(a if isinstance(a, str) else json.dumps(a or {}))
    return n


def measure(sims):
    m = collections.defaultdict(list)
    rows = {}
    for key, s in sims.items():
        cl = calls(s)
        names = [c[0] for c in cl]
        kb = [i for i, (n, _) in enumerate(cl) if n.startswith("KB_search")]
        gold = gold_names(s)
        rew = (s.get("reward_info") or {}).get("reward") or 0.0

        # Under-search: retrieved, then did nothing with it. Decidable as "no tool
        # call of any kind after the last KB_search".
        under = bool(kb) and kb[-1] == len(cl) - 1

        r = {
            "pass": rew == 1.0,
            "kb": len(kb),
            "calls": len(cl),
            "chars": chars(s),
            "msgs": len(s.get("messages") or []),
            "under": under,
            "term": s.get("termination_reason"),
            "called_transfer": TRANSFER in names,
            "called_any_transfer": any("transfer_to_human" in n for n in names),
            "gold_transfer": TRANSFER in gold,
            "gold_any_transfer": any("transfer_to_human" in (g or "") for g in gold),
            "matches": sum(1 for x in s.get("messages") or []
                           if x.get("role") == "tool" and isinstance(x.get("content"), str)
                           and "matches:" in x["content"]),
        }
        rows[key] = r
        for k, v in r.items():
            m[k].append(v)
    return rows, m


def pct(n, d):
    return f"{n}/{d} = {n / d:.1%}" if d else f"{n}/0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    args = ap.parse_args()

    data = {}
    for a in args.arms.split(","):
        print(f"[{a}]")
        data[a] = measure(load(ARMS[a]))
    names = list(data)

    print("\n" + "=" * 74)
    print("Δspurious — gold이 요구하지 않은 이관 (기준선 O8 = 21.2%)")
    print("=" * 74)
    for a in names:
        rows, _ = data[a]
        n = len(rows)
        spur = [k for k, r in rows.items() if r["called_transfer"] and not r["gold_transfer"]]
        spur_w = [k for k, r in rows.items()
                  if r["called_any_transfer"] and not r["gold_any_transfer"]]
        miss = [k for k, r in rows.items() if r["gold_transfer"] and not r["called_transfer"]]
        print(f"  {a:3s} n={n}  이관 호출 {pct(sum(r['called_transfer'] for r in rows.values()), n)}"
              f"  gold-이관 태스크 {sum(r['gold_transfer'] for r in rows.values())}")
        print(f"      Δspurious(엄격·게이트 대상 도구만) {pct(len(spur), n)}")
        print(f"      Δspurious(광의·모든 transfer_to_human*) {pct(len(spur_w), n)}")
        print(f"      gold-이관인데 미호출 {pct(len(miss), n)}")
        print(f"      spurious sims: {sorted(spur)}")

    print("\n" + "=" * 74)
    print("④축 4지표")
    print("=" * 74)
    hdr = "  {:28s}" + "".join(f"{a:>14s}" for a in names)
    print(hdr.format("지표"))

    def line(label, fn):
        vals = []
        for a in names:
            rows, _ = data[a]
            vals.append(fn(rows))
        print("  {:28s}".format(label) + "".join(f"{v:>14s}" for v in vals))

    line("④ [matches] 부착(툴 메시지)", lambda r: str(sum(x["matches"] for x in r.values())))
    line("1. KB_search 호출 총계", lambda r: str(sum(x["kb"] for x in r.values())))
    line("   KB_search / sim 평균", lambda r: f"{sum(x['kb'] for x in r.values()) / len(r):.2f}")
    line("   KB_search 0회 sim", lambda r: str(sum(1 for x in r.values() if not x["kb"])))
    line("2. 문맥 문자수 / sim 평균", lambda r: f"{sum(x['chars'] for x in r.values()) / len(r):,.0f}")
    line("   문맥 문자수 최대", lambda r: f"{max(x['chars'] for x in r.values()):,d}")
    line("   메시지 수 / sim 평균", lambda r: f"{sum(x['msgs'] for x in r.values()) / len(r):.1f}")
    line("   context_window_exceeded", lambda r: str(sum(1 for x in r.values()
                                                         if x["term"] == "context_window_exceeded")))
    line("3. under-search sim", lambda r: str(sum(1 for x in r.values() if x["under"])))
    line("   도구 호출 / sim 평균", lambda r: f"{sum(x['calls'] for x in r.values()) / len(r):.2f}")
    line("pass", lambda r: pct(sum(x["pass"] for x in r.values()), len(r)))

    print("\n=== 종료사유 ===")
    for a in names:
        rows, _ = data[a]
        print(f"  {a:3s} {dict(collections.Counter(x['term'] for x in rows.values()))}")

    if len(names) == 2:
        A, B = (data[n][0] for n in names)
        shared = sorted(set(A) & set(B))
        print(f"\n=== 4. 비-이관 태스크 pass 회귀 (짝 {len(shared)}) ===")
        for label, sel in (("gold-이관 태스크", lambda k: A[k]["gold_transfer"]),
                           ("비-이관 태스크", lambda k: not A[k]["gold_transfer"])):
            ks = [k for k in shared if sel(k)]
            if not ks:
                continue
            a_p = sum(A[k]["pass"] for k in ks)
            b_p = sum(B[k]["pass"] for k in ks)
            up = [k for k in ks if B[k]["pass"] and not A[k]["pass"]]
            dn = [k for k in ks if A[k]["pass"] and not B[k]["pass"]]
            print(f"  {label:16s} n={len(ks):3d}  A {a_p:3d} → B4 {b_p:3d}  "
                  f"(Δ{b_p - a_p:+d})  개선 {len(up)} 회귀 {len(dn)}")
            if up:
                print(f"      A실패→B4통과: {sorted(up)}")
            if dn:
                print(f"      A통과→B4실패: {sorted(dn)}")

        flips = [k for k in shared if A[k]["pass"] != B[k]["pass"]]
        print(f"\n  전체 flip {len(flips)}/{len(shared)} — pass 동률이 무변화가 아니라 "
              f"상쇄인지 확인용")


if __name__ == "__main__":
    main()
