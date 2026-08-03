"""Why a task that passed in arm A failed in arm B4 (or the reverse).

The two arms differ by exactly two things — the transfer toll wording (①) and the
`[axis] matches:` line on retrieval results (④). So for every flipped simulation the
question is narrow: did either artifact touch the trajectory at or before the point
where the two runs diverged? If neither did, the flip is the run's own trial variance,
which arm A already showed to be 25% task-level.

Usage:
    python ab_flip_forensic.py                 # every flip
    python ab_flip_forensic.py --task task_008
"""

import argparse
import glob
import gzip
import io
import json

SD = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
BLIVE = "/home/woori/scratch/tau2-bench/data/simulations/bank_b4_gpu*_20260803h/results.json"
NOTICE = "TRANSFER NOTICE: Would you like"


def load(pattern, gz=False):
    sims = []
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(gzip.open(p, "rt", encoding="utf-8")) if gz \
                else json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        sims.extend(d.get("simulations") or [])
    return {(s.get("task_id"), s.get("trial")): s for s in sims}


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def calls(sim):
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            a = tc.get("arguments")
            if a is None:
                a = (tc.get("function") or {}).get("arguments")
            out.append((n, norm(a)))
    return out


def artifacts(sim):
    """Where ① and ④ appear, as message indices."""
    notice_at, matches_at = [], []
    for i, m in enumerate(sim.get("messages") or []):
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if m.get("role") == "assistant" and NOTICE in c:
            notice_at.append(i)
        if m.get("role") == "tool" and "matches:" in c:
            matches_at.append(i)
    return notice_at, matches_at


def missed(sim):
    ri = sim.get("reward_info") or {}
    out = []
    for c in ri.get("action_checks") or []:
        if c.get("action_match"):
            continue
        a = c.get("action") or {}
        if a.get("requestor") != "assistant":
            continue
        out.append((a.get("name"), json.dumps(a.get("arguments"), ensure_ascii=False)[:90]))
    return out


def first_div(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p[0] != q[0]:
            return i, "tool %s vs %s" % (p[0], q[0])
        if p[1] != q[1]:
            return i, "same tool %s, args differ" % p[0]
    if len(x) != len(y):
        return min(len(x), len(y)), "length %d vs %d" % (len(x), len(y))
    return None, "identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    args = ap.parse_args()

    A = load(f"{SD}/bank_ax33n_gpu*_20260803g.results.json.gz", gz=True)
    B = load(BLIVE)

    def rew(s):
        return (s.get("reward_info") or {}).get("reward") or 0.0

    flips = [k for k in sorted(set(A) & set(B)) if rew(A[k]) != rew(B[k])]
    if args.task:
        flips = [k for k in flips if k[0] == args.task]
    print(f"짝 {len(set(A) & set(B))} sim · 뒤집힘 {len(flips)}\n")

    for k in flips:
        a, b = A[k], B[k]
        ca, cb = calls(a), calls(b)
        idx, why = first_div(ca, cb)
        na, ma = artifacts(a)
        nb, mb = artifacts(b)
        arrow = "A=%s -> B4=%s" % (rew(a), rew(b))
        print("=" * 92)
        print(f"## {k[0]}/t{k[1]}   {arrow}   {'상실' if rew(a) > rew(b) else '획득'}")
        print(f"   A : calls={len(ca):3d} term={a.get('termination_reason')}")
        print(f"   B4: calls={len(cb):3d} term={b.get('termination_reason')}")
        print(f"   ① 새 통행료 발화  A:{na or '-'}  B4:{nb or '-'}")
        print(f"   ④ matches 줄     A:{len(ma)}  B4:{len(mb)}  (B4 최초 @msg {mb[0] if mb else '-'})")
        print(f"   >>> 최초 발산: {why}" + (f" @call#{idx}" if idx is not None else ""))
        # 발산이 ①/④ 이전인가 이후인가 — 귀속의 핵심
        if idx is not None and mb:
            # 발산 call#idx 가 몇 번째 메시지인지 근사: 호출 순서 -> 메시지 인덱스
            n_seen = 0
            div_msg = None
            for i, m in enumerate(b.get("messages") or []):
                if m.get("role") != "assistant":
                    continue
                for _ in m.get("tool_calls") or []:
                    if n_seen == idx:
                        div_msg = i
                        break
                    n_seen += 1
                if div_msg is not None:
                    break
            if div_msg is not None:
                before = [x for x in mb if x < div_msg]
                print(f"   ★발산(msg{div_msg}) 이전의 ④ 발화: {len(before)}개"
                      + ("  ⇒ ④가 문맥에 있었다" if before else "  ⇒ ④는 발산 이후"))
        for lbl, s in (("A ", a), ("B4", b)):
            ms = missed(s)
            print(f"   {lbl} 에이전트-측 MISS: " + (", ".join(f"{n}" for n, _ in ms) if ms else "없음"))
        for lbl, s in (("A ", a), ("B4", b)):
            tail = [m for m in (s.get("messages") or [])
                    if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"]]
            print(f"   {lbl} last: {' '.join(tail[-1]['content'].split())[:120] if tail else ''}")
        print()


if __name__ == "__main__":
    main()
