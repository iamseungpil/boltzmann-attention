#!/usr/bin/env python3
"""x221 — was the decision block pushed back by a later KB search?

HANDOFF 2026-08-10 §5 좌표: "블록은 주입 시점엔 마지막이지만, 그 뒤 에이전트가 KB 검색을
더 돌려 결정 시점엔 블록이 뒤로 밀린다"는 유력 가설(미확정). 여기서는 **가설을 재기만** 한다.

두 소스를 각각 그대로 읽는다 (사이드카 `sim` 은 해시라 results.json 의 uuid 와 대응되지
않는다 — HANDOFF §10. 그래서 시행별 조인을 하지 않고 **두 세기를 나란히** 낸다):

  A. 궤적(results.json.gz) — 계좌 읽기(`call_discoverable_agent_tool`) 이후, 최종
     `submit_referral` 이전에 KB 검색이 몇 번 있었는지. 블록은 결정점(계좌 읽기 뒤 권고 직전)에
     실리므로 이 구간이 "블록 이후"의 상한 근사다.
  B. 사이드카(fb_*.jsonl.gz) — 결정 블록이 실제로 나간 turn, 그 sim 에서 마지막으로 관측된
     turn, 그리고 블록 이후에 우리가 더 밀어 넣은 주입이 몇 건인지.

⛔0 자기점검: 이 스크립트는 **측정만** 한다. 엔진에 아무것도 넣지 않는다.
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIMS = ROOT / "reports/facet_rft_2026/sim_results"

KB = ("KB_search_dense", "KB_search_bm25")
ACCOUNT_READ = "call_discoverable_agent_tool"
FINAL = "submit_referral"

# 결정 블록의 서명 — 격리 서브가 낸 답을 값으로 얹는 문장
BLOCK_SIG = "A separate check was run on the policy constants on record"


def traj_side(tag):
    path = SIMS / f"{tag}.json.gz"
    sims = json.load(gzip.open(path, "rt", encoding="utf-8"))["simulations"]
    rows = []
    for s in sims:
        seq = []
        for m in s["messages"]:
            for t in (m.get("tool_calls") or []):
                fn = (t.get("function") or {}).get("name") or t.get("name")
                seq.append(fn)
        # 계좌 읽기 이후 ~ 최종 제출 이전
        try:
            lo = len(seq) - 1 - seq[::-1].index(ACCOUNT_READ)
        except ValueError:
            lo = None
        try:
            hi = seq.index(FINAL)
        except ValueError:
            hi = len(seq)
        window = seq[lo + 1:hi] if lo is not None and lo < hi else []
        rows.append({
            "task": s["task_id"],
            "trial": s.get("trial"),
            "reward": (s.get("reward_info") or {}).get("reward"),
            "end": s.get("termination_reason"),
            "read_account": lo is not None,
            "kb_after_read": sum(1 for x in window if x in KB),
            "kb_total": sum(1 for x in seq if x in KB),
            "seq": seq,
        })
    return rows


def sidecar_side(tag):
    path = SIMS / f"fb_{tag}.jsonl.gz"
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["sim"]].append(r)
    out = []
    for sim, rs in by.items():
        rs.sort(key=lambda r: r["turn"])
        blocks = [r for r in rs if BLOCK_SIG in r["text"]]
        last_turn = max(r["turn"] for r in rs)
        out.append({
            "sim": sim,
            "n_inject": len(rs),
            "block_turns": [r["turn"] for r in blocks],
            "last_turn": last_turn,
            "after_block": (
                sum(1 for r in rs if r["turn"] > blocks[-1]["turn"]) if blocks else None
            ),
            "answer": [
                (re.search(r"It answers:\s*([^\n]+)", r["text"]) or [None, None])[1]
                for r in blocks
            ],
        })
    return sorted(out, key=lambda d: -len(d["block_turns"]))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_alllevers_20260810"
    print(f"### {tag}")
    print("\n[A] 궤적 — 계좌 읽기 이후 KB 재검색 (블록 이후의 상한 근사)")
    print(f"{'task':10} {'t':>2} {'rew':>4} {'KB_after':>9} {'KB_tot':>7}  end")
    rows = traj_side(tag)
    for r in sorted(rows, key=lambda r: (r["task"], r["trial"])):
        print(f"{r['task']:10} {r['trial']:>2} {r['reward']:>4} "
              f"{r['kb_after_read']:>9} {r['kb_total']:>7}  {r['end']}")
    pas = [r["kb_after_read"] for r in rows if r["reward"] == 1.0]
    fal = [r["kb_after_read"] for r in rows if r["reward"] != 1.0]
    print(f"\n  통과 {len(pas)}건 KB_after 분포 {Counter(pas)}  평균 {sum(pas)/max(1,len(pas)):.2f}")
    print(f"  실패 {len(fal)}건 KB_after 분포 {Counter(fal)}  평균 {sum(fal)/max(1,len(fal)):.2f}")
    print("  ⇒ 가설이 맞으려면 실패 쪽이 확실히 커야 한다.")

    print("\n[B] 사이드카 — 결정 블록의 turn 과 그 뒤 주입")
    print(f"{'sim':14} {'inject':>6} {'block_turns':>14} {'last':>5} {'after':>6}  answers")
    for d in sidecar_side(tag):
        print(f"{d['sim']:14} {d['n_inject']:>6} {str(d['block_turns']):>14} "
              f"{d['last_turn']:>5} {str(d['after_block']):>6}  {d['answer']}")
    print("\n⚠사이드카 sim 은 해시라 위 두 표는 **시행별로 대응되지 않는다**(HANDOFF §10).")


if __name__ == "__main__":
    main()
