# -*- coding: utf-8 -*-
"""x484 — **004 변동성의 원인 확정** (2026-08-22·무료·오프라인·LLM 0).

사용자 지시 축자: *"정밀 포렌식해서 변동성의 원인이 뭔지 확정하라. 경로가 달라져도 실패하면
안되지 않나?"* ⇒ 물음은 두 겹이다.
  (1) 같은 조건에서 pass/fail 이 갈리는 **결정의 자리**가 어디인가 (변동의 소재).
  (2) 경로 차이가 성적 차이가 되는 이유 — 즉 **어떤 경로는 왜 답에 못 닿나** (흡수해야 할 부하).

방법 = 004 sim **전량**(모든 태그·조건 무관)을 같은 모양으로 펴서 pass/fail 대조.
판정 권위는 벤치 `action_match`([[69]]·`t2_forensic.action_diff` 독스트링) — 우리가 다시 채점하지 않는다.
사본 0: 로딩·해제·액션 비교는 전부 정본 `t2_forensic` 에서 온다([[67]]).

출력: JSON(칸 전부) + 사람이 읽는 대조표. 결론은 적지 않는다 — 표만 낸다([[08]]).
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F

TASK = "task_004"
XFER = ("transfer_to_human_agents", "initial_transfer_to_human_agent_0218",
        "initial_transfer_to_human_agent_1822",
        "emergency_credit_bureau_incident_transfer_1114",
        "request_human_agent_transfer")


def xfer_calls(sim):
    """이관 계열 호출 전부 — 래퍼 해제 후 이름 기준. 인자·거절 여부·위치를 함께."""
    out = []
    for t in F.trajectory_actions(sim):
        nm = t["inner"] or t["outer"]
        if nm in XFER:
            out.append({"name": nm, "args": t["args"], "ok": t["ok"], "deny": t["deny"],
                        "msg_i": t["msg_i"]})
    return out


def row(tag, sim):
    ri = sim.get("reward_info") or {}
    gold = F.gold_actions_flat(sim)
    tried = F.trajectory_actions(sim)
    xf = xfer_calls(sim)
    seq = [(t["inner"] or t["outer"]) for t in tried]
    return {
        "tag": tag,
        "simtag": F.simtag(sim),
        "reward": ri.get("reward"),
        "basis": F.reward_basis(sim),
        "bench_action_match": [bool(c.get("action_match"))
                               for c in (ri.get("action_checks") or [])],
        "term": F.term_reason(sim),
        "n_msgs": len(sim.get("messages") or []),
        "n_calls": len(tried),
        "gold": [{"outer": g["outer"], "inner": g["inner"], "args": g["args"],
                  "match": g["bench_match"]} for g in gold],
        "xfer": xf,
        "n_xfer": len(xf),
        "seq": seq,
        "seq_tail": seq[-12:],
        "last_text": (F.assistant_text(sim, last=True) or "")[:400],
    }


def main():
    rows = []
    for tag, sim in F.iter_all_sims(want_tasks=[TASK]):
        if sim.get("reward_info") is None:
            continue
        rows.append(row(tag, sim))
    rows.sort(key=lambda r: (r["tag"], r["simtag"]))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "reports", "facet_rft_2026", "x484_004_variability.json")
    with io.open(os.path.abspath(out), "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "n": len(rows), "rows": rows}, f, ensure_ascii=False, indent=1)

    P = [r for r in rows if r["reward"] == 1.0]
    Ff = [r for r in rows if r["reward"] != 1.0]
    print("== 004 전량 %d sim  ·  pass %d  ·  fail %d" % (len(rows), len(P), len(Ff)))
    print("\n== gold 액션(모든 sim 동일한가)")
    gk = collections.Counter(json.dumps(r["gold"], ensure_ascii=False, sort_keys=True) for r in rows)
    for k, n in gk.most_common():
        print("  %3d회  %s" % (n, k[:300]))

    print("\n== 이관 호출 유무 × 성적")
    ct = collections.Counter((r["reward"] == 1.0, r["n_xfer"] > 0) for r in rows)
    for (p, x), n in sorted(ct.items()):
        print("  pass=%-5s 이관호출=%-5s  %d" % (p, x, n))

    print("\n== 이관을 부른 sim 의 인자 (pass ↔ fail)")
    for lab, grp in (("PASS", P), ("FAIL", Ff)):
        for r in grp:
            if not r["xfer"]:
                continue
            for x in r["xfer"]:
                print("  %-5s %-22s %-46s ok=%-5s args=%s"
                      % (lab, r["tag"][:22], (r["simtag"] or "")[:46], x["ok"],
                         json.dumps(x["args"], ensure_ascii=False)[:200]))

    print("\n== 이관을 **안 부른** fail sim — 종료사유·마지막 호출들")
    for r in Ff:
        if r["n_xfer"]:
            continue
        print("  %-22s %-30s term=%-18s calls=%-3d tail=%s"
              % (r["tag"][:22], (r["simtag"] or "")[:30], str(r["term"])[:18], r["n_calls"],
                 ",".join(r["seq_tail"][-6:])))

    print("\n== 태그별 성적")
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["tag"]][1] += 1
        if r["reward"] == 1.0:
            per[r["tag"]][0] += 1
    for t, (p, n) in sorted(per.items()):
        print("  %-26s %d/%d" % (t, p, n))
    print("\n[JSON] %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()


def reason_report():
    """★변동성의 축 하나만 따로 — gold 는 `reason` enum 한 칸이다."""
    rows = json.load(io.open(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "reports", "facet_rft_2026", "x484_004_variability.json")), encoding="utf-8"))["rows"]
    GOLD = "account_ownership_dispute"

    def chosen(r):
        vals = [x["args"].get("reason") for x in r["xfer"] if x["args"].get("reason")]
        return vals

    print("== reason enum × 성적  (gold = %s)" % GOLD)
    ct = collections.Counter()
    for r in rows:
        v = chosen(r)
        key = ("PASS" if r["reward"] == 1.0 else "FAIL",
               "NO_XFER" if not r["xfer"] else (",".join(sorted(set(v))) or "NO_REASON_ARG"))
        ct[key] += 1
    for (p, v), n in sorted(ct.items(), key=lambda kv: (-kv[1], kv[0])):
        print("  %-5s %-64s %d" % (p, v[:64], n))

    print("\n== sim 별 압축표 (태그 · 시드 · 성적 · 고른 reason · 이관횟수 · 종료)")
    for r in rows:
        v = chosen(r)
        print("  %-24s %-10s %-5s %-52s x%-2d %s"
              % (r["tag"][:24], (r["simtag"] or "").split("#")[-1][:10],
                 "PASS" if r["reward"] == 1.0 else "fail",
                 (",".join(sorted(set(v))) if v else ("(이관 0)" if not r["xfer"] else "(reason 인자 없음)"))[:52],
                 r["n_xfer"], str(r["term"])[:14]))
