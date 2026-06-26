#!/usr/bin/env python
"""pass^4 정체 + r3 개선의 궤적 전수조사 (2026-06-13 사용자 발주, zero-GPU).

질문: ①pass^4가 게이트 arm에서 왜 안 오르나 — trial-간 분산의 원천은 무엇인가
(deny 확률성? user-sim 분산? 대화 길이/종결?) ②r3(G4게이트+중립템플릿)가 r2 대비
pass^1/2/3을 올린 기제는 무엇인가 (태스크-플립 단위로). ③r2의 G4 위반 0이 "운"
이었는지 (transfer 빈도·notice 자발률).

방법(전수):
  A1 per-task pass 패턴 (c=0..4 분포·flaky=1..3)
  A2 flaky 태스크 내부의 deny↔fail 짝비교 (같은 태스크에서 deny 있는 trial이 더 실패하나
     — 태스크 난이도 통제된 within-task 신호; 게이트별 G1/G2/G4 분해)
  A3 flaky 태스크 내 pass-trial vs fail-trial의 assistant 턴수/메시지수
  A4 fail-trial 종결사유 분포
  A5 arm-간 태스크-플립 (r3 vs r2: c 증감 태스크 목록 + transfer-관여/deny-조성 교차)
  A6 transfer 깔때기 (전 arm): transfer 시도→(G4 deny)→notice→실행→pass
Usage: t2_passk_autopsy.py --simdir .../simulations --arms nogate gate gate_r2 gate_r3
"""
import argparse, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_gate import TRANSFER_MSG  # noqa: E402

GATES = ["G1_AUTH_FIRST", "G2_CONFIRM_WRITE", "G3_SINGLE_USER", "G4_TRANSFER_MSG"]


def sim_features(s):
    msgs = s.get("messages") or []
    rid = {m["id"]: m for m in msgs if m.get("role") == "tool" and m.get("id")}
    f = {"denies": Counter(), "n_assist": 0, "n_msgs": len(msgs),
         "transfer_attempt": False, "transfer_exec": False, "notice": False}
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        f["n_assist"] += 1
        mc = m.get("content")
        if isinstance(mc, str) and TRANSFER_MSG in mc:
            f["notice"] = True
        for tc in (m.get("tool_calls") or []):
            res = rid.get(tc.get("id"))
            content = (res or {}).get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            if tc.get("name") == "transfer_to_human_agents":
                f["transfer_attempt"] = True
                if "POLICY GATE" not in content and res is not None \
                        and not res.get("error"):
                    f["transfer_exec"] = True
            if "POLICY GATE" in content:
                for g in GATES:
                    if g in content:
                        f["denies"][g] += 1
    r = (s.get("reward_info") or {}).get("reward")
    f["ok"] = None if r is None else (r >= 1)
    f["term"] = s.get("termination_reason") or "?"
    return f


def load_arm(simdir, arm):
    sims = json.load(open(f"{simdir}/{arm}/results.json"))["simulations"]
    per_task = {}
    for s in sims:
        f = sim_features(s)
        if f["ok"] is None:
            continue
        per_task.setdefault(str(s["task_id"]), []).append(f)
    return per_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    arms = {arm: load_arm(a.simdir, arm) for arm in a.arms}

    for arm, pt in arms.items():
        cs = Counter(sum(x["ok"] for x in v) for v in pt.values() if len(v) == 4)
        flaky = {t: v for t, v in pt.items() if len(v) == 4 and 1 <= sum(x["ok"] for x in v) <= 3}
        print(f"\n===== {arm}: tasks(4-trial)={sum(1 for v in pt.values() if len(v)==4)} "
              f"c-dist={dict(sorted(cs.items()))} flaky={len(flaky)}")

        # A2: within-task deny vs fail (flaky만 — 난이도 통제)
        cell = Counter()
        per_gate = {g: Counter() for g in GATES}
        for t, v in flaky.items():
            for x in v:
                has = sum(x["denies"].values()) > 0
                cell[(has, x["ok"])] += 1
                for g in GATES:
                    if x["denies"][g]:
                        per_gate[g][x["ok"]] += 1
        dp, df = cell[(True, True)], cell[(True, False)]
        np_, nf = cell[(False, True)], cell[(False, False)]
        fr_d = df / max(dp + df, 1)
        fr_n = nf / max(np_ + nf, 1)
        print(f"  [A2 flaky-내 deny↔fail] deny-trial fail율 {fr_d:.0%} ({df}/{dp+df}) vs "
              f"nodeny {fr_n:.0%} ({nf}/{np_+nf})  Δ={fr_d-fr_n:+.0%}")
        for g in GATES:
            c = per_gate[g]
            if c[True] + c[False]:
                print(f"    {g}: trial pass/fail = {c[True]}/{c[False]}")

        # A3/A4: 길이·종결
        pl = [x["n_assist"] for v in flaky.values() for x in v if x["ok"]]
        fl = [x["n_assist"] for v in flaky.values() for x in v if not x["ok"]]
        if pl and fl:
            print(f"  [A3] assistant 턴수: pass-trial 평균 {sum(pl)/len(pl):.1f} vs "
                  f"fail-trial {sum(fl)/len(fl):.1f}")
        terms = Counter(x["term"] for v in flaky.values() for x in v if not x["ok"])
        print(f"  [A4] fail-trial 종결사유: {dict(terms.most_common(4))}")

        # A6: transfer 깔때기
        att = sum(1 for v in pt.values() for x in v if x["transfer_attempt"])
        ex = sum(1 for v in pt.values() for x in v if x["transfer_exec"])
        no = sum(1 for v in pt.values() for x in v if x["transfer_exec"] and x["notice"])
        exok = sum(1 for v in pt.values() for x in v if x["transfer_exec"] and x["ok"])
        print(f"  [A6 transfer] 시도-sim {att} → 실행 {ex} → notice동반 {no} → pass {exok}")

    # A5: 태스크-플립 (마지막 두 arm 비교: arms[-2] -> arms[-1])
    if len(a.arms) >= 2:
        x, y = a.arms[-2], a.arms[-1]
        px, py = arms[x], arms[y]
        common = [t for t in px if t in py and len(px[t]) == 4 and len(py[t]) == 4]
        ups, downs = [], []
        for t in common:
            cx, cy = sum(v["ok"] for v in px[t]), sum(v["ok"] for v in py[t])
            if cy > cx:
                ups.append((t, cx, cy))
            elif cy < cx:
                downs.append((t, cx, cy))
        print(f"\n===== [A5 task-flip {x} → {y}] up={len(ups)} down={len(downs)}")
        for tag, lst in (("UP", ups), ("DOWN", downs)):
            for t, cx, cy in sorted(lst, key=lambda z: -(abs(z[2] - z[1])))[:10]:
                gx = Counter()
                gy = Counter()
                for v in px[t]:
                    gx.update(v["denies"])
                for v in py[t]:
                    gy.update(v["denies"])
                tr = any(v["transfer_attempt"] for v in px[t] + py[t])
                print(f"  {tag} task {t}: c {cx}→{cy} | denies {dict(gx)}→{dict(gy)} "
                      f"| transfer-task={tr}")


if __name__ == "__main__":
    main()
