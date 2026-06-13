#!/usr/bin/env python
"""ⓟ1 폐기의 진짜 원인 전수 궤적조사 (2026-06-13 사용자 발주, zero-GPU).

발견: user-sim temp 0.0(ut0)로도 pass^4=0.018 (r3 temp0.7의 0.054보다 *낮음*).
질문: 결정론화했는데 왜 4-trial이 갈리나? 분산의 진짜 원천은?

★핵심 분리 (trial 간 동일성 분해):
  - seq 다름 ∧ reward 다름 = 에이전트 행동이 trial마다 다름 (생성 비결정 — temp0이어도
    vLLM/하드웨어 비결정성·동시성)
  - **seq 같음 ∧ reward 다름 = 같은 행동인데 채점이 갈림** (judge NL-assertion 비결정 또는
    DB-state 평가 비결정) ← 이게 사실이면 pass^4 정체는 '게이트/모델'이 아니라 '채점기 잡음'
  - seq 같음 ∧ reward 같음 = 완전 결정론 (이상적)
또: arm-교차 실패상관(같은 태스크가 r3·ut0서 같이 실패하면 난이도-결정·user무관),
    종결사유 분포, flaky 내 위 3분해.
Usage: t2_p1_autopsy.py --simdir .../simulations --arms retail_7b_gate_r3 retail_7b_gate_r3_ut0
"""
import argparse, json
from collections import Counter


def trial_feat(s):
    msgs = s.get("messages") or []
    seq = []
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                # 도구명 + 핵심 인자 해시(순서·인자까지 동일성) — name만은 너무 느슨
                args = tc.get("arguments")
                if isinstance(args, (dict, list)):
                    args = json.dumps(args, sort_keys=True)
                seq.append(f"{tc.get('name')}({str(args)[:60]})")
    r = (s.get("reward_info") or {}).get("reward")
    return {"seq": tuple(seq), "name_seq": tuple(x.split("(")[0] for x in seq),
            "ok": None if r is None else (r >= 1), "reward": r,
            "term": s.get("termination_reason") or "?", "n": len(seq)}


def load(simdir, arm):
    sims = json.load(open(f"{simdir}/{arm}/results.json"))["simulations"]
    pt = {}
    for s in sims:
        f = trial_feat(s)
        if f["ok"] is None:
            continue
        pt.setdefault(str(s["task_id"]), []).append(f)
    return pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    arms = {arm: load(a.simdir, arm) for arm in a.arms}

    for arm, pt in arms.items():
        full = {t: v for t, v in pt.items() if len(v) == 4}
        cs = Counter(sum(x["ok"] for x in v) for v in full.values())
        flaky = {t: v for t, v in full.items() if 1 <= sum(x["ok"] for x in v) <= 3}
        print(f"\n===== {arm}: 4-trial tasks={len(full)} "
              f"c-dist={dict(sorted(cs.items()))} flaky={len(flaky)}")

        # ★trial 동일성 3분해 (전 4-trial 태스크)
        seq_same = name_same = rew_same = 0
        for v in full.values():
            seqs = {x["seq"] for x in v}
            names = {x["name_seq"] for x in v}
            rews = {x["reward"] for x in v}
            seq_same += (len(seqs) == 1)
            name_same += (len(names) == 1)
            rew_same += (len(rews) == 1)
        n = len(full)
        print(f"  [동일성] 전4trial 동일: 인자포함-seq {seq_same}/{n} ({seq_same/n:.0%}) · "
              f"도구명-seq {name_same}/{n} ({name_same/n:.0%}) · reward {rew_same}/{n} ({rew_same/n:.0%})")

        # ★flaky(reward 갈림) 분해: seq도 갈리나 vs seq 같은데 reward만?
        seq_diff_rew_diff = seq_same_rew_diff = 0
        for v in flaky.values():
            seqs = {x["seq"] for x in v}
            if len(seqs) == 1:
                seq_same_rew_diff += 1   # ★같은 행동 다른 채점 = 채점기 잡음
            else:
                seq_diff_rew_diff += 1   # 다른 행동 = 생성 비결정
        nf = max(len(flaky), 1)
        print(f"  [flaky 원천] 행동-다름(생성비결정) {seq_diff_rew_diff}/{len(flaky)} "
              f"({seq_diff_rew_diff/nf:.0%}) vs **같은행동-다른채점(채점잡음) {seq_same_rew_diff}/{len(flaky)} "
              f"({seq_same_rew_diff/nf:.0%})**")

        # 종결사유 (전체 fail trial)
        terms = Counter(x["term"] for v in full.values() for x in v if not x["ok"])
        print(f"  [종결] fail-trial: {dict(terms.most_common(5))}")

    # arm-교차 실패상관 (마지막 두 arm)
    if len(a.arms) >= 2:
        x, y = a.arms[0], a.arms[1]
        px, py = arms[x], arms[y]
        common = [t for t in px if t in py and len(px[t]) == 4 and len(py[t]) == 4]
        # c 상관 + 둘 다 0 / 둘 다 4 / 갈림
        both0 = both4 = flip = 0
        for t in common:
            cx = sum(v["ok"] for v in px[t]); cy = sum(v["ok"] for v in py[t])
            if cx == 0 and cy == 0:
                both0 += 1
            elif cx == 4 and cy == 4:
                both4 += 1
            elif (cx >= 2) != (cy >= 2):
                flip += 1
        nc = len(common)
        print(f"\n===== [arm-교차 {x} vs {y}] common={nc} "
              f"both-0(공통실패) {both0} ({both0/nc:.0%}) · both-4(공통성공) {both4} · "
              f"과반-flip {flip} ({flip/nc:.0%})")
        print(f"  해석: both-0 높음 = 난이도-결정(user 무관·agent 능력 바닥) / "
              f"flip 높음 = arm(=user temp)이 결과 좌우")


if __name__ == "__main__":
    main()
