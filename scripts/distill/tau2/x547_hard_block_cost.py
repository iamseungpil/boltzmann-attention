# -*- coding: utf-8 -*-
r"""x547 - **하드 차단이 정말 무엇을 죽이나**. 집계로 말하지 말고 태워서 확정한다 (2026-08-26).

## 사용자 지적 (2026-08-26)

*"하드 차단이 안되는 이유를 격리나 별도 조합 실험으로 원인을 밝혀라. 뭐가 하드 차단이 안된다는
건가? 가장 간단한 케이스부터 만들어서 진짜로 안되는게 뭔지 확정하라."*

맞다. 앞선 진술(*"하드 차단은 못 쓴다"*)은 **gold 중복도 집계**에서 바로 나온 설계 결론이었고
시험한 적이 없다([[08]] 위반). 여기서 태운다.

## 가장 간단한 케이스 — 선언 직독 (x547 §0)

`task_051`(reward_basis=**DB**)의 gold 20 액션 중 [2]와 [17]이 **완전히 같은 호출**이다:

    [2]  submit_credit_limit_increase_request_7392(cc_5e4c1a83b0_bronze, 5e4c1a83b0)
    [12] deny_credit_limit_increase_5848(...)         <- 사이에 **거절**이 있고
    [16] pay_credit_card_from_checking_9182(...)      <- 상환이 있고
    [17] submit_credit_limit_increase_request_7392(...)  <- **같은 인자로 재제출**
    [19] approve_credit_limit_increase_5847(...)

⇒ *"같은 (도구·인자)가 이미 성공했으면 막는다"* 는 술어는 [17]을 막는다. **그러나 그것이
실제로 점수를 깎는지는 재생으로 확인해야 한다** — 이 파일이 그 확인이다.

## 팔 ([[57]])

sim 하나마다:

    A_full        전량 재생        <- **기록된 reward 를 재현해야** 한다. 아니면 그 sim 은 판정 제외([[62]] 2b)
    B_block_all   중복 **전부** 제거  <- 소박한 하드 차단이 하는 일
    C_block_pure  **순수 반복만** 제거 <- 정제된 술어: 앞선 성공 이후 **다른 성공 변이가 하나도
                                       없었을 때만** 반복으로 본다 (051 의 [17]은 사이에 거절·
                                       상환이 있으므로 순수 반복이 **아니다**)

표적은 두 갈래고 **둘 다 봐야** 결론이 난다:

    비용  reward **1.0** 인데 중복이 있는 sim  -> 차단이 점수를 **깎나**
    이득  reward **0.0** 인데 중복이 있는 sim  -> 차단이 점수를 **살리나**(x546 이 B 로 이미 8건)

⇒ 판정표: C 가 이득을 **보존**하면서 비용이 **0** 이면 정제된 하드 차단이 성립한다.
   C 도 비용을 내면 그때야 *"차단은 못 쓴다"* 가 측정된 진술이 된다.

⚠재생이 답하는 것은 **실행을 뺐을 때의 점수**뿐이다(G1/G2). *차단당한 모델이 그 다음에
  무엇을 하는가*(G3)는 재생이 답할 수 없다 — 격리나 런이 필요하다(`x515` §경계).

실행 (리모트 · 정본 tau2):
    R=/home/woori/workspace_common/boltzmann-attention-pi
    cd /home/woori/scratch/tau2-bench && PYTHONPATH=src:$R/scripts/distill/tau2 \
      PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python \
      $R/scripts/distill/tau2/x547_hard_block_cost.py
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F                                           # noqa: E402
from x544_dup_credit_regrade import prune, grade, tool_result_ok  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                   "x547_hard_block_cost_2026_08_26.json")


def dup_detail(sim, mut):
    """성공한 변이의 반복 목록. 각 항목에 **순수 반복인가**(사이에 다른 성공 변이 0)를 단다.

    술어는 전부 닫혀 있다 — 이름 동등성 · 인자 접기(`mut_key`) · 메시지 순서뿐. 도메인 낱말 0."""
    ms = sim.get("messages") or []
    last_at, seq, out = {}, [], []
    for i, m in enumerate(ms):
        if str(m.get("role")) != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            a = F.argsof(tc)
            nm = str(F.nameof(tc))
            tcid = str((tc.get("id") if isinstance(tc, dict) else "") or "")
            target = str(F.inner_name(a) or nm)
            if target not in mut:
                continue
            if not tool_result_ok(ms, i, tcid):
                continue
            key = F.mut_key(nm, a)
            if key in last_at:
                prev = last_at[key]
                between = [k for (j, k) in seq if prev < j < i and k != key]
                out.append({"msg": i, "id": tcid, "tool": target,
                            "prev": prev, "pure": not between,
                            "between": len(between)})
            last_at[key] = i
            seq.append((i, key))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=r"t73\d\d")
    ap.add_argument("--tasks", default="", help="쉼표 목록이면 태그 무관하게 이 태스크도 포함")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    from tau2.domains.banking_knowledge.environment import get_tasks
    tasks = {t.id: t for t in get_tasks()}
    mut = F.mutating_tools("banking_knowledge")
    pat = re.compile(a.tags)
    extra = {t.strip() for t in a.tasks.split(",") if t.strip()}

    cands = []
    for p in sorted(F.all_result_files()):
        tag = F.tag_of_file(p)
        try:
            sims = F.sims(tag, ".results.json.gz")
        except Exception:
            continue
        for s in sims:
            task = str(s.get("task_id") or "?")
            if not (pat.search(tag) or task in extra):
                continue
            rw = (s.get("reward_info") or {}).get("reward")
            if rw is None:
                continue
            try:
                dd = dup_detail(s, mut)
            except Exception:
                continue
            if dd:
                cands.append((tag, s, dd, float(rw)))
    cost = [c for c in cands if c[3] > 0]
    gain = [c for c in cands if c[3] == 0]
    if a.limit:
        cost, gain = cost[:a.limit], gain[:a.limit]

    print("=" * 100)
    print("x547 — 비용(reward 1.0 인데 중복 %d sim) · 이득(reward 0 인데 중복 %d sim)"
          % (len(cost), len(gain)))
    print("=" * 100, flush=True)

    rows, err = [], []
    for label, group in (("비용", cost), ("이득", gain)):
        print("\n── %s 갈래 %d sim" % (label, len(group)), flush=True)
        for n, (tag, s, dd, rw) in enumerate(group, 1):
            task = str(s.get("task_id") or "?")
            t = tasks.get(task)
            if t is None:
                err.append({"tag": tag, "task": task, "why": "선언 없음"})
                continue
            try:
                _m, a_rw = grade(s, t)
            except Exception as e:
                err.append({"tag": tag, "task": task, "why": "A_full: %r" % (e,)})
                continue
            if float(a_rw or 0) != rw:
                print("  [%3d] %-32s %-9s ⛔A_full=%s ≠ 기록 %s — 판정 제외"
                      % (n, tag[:32], task, a_rw, rw), flush=True)
                err.append({"tag": tag, "task": task, "why": "A_full 불일치 %s" % a_rw})
                continue
            alld = [(d["msg"], d["id"], d["tool"]) for d in dd]
            pured = [(d["msg"], d["id"], d["tool"]) for d in dd if d["pure"]]
            try:
                _mb, b_rw = grade(prune(s, alld), t)
                c_rw = b_rw if len(pured) == len(alld) else (
                    rw if not pured else grade(prune(s, pured), t)[1])
            except Exception as e:
                err.append({"tag": tag, "task": task, "why": "B/C: %r" % (e,)})
                continue
            row = {"group": label, "tag": tag, "task": task, "a": rw,
                   "b_all": b_rw, "c_pure": c_rw,
                   "dups": len(alld), "pure": len(pured),
                   "tools": sorted({x for _, _, x in alld})}
            rows.append(row)
            flag = ""
            if label == "비용" and float(b_rw or 0) < rw:
                flag = "  ★B 가 점수를 **깎았다**"
                if float(c_rw or 0) >= rw:
                    flag += " / C 는 지켰다"
            if label == "이득" and float(b_rw or 0) > 0:
                flag = "  ★B 가 **살렸다**"
                if float(c_rw or 0) <= 0:
                    flag += " / C 는 못 살렸다"
            print("  [%3d] %-32s %-9s A=%s B_all=%s C_pure=%s (중복 %d · 순수 %d)%s"
                  % (n, tag[:32], task, rw, b_rw, c_rw, len(alld), len(pured), flag),
                  flush=True)

    def tally(g):
        rs = [r for r in rows if r["group"] == g]
        return rs

    cst, gn = tally("비용"), tally("이득")
    b_drop = [r for r in cst if float(r["b_all"] or 0) < r["a"]]
    c_drop = [r for r in cst if float(r["c_pure"] or 0) < r["a"]]
    b_save = [r for r in gn if float(r["b_all"] or 0) > 0]
    c_save = [r for r in gn if float(r["c_pure"] or 0) > 0]

    print("\n" + "=" * 100)
    print("판정 (오류·제외 %d)" % len(err))
    print("  비용 갈래 %d sim | 소박한 차단이 깎은 것 **%d** · 정제된 차단이 깎은 것 **%d**"
          % (len(cst), len(b_drop), len(c_drop)))
    print("  이득 갈래 %d sim | 소박한 차단이 살린 것 **%d** · 정제된 차단이 살린 것 **%d**"
          % (len(gn), len(b_save), len(c_save)))
    if b_drop:
        print("  깎인 것(B): %s" % [(r["task"], r["tag"][:26], r["tools"]) for r in b_drop][:8])
    if c_drop:
        print("  깎인 것(C): %s" % [(r["task"], r["tag"][:26], r["tools"]) for r in c_drop][:8])
    print("  ⇒ C 가 이득을 보존하며 비용 0 이면 **정제된 하드 차단이 성립**한다.")
    print("  ⚠재생은 *실행을 뺀 점수*만 답한다 — 차단당한 모델의 다음 수(G3)는 격리·런의 몫이다.")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "errors": err,
                   "summary": {"cost_sims": len(cst), "b_drop": len(b_drop),
                               "c_drop": len(c_drop), "gain_sims": len(gn),
                               "b_save": len(b_save), "c_save": len(c_save)}},
                  fh, ensure_ascii=False, indent=2)
    print("산출: %s" % os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
