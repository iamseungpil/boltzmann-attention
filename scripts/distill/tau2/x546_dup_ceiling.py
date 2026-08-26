# -*- coding: utf-8 -*-
r"""x546 - 쓰기 중복 가드의 **상한**: 중복만 빼면 몇 sim 이 1.0 으로 뒤집히나. 모델 0회·GPU 0.

## 왜 (2026-08-26 · x544 → x545 다음)

x544: 074 는 **중복만 빼면 reward 1.0** (A_full False ↔ B_nodup True ↔ C_one False ↔ N_reads False).
x545: 영속 13,648 sim 중 **685** 이 성공한 변이를 중복 실행했고, 그중 **89 는 그래도 만점**이다
      ⇒ 중복이 항상 치명적이지 않다 ⇒ **상한은 685 가 아니다**.

이 파일이 그 상한을 센다: *중복이 있고 reward 0 인 sim* 을 **중복만 빼고 다시 채점**해
1.0 으로 뒤집히는 수를 세는 것. 이것이 쓰기 가드가 살 수 있는 **최대치**다
(가드가 완벽히 작동한다고 가정한 값이므로 상한이지 기대치가 아니다).

## 공정성 게이트 ([[62]] 2b — 어기면 판정하지 않는다)

`A_full`(전량 재생)이 **기록된 reward 를 재현하지 못하면 그 sim 은 세지 않는다**.
옛 런은 검색 변종이 다를 수 있고(러너 기본값은 `openai_embeddings`, 우리 런은 `alltools`),
그러면 재생 환경이 런과 달라 어떤 숫자도 의미가 없다. 재현 실패 수를 **따로 인쇄**한다.

## 세대

기본 표적은 `t73\d\d` 세대뿐이다([[74]]-b 세대 뭉개기 금지). 결과는 **런 태그별**로 남긴다.

술어(`scan`)·재채점(`grade`)·가지치기(`prune`)는 전부 x544 에서 **import** 한다([[67]]).

실행 (리모트 · 정본 tau2 로만):
    R=/home/woori/workspace_common/boltzmann-attention-pi
    cd /home/woori/scratch/tau2-bench && PYTHONPATH=src:$R/scripts/distill/tau2 \
      PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python \
      $R/scripts/distill/tau2/x546_dup_ceiling.py
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F                                        # noqa: E402
from x544_dup_credit_regrade import scan, prune, grade         # noqa: E402

OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                   "x546_dup_ceiling_2026_08_26.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=r"t73\d\d", help="런 태그 정규식")
    ap.add_argument("--limit", type=int, default=0, help="0 = 전부")
    a = ap.parse_args(argv)

    from tau2.domains.banking_knowledge.environment import get_tasks
    tasks = {t.id: t for t in get_tasks()}
    mut = F.mutating_tools("banking_knowledge")
    pat = re.compile(a.tags)

    cands = []
    for p in sorted(F.all_result_files()):
        tag = F.tag_of_file(p)
        if not pat.search(tag):
            continue
        try:
            sims = F.sims(tag, ".results.json.gz")
        except Exception:
            continue
        for s in sims:
            rw = (s.get("reward_info") or {}).get("reward")
            if rw is None or float(rw) > 0:
                continue
            try:
                dups, _ = scan(s, mut)
            except Exception:
                continue
            if dups:
                cands.append((tag, s, dups))
    if a.limit:
        cands = cands[:a.limit]

    print("=" * 98)
    print("x546 상한 — 표적 %d sim (중복 있고 reward 0 · 태그 /%s/)" % (len(cands), a.tags))
    print("=" * 98, flush=True)

    flip, stay, unfair, err = [], [], [], []
    per_tag = collections.Counter()
    per_task = collections.Counter()
    for n, (tag, s, dups) in enumerate(cands, 1):
        task = str(s.get("task_id") or "?")
        t = tasks.get(task)
        if t is None:
            err.append((tag, task, "태스크 선언 없음"))
            continue
        try:
            a_match, a_rw = grade(s, t)
        except Exception as e:
            err.append((tag, task, "A_full 재생 실패: %r" % (e,)))
            print("  [%3d/%d] %-34s %-9s A_full 재생 실패 — 제외" % (n, len(cands), tag[:34], task),
                  flush=True)
            continue
        if float(a_rw or 0) != 0.0:
            unfair.append((tag, task, a_rw))
            print("  [%3d/%d] %-34s %-9s ⛔A_full=%s ≠ 기록 0.0 — **판정 제외**([[62]] 2b)"
                  % (n, len(cands), tag[:34], task, a_rw), flush=True)
            continue
        try:
            b_match, b_rw = grade(prune(s, dups), t)
        except Exception as e:
            err.append((tag, task, "B_nodup 재생 실패: %r" % (e,)))
            continue
        row = {"tag": tag, "task": task, "dups": len(dups),
               "tools": sorted({x for _, _, x in dups}), "b_reward": b_rw}
        if float(b_rw or 0) > 0:
            flip.append(row)
            per_tag[tag] += 1
            per_task[task] += 1
            print("  [%3d/%d] %-34s %-9s ★뒤집힘 0.0 → %s  (중복 %d · %s)"
                  % (n, len(cands), tag[:34], task, b_rw, len(dups),
                     ",".join(row["tools"])[:40]), flush=True)
        else:
            stay.append(row)
            print("  [%3d/%d] %-34s %-9s  그대로 0.0 (중복 %d)"
                  % (n, len(cands), tag[:34], task, len(dups)), flush=True)

    judged = len(flip) + len(stay)
    print("\n" + "=" * 98)
    print("판정 가능 %d · **뒤집힘 %d** · 그대로 %d | 공정성 탈락 %d · 오류 %d"
          % (judged, len(flip), len(stay), len(unfair), len(err)))
    if judged:
        print("⇒ 이 세대에서 쓰기 중복 가드의 **상한** = 판정분의 %.0f%% (%d/%d)"
              % (100.0 * len(flip) / judged, len(flip), judged))
    print("뒤집힌 태스크: %s" % dict(per_task.most_common()))
    print("뒤집힌 런:     %s" % dict(per_tag.most_common(12)))
    print("⚠상한이지 기대치가 아니다 — 가드가 **완벽히** 막았을 때의 값이다([[70]]).")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"pattern": a.tags, "targets": len(cands), "judged": judged,
                   "flip": flip, "stay": stay,
                   "unfair": [{"tag": t, "task": k, "a_reward": r} for t, k, r in unfair],
                   "errors": [{"tag": t, "task": k, "why": w} for t, k, w in err],
                   "per_task": dict(per_task), "per_tag": dict(per_tag)},
                  fh, ensure_ascii=False, indent=2)
    print("산출: %s" % os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
