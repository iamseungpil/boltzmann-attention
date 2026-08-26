# -*- coding: utf-8 -*-
r"""x545 - 중복 실행: **무엇을 파나**(§1)와 **얼마를 사나**(§2). 런 0회·모델 0회.

## 왜 (2026-08-26 · x544 판정 직후)

x544 가 074 에서 확정했다: 궤적은 **중복 실행만 빼면 reward 1.0** 이다
(A_full False ↔ B_nodup **True** ↔ C_one False ↔ N_reads False). env 의 크레딧 도구는
멱등이 아니고(부를 때마다 잔액에 더한다) 거래 id 는 인자에서 결정돼 두 번째가 첫 번째를
덮으므로, 이중 청구는 **도구 출력에도 궤적에도 안 보이고** DB 해시로만 드러난다.

우리 층에는 **읽기** 중복 가드만 있다(`[DUPLICATE-READ]`·`[NEAR-DUPLICATE-READ]`).
쓰기 가드를 짓기 전에 [[70]] 의 두 물음을 **먼저 수치로** 답한다([[62]]①):

    §1 무엇을 파나 — gold 가 **같은 (도구·인자) 변이를 두 번** 요구하는 태스크가 있나?
                     있으면 fail-closed 차단은 그 태스크를 죽인다 ⇒ 가드는 '알려주기' 형태여야 한다.
    §2 얼마를 사나 — 영속된 코퍼스에서 **성공한 변이가 중복된** sim 이 몇 건인가?
                     그 sim 들의 reward 분포가 이 레버의 상한이다.

⚠**세대를 뭉개지 않는다**([[74]]-b): §2 는 **런 태그별로** 인쇄한다. 태그를 가로질러 더한
  수는 레버 구성이 다른 런들을 섞은 것이라 원인 진술에 못 쓴다.

## 술어는 x544 의 것을 **그대로 import** 한다([[67]] 사본 금지)

`scan()` = 같은 `t2_forensic.mut_key(도구, 인자)` 가 **앞서 성공한 적이 있으면** 뒤엣것을
중복으로 센다. 성공 판정은 그 호출의 결과 메시지(`id` 로 짝지음)가 오류가 아닌지뿐이다.
도메인 낱말 0 · 태스크 id 0 · gold 미접촉([[23]]).

§1 의 gold 는 각 sim 의 `reward_info.action_checks[].action` — **벤치 자신이 실은 gold 액션
목록**이다(우리가 다시 판정하지 않는다·`action_diff` 독스트링과 같은 권위).

실행: PYTHONIOENCODING=utf-8 py -3 x545_duplicate_mutations.py
"""
import collections
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F                                       # noqa: E402
from x544_dup_credit_regrade import scan                      # noqa: E402

OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                   "x545_duplicate_mutations_2026_08_26.json")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        d = json.load(fh)
    if isinstance(d, dict):
        d = d.get("simulations") or d.get("results") or []
    return d if isinstance(d, list) else []


def gold_mut_keys(sim, mut):
    """이 sim 의 gold 변이 키 목록 — 벤치가 실은 액션 목록에서 읽는다."""
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        act = (ck or {}).get("action") or {}
        nm = str(act.get("name") or "")
        args = act.get("arguments") or {}
        if not nm:
            continue
        target = str(F.inner_name(args) or nm)
        if target not in mut:
            continue
        out.append(F.mut_key(nm, args))
    return out


def main():
    mut = F.mutating_tools("banking_knowledge")
    files = F.all_result_files()
    print("=" * 98)
    print("x545 — 중복 실행 · 결과 파일 %d 개 · 변이 도구 %d 종" % (len(files), len(mut)))
    print("=" * 98)

    gold_multi = {}          # task -> 최대 중복도
    gold_seen_tasks = set()
    per_tag = collections.OrderedDict()
    per_task = collections.Counter()
    per_task_sims = collections.Counter()
    dup_reward = collections.Counter()
    examples = []

    for p in sorted(files):
        tag = F.tag_of_file(p)
        try:
            sims = load(p)
        except Exception as e:
            print("  [%s] 못 읽음: %r" % (tag, e))
            continue
        for s in sims:
            task = str(s.get("task_id") or "?")
            per_task_sims[task] += 1
            # §1 gold 쪽
            gk = gold_mut_keys(s, mut)
            if gk:
                gold_seen_tasks.add(task)
                c = collections.Counter(gk).most_common(1)[0][1]
                gold_multi[task] = max(gold_multi.get(task, 0), c)
            # §2 궤적 쪽
            try:
                dups, _reads = scan(s, mut)
            except Exception:
                continue
            row = per_tag.setdefault(tag, {"sims": 0, "dup_sims": 0, "dup_calls": 0})
            row["sims"] += 1
            if dups:
                row["dup_sims"] += 1
                row["dup_calls"] += len(dups)
                per_task[task] += 1
                rw = (s.get("reward_info") or {}).get("reward")
                dup_reward[str(rw)] += 1
                if len(examples) < 25:
                    examples.append({"tag": tag, "task": task, "dups": len(dups),
                                     "reward": rw,
                                     "tools": sorted({t for _, _, t in dups})})

    print("\n§1 무엇을 파나 — gold 가 같은 변이를 두 번 요구하는 태스크")
    print("  gold 변이를 가진 태스크 %d 종" % len(gold_seen_tasks))
    twice = {t: c for t, c in gold_multi.items() if c >= 2}
    if twice:
        print("  ⚠중복도 2 이상: %s" % twice)
        print("  ⇒ fail-closed 차단은 이 태스크들을 죽인다. 가드는 **알려주기** 형태여야 한다.")
    else:
        print("  ★없다 (전 태스크에서 최대 중복도 1)")
        print("  ⇒ 같은 (도구·인자) 변이의 반복은 gold 가 **한 번도** 요구하지 않는다.")

    print("\n§2 얼마를 사나 — 런 태그별 (세대를 뭉개지 않는다)")
    hit = [(t, r) for t, r in per_tag.items() if r["dup_sims"]]
    print("  중복이 관측된 런 %d / %d" % (len(hit), len(per_tag)))
    for t, r in hit:
        print("     %-42s sim %3d 중 **%2d** (중복 호출 %d)"
              % (t[:42], r["sims"], r["dup_sims"], r["dup_calls"]))

    print("\n  태스크별 중복 sim 수 (전 런 합 — 빈도 파악용이지 원인 진술용 아님):")
    for t, n in per_task.most_common(15):
        print("     %-10s %3d / %3d sim" % (t, n, per_task_sims[t]))

    print("\n  중복이 있던 sim 의 reward 분포: %s" % dict(dup_reward))
    print("  ⚠reward>0 인 중복 sim 이 있으면 중복이 **항상** 치명적이지는 않다는 뜻이다.")

    print("\n  실물 몇 개:")
    for e in examples[:12]:
        print("     %-38s %-10s dup=%-2d reward=%-5s %s"
              % (e["tag"][:38], e["task"], e["dups"], e["reward"],
                 ",".join(e["tools"])[:44]))

    rep = {"files": len(files), "gold_multiplicity": gold_multi,
           "gold_twice": twice, "per_tag": per_tag, "per_task": dict(per_task),
           "per_task_sims": dict(per_task_sims),
           "dup_reward": dict(dup_reward), "examples": examples}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print("\n산출: %s" % os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
