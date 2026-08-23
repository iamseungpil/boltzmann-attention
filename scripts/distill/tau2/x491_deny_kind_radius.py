# -*- coding: utf-8 -*-
"""A-7⑵ 의 반경: 실패한 write 를 성공으로 세던 술어가 무엇을 오분류했나 (2026-08-23).

`t2_forensic.deny_kind` 는 env 거절을 `Error:` 접두로만 알아봤다. 그런데 이 환경은
`Failed to …` 로도 거절한다. 그 본문을 성공으로 세면 **그 호출이 MATCHED 가 되고, 앞선
성공이 DUP 으로 재분류**된다 — 079 의 DUP 주장이 그렇게 태어났다(마스터 §2.2 축 H).

이 프로브는 구판 술어(`Error:` 만)와 신판(`Failed to ` 포함)을 **같은 코퍼스에 나란히**
돌려 분류가 갈리는 sim 을 전수로 센다. 판정은 [[69]] 기준이다: reward 는 궤적 재실행 후
**DB 해시 비교**이므로, 실패한 write 는 상태를 안 바꿔 해시에 안 남는다 ⇒ BLOCKED 가 맞고
MATCHED/DUP 은 틀리다. **성적은 벤치가 매기므로 이 변경은 점수를 바꾸지 않는다** —
바뀌는 것은 우리가 실패를 무엇이라 불렀는가뿐이다([[25]] 우리 계기 100% 정답 의무).

출력: `x491_deny_kind_radius.json`
"""

import collections
import glob
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F          # noqa: E402

NEW = tuple(F.ENV_FAIL_PREFIX)
OLD = ("Error:",)
MUT = F.mutating_tools()
KEYS = ("blocked", "missing", "wrongarg", "extra", "matched", "dup")


def snap(sim):
    d = F.mutation_diff(sim, MUT)
    return {k: (len(v) if isinstance(v, (list, tuple)) else v)
            for k, v in d.items() if k in KEYS}


def main():
    pats = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                        "sim_results", "bank_t73*_2026*.results.json.gz")
    files = sorted(glob.glob(pats))
    bodies = collections.Counter()
    changed, nsim = [], 0
    for p in files:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        run = os.path.basename(p).split(".")[0]
        for s in (d.get("simulations") or []):
            nsim += 1
            for m in (s.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                b = " ".join(str(m.get("content") or "").split()).lstrip()
                for pre in NEW:
                    if pre not in OLD and b.startswith(pre):
                        bodies[b[:70]] += 1
            F.ENV_FAIL_PREFIX = OLD
            a = snap(s)
            F.ENV_FAIL_PREFIX = NEW
            b2 = snap(s)
            if a != b2:
                changed.append({"run": run, "task": s.get("task_id"),
                                "reward": (s.get("reward_info") or {}).get("reward"),
                                "old": a, "new": b2})
    F.ENV_FAIL_PREFIX = NEW

    by_task = collections.Counter(c["task"] for c in changed)
    dup_gone = sum(1 for c in changed if c["old"].get("dup", 0) > c["new"].get("dup", 0))
    matched_gone = sum(c["old"].get("matched", 0) - c["new"].get("matched", 0)
                       for c in changed)
    out = {"n_sim": nsim, "n_files": len(files),
           "n_changed_sim": len(changed),
           "n_sim_losing_dup": dup_gone,
           "matched_reclassified_as_blocked": matched_gone,
           "shapes": dict(bodies.most_common()),
           "by_task": dict(by_task.most_common()),
           "changed": changed}
    with io.open(os.path.join(HERE, "x491_deny_kind_radius.json"), "w",
                 encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("코퍼스 %d파일 · %d sim" % (len(files), nsim))
    print("`Failed to ` 본문 %d건 (형상 %d종)" % (sum(bodies.values()), len(bodies)))
    print("분류가 갈린 sim **%d건** · 그중 DUP 이 사라진 sim **%d건**" % (len(changed), dup_gone))
    print("MATCHED → BLOCKED 재분류 **%d건**" % matched_gone)
    print("\n태스크별 갈림:")
    for t, n in by_task.most_common():
        print("  %-10s %d" % (t, n))
    print("\n형상:")
    for b, n in bodies.most_common(5):
        print("  %4d  %s" % (n, b))
    print("\n★reward 는 벤치가 매긴다 — 이 변경은 성적을 바꾸지 않는다.")
    print("★영향은 마스터의 DUP/MATCHED 서술이다: 위 태스크의 그 칸은 다시 읽어야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
