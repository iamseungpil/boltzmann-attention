# -*- coding: utf-8 -*-
"""97-태스크 스윕의 **전체 통계** — pass가 얼마인가.

이 스윕은 한 번에 돌지 않았다: 전반부(`*_main_20260806`·사이드카 없음)를 중단·영속화하고 잔여를
사이드카 ON으로 재발사했으며(`*_main_20260806b`), 그 사이 빈-예약 버그로 뜬 런의 완료분 4 sim이
따로 있다. 그래서 단순 평균을 내면 **같은 태스크가 두 번 세어진다**. 여기서는 arm 구조를 먼저 찍고,
태스크 단위로 접은 뒤 지표를 낸다.

지표(리더보드 규약과 같은 정의):
  pass^1  태스크의 trial 중 **평균 통과율**(= 전체 sim pass 비율을 태스크로 가중)
  pass^2  그 태스크의 **모든 trial이 통과**한 비율(2 trial 이상인 태스크만)
  ⚠trial 수가 태스크마다 다르므로(2 또는 3) pass^2는 "전 trial 통과"로 정의한다 — 표에 trial 수를 함께 찍는다.

  usage:  x114_sweep_scoreboard.py [--tag 20260806] [--by-arm]
"""

import collections
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x109_task_dossier import load_sims                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    sims = load_sims()
    if not sims:
        print("데이터 없음")
        return
    print("== arm 구성 ==")
    by_arm = collections.defaultdict(list)
    for s in sims:
        by_arm[s["_src"]].append(s)
    for arm, ss in sorted(by_arm.items()):
        npass = sum(1 for s in ss if (s.get("reward_info") or {}).get("reward") == 1.0)
        print("  %-34s sim %3d · pass %3d (%.1f%%) · 태스크 %d종"
              % (arm, len(ss), npass, 100.0 * npass / len(ss), len({s["task_id"] for s in ss})))

    print("\n== sim 단위 (arm 전부 합산·중복 포함) ==")
    npass = sum(1 for s in sims if (s.get("reward_info") or {}).get("reward") == 1.0)
    print("  sim %d · pass %d · **%.1f%%**" % (len(sims), npass, 100.0 * npass / len(sims)))
    term = collections.Counter(s.get("termination_reason") for s in sims)
    print("  종료사유: %s" % dict(term))

    bytask = collections.defaultdict(list)
    for s in sims:
        bytask[s["task_id"]].append((s.get("reward_info") or {}).get("reward") == 1.0)

    print("\n== 태스크 단위 (%d 태스크) ==" % len(bytask))
    p1 = sum(sum(v) / float(len(v)) for v in bytask.values()) / len(bytask)
    allpass = sum(1 for v in bytask.values() if all(v))
    anypass = sum(1 for v in bytask.values() if any(v))
    nonepass = sum(1 for v in bytask.values() if not any(v))
    print("  pass^1 (태스크 평균 통과율)      **%.1f%%**" % (100.0 * p1))
    print("  전 trial 통과 태스크             %d / %d (**%.1f%%**)"
          % (allpass, len(bytask), 100.0 * allpass / len(bytask)))
    print("  한 번이라도 통과한 태스크        %d (%.1f%%)" % (anypass, 100.0 * anypass / len(bytask)))
    print("  한 번도 통과 못한 태스크         %d (%.1f%%)" % (nonepass, 100.0 * nonepass / len(bytask)))
    tri = collections.Counter(len(v) for v in bytask.values())
    print("  태스크별 trial 수 분포: %s" % dict(sorted(tri.items())))

    print("\n== 흔들리는 태스크(일부만 통과) ==")
    flaky = sorted(t for t, v in bytask.items() if any(v) and not all(v))
    print("  %d종: %s" % (len(flaky), ", ".join(t.replace("task_", "") for t in flaky)))

    print("\n== 통과 태스크 ==")
    ok = sorted(t for t, v in bytask.items() if all(v))
    print("  %d종: %s" % (len(ok), ", ".join(t.replace("task_", "") for t in ok)))


if __name__ == "__main__":
    main()
