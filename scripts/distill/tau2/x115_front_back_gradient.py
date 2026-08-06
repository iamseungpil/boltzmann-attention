# -*- coding: utf-8 -*-
"""태스크 번호 구간별 통과율 — "앞은 되고 뒤가 무너진다"를 수치로 확인한다.

주장(2026-08-06 사용자): *"front32에서는 40% 가까이 되던 게 지금 안 된다 · 뒷부분이 무너지고 있다."*
확인하려면 두 가지가 필요하다 — ①현 스윕의 **구간별** 통과율 ②과거 32-태스크 런의 통과율.
둘 다 이 스크립트로 낸다. 임의의 arm glob을 받아 같은 방식으로 계산하므로 비교가 같은 규약을 쓴다.

⚠비교 규약: 태스크 집합이 다르면 통과율 차이를 스택 차이로 읽으면 안 된다. 그래서 arm마다
**태스크 번호 범위와 개수**를 함께 찍는다.

  usage:  x115_front_back_gradient.py ["<arm glob>" ...]
          기본 = 현 스윕(bank_n97_gpu*_20260806*)
"""

import collections
import glob
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIRS = ["/home/woori/scratch/tau2-bench/data/simulations",
            os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                         "reports", "facet_rft_2026", "sim_results"))]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATTERNS = [a for a in sys.argv[1:] if not a.startswith("-")] or ["bank_n97_gpu*_20260806*"]
BUCKETS = [(1, 32), (33, 64), (65, 102)]


def load(pattern):
    out = []
    for base in SIM_DIRS:
        for p in sorted(glob.glob(os.path.join(base, pattern + ".results.json.gz"))
                        + glob.glob(os.path.join(base, pattern, "results.json"))):
            op = gzip.open if p.endswith(".gz") else io.open
            try:
                d = json.load(op(p, "rt", encoding="utf-8"))
            except Exception as e:
                print("  (읽기 실패 %s: %s)" % (os.path.basename(p), e))
                continue
            tag = os.path.basename(p).replace(".results.json.gz", "")
            if tag == "results.json":
                tag = os.path.basename(os.path.dirname(p))
            for s in d.get("simulations") or []:
                out.append((tag, s))
    return out


def num(task_id):
    m = re.search(r"(\d+)", task_id or "")
    return int(m.group(1)) if m else -1


def main():
    for pat in PATTERNS:
        rows = load(pat)
        if not rows:
            print("== %s == 데이터 없음\n" % pat)
            continue
        tags = sorted({t for t, _ in rows})
        nums = [num(s["task_id"]) for _, s in rows]
        print("== %s ==" % pat)
        print("  arm %d개: %s" % (len(tags), ", ".join(tags)))
        print("  sim %d · 태스크 %d종 · 번호 범위 %d~%d"
              % (len(rows), len({s["task_id"] for _, s in rows}), min(nums), max(nums)))

        # 태스크 단위로 접는다(같은 태스크가 여러 arm에 있으면 trial을 합친다).
        bytask = collections.defaultdict(list)
        for _, s in rows:
            bytask[s["task_id"]].append((s.get("reward_info") or {}).get("reward") == 1.0)
        p1 = sum(sum(v) / float(len(v)) for v in bytask.values()) / len(bytask)
        allp = sum(1 for v in bytask.values() if all(v))
        print("  전체: pass^1 **%.1f%%** · 전 trial 통과 %d/%d (%.1f%%)"
              % (100 * p1, allp, len(bytask), 100.0 * allp / len(bytask)))

        print("  구간별:")
        for lo, hi in BUCKETS:
            sub = {t: v for t, v in bytask.items() if lo <= num(t) <= hi}
            if not sub:
                continue
            sp1 = sum(sum(v) / float(len(v)) for v in sub.values()) / len(sub)
            sall = sum(1 for v in sub.values() if all(v))
            print("    %3d~%-3d 태스크 %2d종 · pass^1 %5.1f%% · 전 trial 통과 %2d (%.1f%%)"
                  % (lo, hi, len(sub), 100 * sp1, sall, 100.0 * sall / len(sub)))
        print()


if __name__ == "__main__":
    main()
