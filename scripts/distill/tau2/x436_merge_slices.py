# -*- coding: utf-8 -*-
r"""x436 — 슬라이스 3개를 합쳐 사실표를 완성한다 (2026-08-20)

`x435 --slice i/3` 이 각자 자기 몫만 채워 `..._corpuswide_s{i}.json` 로 낸다. 각 파일은 **표 전체**를
담되 자기 슬라이스의 클래스만 갱신돼 있으므로, 합치는 규칙은 하나다 — **값이 있는 칸이 이긴다**.
같은 칸에 서로 다른 값이 있으면 **충돌로 표시하고 둘 다 남긴다**(우리가 고르지 않는다).

사용: py -3 x436_merge_slices.py [--n 3]
"""
import argparse
import collections
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

import x431_spec_selects as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = os.path.abspath(S.TBL)
    parts = []
    for i in range(a.n):
        p = base.replace(".json", "_corpuswide_s%d.json" % i)
        if not os.path.exists(p):
            print("  ⛔없음: %s" % os.path.basename(p))
            continue
        with io.open(p, encoding="utf-8") as f:
            parts.append((i, json.load(f)))
        print("  슬라이스 %d 로드" % i)
    if not parts:
        print("합칠 것이 없다")
        return 1

    merged = {}
    tal = collections.Counter()
    keys = set()
    for _i, d in parts:
        keys |= set(d)
    for cls in sorted(keys):
        row = {}
        for _i, d in parts:
            src = d.get(cls)
            if not isinstance(src, dict):
                continue
            for k, v in src.items():
                if k.startswith("_") or not isinstance(v, dict):
                    row.setdefault(k, v)
                    continue
                cur = row.get(k)
                if (v.get("values") or []) and not ((cur or {}).get("values") or []):
                    row[k] = v
                    tal["채움"] += 1
                elif (v.get("values") or []) and (cur or {}).get("values"):
                    if v["values"][0] != cur["values"][0]:
                        cur.setdefault("conflict_values", []).append(v["values"][0])
                        cur["conflict"] = True
                        tal["충돌"] += 1
                elif cur is None:
                    row[k] = v
        merged[cls] = row
    v = ab = un = 0
    for cls, row in merged.items():
        for k, c in row.items():
            if k.startswith("_") or not isinstance(c, dict):
                continue
            if c.get("values"):
                v += 1
            elif c.get("absent"):
                ab += 1
            else:
                un += 1
    print("\n클래스 %d · 값있음 %d · 미기재 확정 %d · 미해결 %d · %s"
          % (len(merged), v, ab, un, dict(tal)))
    out = a.out or base.replace(".json", "_filled.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print("→ %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
