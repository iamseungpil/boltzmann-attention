# -*- coding: utf-8 -*-
r"""x233 — **발견도구 계열 37 태스크의 결손은 정확히 무엇인가** (전수 포렌식 · 유료 0 · 엔진 0).

## 왜 (사용자 지시 2026-08-10)

x232: 계열별로 **발견도구 호출이 37 태스크·1,103 sim·pass 1%** 로 가장 크고 가장 안 된다.
레버를 짓기 전에 **결손의 모양**을 먼저 본다(⛔0 ①·[[08]]).

## 무엇을 세는가 (실패를 네 통에 나눈다)

  A. **미시도**      — `unlock_discoverable_agent_tool` 을 한 번도 안 불렀다
  B. **열고 안 씀**  — unlock 은 했는데 `call_discoverable_agent_tool` 이 없다 (C398 형)
  C. **틀린 이름**   — 호출은 했는데 gold 가 요구한 도구가 아니다
  D. **불렀는데 실패**— gold 도구를 실제로 불렀는데도 실패 (=다른 축의 결손)

gold 가 요구하는 도구 이름은 **권위본**(`tasks/task_*.json`)의 `evaluation_criteria.actions`
인자에서 읽는다(101/102 포렌식 §0b: 비권위본을 읽으면 런이 쓰지도 않은 gold 로 센다).

실행: python x233_discovery_forensic.py [출력.json]
"""
import collections
import glob
import gzip
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
SIMDIRS = ["/home/woori/scratch/tau2-bench/data/simulations/*/results.json"]
UNLOCK = "unlock_discoverable_agent_tool"
CALL = "call_discoverable_agent_tool"


def auth_tasks():
    out = {}
    for p in sorted(glob.glob(os.path.join(DOM, "tasks", "task_*.json"))):
        try:
            o = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        o = o[0] if isinstance(o, list) and o else o
        if isinstance(o, dict) and o.get("id"):
            out[o["id"]] = o
    return out


def gold_disc(t):
    """gold 가 이름으로 지목한 **발견 도구**들 (인자 안에 들어 있다)."""
    names = set()
    for a in ((t.get("evaluation_criteria") or {}).get("actions") or []):
        if not isinstance(a, dict):
            continue
        if a.get("name") in (CALL, UNLOCK):
            for v in (a.get("arguments") or {}).values():
                if isinstance(v, str) and v:
                    names.add(v)
    return names


def calls_of(s):
    out = []
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            f = tc.get("function") or {}
            out.append(((f.get("name") or tc.get("name")),
                        str(f.get("arguments") or tc.get("arguments") or "")))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "x233_discovery.json"
    tasks = auth_tasks()
    fam = {tid: t for tid, t in tasks.items()
           if any(a.get("name") == CALL
                  for a in ((t.get("evaluation_criteria") or {}).get("actions") or [])
                  if isinstance(a, dict))}
    print("발견도구 계열 %d 태스크 (권위본 기준)" % len(fam))
    bucket = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    seen = set()
    for pat in SIMDIRS:
        for p in sorted(glob.glob(pat)):
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for s in (d.get("simulations") or []):
                tid = s.get("task_id")
                if tid not in fam or s.get("id") in seen:
                    continue
                seen.add(s.get("id"))
                ok = (s.get("reward_info") or {}).get("reward") == 1
                cs = calls_of(s)
                names = [n for n, _ in cs]
                want = gold_disc(fam[tid])
                unlocked = names.count(UNLOCK)
                called = [a for n, a in cs if n == CALL]
                hit = any(any(w in a for w in want) for a in called) if want else bool(called)
                if ok:
                    b = "PASS"
                elif unlocked == 0:
                    b = "A_미시도"
                elif not called:
                    b = "B_열고_안씀"
                elif not hit:
                    b = "C_틀린이름"
                else:
                    b = "D_불렀는데_실패"
                bucket[b] += 1
                per[tid][b] += 1
    tot = sum(bucket.values())
    print("\n전체 %d sim" % tot)
    for k, v in bucket.most_common():
        print("  %-16s %5d  (%.0f%%)" % (k, v, 100.0 * v / max(1, tot)))
    print("\n태스크별 (실패 통이 가장 큰 것부터·상위 20)")
    rows = sorted(per.items(), key=lambda kv: -(sum(kv[1].values()) - kv[1]["PASS"]))
    for tid, c in rows[:20]:
        n = sum(c.values())
        print("  %-9s n=%3d pass=%2d | %s"
              % (tid, n, c["PASS"],
                 " ".join("%s=%d" % (k.split("_")[0], v)
                          for k, v in c.most_common() if k != "PASS")))
    json.dump({t: dict(c) for t, c in per.items()},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: %s" % out_path)
    print("\n※ A/B 가 크면 결손은 **호출 자체**(C398 축·격리 프로브가 이미 7/8 대 0 을 쟀다),"
          "\n  C 가 크면 **이름 매핑**, D 가 크면 발견은 됐고 **다른 축**이 남은 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
