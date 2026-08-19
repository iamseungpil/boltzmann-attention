# -*- coding: utf-8 -*-
r"""x409 - R3 per-step 포렌식: 배달된 그 순간 전후에 무엇이 일어났나 (프로브와 독립)

사용자 지시 2026-08-19: "프로브와 별도로 포렌식으로 R3 이유를 확정하라."

R3 48건마다 궤적을 한 단계씩 읽어 다음을 축자로 센다 (해석 0):
  (1) 무엇이 그 이름을 물어왔나  - 배달 호출의 도구 이름
  (2) 슬라이스가 **지시문**인가 단순 언급인가 - 축자 문형 목록으로 분류(목록 인쇄)
  (3) 배달이 **잘렸나** - 도구 결과 본문이 절단 표지로 끝나나
  (4) 배달 직후 assistant 가 한 것 - 호출이면 그 이름 / 산문 / 질문
  (5) 배달 이후 남은 assistant 턴 수 - 고칠 기회가 몇 번 있었나
  (6) 배달 이후 손님이 화제를 바꿨나 - 손님 발화에 새 요구 문형
"""
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F
import x396_saying_vs_doing as C

# 슬라이스가 그 도구를 **지시**하는 문형 (축자 목록 - 인쇄된다)
DIRECTIVE = [
    ("GIVE_USER", ("give them the", "give the user", "provide access to", "give them access")),
    ("CALL_IT", ("use the", "call the", "call `", "invoke", "using:", "you must", "must unlock")),
    ("ORDER", ("in this exact order", "before completing", "do not skip", "first,", "then call")),
]
TRUNC = ("...", "…", "[truncated", "truncated]")


def dir_kind(sl, tool):
    low = " ".join((sl or "").split()).lower()
    i = low.find(tool.lower())
    near = low[max(0, i - 160):i + 60] if i >= 0 else low
    for name, keys in DIRECTIVE:
        if any(k in near for k in keys):
            return name
    return "MENTION_ONLY"


def main():
    print("=" * 118)
    print("x409 · R3 per-step 포렌식")
    print("지시 문형(축자):")
    for n, k in DIRECTIVE:
        print("   %-10s %s" % (n, ", ".join(k)))
    print("=" * 118)

    rows = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            msgs = sim.get("messages") or []
            body = " ".join(" ".join(str(m.get("content") or "").split())
                            for m in msgs if m.get("role") == "assistant" and m.get("content"))
            calls = C.called(sim)
            # tool_call id -> (호출 도구 이름, 그 호출이 있던 메시지 idx)
            src = {}
            for i, m in enumerate(msgs):
                for tc in (m.get("tool_calls") or []):
                    src[tc.get("id")] = (str(F.nameof(tc)), i)
            for g in C.gold_rows(sim):
                if g["match"] or calls.get(g["name"]):
                    continue
                ops = C.operand_tokens(g["args"])
                if g["name"] in body or (ops and any(o in body for o in ops)):
                    continue
                hit = None
                for i, m in enumerate(msgs):
                    if m.get("role") != "tool":
                        continue
                    c = " ".join(str(m.get("content") or "").split())
                    j = c.find(g["name"])
                    if j >= 0:
                        hit = (i, c, j)
                        break
                if hit is None:
                    continue
                i, c, j = hit
                sl = c[max(0, j - 600):j + 600]
                after = msgs[i + 1:]
                nxt_call, nxt_kind = None, "?"
                for m in after:
                    if m.get("role") != "assistant":
                        continue
                    tcs = m.get("tool_calls") or []
                    if tcs:
                        nxt_call = str(F.nameof(tcs[0]))
                        nxt_kind = "CALL"
                    else:
                        t = " ".join(str(m.get("content") or "").split())
                        nxt_kind = "ASK" if t.endswith("?") else "PROSE"
                    break
                rows.append({
                    "task": F.task_id(sim), "trial": sim.get("trial"), "tool": g["name"],
                    "type": g["type"],
                    "src": src.get(msgs[i].get("id"), ("?", -1))[0],
                    "at": i, "nmsg": len(msgs),
                    "dirk": dir_kind(sl, g["name"]),
                    "trunc": any(c.rstrip().endswith(t) for t in TRUNC),
                    "reslen": len(c),
                    "nxt": (nxt_call or nxt_kind),
                    "n_after_a": sum(1 for m in after if m.get("role") == "assistant"),
                    "n_after_call": sum(1 for m in after if (m.get("tool_calls") or [])),
                })

    print("\n## R3 %d건" % len(rows))
    print("\n### (1) 무엇이 그 이름을 물어왔나")
    for k, v in collections.Counter(r["src"] for r in rows).most_common():
        print("   %-34s %2d" % (k, v))
    print("\n### (2) 슬라이스가 지시문인가")
    for k, v in collections.Counter(r["dirk"] for r in rows).most_common():
        print("   %-14s %2d  (%.0f%%)" % (k, v, 100.0 * v / len(rows)))
    print("\n### (3) 그 도구 결과가 절단됐나 · 길이")
    print("   절단 표지로 끝남 %d/%d · 결과 길이 중앙값 %d자"
          % (sum(r["trunc"] for r in rows), len(rows),
             sorted(r["reslen"] for r in rows)[len(rows) // 2]))
    print("\n### (4) 배달 직후 assistant 가 한 것")
    for k, v in collections.Counter(r["nxt"] for r in rows).most_common():
        print("   %-40s %2d" % (k, v))
    print("\n### (5) 배달 이후 남은 기회")
    d = collections.Counter(r["n_after_a"] for r in rows)
    print("   남은 assistant 턴: %s" % dict(sorted(d.items())))
    print("   남은 도구호출 턴 중앙값: %d"
          % sorted(r["n_after_call"] for r in rows)[len(rows) // 2])
    print("   배달 이후 호출이 0회인 건: %d" % sum(1 for r in rows if r["n_after_call"] == 0))

    print("\n### (6) 지시문 종류 × 직후 행동")
    x = collections.defaultdict(collections.Counter)
    for r in rows:
        x[r["dirk"]][("CALL" if r["nxt"] not in ("PROSE", "ASK", "?") else r["nxt"])] += 1
    for k in x:
        print("   %-14s %s" % (k, dict(x[k])))

    print("\n### 전량")
    print("  %-9s %-3s %-40s %-8s %-13s %-5s %-24s %s"
          % ("task", "tr", "tool", "지시", "물어온것", "위치", "직후", "이후A/호출"))
    for r in sorted(rows, key=lambda z: (z["task"], str(z["trial"]))):
        print("  %-9s %-3s %-40s %-8s %-13s %-5s %-24s %d/%d"
              % (r["task"], r["trial"], r["tool"][:40], r["dirk"][:8], r["src"][:13],
                 "%d/%d" % (r["at"], r["nmsg"]), r["nxt"][:24], r["n_after_a"], r["n_after_call"]))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x409_r3_perstep.json"))
    io.open(o, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


sys.exit(main())
