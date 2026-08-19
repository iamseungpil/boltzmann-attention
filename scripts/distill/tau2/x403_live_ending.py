# -*- coding: utf-8 -*-
r"""x403 — 라이브 실패의 **종결 국면**: 왜 그 자리에서 호출이 멎었나

x402 로 확인: 라이브 33 실패 sim 전량이 `user_stop` + 마지막 assistant = 산문.
격리의 `{"tool": null}` 은 **라이브에 대응물이 없다**(프로브 구성물). 그러면 라이브의 멎음은 무엇인가.

측정(전부 궤적 축자·해석 0):
  ⒜ 마지막 도구호출 이후 남은 assistant 턴 수  (= 몇 턴을 말만 하다 끝났나)
  ⒝ 마지막 국면의 문면 유형 — 이관 / 손님대기(질문) / 우리 게이트 차단 / 완료선언
  ⒞ 우리 층 차단·거부가 궤적에 몇 번 발화됐나
"""
import collections, io, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_forensic as F
import x396_saying_vs_doing as C

OURS = ("blocked by a policy gate", "were NOT executed", "[Note:", "T2_", "Satisfy")
TRANSFER = ("transferred to a human", "transfer to a human", "TRANSFER NOTICE",
            "human agent will", "connect you", "transfer you to")
ASKQ = ("?",)


def main():
    print("=" * 112); print("x403 · 라이브 실패의 종결 국면 (n=실패 sim)"); print("=" * 112)
    rows = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            msgs = sim.get("messages") or []
            idx_call = [i for i, m in enumerate(msgs) if (m.get("tool_calls") or [])]
            last_call = idx_call[-1] if idx_call else -1
            tail_a = [m for m in msgs[last_call + 1:] if m.get("role") == "assistant"]
            tail_u = [m for m in msgs[last_call + 1:] if m.get("role") == "user"]
            last_txt = " ".join(str((tail_a[-1] if tail_a else {}).get("content") or "").split())
            # 우리 층 차단 발화 (도구 결과 본문에 우리 문구)
            ours = sum(1 for m in msgs if m.get("role") == "tool"
                       and any(p in str(m.get("content") or "") for p in OURS))
            body_tail = " ".join(" ".join(str(m.get("content") or "").split()) for m in tail_a)
            if any(p in body_tail for p in OURS):
                kind = "GATE_BLOCKED"
            elif any(p in body_tail for p in TRANSFER):
                kind = "TRANSFER"
            elif C.DONE_RE.search(body_tail):
                kind = "DECLARED_DONE"
            elif body_tail.rstrip().endswith("?") or "?" in body_tail[-160:]:
                kind = "AWAIT_USER"
            else:
                kind = "OTHER"
            miss = [g for g in C.gold_rows(sim) if not g["match"]]
            rows.append({"task": F.task_id(sim), "trial": sim.get("trial"),
                         "tail_assistant": len(tail_a), "tail_user": len(tail_u),
                         "ncalls": len(idx_call), "ours_deny": ours, "kind": kind,
                         "miss": len(miss), "last": last_txt[:80]})

    print("\n## ⒝ 종결 국면 유형")
    for k, v in collections.Counter(r["kind"] for r in rows).most_common():
        sub = [r for r in rows if r["kind"] == k]
        print("  %-14s %2d  (%.0f%%)   미매치 gold 합 %3d · 꼬리 assistant 턴 중앙 %s"
              % (k, v, 100.0 * v / len(rows), sum(r["miss"] for r in sub),
                 sorted(r["tail_assistant"] for r in sub)[len(sub) // 2]))

    print("\n## ⒜ 마지막 호출 이후 말만 한 턴 수")
    d = collections.Counter(r["tail_assistant"] for r in rows)
    for k in sorted(d):
        print("  %2d턴 : %s (%d)" % (k, "#" * d[k], d[k]))

    print("\n## ⒞ 우리 층 차단·거부 발화가 있는 sim")
    hv = [r for r in rows if r["ours_deny"]]
    print("  %d/%d sim · 발화 총 %d회" % (len(hv), len(rows), sum(r["ours_deny"] for r in rows)))
    for r in sorted(hv, key=lambda z: -z["ours_deny"])[:12]:
        print("    %-9s t%-2s deny%-3d 호출%-3d 미매치%-3d %s"
              % (r["task"], r["trial"], r["ours_deny"], r["ncalls"], r["miss"], r["kind"]))

    print("\n## 전량")
    print("  %-9s %-3s %-14s %-6s %-6s %-6s %-5s %s"
          % ("task", "tr", "kind", "호출", "꼬리A", "꼬리U", "미매치", "우리deny"))
    for r in sorted(rows, key=lambda z: (z["kind"], z["task"])):
        print("  %-9s %-3s %-14s %-6d %-6d %-6d %-5d %d"
              % (r["task"], r["trial"], r["kind"], r["ncalls"], r["tail_assistant"],
                 r["tail_user"], r["miss"], r["ours_deny"]))
    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x403_live_ending.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % out)
    return 0

sys.exit(main())
