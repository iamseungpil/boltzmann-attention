# -*- coding: utf-8 -*-
r"""x404 — ⑴ 이관이 정당한가(gold 대조) ⑵ 게이트 차단 문구가 어느 층에 사는가"""
import collections, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_forensic as F
import x396_saying_vs_doing as C

print("=" * 108); print("x404 ⑴ 이관 정당성 — gold 에 transfer 가 있나"); print("=" * 108)
for tag in C.TAGS:
    for sim in F.scored(tag, C.SUF):
        rw = ((sim.get("reward_info") or {}).get("reward") or 0)
        if rw >= 1.0: continue
        gold = C.gold_rows(sim)
        gnames = [g["name"] for g in gold]
        gold_xfer = [g for g in gold if "transfer" in g["name"].lower()]
        called = C.called(sim)
        did_xfer = [k for k in called if "transfer" in k.lower()]
        body = " ".join(" ".join(str(m.get("content") or "").split())
                        for m in (sim.get("messages") or [])
                        if m.get("role") == "assistant" and m.get("content"))
        said_xfer = bool(re.search(r"transfer(?:red|ring)? (?:you )?to a human|TRANSFER NOTICE|human agent", body))
        if said_xfer or did_xfer or gold_xfer:
            print("  %-9s t%-2s  gold에transfer=%-5s 호출=%-24s 본문언급=%-5s 미매치=%d"
                  % (F.task_id(sim), sim.get("trial"), bool(gold_xfer),
                     ",".join(did_xfer)[:24] or "-", said_xfer,
                     sum(1 for g in gold if not g["match"])))

print("\n" + "=" * 108); print("x404 ⑵ '[Note: ... policy gate' 문구가 사는 층"); print("=" * 108)
for tag in C.TAGS:
    for sim in F.scored(tag, C.SUF):
        for i, m in enumerate(sim.get("messages") or []):
            c = " ".join(str(m.get("content") or "").split())
            if "blocked by a policy gate" in c:
                print("  %-9s t%-2s idx%-4d role=%-10s tool_calls=%-5s %s"
                      % (F.task_id(sim), sim.get("trial"), i, m.get("role"),
                         bool(m.get("tool_calls")), c[:120]))
