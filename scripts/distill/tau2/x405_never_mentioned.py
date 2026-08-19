# -*- coding: utf-8 -*-
r"""x405 — **완전 무언급** 집합을 내 손으로 다시 정의하고, KB 조회용 표적 목록을 뽑는다.

정의(전부 축자·해석 0):
  대상 = 미매치 gold 중 **그 도구를 한 번도 호출하지 않은 것**
  name_in_body = 도구 이름이 assistant 본문에 축자로 등장
  ops_in_body  = 그 액션의 목적어 id(chk_/sav_/dbc_/cc_/txn_/acc_)가 assistant 본문에 등장
  NEVER_MENTIONED = (not name_in_body) and (not ops_in_body)

⚠ops 가 애초에 0개인 액션은 ops_in_body 가 정의상 False 다 — 그 사실을 따로 인쇄한다
  (없는 것을 '무언급'으로 세면 과대계상이다).
"""
import collections, io, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_forensic as F
import x396_saying_vs_doing as C

rows = []
for tag in C.TAGS:
    for sim in F.scored(tag, C.SUF):
        if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
            continue
        body = " ".join(" ".join(str(m.get("content") or "").split())
                        for m in (sim.get("messages") or [])
                        if m.get("role") == "assistant" and m.get("content"))
        calls = C.called(sim)
        for g in C.gold_rows(sim):
            if g["match"] or calls.get(g["name"]):
                continue
            ops = C.operand_tokens(g["args"])
            rows.append({"task": F.task_id(sim), "trial": sim.get("trial"), "name": g["name"],
                         "type": g["type"], "nops": len(ops),
                         "name_in_body": g["name"] in body,
                         "ops_in_body": bool(ops and any(o in body for o in ops))})

print("=" * 100); print("x405 · 호출 0회인 미매치 gold %d건" % len(rows)); print("=" * 100)
q = collections.Counter((r["name_in_body"], r["ops_in_body"]) for r in rows)
print("\n## 이름언급 × 엔티티언급")
for k in [(True, True), (True, False), (False, True), (False, False)]:
    print("  이름=%-5s 엔티티=%-5s : %3d" % (k[0], k[1], q[k]))
nm = [r for r in rows if not r["name_in_body"] and not r["ops_in_body"]]
print("\n  ⇒ NEVER_MENTIONED %d건" % len(nm))
print("     그중 ops 가 애초에 0개(=엔티티 검사 공허) %d건" % sum(1 for r in nm if not r["nops"]))
print("     ops 가 있는데도 본문에 안 나온 것          %d건" % sum(1 for r in nm if r["nops"]))

print("\n## NEVER_MENTIONED · 태스크별")
for k, v in collections.Counter(r["task"] for r in nm).most_common():
    print("  %-9s %2d   %s" % (k, v, ", ".join(sorted({r["name"] for r in nm if r["task"] == k}))[:100]))
print("\n## NEVER_MENTIONED · 도구별")
for k, v in collections.Counter(r["name"] for r in nm).most_common():
    print("  %-46s %2d" % (k, v))

io.open("_x405_targets.json", "w", encoding="utf-8").write(json.dumps(
    sorted({(r["task"], r["name"]) for r in nm}), ensure_ascii=False))
json.dump(nm, io.open(os.path.join("..", "..", "..", "reports", "facet_rft_2026",
                                   "x405_never_mentioned.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n표적 (task,tool) 쌍 %d개 -> _x405_targets.json" % len({(r["task"], r["name"]) for r in nm}))
