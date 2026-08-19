# -*- coding: utf-8 -*-
r"""x400 — **계획은 어디서 오는가**: 언급조차 없던 단계가 정책에 적혀 있었나.

## 왜 (사용자 물음 2026-08-19)
> *"계획 언급 부재는 계획이 없는 건데, 계획을 정책 문서에서도 만들 수 없나?
>   절차나 리스트가 정책에 없고 LLM 이 생짜로 만들어야 하는 건가?"*

`x399` 가 가른 **계획·언급 부재**(도구 이름도 그 엔티티도 본문에 한 번도 안 나온 자리)에 대해
**4단 사다리**로 센다 — 어디서 끊기는지가 처방을 정한다.

    ①정책에 그 도구가 이름으로 적혀 있나            (없으면 LLM 이 생짜로 만들어야 한다)
    ②그 문서에 **절차/선행 조건 문장**이 있나        (있으면 선언 저작이 정책 출처로 가능 [[23]])
    ③그 문서가 그 sim 에 **배달**됐나                (안 왔으면 결손은 회수)
    ④그 절차 문장이 **축자로** 도달했나              (왔는데도 언급조차 없으면 결손은 이행/계획)

⚠gold 는 표적 선정에만 쓰고 프롬프트에는 안 들어간다(계측 전용).

사용: py -3 x400_plan_source.py
"""
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

import t2_forensic as F  # noqa: E402
import x395_compliance_iso as X  # noqa: E402
import x396_saying_vs_doing as C  # noqa: E402
import x399_closure_forms as Z  # noqa: E402
import x393_policy_reach as R  # noqa: E402

DOCID = R.DOCID_RE


def main():
    docs = R.load_docs()
    owner = {}
    print("정책 문서 %d개" % len(docs))

    rows = []
    for tag in X.TAGS:
        for sim in F.scored(tag, X.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            gold = C.gold_rows(sim)
            texts = [str(m.get("content") or "") for m in (sim.get("messages") or [])
                     if m.get("role") == "assistant" and m.get("content")]
            body = " ".join(" ".join(t.split()) for t in texts)
            tool_body = " ".join(" ".join(str(m.get("content") or "").split())
                                 for m in (sim.get("messages") or []) if m.get("role") == "tool")
            delivered = set(DOCID.findall(tool_body))
            for g in gold:
                if g["match"]:
                    continue
                code, _ = Z.form_of(sim, g, gold)
                if code not in ("NO_CLAIM", "PLAN_ONLY"):
                    continue
                ops = C.operand_tokens(g["args"])
                if (g["name"] in body) or (ops and any(o in body for o in ops)):
                    continue                       # 언급이 있으면 이 부류가 아니다
                nm = g["name"]
                if nm not in owner:
                    owner[nm] = [k for k, d in docs.items() if nm in (d["content"] if isinstance(d, dict) else d[1])]
                ods = owner[nm]
                pl = X.proc_lines({k: (d if isinstance(d, dict) else {"title": d[0], "content": d[1]})
                                  for k, d in docs.items()}, nm)
                hit_doc = [d for d in ods if d in delivered]
                hit_line = [s for s in pl if s.split("] ", 1)[-1][:55] in tool_body]
                if not ods:
                    lvl = "①정책에 없음"
                elif not pl:
                    lvl = "②절차문장 없음"
                elif not hit_doc and not hit_line:
                    lvl = "③문서 미배달"
                elif not hit_line:
                    lvl = "③문서만 배달"
                else:
                    lvl = "④문장 도달했는데 무언급"
                rows.append({"task": F.task_id(sim), "trial": sim.get("trial"), "name": nm,
                             "type": g["type"], "level": lvl, "ndoc": len(ods), "nline": len(pl),
                             "ex": (hit_line[0] if hit_line else (pl[0] if pl else ""))[:96]})

    print("\n## 계획·언급 부재 %d건 — 계획의 출처는 어디까지 있었나" % len(rows))
    cc = collections.Counter(r["level"] for r in rows)
    for k in ("①정책에 없음", "②절차문장 없음", "③문서 미배달", "③문서만 배달", "④문장 도달했는데 무언급"):
        if cc[k]:
            print("  %-24s %3d  (%.0f%%)" % (k, cc[k], 100.0 * cc[k] / len(rows)))

    print("\n## 도구별")
    bt = collections.defaultdict(collections.Counter)
    for r in rows:
        bt[r["name"]][r["level"]] += 1
    for nm in sorted(bt, key=lambda x: -sum(bt[x].values())):
        print("  %-44s %s" % (nm[:44], " · ".join("%s×%d" % (k, v) for k, v in bt[nm].most_common())))

    print("\n## 태스크별")
    tt = collections.defaultdict(collections.Counter)
    for r in rows:
        tt[r["task"]][r["level"]] += 1
    for t in sorted(tt):
        print("  %-9s %s" % (t, " · ".join("%s×%d" % (k, v) for k, v in tt[t].most_common())))

    print("\n## ④ 실물 — 정책 문장이 궤적에 축자로 왔는데도 그 단계를 입에 안 올린 자리")
    n = 0
    for r in rows:
        if r["level"].startswith("④") and n < 8:
            n += 1
            print("  %-9s t%-2s %-40s ← %s" % (r["task"], r["trial"], r["name"][:40], r["ex"]))

    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x400_plan_source.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
