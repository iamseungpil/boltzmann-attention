# -*- coding: utf-8 -*-
r"""x268 — 071 MULTI 의 범인 문장 + 070 접미사의 생존 여부 (유료 0 · 엔진 0).

## ① 071 MULTI — leave-one-out (x267 후속)

x267: 라이브 창(색인<47) ask → 두 축 **0/8 MULTI** ↔ 끝 창 → **8/8**.
두 창의 차이 = 라이브 창에만 있는 **msg 27**: 손님이 오답 두 개를 목록으로 수락한 발화
(*"Yes — let's proceed with those: True Blue … Gold Saver …"*). 끝 창의 msg 59 는 `True Blue`
를 **단일 지목**하는데도 8/8 이므로, 무너뜨리는 것은 손님 지목 일반이 아니라 **그 목록**이라는
가설. x231(이름 목록 8/8→0/8)·[[42]](copy=induction) 과 같은 모양이다.

  A_LIVE    라이브 창 4개 그대로 (msg 22·27·29·31)     ← 0/8 재현 기대
  B_NO27    msg 27 만 뺀 3개                            ← 이것만 8/8 이면 범인 확정
  C_ONLY27  msg 27 하나만                                ← 목록 단독의 독성
  D_NO22    msg 22 만 뺀 3개 (대조 — 요건 발화를 빼면)   ← 요건이 실려 있는지 검사

## ② 070 접미사 — 서브의 라이브 형태가 인자까지 살아남는가

라이브 서브는 늘 `'Sky Blue Account'`(접미사)를 낸다. 채점 칸은 `account_class='Sky Blue'`.
x256 체인으로 값만 갈아 끼워 잰다:

  S_BARE    [DECIDED] … is: Sky Blue          ← C435 의 B_SUB 재현(7/8 기대)
  S_SUFF    [DECIDED] … is: Sky Blue Account  ← 라이브 형태. HIT 면 모델이 정규화한다(무해)
                                                  CLASS_WRONG(Sky Blue Account) 면 해악 확정

⚠엔진은 접미사를 떼지 않는다([[59]]·C376). 여기서 재는 것은 **모델이 떼 주는가**다.

실행(리모트·GPU1): T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x268_multi_and_suffix.py [N]
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as S                                               # noqa: E402
from x216_read_and_offset import chat                               # noqa: E402
from x266_decide_ask_axis import a2, classify, GOLD, DOCS           # noqa: E402
import x256_dispatcher_write_probe as X256                          # noqa: E402

RES = os.environ.get(
    "X268_RESULTS",
    "/home/woori/scratch/tau2-bench/data/simulations/bank_all6b_20260811/results.json")
NOW = "2025-11-14"


def part1(n):
    A = a2()
    po = A["policy_ontology"]
    d = json.load(io.open(RES, encoding="utf-8"))
    sim = [x for x in d["simulations"] if x["task_id"] == "task_071"][0]
    us = [(i, str(m.get("content") or "")) for i, m in enumerate(sim["messages"])
          if m.get("role") == "user"]
    live = [u for u in us if u[0] < 47][-4:]                # 색인 22·27·29·31
    idx = [i for i, _ in live]
    print("① 071 leave-one-out · 라이브 창 색인 %s" % idx)
    arms = (("A_LIVE", [c for _, c in live]),
            ("B_NO27", [c for i, c in live if i != 27]),
            ("C_ONLY27", [c for i, c in live if i == 27]),
            ("D_NO22", [c for i, c in live if i != 22]))
    for g in ("business_checking_accounts", "business_savings_accounts"):
        gold = GOLD[g]
        mat, info = S.material_for(A, g, DOCS, NOW)
        allnames = sorted(set(GOLD.values()))
        print("  축 %s · gold=%r · 문서 %d(뺀 것 %d)" % (g, gold, info["kept"], len(info["dropped"])))
        for label, msgs in arms:
            ask = " --- ".join(msgs)[-6000:]
            c = collections.Counter()
            for _i in range(n):
                try:
                    body = po["doc_decide_prompt"].format(ask=ask[:3000], material=mat)
                    r = chat(body, None, 0.0 if _i == 0 else 0.7, 40).get("content")
                except Exception as e:
                    r = "ERR %s" % type(e).__name__
                c[classify(r, gold, allnames)] += 1
            print("    %-9s EXACT %d/%d  %s" % (label, c["EXACT"], n, c.most_common(3)))


def part2(n):
    print("\n② 070/071형 접미사 — 값 형태가 인자까지 살아남는가 (x256 체인)")
    d = json.load(io.open(X256.RES, encoding="utf-8"))
    sim = [s for s in d["simulations"] if s["task_id"] == X256.TASK][0]
    gold = X256.gold_args(sim)
    cut = None
    for i, m in enumerate(sim["messages"]):
        for tc in (m.get("tool_calls") or []):
            if (tc.get("name") or "") == X256.DISPATCH and X256.TOOL in json.dumps(
                    tc.get("arguments"), ensure_ascii=False) and cut is None:
                cut = i
    tools = X256.U.tools_of(sim)
    live = X256.build(sim, cut, True)
    base = ("[DECIDED] The account class the retrieved documents support for this request "
            "is: %s")
    for label, val in (("S_BARE", str(gold.get("account_class"))),
                       ("S_SUFF", str(gold.get("account_class")) + " Account")):
        body = live + "\n[system] " + base % val
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 300)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[X256.score(r, gold)] += 1
        print("  %-7s 값=%r · HIT %d/%d  %s" % (label, val, c["HIT"], n, c.most_common(3)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    part1(n)
    part2(n)
    print("\n※ ①: `B_NO27` 만 높으면 범인은 msg 27 의 수락 목록이다(문장 하나가 0/8↔8/8)."
          "\n  ②: `S_SUFF` 가 HIT 면 모델이 접미사를 정규화한다(무해) — CLASS_WRONG 이면 해악.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
