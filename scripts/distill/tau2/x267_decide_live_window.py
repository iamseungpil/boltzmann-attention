# -*- coding: utf-8 -*-
r"""x267 — 라이브가 **실제로 발화한 창**에서 결정 서브를 다시 잰다 (유료 0 · 엔진 0).

## 왜 (x266 의 자기 반증)

x266 은 대화 **끝**에서 ask 를 뽑았고 출시본 ask 가 두 축 **8/8** 이었다. 그런데 라이브는
**이름 다중**을 냈다(`True Blue Business Checking Gold Saver Business Savings`). 두 수가 같은
것을 재고 있지 않다 — **관측 창이 다르다.** 설계 §7 축자: *"프로브의 관측 창이 표적의 창과
같은가를 먼저 확인하라."*

## 창을 어떻게 찾나 — 사이드카 `turn` 을 믿지 않는다

C429⒟: `trace.turn` 과 사이드카 `turn` 은 **기준이 다르고** 한 턴에 재생성이 여러 번 돈다.
그래서 궤적에서 **문자열로** 찾는다 — 엔진이 실제로 내보낸 결정 문장(`decided_by_docs_text` 의
고정 앞부분)이 들어 있는 메시지의 색인 `i` 가 곧 그 발화 자리이고, 엔진이 그때 쓴 ask 는
**`i` 이전의 손님 발화 마지막 4개**다(`_search_material` 축자 재현).

## 팔

  A_LIVEWIN    ask = 그 창의 손님 발화 4개            ← 라이브가 실제로 준 것
  B_ENDWIN     ask = 대화 끝의 손님 발화 4개          ← x266 이 쟀던 것(대조)

`A_LIVEWIN` 이 낮고 `B_ENDWIN` 이 높으면 원인은 **그 창의 ask 내용**이다.
둘 다 높으면 원인은 ask 가 아니다 — 전송 경로(라이브 `la.generate` + agent `llm_args`)로 간다.

⚠엔진 문구·재료는 **출시본 그대로**(A2 템플릿 · `t2_search.material_for`).
⚠라이브가 그때 쓴 `now` 는 로그가 말해 준다(`now=2025-11-14`) — 프로브가 지어내지 않는다.

실행(리모트·GPU1):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x267_decide_live_window.py [N]
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

import t2_search as S                                              # noqa: E402
from x216_read_and_offset import chat                              # noqa: E402
from x266_decide_ask_axis import a2, classify, live_ask, pick_groups, GOLD, DOCS  # noqa: E402

RES = os.environ.get(
    "X267_RESULTS",
    "/home/woori/scratch/tau2-bench/data/simulations/bank_all6b_20260811/results.json")
NOW = "2025-11-14"          # 라이브 로그가 찍은 값 — 프로브가 정하지 않는다
MARK = "A separate check was run on the policy documents"


def firing_points(sim):
    """엔진이 결정 문장을 **실제로 내보낸** 메시지 색인들 (문자열로 찾는다)."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if MARK in str(m.get("content") or ""):
            out.append(i)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    A = a2()
    po = A["policy_ontology"]
    d = json.load(io.open(RES, encoding="utf-8"))

    found = 0
    for sim in d["simulations"]:
        pts = firing_points(sim)
        if not pts:
            continue
        found += 1
        end_ask = live_ask(sim, len(sim["messages"]))
        print("=" * 92)
        print("%s · sim %s · 발화 색인 %s (총 메시지 %d)"
              % (sim.get("task_id"), str(sim.get("id") or "")[:12], pts, len(sim["messages"])))
        for i in pts:
            ask = live_ask(sim, i)
            gs = pick_groups(po, list(po.get("doc_index") or {}), ask)
            print("  [색인 %d] 그 창의 축: %s · ask %d자" % (i, gs or "없음", len(ask)))
            for g in gs:
                gold = GOLD.get(g)
                if not gold:
                    continue
                mat, info = S.material_for(A, g, DOCS, NOW)
                allnames = sorted(set(GOLD.values()))
                for label, a_ in (("A_LIVEWIN", ask), ("B_ENDWIN", end_ask)):
                    c = collections.Counter()
                    for _i in range(n):
                        try:
                            body = po["doc_decide_prompt"].format(
                                ask=str(a_)[:3000], material=mat)
                            r = chat(body, None, 0.0 if _i == 0 else 0.7, 40).get("content")
                        except Exception as e:
                            r = "ERR %s" % type(e).__name__
                        c[classify(r, gold, allnames)] += 1
                    print("      %-10s %-28s EXACT %d/%d  %s"
                          % (label, g, c["EXACT"], n, c.most_common(3)))
    if not found:
        print("결정 문장을 낸 sim 이 없다 — 이 결과 파일에서는 잴 것이 없다.")
        return 2
    print("\n※ `A_LIVEWIN` 낮고 `B_ENDWIN` 높음 ⇒ 원인은 그 창의 ask 내용."
          "\n  둘 다 높음 ⇒ ask 가 아니다 — 전송 경로(la.generate·llm_args)로 좁힌다."
          "\n  둘 다 낮음 ⇒ x266 의 8/8 이 창 운이었다(그 결론부터 재검).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
