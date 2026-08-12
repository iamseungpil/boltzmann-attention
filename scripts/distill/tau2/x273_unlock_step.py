# -*- coding: utf-8 -*-
r"""x273 — 체인 (2)단계: **검색 결과를 그대로 주면 unlock 하는가** ([[18]] 정보-맞춘 격리).

## 왜 (x272 의 자기 교정)

x272 는 `KB_search 또는 unlock` 을 "체인 진입"으로 셌고 네 팔 전부 8/8 이 나왔다. 그런데
라이브는 `KB_search` **7회**를 하고도 `unlock` **0회**다 — 즉 (1)단계는 이미 되고 있었고
**내 계기가 (1)과 (2)를 뭉갰다**. 실패는 (2)다: 검색 결과에서 도구 이름을 얻어 unlock 으로
넘어가는 자리.

## 팔 (계기는 **다음 응답이 `unlock_discoverable_agent_tool` 을 부르는가** 하나)

  A_LIVEHITS   라이브가 **실제로 받은** KB 결과 그대로 + 요청            ← 정보-맞춤(핵심)
  B_NAMEDOC    도구 이름이 **적힌 문서**를 회수했다고 가정한 결과        ← 회수가 성공했다면?
  C_HITS_ONLY  라이브 결과만 (발견 문구 없음·부정 통제)                  ← 문구 공로 분리
  D_NAMEONLY   이름 한 줄만 (`open_bank_account_4821`)                   ← 상한

`A` 낮고 `B` 높으면 병은 **회수**(질의가 그 문서를 못 물어온다)다. `A` 도 높으면 병은
라이브의 다른 것(전달·경쟁·순서)이고 회수는 무죄다.

⚠KB 결과·문구 전부 **라이브 궤적/사이드카에서 축자로** 가져온다 — 프로브가 짓지 않는다([[03b]]).
⚠`B`/`D` 의 이름은 **환경 레지스트리**에서 온다(gold 아님·`_agent_discoverable` 과 같은 출처).

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x273_unlock_step.py [N]
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                               # noqa: E402
import x256_dispatcher_write_probe as X256                          # noqa: E402
from x272_discovery_adoption import shipped_text, SIMS, TAG, TASK   # noqa: E402

DOCS = os.environ.get(
    "X273_DOCS",
    "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents")


def live_kb_hits(sim, cut):
    """라이브가 실제로 받은 KB 결과 축자 (그 시점까지)."""
    out, byid = [], {}
    for m in sim["messages"][:cut]:
        for tc in (m.get("tool_calls") or []):
            byid[tc.get("id")] = tc.get("name")
    for m in sim["messages"][:cut]:
        if m.get("role") == "tool" and str(byid.get(m.get("id")) or "").startswith("KB_search"):
            out.append(str(m.get("content") or ""))
    return out


def name_doc():
    """도구 이름이 **적힌** 문서를 코퍼스에서 찾는다(회수가 성공했을 때의 재료)."""
    for fn in sorted(os.listdir(DOCS)):
        if not fn.endswith(".json"):
            continue
        try:
            d = json.load(io.open(os.path.join(DOCS, fn), encoding="utf-8"))
        except Exception:
            continue
        blob = json.dumps(d, ensure_ascii=False)
        if "open_bank_account_4821" in blob:
            return fn, (d.get("content") or blob)
    return None, None


def called_unlock(msg):
    for tc in ((msg or {}).get("tool_calls") or []):
        if "unlock_discoverable" in str((tc.get("function") or {}).get("name") or ""):
            return True
    return False


def classify(msg):
    if called_unlock(msg):
        return "UNLOCK"
    names = [str((tc.get("function") or {}).get("name") or "")
             for tc in ((msg or {}).get("tool_calls") or [])]
    if any("KB_search" in n for n in names):
        return "KB_AGAIN"
    if names:
        return "OTHER(%s)" % names[0][:20]
    return "PROSE"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    txt = shipped_text()
    d = json.load(io.open(os.path.join(SIMS, TAG, "results.json"), encoding="utf-8"))
    sim = [s for s in d["simulations"]
           if s["task_id"] == TASK and (s.get("reward_info") or None) is not None][0]
    cut = max(i for i, m in enumerate(sim["messages"]) if m.get("role") == "user")
    users = [str(m.get("content") or "") for m in sim["messages"][:cut + 1]
             if m.get("role") == "user"]
    ask = " --- ".join(users[-2:])
    hits = live_kb_hits(sim, cut)
    fn, doc = name_doc()
    tools = X256.U.tools_of(sim)
    print("라이브 KB 결과 %d건(%d자) · 이름 문서=%s" % (len(hits), sum(len(h) for h in hits), fn))
    print("라이브 결과에 'open_bank_account_4821' 포함:",
          any("open_bank_account_4821" in h for h in hits), "\n")

    hits_txt = "\n\n".join(h[:2500] for h in hits[-4:])
    arms = (("A_LIVEHITS", "%s\n\nKnowledge base results:\n%s\n\n[system] %s" % (ask, hits_txt, txt)),
            ("B_NAMEDOC", "%s\n\nKnowledge base results:\n%s\n\n[system] %s"
             % (ask, str(doc)[:2500], txt)),
            ("C_HITS_ONLY", "%s\n\nKnowledge base results:\n%s" % (ask, hits_txt)),
            ("D_NAMEONLY", "%s\n\nKnowledge base results:\nThe tool to use is "
                           "open_bank_account_4821.\n\n[system] %s" % (ask, txt)))
    for label, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 300)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[classify(r)] += 1
        print("  %-12s UNLOCK %d/%d   %s" % (label, c["UNLOCK"], n, c.most_common(3)))
    print("\n※ A 낮고 B 높음 ⇒ 병은 **회수**(질의가 그 문서를 못 물어온다)."
          "\n  A 도 높음 ⇒ 회수는 무죄이고 병은 라이브의 전달·경쟁·순서다."
          "\n  C 가 A 만큼 높으면 문구 공로가 아니다(부정 통제).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
