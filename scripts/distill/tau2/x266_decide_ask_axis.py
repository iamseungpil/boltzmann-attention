# -*- coding: utf-8 -*-
r"""x266 — 결정 서브의 `ask` 에 **축이 머리에 있는가**가 라이브 답을 가르는가 (유료 0 · 엔진 0).

## 왜 (라이브 실측 2026-08-12 · `bank_all6b_20260811`)

배달된 결정 값 3건을 전수로 읽었더니 **쓸 수 있는 것이 0** 이었다:

    sim 840dd04ac4a0 t42 → "Sky Blue Account."                          ← 접미사
    sim b8e8cbdc74df t48 → "True Blue Business Checking Gold Saver …"    ← 이름 다중
    sim b8e8cbdc74df t50 → "Gold Saver Business Savings True Blue …"     ← 이름 다중

**밀린 것은 0이었다** — 말한 3번 전부 배달됐다. 즉 이 계열의 병은 rank 도 억제도 아니고
**서브가 라이브에서 내는 답 자체**다.

코드를 보면 원인이 하나로 좁혀진다. `group_prompt` 은 축을 **여러 개** 대라 하고 엔진은 축
하나(`_g`)의 재료만 만드는데, `decide_from_docs(ask=_ask)` 의 `_ask` 는 **마지막 손님 발화 4개**
= 두 요청 전부이고 축이 머리에 없다. x248 이 두 축 8/8 을 낸 구성은 축이 머리에 있는 쪽이다
(C417⒠ 축자: *"결정점은 요청 하나에 하나이므로 지금 무엇을 묻는지가 앞에 와야 한다"*).
그 처방이 `formalize_groups` 에는 들어갔고 **`decide_from_docs` 에는 안 들어갔다.**

## 무엇을 재나 — **출시 경로로** 잰다

엔진의 실제 함수(`t2_search.material_for` · `decide_from_docs`)와 실제 A2 템플릿을 쓴다.
프로브가 제 문구를 새로 쓰면 C435 에서 내가 밟은 함정(측정한 경로 ≠ 출시 경로)이 재생된다.

  A_SHIPPED   ask = 라이브 그대로(손님 발화 마지막 4개·축 머리 없음)   ← 현행
  B_AXIS      ask = **축 한 줄** + 같은 것                              ← 처방
  C_AXIS_ONLY ask = 축 한 줄만 (부정 통제 — 요청 본문 없이도 사는가)

축 문장은 **A2 슬롯**에서 온다(엔진 리터럴 0). 축 이름은 LLM 이 `formalize_groups` 로 낸
A3 키다 — LLM 출력을 LLM 입력으로 나르는 것이지 엔진이 고르는 것이 아니다.

계기: `EXACT`(gold 축자) · `SUFFIX`(gold + 접미사) · `MULTI`(이름 2개 이상) · `OTHER`.
⚠`SUFFIX` 를 성공으로 세지 않는다 — 채점 칸은 `account_class` 축자다.

실행(리모트·GPU1):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x266_decide_ask_axis.py [N]
"""
import collections
import json
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as S                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get(
    "X266_RESULTS",
    "/home/woori/scratch/tau2-bench/data/simulations/bank_all6b_20260811/results.json")
DOM = os.environ.get("X266_DOMAIN",
                     "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge")
DOCS = os.path.join(DOM, "documents")
NOW = "2025-11-14"
# gold = 태스크 채점 칸 축자 (x248 과 같은 출처)
GOLD = {"business_checking_accounts": "Sky Blue",
        "business_savings_accounts": "Gold Saver Account"}


def a2():
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    return json.load(io.open(p, encoding="utf-8"))


def live_ask(sim, upto):
    """엔진이 짓는 것과 **같은** ask: 손님 발화 마지막 4개 · ' --- ' 결합 · 6000자 컷."""
    us = [str((m.get("content") or "")) for m in sim["messages"][:upto]
          if m.get("role") == "user"]
    return " --- ".join(us[-4:])[-6000:]


def pick_groups(po, groups, ask):
    """① 군 형식화 — **출시본 `group_prompt` 그대로**. 엔진은 답이 A3 키 집합의 원소인지만 본다."""
    from x216_read_and_offset import chat
    listing = "\n".join("  %s" % g for g in sorted(groups))
    body = po["group_prompt"].format(groups=listing, text=str(ask)[:6000])
    raw = " ".join(str(chat(body, None, 0.0, 60).get("content", "") or "").split())
    return [g for g in groups if g and g.lower() in raw.lower()]


def classify(raw, gold, allnames):
    t = " ".join(str(raw or "").split()).strip().strip("*.").strip()
    if not t:
        return "EMPTY"
    hits = [n for n in allnames if n and n.lower() in t.lower()]
    # 이름이 둘 이상 실리면 결정이 아니다 (라이브 실패형)
    if len({h.lower() for h in hits}) >= 2:
        return "MULTI"
    if t.lower() == gold.lower():
        return "EXACT"
    if t.lower().startswith(gold.lower()):
        return "SUFFIX(%s)" % t[:34]
    return "OTHER(%s)" % t[:34]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    from x216_read_and_offset import chat                           # noqa: F401
    A = a2()
    po = A["policy_ontology"]
    d = json.load(io.open(RES, encoding="utf-8"))
    sims = {s["task_id"]: s for s in d["simulations"]}
    corpus = None

    # 축 문장은 A2 슬롯 — 없으면 프로브가 만들지 않고 멈춘다(문구 신설 금지·[[03b]])
    axis_tpl = po.get("decide_axis_text")
    if not axis_tpl:
        print("A2 에 `decide_axis_text` 슬롯이 없다 — 프로브가 문구를 만들지 않는다.\n"
              "  이 프로브는 그 슬롯을 둔 뒤에 돌린다(측정한 문구 = 출시할 문구·[[03b]]).")
        return 2

    for task in ("task_071",):
        sim = sims.get(task)
        if not sim:
            print("%s 없음" % task)
            continue
        # 결정점 = 첫 open_bank_account 디스패처 호출 직전
        cut = len(sim["messages"])
        ask = live_ask(sim, cut)
        groups = list(po.get("doc_index") or {})
        gs = pick_groups(po, groups, ask)
        print("=" * 92)
        print("%s · 손님발화로 뽑은 축: %s" % (task, gs or "없음"))
        for g in gs:
            gold = GOLD.get(g)
            if not gold:
                print("  %s — gold 미상, 건너뜀" % g)
                continue
            mat, info = S.material_for(A, g, DOCS, NOW, corpus=corpus)
            allnames = sorted(set(GOLD.values()))
            arms = (("A_SHIPPED", ask),
                    ("B_AXIS", axis_tpl.format(group=g) + "\n\n" + ask),
                    ("C_AXIS_ONLY", axis_tpl.format(group=g)))
            print("  축 %s · gold=%r · 재료 %d자 · 문서 %d(뺀 것 %d)"
                  % (g, gold, len(mat), info["kept"], len(info["dropped"])))
            for label, a in arms:
                c = collections.Counter()
                for _i in range(n):
                    try:
                        body = po["doc_decide_prompt"].format(
                            ask=str(a)[:3000], material=mat)
                        r = chat(body, None, 0.0 if _i == 0 else 0.7, 40).get("content")
                    except Exception as e:
                        r = "ERR %s" % type(e).__name__
                    c[classify(r, gold, allnames)] += 1
                print("    %-12s EXACT %d/%d   %s"
                      % (label, c["EXACT"], n, c.most_common(3)))
    print("\n※ 읽는 법 — `B_AXIS` 만 EXACT 가 높으면 병은 **ask 구성**이고 처방은 축 한 줄이다."
          "\n  `A_SHIPPED` 도 높으면 라이브 실패의 원인은 다른 데 있다(ask 가 아니다)."
          "\n  `C_AXIS_ONLY` 가 `B_AXIS` 만큼 높으면 요청 본문은 이 결정에 기여하지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
