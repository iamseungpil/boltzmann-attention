# -*- coding: utf-8 -*-
r"""x201 — task_098 의 잔여 결손은 **타입**인가, 그리고 레버는 문구인가 필터인가 (유료 0).

## 어디까지 왔나

A3 의 예치 문턱을 채우자(x199·C387) `Gold Years` 는 통과 표에서 빠졌고 `B_sum` 은 0/8 → 8/8 이
됐다. 그런데 `A_iso` 는 여전히 0/8 이고, 이번 오답은 **`Business Platinum Rewards Card`** 다 —
카드에는 예치 축이 없어 *"모름 ≠ 탈락"* 규칙([[25]])대로 표에 남고, 모델은 합산 대신 **단일
최대 수(referrer 300)** 를 집는다. 손님은 친구가 **계좌를 여는** 이야기를 하고 있다.

## ⛔0 — 무엇을 재는가 (짓기 전에)

*"카드가 계좌 질문에 남는다"* 를 고치는 길은 둘이고, **어느 쪽이 되는지 모른 채 지으면 안 된다**:

  E_hint   표는 그대로 두고 **한 줄**로 무엇을 묻는지 말해 준다      → 되면 레버 = **전달**
  F_kind   엔진이 **타입으로 거른** 표를 준다                        → E 가 실패할 때만 정당
  G_llm    2단계: 모델이 타입을 **고르고**(해석) 엔진은 그 타입으로  → 실제 설계 후보
           행을 거른다(원소 검사·[[22]])

  A_iso    현행 통과 표 (기준선·0/8 이 재현되나)
  D_null   표 없이 질문만 (부정 통제)

**타입은 지어내지 않는다** — `source.doc` 의 문서군에서 기계적으로 나온다(§4b·전 주어가 정확히
한 군에 속함을 확인함). 엔진은 타입의 *뜻* 을 모르고 문자열로 묶기만 한다.

⚠어느 팔에서도 **정답을 말해 주지 않는다**. 순위·최댓값·argmax 는 이 파일에 없다([[62]]).

실행: python x201_type_axis.py [N] [--show]
"""
import collections
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402
import t2_factdag as FD                                         # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
GOLD = "Blue"                                                    # 진단 전용([[23]])
CASE = {"days": 400, "stated": {"qualifying_deposit_usd": 600}}
Q = ("Hey! My roommate just got her first real job and she's going to deposit her first paycheck "
     "- around $600 I think. I want to refer her to open one of the accounts you guys offer. "
     "Which one should I use to get the best possible combined bonus for both of us?")

# 문서군 → 타입. 엔진은 이 이름들의 **뜻을 모른다**; 문서 id 의 접두사를 그대로 쓴다.
GROUPS = ("business_checking_accounts", "checking_accounts", "business_credit_cards",
          "credit_cards", "savings_accounts", "bank_accounts")


def kind_of(rows, subject):
    """그 주어의 행이 인용한 **문서군**을 돌려준다 (없거나 갈리면 None = 강제하지 않는다)."""
    ks = set()
    for r in rows:
        if r.get("subject") != subject:
            continue
        doc = ((r.get("source") or {}).get("doc") or "")
        for g in GROUPS:
            if doc.startswith("doc_" + g + "_"):
                ks.add(g)
                break
    return sorted(ks)[0] if len(ks) == 1 else None


def scope_of(rows, subject):
    for r in rows:
        if r.get("subject") == subject and "_(general)_" in ((r.get("source") or {}).get("doc") or ""):
            return "general"
    return "product"


def ask(prompt, choices=None, temp=0.0, mx=24):
    body = {"model": MODEL, "temperature": temp, "max_tokens": mx,
            "messages": [{"role": "user", "content": prompt}]}
    if choices:
        body["guided_choice"] = list(choices)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = 8
    show = "--show" in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
    a2 = load_domain_a2("banking_knowledge")
    sp = next(x for x in a2["ledger_metrics"] if x.get("eligible_text"))
    rows = (a2.get("policy_ontology") or {}).get("rows") or []
    axes = (sp.get("eligible") or {}).get("show_axes") or []
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    tbl = (LG.eligible_text(CASE["days"], None, maps, sp, CASE["stated"]) or "").strip()
    body = [l for l in tbl.splitlines() if l.startswith("  ") and ":" in l]
    names = [l.strip().split(":")[0].strip() for l in body]
    kinds = {s: kind_of(rows, s) for s in names}
    print("통과 표 %d행 · 타입 분포: %s"
          % (len(names), dict(collections.Counter(v or "?" for v in kinds.values()))))
    for s in names:
        print("   %-32s %s / %s" % (s, kinds[s], scope_of(rows, s)))

    facts = "The customer says the deposit will be about %d." % CASE["stated"]["qualifying_deposit_usd"]
    tail = ("\n\nThe customer asked:\n%s\n\nAnswer with one name copied exactly from the list "
            "above, and nothing else." % Q)

    def filtered(keep):
        out = []
        for l in tbl.splitlines():
            if l.startswith("  ") and ":" in l:
                if kinds.get(l.strip().split(":")[0].strip()) not in keep:
                    continue
            out.append(l)
        return "\n".join(out)

    # E: 표는 그대로, **한 줄**만 덧붙인다. 이름을 고르지 않고 무엇을 묻는지만 말한다.
    HINT = ("\nThe customer is asking about an account their friend would open, so a product that "
            "is not an account of that sort is not an answer to this question.")

    # G: 모델이 타입을 고른다 → 엔진은 그 문자열로 행을 거른다(원소 검사).
    kind_choices = sorted({v for v in kinds.values() if v})
    pick_prompt = ("These are the product groups on record:\n%s\n\nConversation:\n%s\n\nWhich ONE "
                   "group does the product the customer is asking about belong to? Reply with that "
                   "group name exactly and nothing else." % ("\n".join("  " + k for k in kind_choices), Q))

    arms = [("A_iso", tbl), ("E_hint", tbl + HINT), ("F_kind", filtered({"checking_accounts"})),
            ("D_null", "")]
    res = {}
    print("\n" + "=" * 96)
    print("task_098 타입 축  gold=%r  (n=%d · %s)" % (GOLD, n, MODEL))
    print("=" * 96)
    allowed = sorted(set(names))
    for label, block in arms:
        c = collections.Counter()
        for i in range(n):
            p = (block + "\n\n" if block else "") + facts + tail
            try:
                c[ask(p, allowed, 0.0 if i == 0 else 0.7)] += 1
            except Exception as e:
                c["ERR %s" % type(e).__name__] += 1
        hit = sum(v for k, v in c.items() if str(k).strip() == GOLD)
        res[label] = [hit, n]
        print("  %-8s gold %d/%d   최빈: %s" % (label, hit, n, c.most_common(2)))

    # G_llm — 2단계
    c, picks = collections.Counter(), collections.Counter()
    for i in range(n):
        try:
            k = ask(pick_prompt, kind_choices, 0.0 if i == 0 else 0.7)
        except Exception:
            k = ""
        picks[k] += 1
        blk = filtered({k.strip()}) if k.strip() in kind_choices else tbl
        try:
            c[ask(blk + "\n\n" + facts + tail, allowed, 0.0 if i == 0 else 0.7)] += 1
        except Exception as e:
            c["ERR %s" % type(e).__name__] += 1
    hit = sum(v for k, v in c.items() if str(k).strip() == GOLD)
    res["G_llm"] = [hit, n]
    print("  %-8s gold %d/%d   최빈: %s" % ("G_llm", hit, n, c.most_common(2)))
    print("           타입 선택: %s" % picks.most_common(3))
    if show:
        print("\n[F_kind 표]\n%s" % filtered({"checking_accounts"}))

    json.dump(res, open(os.environ.get("T2_X201_OUT", "x201_out.json"), "w"), indent=1)
    print("\n※ E 가 되면 레버는 **전달(한 줄)** 이고 필터를 지을 이유가 없다."
          "\n  E 가 실패하고 F·G 가 되면 그때 타입 필터가 정당하다 — G 가 설계형이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
