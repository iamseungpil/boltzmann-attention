# -*- coding: utf-8 -*-
r"""x534 — 057·055 추천 턴 격리: **우리 도구 하나가 면접을 납치하나** (무료·2026-08-25)

## 관측 (궤적 축자 · t7348 halfB task_057)

    U1  "looking to open a new checking account in addition to my old one year old
         checking account. Can you help me figure out the best option and get it opened?"
    A2  "I'll need a bit more information about your **expected usage**:
         1. How many **months** ... 2. How many **ATM withdrawals** per month ...
         3. What is the typical **amount** per withdrawal"
    A4  [tool] get_checking_atm_fee_totals
    A6  "the **Green Fee-Free Account** and Light Blue Account both have zero
         out-of-network and foreign ATM fees, making them ideal choices"
    A8  "the **Green Fee-Free Account** seems to be the best fit"      ← gold 는 `Blue Account`
    U9  "Let's do it! What info do you need to open the **Green Fee-Free Account**?"
        ⇒ 손님이 오답을 이름으로 채택한다. 그 뒤 턴은 전부 하류다.

A2 의 세 질문은 우리 A2 선언 `get_checking_atm_fee_totals` 의 **세 파라미터와 정확히 일치**한다
(`months` · `withdrawals_per_month` · `withdrawal_amount`). 그리고 그 선언의 첫 낱말이
*"**MANDATORY** before recommending which personal checking account class to open (or keep)
**when the customer's stated criterion involves ATM fees**"* 인데 — 057 에서 손님은 그 시점까지
ATM 을 **한 번도 말하지 않았다**. 조건이 미충족인데 도구가 면접을 끌고 갔다는 것이 가설이다.

## 폭발 반경 (실측·t7346+t7348)

    task_057  호출 4회  reward [0,0,0,0]
    task_055  호출 3회  reward [0,0,0]
    그 밖     호출 0회
⇒ 이 도구는 **정확히 두 태스크에서만 발화하고 둘 다 0** 이다. 그리고 도입 근거 문서
(`CALC_LEVER_PASS_PROVENANCE_2026_08_19.md`)의 짝비교 칸이 **"없음"** — 효과가 한 번도
측정된 적이 없다([[70]] 레버 판정 의무 ①).

## 팔 (한 번에 한 줄만 바꾼다 · [[57]] 부정통제 포함)

    A_asis  A2 가 선언한 도구 전량                          ← 라이브 산출을 재현해야 한다
    B_excl  거기서 `get_checking_atm_fee_totals` **하나만 제거**   ← [[63]] 제거형
    C_cond  그 도구는 두되 **선언의 자기 조건만 살린 문면**으로 교체(MANDATORY 삭제)
    N_neg   같은 수만큼 **다른 도구 하나** 제거              ← 목록이 짧아진 효과 통제

★B_excl 은 라이브에도 같은 스위치가 이미 있다 — `T2_SG_EXCLUDE`(정본 `t2_scaffold_get.py`
:1894 부근·이름 필터뿐·도메인 리터럴 0). 즉 이 격리가 통과하면 **새 코드 0 으로** 라이브 A/B 가 된다.

## 채점 — 닫힌 술어만·gold 미접촉([[23]])

    asks_usage   응답이 그 도구의 **자기 파라미터 낱말**(months·withdrawal(s)·per month)을 묻는가
    names_class  응답이 계좌 클래스 이름을 **바로 지목**하는가(선언된 이름 집합 소속)
    asks_any     물음표가 있는가
어느 클래스가 옳은지는 채점하지 않는다. 지목한 이름은 **분포로만** 남긴다([[69]] gold 는 진단 보조).

사용: (리모트·cwd=scripts/distill/tau2) py -3 x534_recommend_turn_iso.py --port 8141 --n 4
"""
import argparse
import gzip
import io
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
A2P = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TOOL = "get_checking_atm_fee_totals"
RUNS = ("bank_t7348_halfB_20260824", "bank_t7348_halfA_20260824")
TASKS = ("task_057", "task_055")

USAGE_WORDS = ("withdrawal", "withdrawals", "per month", "months")


def gen(port, body, maxtok=420):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def a2_tools():
    """선언에서 (이름, 설명) — 지어내지 않는다([[71]]②)."""
    d = json.load(io.open(A2P, encoding="utf-8"))
    out = []
    for t in (d.get("scaffold_get_tools") or []):
        n, ds = t.get("name"), t.get("description")
        if n and ds:
            out.append((str(n), str(ds)))
    return out


def class_names():
    """선언된 계좌 클래스 이름 집합 — 채점용(닫힌 집합·엔진 판단 0)."""
    p = os.path.join(REP, "x430_account_facts_full_2026_08_25.json")
    d = json.load(io.open(p, encoding="utf-8"))
    out = set()
    for k in d:
        out.add(k.replace("_", " ").title())
        out.add(k.replace("-", " ").replace("_", " ").title())
    return {s for s in out if s}


def opening(task):
    """그 태스크의 **첫 손님 요구**까지 — 궤적 축자."""
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            if s.get("task_id") != task:
                continue
            for m in (s.get("messages") or []):
                if m.get("role") == "user" and str(m.get("content") or "").strip():
                    return str(m.get("content")).strip()
    return None


def render(tools, drop=None, replace=None):
    out = []
    for n, ds in tools:
        if drop and n == drop:
            continue
        if replace and n == replace[0]:
            ds = replace[1]
        out.append("- %s: %s" % (n, ds))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(REP, "x534_recommend_turn_2026_08_25.json"))
    a = ap.parse_args(argv)

    tools = a2_tools()
    names = {n for n, _ in tools}
    if TOOL not in names:
        print("선언에 %s 가 없다 — 중단" % TOOL)
        return 1
    # 부정통제로 뺄 도구 = 이 두 태스크의 궤적에 **한 번도 안 나온** 도구 하나(닫힌 선택).
    other = next((n for n, _ in tools if n != TOOL and "card" in n), None)
    # C_cond: 선언 자신의 조건절만 남기고 'MANDATORY' 를 뺀다 — 새 규칙 0([[66]]).
    orig = dict(tools)[TOOL]
    cond = orig.replace("MANDATORY before recommending",
                        "Use this only after the customer has already told you that ATM fees "
                        "matter to them. Do not ask them for usage numbers in order to use it. "
                        "It is for recommending")
    CLS = class_names()
    rows, agg = [], {}
    for task in TASKS:
        first = opening(task)
        if not first:
            print("창 없음: %s" % task)
            continue
        arms = {"A_asis": render(tools),
                "B_excl": render(tools, drop=TOOL),
                "C_cond": render(tools, replace=(TOOL, cond)),
                "N_neg": render(tools, drop=other) if other else None}
        for arm, tl in arms.items():
            if tl is None:
                continue
            for k in range(a.n):
                body = ("You are a bank customer-service agent. In addition to the bank's own "
                        "tools you have these:\n" + tl +
                        "\n\nThe customer just said:\n\"" + first + "\"\n\n"
                        "Write your next reply to the customer. Reply with the message text only.")
                try:
                    txt = gen(a.port, body)
                except Exception as e:
                    txt = "!!%r" % (e,)
                low = txt.lower()
                asks_usage = any(w in low for w in USAGE_WORDS)
                named = sorted({c for c in CLS if c and c.lower() in low and len(c) > 8})
                rows.append({"task": task, "arm": arm, "k": k,
                             "asks_usage": asks_usage, "named": named,
                             "asks_any": "?" in txt, "text": txt[:500]})
                d0 = agg.setdefault((task, arm), {"n": 0, "usage": 0, "named": 0, "q": 0})
                d0["n"] += 1
                d0["usage"] += 1 if asks_usage else 0
                d0["named"] += 1 if named else 0
                d0["q"] += 1 if "?" in txt else 0
                print("%-9s %-7s k=%d usage=%-5s named=%s" % (task, arm, k, asks_usage, named),
                      flush=True)
    out = {"probe": "x534", "date": "2026-08-25", "tool": TOOL, "tasks": list(TASKS),
           "blast_radius": "t7346+t7348 에서 이 도구를 부른 태스크는 057(4회)·055(3회) 뿐이고 "
                           "reward 는 전부 0. 도입 근거 문서의 짝비교 칸은 '없음'.",
           "live_switch": "T2_SG_EXCLUDE (정본 t2_scaffold_get.py · 이름 필터뿐) — "
                          "B_excl 이 이기면 새 코드 0 으로 라이브 A/B 가 된다.",
           "agg": {"%s|%s" % k: v for k, v in agg.items()}, "rows": rows}
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n== agg ==")
    for k, v in agg.items():
        print(" %-9s %-7s usage=%d/%d named=%d q=%d" % (k[0], k[1], v["usage"], v["n"],
                                                        v["named"], v["q"]))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
