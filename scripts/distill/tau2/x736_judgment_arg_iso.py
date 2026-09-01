# -*- coding: utf-8 -*-
"""x736 — **인자 한 칸**을 격리로 묻는다 — 계산기를 짓기 전에 LLM 이 맞히는지부터 (사용자 지시 2026-09-01: *"039 가 열린 판단이면
격리로 되는지부터 보라"*).

배경: A2 `field_ops` 는 이 칸을 **judgment** 로 분류한다(`compute` 아님) —
`eligible_for_provisional_credit` · `provisional_credit_eligible` · `partial_refund_amount`.
039 는 gold=false 인데 우리는 true 를 썼다(msg57). 084 도 같은 칸을 틀렸다.

재는 것: **쓰기 직전까지 라이브가 실제로 받은 재료**를 그대로 주면 모델이 그 불리언을 맞히나.
  · 맞히면 결손은 능력이 아니라 **거리·전달**이다(→ 쓰기 시점 전달 레버).
  · 못 맞히면 사용자 규율대로 **넘어간다**(*"격리로 안되면 넘어가야 한다"*).

팔:
  A_LIVE      쓰기 직전까지의 손님 발화 + 도구 출력 전부 (= 정보-맞춘 격리 [[18]])
  B_CUSTOMER  손님 발화만 · **부정통제**([[57]]) — 도구 출력이 실제로 쓰이는지 확인

규율: 서브는 gold 를 보지 않는다([[23]]) · 엔진은 **비교만** 한다([[10]]/[[52]]) ·
  문서 id·정책 문장을 코드에 적지 않는다 — **궤적이 받은 것만** 준다([[71]]) ·
  한도는 모델 프로필 선언에서 읽는다([[18]] 라이브와 같은 짝).

⚠사용자 지시(2026-09-01): *"계산하기 전에 격리로 LLM 이 정답을 맞출 수 있나를 먼저 확인하라."*
   그래서 `judgment` 칸뿐 아니라 `compute` 칸(금액·요율)도 같은 방식으로 묻는다 — 맞히면
   계산기는 필요 없고, 못 맞히면 그때 비로소 이관을 논한다([[62]] 결손을 먼저 재라).

사용: x736_judgment_arg_iso.py <base_url> <model> <tag> <task_id> <arg_name> [반복]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F
from x733_048_percard_probe import TRUNC, ask, limits, persist

HERE = os.path.dirname(os.path.abspath(__file__))

TOOL_CAP = int(os.environ.get("T2_PROBE_TOOL_CAP") or 0)   # 기본 무제한(리뷰 X-1)


def find_write(sim, arg_name):
    """그 인자를 담은 **우리 쓰기**와 **짝이 되는 gold 행**을 찾는다. 판정은 하지 않는다.

    ⚠첫 판에서 039 를 `gold=true` 로 찍었는데 실물 대조는 `gold=false ↔ ours=true` 였다 —
      한 태스크에 같은 도구의 행이 여럿인데 **첫 행을 집었기** 때문이다. 짝짓기는 **대상 id**
      (transaction_id 등)로 한다. id 가 겹치지 않으면 **인자 일치 수가 최대**인 행을 고르고,
      후보가 둘 이상 동률이면 **모른다**로 두고 실행하지 않는다(잘못된 gold 로 재면 무의미하다).
    """
    d = F.mutation_diff(sim)
    ours = None
    for r in (d.get("wrongarg") or []) + (d.get("extra") or []) + (d.get("done") or []):
        if arg_name in (r.get("args") or {}):
            ours = r
            break
    if ours is None:
        return None, None
    oa = ours.get("args") or {}
    cands = [g for g in (d.get("gold") or [])
             if g.get("name") == ours.get("name") and arg_name in (g.get("args") or {})]
    if not cands:
        return ours, None
    if len(cands) == 1:
        return ours, cands[0]
    scored = []
    for g in cands:
        ga = g.get("args") or {}
        same = sum(1 for k in ga if k != arg_name and str(ga.get(k)) == str(oa.get(k)))
        scored.append((same, g))
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        print("  ⚠gold 행 짝짓기 모호(동률 %d) — 실행하지 않는다" % scored[0][0])
        return ours, None
    return ours, scored[0][1]


def materials(sim, upto, with_tools=True):
    out = []
    for m in (sim.get("messages") or [])[:upto]:
        r, c = m.get("role"), str(m.get("content") or "").strip()
        if not c:
            continue
        if r == "user":
            out.append("CUSTOMER: " + c)
        elif r == "tool" and with_tools:
            out.append("TOOL RESULT: " + (c[:TOOL_CAP] if TOOL_CAP else c))
    return "\n\n".join(out)


def a3_quotes(arg_name):
    """A3 `policy_facts` 에서 **그 축의 선언된 축자**를 읽는다 — 우리가 고르지 않는다([[71]]).

    선언이 없으면 빈 목록을 돌려주고, 그 사실 자체가 결손의 신호다([[62]]).
    """
    fn = os.path.join(HERE, "a2", "banking_knowledge.policy_facts.json")
    if not os.path.exists(fn):
        return []
    with io.open(fn, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for r in d.get("rows") or []:
        if r.get("axis") != arg_name:
            continue
        for src in r.get("sources") or []:
            q = str(src.get("quote") or "").strip()
            if q:
                out.append("[%s]\n%s" % (src.get("doc"), q))
    return out


def call_facts(ours, arg_name):
    """같은 호출의 **다른 인자들** = 모델이 이미 형식화해 둔 사건 사실."""
    a = (ours or {}).get("args") or {}
    return "\n".join("- %s: %s" % (k, v) for k, v in sorted(a.items()) if k != arg_name)


def build_q(mat, arg):
    """근거를 **문서 id + 축자**로만 받는다 — 에코 문구를 넣지 않는다(리뷰 X-2).

    종전 프롬프트는 "if no policy sentence in the material above decides it, say so in that
    field" 라고 시켰고, 돌아온 근거 16줄이 그 문장의 **바이트 동일 반향**이었다. 그래서
    *"근거를 못 댔다"* 가 관측이 아니라 **지시의 메아리**였다. 이제 모델은 문장을 **옮겨 적기만**
    하고, 그것이 재료 안에 실재하는지는 **엔진이 substring 으로** 판정한다([[52]]).
    """
    parts = ["You are a bank support agent. Below is what you have seen in this conversation.",
             "", "=== CONVERSATION ===", mat, "=== END ===", "",
             "You are filing the customer's dispute. One field of that filing is: " + arg,
             "Decide its value.",
             "Then report the document id you relied on and the exact sentence from it.",
             "Copy that sentence character for character from the material above;",
             "do not paraphrase and do not summarise."]
    return chr(10).join(parts)


def grounded(quote, mat):
    """엔진의 유일한 판정 = **축자 실재**. 해석·순위 0([[59]])."""
    q = " ".join(str(quote or "").split())
    m = " ".join(str(mat or "").split())
    return bool(q) and len(q) >= 20 and q in m


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8141/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.8-27B-FP8"
    tag = sys.argv[3] if len(sys.argv) > 3 else "bank_x725_t3prime_A_20260901"
    task = sys.argv[4] if len(sys.argv) > 4 else "task_039"
    arg = sys.argv[5] if len(sys.argv) > 5 else "eligible_for_provisional_credit"
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 3

    sim = next((s for s in F.sims(tag) if s.get("task_id") == task), None)
    if sim is None:
        print("sim 없음: %s / %s" % (tag, task))
        return 2

    ours, gold = find_write(sim, arg)
    if gold is None:
        print("gold 에 그 인자를 담은 행이 없다: %s" % arg)
        return 3
    gv = str((gold.get("args") or {}).get(arg)).lower()
    ov = str((ours.get("args") or {}).get(arg)).lower() if ours else "(없음)"
    cut = (ours or {}).get("msg_i")
    if cut is None:
        print("우리 쓰기 위치를 못 찾았다 — 궤적 끝까지를 재료로 쓴다")
        cut = len(sim.get("messages") or [])

    MT, TB = limits(model)
    live = materials(sim, cut, True)
    cust = materials(sim, cut, False)
    print("%s / %s / %s" % (tag, task, arg))
    print("  gold=%s · 라이브가 쓴 값=%s · 쓰기 위치 msg%s" % (gv, ov, cut))
    print("  재료: A_LIVE %d자 · B_CUSTOMER %d자 · 한도 %s/%s" % (len(live), len(cust), MT, TB))

    # 불리언 칸이면 enum 으로 조이고, 그 밖의 칸(금액·요율·범주)은 자유 문자열로 둔다.
    # ⚠후보 목록을 우리가 만들지 않는다 — 만드는 순간 엔진이 선택지를 좁힌 것이 된다([[62]]).
    if gv in ("true", "false"):
        vsch = {"type": "string", "enum": ["true", "false"]}
    else:
        vsch = {"type": "string"}
    sch = {"type": "object", "required": ["value", "source_doc", "quote"], "properties": {
        "value": vsch, "source_doc": {"type": "string"}, "quote": {"type": "string"}}}

    quotes = a3_quotes(arg)
    facts = call_facts(ours, arg)
    arms = [("A_LIVE", live), ("B_CUSTOMER", cust)]
    if quotes:
        sep = chr(10) + chr(10)
        decl = ("=== BANK POLICY (verbatim, as declared for this field) ===" + chr(10)
                + sep.join(quotes)
                + chr(10) + "=== END POLICY ===" + sep
                + "=== THE CASE, AS ALREADY ESTABLISHED ===" + chr(10) + facts)
        arms.append(("C_A3", decl))
        arms.append(("D_A3_TOOLS", decl + sep + "=== TOOL RESULTS ===" + chr(10) + live))
        print("  A3 선언 축자 %d개 · %d자 · 사건 사실 %d자" % (len(quotes), len(decl), len(facts)))
    else:
        print("  ⚠A3 에 이 축의 선언이 **없다** — 선언 팔을 돌리지 않는다(결손 자체가 관측이다)")

    res = {}
    for armname, mat in arms:
        hits, gnd, preds, bases = 0, 0, [], []
        for rep in range(reps):
            r = ask(base, model, build_q(mat, arg), sch, max_tokens=MT, tb=TB)
            if r is None:
                preds.append("무응답")
                continue
            v = str(r.get("value")).strip().lower()
            g = grounded(r.get("quote"), mat)   # ★엔진의 판정은 축자 실재뿐
            preds.append(v)
            gnd += 1 if g else 0
            bases.append({"doc": str(r.get("source_doc"))[:60],
                          "quote": str(r.get("quote"))[:200], "grounded": g})
            hits += 1 if v == gv else 0
            print("  %s rep%d = %-24s 근거실재=%s" % (armname, rep, v, g), flush=True)
        res[armname] = (hits, preds, bases, gnd)

    print("\n=== 결과 (gold=%s · %d회) ===  ⚠무응답 %d건" % (gv, reps, TRUNC["n"]))
    for armname, _mat in arms:
        hits, preds, bases, gnd = res[armname]
        print("%-11s 정답 %d/%d · 근거실재 %d/%d  예측=%s"
              % (armname, hits, reps, gnd, reps, preds))
        for b in bases[:2]:
            print("      근거[%s] 실재=%s: %s" % (b["doc"], b["grounded"], " ".join(b["quote"].split())[:150]))
    persist("x736_%s_%s_%s" % (task, arg, tag), {
        "probe": "x736", "tag": tag, "task": task, "arg": arg, "model": model,
        "limits": {"max_tokens": MT, "thinking_token_budget": TB},
        "tool_cap": TOOL_CAP, "gold": gv, "live": ov, "cut_msg": cut,
        "a3_quotes": len(quotes), "trunc": TRUNC["n"],
        "materials": dict((k, len(m)) for k, m in arms),
        "arms": dict((k, {"hits": res[k][0], "preds": res[k][1],
                          "bases": res[k][2], "grounded": res[k][3]}) for k in res)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
