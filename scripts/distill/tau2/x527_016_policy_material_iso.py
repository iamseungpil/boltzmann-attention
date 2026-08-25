# -*- coding: utf-8 -*-
r"""x527 — 016 격리 음성이 **재료 결손**인가 (2026-08-25·무료·x516/x517 후속)

## 왜
`x516`(후보집합)·`x517`(질문 프레임)이 016 에서 gold `submit_transaction` **0/39** 를 냈고,
큐는 그것을 *"⑦유도 경로 없음"* 으로 적었다. 그런데 사용자 확정(2026-08-24 축자):
*"격리에서 실패하는 경우는 거의 모두 정보가 부족하거나 뭔가 빠뜨렸을 경우일거다. 100%다."*

016 의 gold 는 `submit_transaction {user_id: friend_user_5839, amount: 750}` — **손님이 아니라
그 친구**가 입금해야 추천이 자격을 얻는다. 그 조건은 **정책 문서**에 축자로 있다
(*"### Qualification criteria To earn a referral bonus: 1. …"*). 그런데 `formalize_intent_tool`
이 보는 것은 **마지막 손님 발화 6개뿐**이다 ⇒ 답에 필요한 사실이 창 안에 **원리상 없다**.

## 팔 (재료 한 칸만 민다 · 정본 함수를 그대로 호출·[[67]])
    A_asis      x516 과 동일한 창(손님 발화 6개)                     → 0/39 재현 확인
    B_policy    창 + **그 sim 이 실제로 받은 자격조건 문서 축자**     → 재료를 주면 갈리나
    C_neutral   B_policy + 중립 물음(x517 의 `C_neutral` 문면·`ask` 인자)
    N_sham      창 + **같은 길이의 무관 문서**                        → 부정통제([[57]])

## ⚠구조적 한계 — 그리고 그것이 결과의 일부다
`formalize_intent_tool` 은 프롬프트를 **손님 발화에서만** 만든다. 그래서 정책을 넣을 채널이
**손님 발화 자리밖에 없다**. 즉 이 프로브가 B 를 만들 수 있다는 사실 자체가
*"라이브 서브에는 정책이 들어갈 입구가 없다"* 는 배선 결손의 증거다([[71]] 재료는 선언에서
읽어 전달 — 이 서브는 그 전달 경로가 아예 없다).

## 채점 (닫힌 술어)
    gold_hit   산출이 `submit_transaction` 인가 (gold 는 **채점에만** · 프롬프트 미접촉·[[23]])
    in_cands   산출이 후보집합 안인가 (날조 검출)

사용: (리모트·cwd=tau2) py -3 x527_016_policy_material_iso.py --port 8140 --limit 12
"""
import argparse
import collections
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x516_induction_target_iso as X16          # noqa: E402  정본 재사용([[67]])
import t2_resolve as R                            # noqa: E402  정본 함수 그대로 호출

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GOLD = "submit_transaction"
# ★재료 정정 (2026-08-25): 어제는 `Qualification criteria`(추천 **입금** 자격)를 넣었는데
#   016 의 gold 는 입금이 아니라 **카드 적격 지출**이다 —
#     gold 016_1 = user · submit_transaction {credit_card_type:"Silver Rewards Card",
#                    merchant_name:"Best Buy", amount:750, category:"Shopping"}
#   그리고 그 요건은 궤적에 축자로 있다(도착이 **늦어** 서브 창에는 원리상 안 들어간다):
#     "The referred person must be approved and spend at least $750 within 60 days of account opening"
#   ⇒ 어제의 0/24 는 *재료를 줘도 안 된다*가 아니라 **틀린 재료를 준 것**이다.
NEEDLE = "must be approved and spend at least"
ASK_NEUTRAL = ("which ONE of these tools must be EXECUTED to fulfill the request "
               "(by the agent, or by the customer themselves)? ")


def sim_docs(needle=NEEDLE):
    """simtag → (자격조건 문서 축자, 같은 길이의 무관 문서). 전부 **그 sim 의 궤적**에서 뜬다."""
    out = {}
    for tag in X16.RUNS:
        rp = os.path.join(X16.SIMS, tag + ".results.json.gz")
        if not os.path.exists(rp):
            continue
        d = json.load(gzip.open(rp, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            if s.get("task_id") != X16.TASK:
                continue
            key = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            pol, other = None, None
            for m in (s.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                c = str(m.get("content") or "")
                if pol is None and needle in c:
                    pol = c
                elif other is None and needle not in c and len(c) > 400:
                    other = c
            if pol:
                out[key] = (pol, other or "")
    return out


def run_case(la, agent, case, arm, docs):
    msgs = list(case["msgs"])
    ask = None
    pol, sham = docs.get(case["simtag"], ("", ""))
    if arm in ("B_policy", "C_neutral"):
        if not pol:
            return None
        msgs = msgs + [X16._Msg("user", pol)]
    if arm == "C_neutral":
        ask = ASK_NEUTRAL
    if arm == "N_sham":
        if not sham:
            return None
        msgs = msgs + [X16._Msg("user", sham[:len(pol) or 2000])]
    try:
        return R.formalize_intent_tool(agent, la, X16._UM, msgs, case["cands"], ask=ask)
    except TypeError:                      # `ask` 인자가 없는 판본 폴백
        return R.formalize_intent_tool(agent, la, X16._UM, msgs, case["cands"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--needle", default=NEEDLE, help="정책 축자를 찾을 바늘(궤적 직독)")
    ap.add_argument("--arms", default="A_asis,B_policy,C_neutral,N_sham")
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "..", "..", "reports", "facet_rft_2026",
        "x527_016_policy_material_2026_08_25.json"))
    a = ap.parse_args()

    cases = X16.windows()
    if a.limit:
        cases = cases[:a.limit]
    docs = sim_docs(a.needle)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    la, agent = X16._LA(a.port), X16._Agent()
    print("=" * 96)
    print("x527 · 창 %d · 자격조건 문서 보유 sim %d · 팔 %s" % (len(cases), len(docs), arms))
    print("=" * 96)

    rows, tally = [], {arm: collections.Counter() for arm in arms}
    for i, c in enumerate(cases):
        line = "%3d %-22s " % (i, c["simtag"])
        row = {"simtag": c["simtag"], "turn_k": c["turn_k"], "run": c["run"]}
        for arm in arms:
            got = run_case(la, agent, c, arm, docs)
            got = str(got) if got is not None else "None"
            row[arm] = got
            tally[arm][got] += 1
            if got == GOLD:
                tally[arm]["__GOLD__"] += 1
            line += "%s=%-24s " % (arm, got[:24])
        rows.append(row)
        print(line)

    print("=" * 96)
    for arm in arms:
        t = tally[arm]
        print("  %-10s gold %2d/%d · 최빈 %s"
              % (arm, t.get("__GOLD__", 0), len(rows),
                 [x for x in t.most_common(3) if x[0] != "__GOLD__"][:2]))
    with io.open(os.path.normpath(a.out), "w", encoding="utf-8") as f:
        json.dump({"probe": "x527", "date": "2026-08-25", "task": X16.TASK, "gold": GOLD,
                   "n_windows": len(rows), "arms": arms,
                   "tally": {k: dict(v) for k, v in tally.items()}, "rows": rows,
                   "limits": [
                       "정책은 **손님 발화 자리**로 들어간다 — 이 함수에 다른 입구가 없다(그 사실이 결손이다).",
                       "창·후보집합은 x516 정본에서 그대로 온다(사본 0).",
                       "gold 는 채점에만 썼다([[23]])."]},
                  f, ensure_ascii=False, indent=1)
    print("→ %s" % os.path.normpath(a.out))


if __name__ == "__main__":
    main()
