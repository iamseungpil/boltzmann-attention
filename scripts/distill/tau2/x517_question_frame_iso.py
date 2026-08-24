# -*- coding: utf-8 -*-
r"""x517 — S3b 격리: **결손이 후보집합이 아니라 물음의 프레임인가** (x509 S3b · 사용자 승인 2026-08-24 *"돌려라"*).

## 물음

`x516` 이 후보집합 가설을 기각했다 — `submit_referral` 을 빼도 서브는 gold `submit_transaction`
으로 가지 않고 **에이전트 디스패처로** 간다(38/39). 그 자리에서 드러난 것:

    "which ONE of these tools must the **agent CALL** to fulfill the request?"

`submit_transaction` 은 **손님-실행** 도구다. 그러니 이 물음의 옳은 답이 **될 수 없다**.
서브는 틀린 게 아니라 **틀린 질문에 맞게** 답하고 있었다.

⇒ 물음을 **모드-무관**으로 바꾸면 gold 를 고르는가. 그것만 잰다.

## 팔 (후보집합은 세 팔 모두 **동일** — 이번에 가르는 것은 문장뿐)

    A_agent    정본 문장 그대로(에이전트-프레임)         ← x516 의 A 재현이 검산
    C_neutral  "must be EXECUTED … (by the agent, or by the customer themselves)"
    N_word     같은 에이전트-프레임인데 **어휘만** 교체    ← 부정통제([[57]])

N_word 가 C_neutral 만큼 움직이면 산 것은 프레임이 아니라 **문장을 건드린 것**이다.

## 계기 — 사본 0 ([[67]])

정본 `t2_resolve.formalize_intent_tool` 에 `ask` 인자를 **추가**했고(기본값은 종전 문장과
바이트 동일 — 라이브 거동 불변), 프로브는 그 인자만 바꾼다. 창·후보집합·어댑터는
`x516_induction_target_iso` 를 그대로 import 한다.

## [[66]] 준수

케이스 규칙(*"아직 질문 중이면 none"* 류)은 **넣지 않는다** — 과거 af8c1e21 이 그러다 098
4/4→0/4 로 무너졌다. 여기서 바꾸는 것은 **일반 어법**뿐이다.

## 실행 (리모트 · GPU1 · 무료)

    PYTHONIOENCODING=utf-8 python x517_question_frame_iso.py --port 8141
"""
import argparse
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

import x516_induction_target_iso as X16      # 창·어댑터·재료 (사본 0)

OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")

ASKS = {
    # 정본 문장 — `t2_resolve.ASK_AGENT_CALL` 을 **인용**한다(여기 적지 않는다).
    "A_agent": None,
    "C_neutral": ("which ONE of these tools must be EXECUTED to fulfill the request "
                  "(by the agent, or by the customer themselves)? "),
    "N_word": ("which ONE of these tools must the agent INVOKE in order to satisfy "
               "the request? "),
}
ORDER = ("A_agent", "C_neutral", "N_word")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    import t2_resolve as RZ

    cases = X16.windows()
    if a.limit:
        cases = cases[:a.limit]
    if not cases:
        print("재료 없음 — 돌리지 않는다([[25]]).")
        return 1
    gold = ["submit_transaction"]      # x516 이 env `tasks[]` 선언에서 읽은 값. 채점에만 쓴다([[23]]).

    print("표적 %s · gold(requestor=user) = %s · 창 %d개" % (X16.TASK, gold, len(cases)))
    print("A_agent 문장(정본) = %r" % (RZ.ASK_AGENT_CALL,))
    for k in ORDER[1:]:
        print("%-9s 문장 = %r" % (k, ASKS[k]))
    print("")

    la = X16._LA(a.port)
    ag = X16._Agent()
    res = collections.defaultdict(collections.Counter)
    rows = []
    for i, c in enumerate(cases):
        row = {"run": c["run"], "simtag": c["simtag"], "turn_k": c["turn_k"],
               "n_cands": len(c["cands"])}
        for arm in ORDER:
            got = RZ.formalize_intent_tool(ag, la, X16._UM, c["msgs"], c["cands"],
                                           ask=ASKS[arm])
            row[arm] = got
            res[arm][str(got)] += 1
            if got in gold:
                res[arm]["__GOLD__"] += 1
        rows.append(row)
        print("  [%2d] %s k=%-2d A=%-26s C=%-26s N=%s"
              % (i, c["simtag"], c["turn_k"], row["A_agent"], row["C_neutral"], row["N_word"]))

    print("")
    print("=" * 100)
    print("결과 — 팔별 산출 분포 (n=%d 창 · 서브콜 %d회 · 후보집합 동일)"
          % (len(cases), la.calls))
    print("=" * 100)
    for arm in ORDER:
        g = res[arm].pop("__GOLD__", 0)
        dist = " · ".join("%s×%d" % kv for kv in res[arm].most_common(4))
        print("  %-10s gold %2d/%d   %s" % (arm, g, len(cases), dist))
        res[arm]["__GOLD__"] = g
    print("")
    print("판독:")
    print("  A_agent 가 x516 의 A(gold 0/39 · submit_referral 24)를 재현하면 계기가 살아 있다.")
    print("  C_neutral 이 gold 를 사고 **N_word 는 안 사면** 결손은 **프레임**이다 —")
    print("    선택을 모드-무관으로 두고 호출 형태만 뒤에서 가르는 배치가 실측 근거를 얻는다.")
    print("  둘 다 오르면 산 것은 프레임이 아니라 **문장을 건드린 것**이다([[57]]).")
    print("  C_neutral 도 0 이면 016 의 결손은 유도 층 **위**에 있고 ⑦유도 축은 큐에서 내려간다.")

    out = {"probe": "x517_question_frame_iso", "date": "2026-08-24",
           "task": X16.TASK, "gold_user": gold,
           "asks": {"A_agent": RZ.ASK_AGENT_CALL, "C_neutral": ASKS["C_neutral"],
                    "N_word": ASKS["N_word"]},
           "n_windows": len(cases), "subcalls": la.calls,
           "arms": {k: dict(v) for k, v in res.items()},
           "rows": rows,
           "limits": ["후보집합은 세 팔 동일 — 이 프로브가 가르는 것은 문장뿐이다.",
                      "temperature 0 · 창마다 1회. n 은 창 수이지 재시행 수가 아니다.",
                      "gold 는 채점에만 썼다([[23]]).",
                      "격리에서 되면 라이브도 된다는 뜻이 아니다 — 라이브는 부하가 다르다([[62]])."]}
    dst = os.path.join(OUT, "x517_question_frame_iso_2026_08_24.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
