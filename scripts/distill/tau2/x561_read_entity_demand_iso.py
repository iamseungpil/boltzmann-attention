# -*- coding: utf-8 -*-
r"""x561 — `T2_READ_PER_ENTITY` 의 **효과**를 격리로 잰다 ([[62]] ②·유료 0).

## 무엇을 아직 모르나

`x560` 은 **발화면**만 쟀다(7/33·전부 reward 0). 모르는 것은 하나다 —
*"주체를 지목해 선행 read 를 다시 요구하면, 모델이 그 read 를 **그 주체로** 부르는가."*
안 부르면 이 레버는 턴만 먹는다([[70]] 선언 주석 축자: *"read 강제는 턴을 먹는다"*).

## 문맥과 팔

문맥 = `task_016#s626729`(t7363) 의 **친구 id 가 도착한 직후**. 라이브 축자 —
    msg[37] user  *"Yes — my friend's user ID is **friend_user_5839**."*
그 다음 라이브가 한 것은 `get_credit_card_transactions_by_user{user_id: friend_user_5839}` 였고
계좌 read 는 끝내 안 돌았다(그래서 카드 종류를 못 얻고 원장 15행 중 어느 행인지 못 정한다).

    A_asis   그대로                                   ← 재현 게이트(계좌 read 를 **안 불러야** 한다)
    B_demand + 라이브가 실제로 낼 문장(`merged_text`)  ← 저작 0·엔진이 조립한 그대로
    N_len    길이만 맞춘 무관 문장([[57]])

## 채점 — 닫힌 술어 · gold 무참조([[23]])

답한 도구 이름이 선언된 그 read 이고 **인자에 그 주체 값이 있는가**. 값은 손님이 준 것이고
이름은 선언에서 온다. 엔진은 어느 도구가 옳은지 고르지 않는다.

사용: PYTHONPATH=. py -3 x561_read_entity_demand_iso.py --port 8140
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gate_interpreter as GI                                       # noqa: E402
import t2_dominance as DOM                                          # noqa: E402
import t2_forensic as F                                             # noqa: E402
import x559_016_row_pick_iso as X559                                # noqa: E402
import x560_read_entity_gap_scan as X560                            # noqa: E402

NL = chr(10)
READ = "get_all_user_accounts_by_user_id"
ASK = (NL + NL + "What is the very next tool call you make? Reply with one line only, "
       "in the form `tool_name {\"arg\": \"value\"}`. Nothing else.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tag", default="bank_t7363_hard0_20260827")
    ap.add_argument("--sim", default="task_016#s626729")
    ap.add_argument("--target", default="submit_referral")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    a2 = GI.load_domain_a2("banking_knowledge") or {}
    sims = [s for s in F.sims(a.tag) if F.simtag(s) == a.sim]
    if not sims:
        print("그 sim 이 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2
    ms = sims[0].get("messages") or []
    # 결정점 = **레버가 실제로 서는 첫 자리**. 손님이 주체를 말한 직후가 아니다 —
    # 술어는 호출 인자만 보므로(NL 미독·[[59]]) 모델이 그 주체를 어딘가에 **넣은 뒤**에야
    # 선다. 그 자리를 코드로 짚지 말고 **찾아서** 쓴다(내가 msg[38] 로 짐작했다가 요건 0 을 봤다).
    w, reqs = None, []
    os.environ["T2_READ_PER_ENTITY"] = "1"
    for i in range(2, len(ms) + 1):
        rq = [r for r in DOM.requirements_for(a2, [X560._M(m) for m in ms[:i]], a.target,
                                              unwrap=X560._unwrap) if "@" in str(r.get("id"))]
        if rq:
            w, reqs = i, rq
            break
    os.environ.pop("T2_READ_PER_ENTITY", None)
    if not w:
        print("이 sim 어디에서도 레버가 안 선다", file=sys.stderr)
        return 2
    demand = DOM.merged_text(a2, reqs, a.target)
    print("# x561 — 결정점 msg[%d] · 그 자리 요건 %d" % (w, len(reqs)))
    for r in reqs:
        print("   ·", r.get("id"))
    if not demand:
        print("요건이 안 선다 — 이 자리에서는 레버가 침묵한다. 판정하지 않는다.", file=sys.stderr)
        return 2
    print("--- B_demand 문면 ---")
    print(demand)
    base = X559.render(ms, w)
    print("--- 문맥 %d자 · 라이브의 실제 다음 호출 ---" % len(base))
    nxt = next((tc for m in ms[w:] for tc in (m.get("tool_calls") or ())), None)
    print("   ", F.label(F.nameof(nxt), F.argsof(nxt)) if nxt else "없음",
          F.argsof(nxt) if nxt else "")
    if a.wiring_only:
        return 0

    adds = {"A_asis": "",
            "B_demand": NL + NL + demand,
            "N_len": (NL + NL + "[note] the details above were gathered earlier in this "
                      "conversation and have not changed since; they remain current and "
                      "complete for this customer and for the party they are asking about.")}
    print()
    print("%-10s %-5s %-58s %s" % ("팔", "temp", "지목한 다음 호출", "판정"))
    print("-" * 100)
    tally = collections.defaultdict(lambda: [0, 0])
    for nm in ("A_asis", "B_demand", "N_len"):
        body = base + adds[nm] + ASK
        for tp, k in ((0.0, 1), (a.temp, a.n)):
            for _ in range(k):
                try:
                    rep = " ".join(str(X559.gen(a.port, body, 96, tp)).split())
                except Exception as e:
                    print("%-10s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                low = rep.lower()
                ok = READ in low and "friend_user_5839" in low
                tally[nm][1] += 1
                tally[nm][0] += 1 if ok else 0
                print("%-10s %-5s %-58s %s" % (nm, tp, rep[:58], "그 주체로 계좌 read" if ok else "-"))
    print()
    print("## 판정 (선언된 read 를 **그 주체로** 부른 비율)")
    for nm in ("A_asis", "B_demand", "N_len"):
        print("   %-10s %d/%d" % (nm, tally[nm][0], tally[nm][1]))
    print()
    print("⚠A_asis 가 이미 부르면 결손이 아니다 — 판정하지 마라([[62]] 2b).")
    print("⚠N_len 이 B_demand 와 같으면 그 이득은 **길이**다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
