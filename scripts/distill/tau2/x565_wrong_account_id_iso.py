# -*- coding: utf-8 -*-
r"""x565 — 잘못된 `account_id` 를 **문맥에 이미 있는 옳은 id 로** 바꾸는가 (유료 0).

## 결손 (x564 코퍼스 계수 · 2026-08-27)

선언(`arg_source_reads.account_id = [get_all_user_accounts_by_user_id_3847]`)이 그 값의 생산자를
적어 뒀는데 집행이 없다. `[PROVENANCE]` 술어는 `_ctx_has` — *"문맥 어딘가에 있나"* 라서 **다른
도구가 낸 값**이 그대로 통과한다. 코퍼스: 채점 37 sim 중 **15(41%)** 에서 발생하고 6 태스크
(040·057·072·074·079·085) · **전부 reward 0** · 걸린 인자는 **전부 `account_id`**.

## 이 프로브가 잰 자리

그 sim 에서 **생산자 read 가 이미 돌아 실제 id 를 문맥에 내놓은 뒤**, 모델이 `account_id` 에
다른 출처의 값을 넣는 첫 호출. 즉 **옳은 값이 이미 문맥에 있다** — 그래서 이것은 전달 결손이
아니라 결속 결손이고, 물어볼 값이 있다.

    A_asis   그대로                                    ← 재현 게이트(틀린 값이 나와야 한다)
    B_say    선언이 아는 것만 말한다 — 그 인자의 값은
             생산자 read 가 낸 것이어야 한다(그 목록은 위에 있다)   ← 값 지목 0
    N_len    길이만 맞춘 무관 문장([[57]])

⛔B_say 는 **어느 id 를 쓰라고 말하지 않는다.** 고르는 것은 끝까지 모델이다([[62]]③④).

## 채점 — 닫힌 술어

다음 호출의 `account_id` 값이 **생산자 read 출력에 축자로 있는가**. 엔진은 그 값이 무엇을
뜻하는지 모르고, 어느 계좌가 옳은지도 모른다.

사용: PYTHONPATH=. py -3 x565_wrong_account_id_iso.py --port 8140
"""
import argparse
import collections
import json
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
import x564_arg_producer_census as X564                             # noqa: E402

NL = chr(10)
ARG = "account_id"
ASK = (NL + NL + "What is the very next tool call you make? Reply with one line only, "
       "in the form `tool_name {\"arg\": \"value\"}`. Nothing else.")


def producer_values(ms, upto, prods, arg):
    """생산자 read 출력에서 **그 인자 이름으로 실제 쓰인 적이 있는** 값들 — 축자 대조용.

    ⚠엔진이 id 를 *추출*하지 않는다(그건 패턴매칭이다·[[59]]). 대화 안에서 그 인자에 쓰였고
      생산자 출력에 축자로 있는 값만 모은다 — 채점에만 쓰고 프롬프트에는 안 넣는다."""
    outs = []
    pend = {}
    for i, m in enumerate(ms[:upto]):
        for tc in (m.get("tool_calls") or ()):
            pend[i] = F.label(F.nameof(tc), F.argsof(tc))
        if m.get("role") == "tool":
            nm = ""
            for j in range(i - 1, max(-1, i - 3), -1):
                if j in pend:
                    nm = pend[j]
                    break
            if any(p.split("_by_")[0] in nm or nm in p for p in prods):
                outs.append(str(m.get("content") or ""))
    vals = set()
    for m in ms:
        for tc in (m.get("tool_calls") or ()):
            v = (DOM._args_dict(X564._TC(tc)) or {}).get(arg)
            if isinstance(v, str) and v.strip() and any(v.strip() in o for o in outs):
                vals.add(v.strip())
    return vals, outs


def find_spot(a2, tag, arg=ARG):
    """생산자가 이미 돈 뒤 잘못된 값이 들어가는 **첫** 호출. 없으면 None."""
    prods = [str(x) for x in ((a2.get("arg_source_reads") or {}).get(arg) or ())]
    for s in F.scored(tag):
        ms = s.get("messages") or []
        outs_seen = False
        pend = {}
        outs = []
        for i, m in enumerate(ms):
            for tc in (m.get("tool_calls") or ()):
                pend[i] = F.label(F.nameof(tc), F.argsof(tc))
            if m.get("role") == "tool":
                nm = ""
                for j in range(i - 1, max(-1, i - 3), -1):
                    if j in pend:
                        nm = pend[j]
                        break
                if any(p.split("_by_")[0] in nm or nm in p for p in prods):
                    outs_seen = True
                outs.append((i, nm, str(m.get("content") or "")))
            if not outs_seen:
                continue
            for tc in (m.get("tool_calls") or ()):
                v = (DOM._args_dict(X564._TC(tc)) or {}).get(arg)
                if not (isinstance(v, str) and len(v.strip()) >= 4):
                    continue
                val = v.strip()
                sp = any(val in t and any(p.split("_by_")[0] in nm or nm in p for p in prods)
                         for j, nm, t in outs if j < i)
                st = any(val in t for j, nm, t in outs if j < i)
                if (not sp) and st:
                    return s, i, val, prods
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tags", default="bank_t7356_grpA3_20260826,bank_t7363_hard0_20260827")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)
    a2 = GI.load_domain_a2("banking_knowledge") or {}

    spots = []
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        got = find_spot(a2, tag)
        if got:
            spots.append((tag,) + got)
    if not spots:
        print("그런 자리가 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2

    print("# x565 — 잘못된 %s 결속" % ARG)
    for tag, s, i, val, prods in spots:
        ok, _outs = producer_values(s.get("messages") or [], i, prods, ARG)
        print("  %-26s %-9s msg[%d] 잘못 쓴 값=%s · 생산자가 낸 옳은 값 %d개: %s"
              % (tag[:26], F.task_id(s), i, val, len(ok), sorted(ok)[:4]))
    if a.wiring_only:
        return 0

    print()
    print("%-26s %-8s %-5s %-46s %s" % ("tag", "팔", "temp", "다음 호출", "판정"))
    print("-" * 110)
    tally = collections.defaultdict(lambda: [0, 0])
    for tag, s, i, val, prods in spots:
        ms = s.get("messages") or []
        good, _o = producer_values(ms, i, prods, ARG)
        base = X559.render(ms, i)
        say = (NL + NL + "Error: [SOURCE] the value you are about to pass as '%s' did not come from "
               "%s, which is where the values for that argument come from. Its output is above in "
               "this conversation - take the value from there, and if the right one is not there, "
               "run that tool first." % (ARG, prods[0] if prods else "the lookup tool"))
        adds = {"A_asis": "", "B_say": say,
                "N_len": NL + NL + ("[note] " + "the details gathered so far in this conversation "
                                    "remain current and complete. " * 4)[:len(say)]}
        for nm in ("A_asis", "B_say", "N_len"):
            body = base + adds[nm] + ASK
            for tp, k in ((0.0, 1), (a.temp, a.n)):
                for _ in range(k):
                    try:
                        rep = " ".join(str(X559.gen(a.port, body, 96, tp)).split())
                    except Exception as e:
                        print("%-26s %-8s %-5s 호출 실패: %r" % (tag[:26], nm, tp, e))
                        continue
                    hit = [g for g in good if g in rep]
                    tally[(tag, nm)][1] += 1
                    if hit:
                        tally[(tag, nm)][0] += 1
                    print("%-26s %-8s %-5s %-46s %s"
                          % (tag[:26], nm, tp, rep[:46],
                             ("생산자 값 %s" % hit[0]) if hit else ("옛 값 그대로" if val in rep else "-")))
    print()
    print("## 판정 (생산자가 낸 값을 쓴 비율)")
    for tag, s, i, val, prods in spots:
        print("   %-26s %s" % (tag[:26], " · ".join(
            "%s %d/%d" % (nm, tally[(tag, nm)][0], tally[(tag, nm)][1])
            for nm in ("A_asis", "B_say", "N_len"))))
    print()
    print("⚠A_asis 가 이미 옳게 쓰면 결손이 아니다([[62]] 2b). N_len 이 같으면 길이다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
