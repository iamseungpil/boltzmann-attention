# -*- coding: utf-8 -*-
r"""x562 — 요구와 **그 부정이 한 메시지에** 있으면 순종이 사라지는가 (유료 0).

## 왜 (t7364 실측 · 2026-08-27)

`T2_READ_PER_ENTITY` 는 격리에서 4/4 였는데(`x561`) 라이브 t7364 에서는 **0/1** 이었다.
사이드카가 그 차이를 축자로 보여 준다 — `fb_bank_t7364_hard0_20260827` · `task_016#s1567` ·
turn 38 · 3501자 한 메시지 안에서:

    *"Do that now with the real tool calls."*                       ← 요구
    *"…was called but has not succeeded yet - its result above says why"*   ← 그 두 줄 뒤
    *"Steps that are possible right now (any of them, your choice): (none available)"*

격리에는 뒤의 둘이 **없었다**. 그 둘은 이웃 조각이 아직 **이름만** 보기 때문에 생긴다
(`_dn`·`frontier` 가 엔티티-무관 · `_tried`(접미사 제거) − `_dn`(정확 이름) 뺄셈이
성공한 discoverable read 를 실패로 만든다).

## 팔 — 저작 0. 라이브가 보낸 **그 바이트**에서 만든다

    A_asis    우리 메시지 없음                       ← 대조
    B_live    사이드카 축자 3501자 그대로            ← 재현 게이트(순종이 없어야 한다)
    C_fixed   같은 문면에서 **모순 두 곳만** 교체     ← 수리 후 코드가 내는 문면과 같다
    N_len     길이만 맞춘 무관 문장([[57]])

## 채점 — 닫힌 술어

다음 호출이 선언된 그 read 이고 인자에 그 주체가 있는가. 이름은 선언·값은 손님이 준 것이다.

사용: PYTHONPATH=. py -3 x562_order_contradiction_iso.py --port 8140
"""
import argparse
import collections
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                             # noqa: E402
import x559_016_row_pick_iso as X559                                # noqa: E402

NL = chr(10)
TAG = "bank_t7364_hard0_20260827"
SIM = "task_016#s1567"
READ = "get_all_user_accounts_by_user_id"
WHO = "friend_user_5839"
ASK = (NL + NL + "What is the very next tool call you make? Reply with one line only, "
       "in the form `tool_name {\"arg\": \"value\"}`. Nothing else.")

# 수리 후 코드가 내는 문면 — 두 곳만 다르다(`t2_gate_patch` 의 `_pe_fams` 분기).
BAD_WHY = "was called but has not succeeded yet - its result above says why"
FIX_WHY = "has not been called for %s in this conversation" % WHO
BAD_OPT = "(none available)"
FIX_OPT = ('get_all_user_accounts_by_user_id_3847 (not in your tool list - it is a discoverable '
           'tool; the way to run it is call_discoverable_agent_tool'
           '(agent_tool_name="get_all_user_accounts_by_user_id_3847"))')


def live_message():
    """라이브가 그 턴에 실제로 보낸 문면 — 사이드카 축자."""
    p = F.path_for(TAG, ".jsonl.gz")  # 미사용(경로 규약이 다름) — 아래에서 직접 연다
    del p
    here = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                        "fb_%s.jsonl.gz" % TAG)
    out = []
    with gzip.open(here, "rt", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("simtag") == SIM and "other party" in str(r.get("text") or ""):
                out.append(r)
    return out[0] if out else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    rec = live_message()
    if not rec:
        print("그 문면이 사이드카에 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2
    live = str(rec.get("text") or "")
    turn = int(rec.get("turn") or 0)
    fixed = live.replace(BAD_WHY, FIX_WHY).replace(BAD_OPT, FIX_OPT)
    if fixed == live:
        print("교체가 하나도 안 일어났다 — 문면이 내가 아는 것과 다르다. 중단.", file=sys.stderr)
        return 2

    sims = [s for s in F.sims(TAG) if F.simtag(s) == SIM]
    if not sims:
        print("궤적이 없다", file=sys.stderr)
        return 2
    ms = sims[0].get("messages") or []
    w = min(turn, len(ms))
    base = X559.render(ms, w)
    print("# x562 — turn %d · 문맥 %d자 · 라이브 문면 %d자 (수리 후 %d자)"
          % (turn, len(base), len(live), len(fixed)))
    nxt = next((tc for m in ms[w:] for tc in (m.get("tool_calls") or ())), None)
    print("  라이브가 그 뒤에 실제로 한 것: %s"
          % (F.label(F.nameof(nxt), F.argsof(nxt)) if nxt else "없음"))
    if a.wiring_only:
        for k, v in (("BAD_WHY", BAD_WHY in live), ("BAD_OPT", BAD_OPT in live)):
            print("  %-8s 라이브에 존재: %s" % (k, v))
        print("--- 교체된 두 곳 ---")
        for seg in (FIX_WHY, FIX_OPT[:80]):
            print("   ", seg)
        return 0

    adds = {"A_asis": "",
            "B_live": NL + NL + live,
            "C_fixed": NL + NL + fixed,
            "N_len": NL + NL + ("[note] " + "the details gathered so far in this conversation "
                                "remain current and complete. " * 12)[:len(live)]}
    print()
    print("%-9s %-5s %-56s %s" % ("팔", "temp", "지목한 다음 호출", "판정"))
    print("-" * 100)
    tally = collections.defaultdict(lambda: [0, 0])
    for nm in ("A_asis", "B_live", "C_fixed", "N_len"):
        body = base + adds[nm] + ASK
        for tp, k in ((0.0, 1), (a.temp, a.n)):
            for _ in range(k):
                try:
                    rep = " ".join(str(X559.gen(a.port, body, 96, tp)).split())
                except Exception as e:
                    print("%-9s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                low = rep.lower()
                ok = READ in low and WHO in low
                tally[nm][1] += 1
                tally[nm][0] += 1 if ok else 0
                print("%-9s %-5s %-56s %s" % (nm, tp, rep[:56], "그 주체로 계좌 read" if ok else "-"))
    print()
    print("## 판정")
    for nm in ("A_asis", "B_live", "C_fixed", "N_len"):
        print("   %-9s %d/%d" % (nm, tally[nm][0], tally[nm][1]))
    print()
    print("⚠B_live 가 이미 순종하면 라이브 실패의 원인은 이 문면이 아니다 — 판정하지 마라.")
    print("⚠N_len 이 C_fixed 와 같으면 그 이득은 **길이**다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
