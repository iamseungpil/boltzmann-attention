# -*- coding: utf-8 -*-
r"""x590 - 같은 결정점에서 **더하기 vs 빼기** (격리 · 라이브 프롬프트 위).

## 왜 (2026-08-28 밤)

`x583`: 표적 쓰기 도구를 못 부른 20 sim **전부**에서 그 이름은 이미 배달돼 있었다 = **부하**.
`x585`: 그 이름을 **되짚어 줘도** 아무것도 안 바뀐다 - A_asis · B_restate · N_len 이 **0/5 동률**.
   [[57]] 부정통제까지 같으니 길이 탓도 아니고 **더하기가 안 먹는** 것이다.

[[63]] 은 정확히 이것을 말한다: *모델은 더하기·지시는 안 듣고 **제거만** 닫는다*(0/8 <-> 8/8).
공통 기전은 하나 - **닫힌 술어로 후보를 제거**하고 기준만 갈아 끼운다.

=> 그러면 같은 자리에서 **빼면** 달라지나? 그것만 잰다.

## 팔 (전부 같은 라이브 프롬프트 위 · 새 사실 0)

    A_asis   회수된 프롬프트 그대로                      <- 재현 게이트
    B_add    + 배달된 미호출 이름 **되짚기**(x585 의 그 문면)   <- 더하기 (알려진 음성)
    C_sub    + **이관 계열 도구를 이 단계에서 제외**한다는 한 줄  <- 빼기
    N_len    + 길이만 맞춘 무관 문장                      <- [[57]]

제외 집합은 닫혀 있다: 프롬프트에 실재하는 발견형 이름 중 `transfer` 를 포함하는 것.
엔진이 고르지 않는다 - **이름 규칙 하나**다.

⚠C_sub 의 문면은 *"이 단계에서 부를 수 없다"* 고 말한다. 라이브로 배선하면 엔진이 실제로
  그 호출을 거절해야 참이 된다([[25]]). 여기서는 **빼기가 먹는지**를 재는 프로브다.

## 채점 (닫힌 술어)

    표적호출  출력에 `apply_checking_account_credit_5829` 가 나오나
    이관지목  출력에 이관 계열 이름이 나오나
    타도구    그 밖의 발견형 이름 수
"""
import argparse
import collections
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x585_delivered_name_restate_iso as X5                            # noqa: E402

NL = chr(10)
RE_DISC = re.compile(r"\b[a-z][a-z0-9_]*_\d{4}\b")
TARGET = "apply_checking_account_credit_5829"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--simtag", default="task_072#s373753")
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    recs = X5.prompts(a.dump, a.simtag)
    r = X5.pick_turn(recs, "last")
    if r is None:
        sys.stderr.write("[STOP] 표적이 실린 미호출 메인 프롬프트가 없다\n")
        return 2
    base = str(r.get("text"))
    add, names = X5.restate_line(base)

    present = sorted(set(RE_DISC.findall(base)))
    excl = [n for n in present if "transfer" in n]
    if not excl:
        sys.stderr.write("[STOP] 뺄 이관 계열 이름이 프롬프트에 없다 - 판정하지 않는다\n")
        return 2
    sub = (NL + NL + "[note] For this step the following tools are excluded and cannot be "
           "called: %s." % ", ".join(excl))

    print("# x590 - turn=%s · %d자 · sim=%s" % (r.get("turn"), len(base), a.simtag))
    print("   프롬프트의 발견형 이름 %d개" % len(present))
    print("   B_add 가 되읊는 것 %d개: %s" % (len(names), ", ".join(names)))
    print("   C_sub 가 빼는 것 %d개: %s" % (len(excl), ", ".join(excl)))
    print("   표적이 제외 집합에 있나(있으면 프로브 무효): %s" % (TARGET in excl))
    if a.wiring_only:
        print("--- C_sub 문면 ---")
        print("   " + " ".join(sub.split()))
        return 0

    import x559_016_row_pick_iso as X559
    fill = ("[note] the information gathered so far in this conversation remains current. " * 8)
    arms = {"A_asis": "", "B_add": add, "C_sub": sub,
            "N_len": NL + NL + fill[:max(len(add), len(sub))]}
    print("")
    print("%-9s %-5s %-8s %-8s %-6s %s" % ("팔", "temp", "표적호출", "이관지목", "타도구", "답"))
    print("-" * 100)
    tally = collections.defaultdict(collections.Counter)
    for nm in ("A_asis", "B_add", "C_sub", "N_len"):
        for tp, cnt in ((0.0, 1), (a.temp, a.n)):
            for _ in range(cnt):
                try:
                    rep = " ".join(str(X559.gen(a.port, base + arms[nm], 300, tp)).split())
                except Exception as e:
                    print("%-9s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                hit = TARGET in rep
                xfer = any(x in rep for x in excl) or "transfer_to_human" in rep
                others = sorted(set(RE_DISC.findall(rep)) - set([TARGET]) - set(excl))
                tally[nm]["표적"] += 1 if hit else 0
                tally[nm]["이관"] += 1 if xfer else 0
                tally[nm]["n"] += 1
                print("%-9s %-5s %-8s %-8s %-6d %s"
                      % (nm, tp, "O" if hit else "-", "O" if xfer else "-", len(others), rep[:42]))
    print("")
    print("## 집계")
    for nm in ("A_asis", "B_add", "C_sub", "N_len"):
        c = tally[nm]
        if c["n"]:
            print("   %-9s 표적 %d/%d · 이관지목 %d/%d" % (nm, c["표적"], c["n"], c["이관"], c["n"]))
    print("")
    print("[읽기] C_sub 가 A·B·N 보다 표적을 많이 부르면 **빼기가 먹는다**([[63]]).")
    print("[읽기] 넷이 같으면 이 결정점은 빼기로도 안 열린다 - 다른 자리를 봐야 한다.")
    print("[읽기] N_len 이 C 와 같으면 산 것은 내용이 아니라 길이다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
