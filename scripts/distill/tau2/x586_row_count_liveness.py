# -*- coding: utf-8 -*-
r"""x586 - `T2_SG_ROW_COUNT` 는 죽은 배선인가, 조건이 사라진 것인가 (모델 0 · 무료 · 계수만).

## 왜 (2026-08-28 밤)

t7376 20 sim 전수에서 `[T2_SG_ROW_COUNT]` 가 **0회** 발화했다. 그런데 틀린 총액은 고쳐졌다.
핸드오프 §3 표는 이 레버에 *"틀린 5.0 차단"* 을 근거로 달아 뒀는데 그 런에서는 성립하지 않는다.
[[60]] 은 레버를 끄지 말라고 하지만, **조용한 레버가 죽은 것인지 할 일이 없는 것인지**는 갈라야
한다 - 죽었으면 조용한 능력 상실이고, 할 일이 없으면 그물로 남겨 두는 것이 맞다.

## 무엇을 세나 (닫힌 술어 · 로그 축자)

각 런 로그에서 `operand-size <도구>: sub=N rows · source=M rows · <종류>=K rows` 를 뽑아

    종류선언   K 가 찍힌 관측 수 (이 계기는 t7370 부터 존재한다 - 그 전 런은 분모가 없다)
    모자람     sub < K        <- ROW_COUNT 의 술어
    초과       sub > K        <- 아직 검산이 **없는** 방향
    실발화     `[T2_SG_ROW_COUNT]` 축자 계수
    자기완결   `형태=자기완결` 계수 (= `T2_SG_CLOSE_SELF` 의 유일한 라이브 마커)

## 무엇이 나왔나 (모든 bank 런)

    종류선언 53 · 모자람 11 · 초과 16 · 실발화 8 · 자기완결 24

    t7370  모자람 6 -> 실발화 6      (ROW_COUNT 단독 측정 · CLOSE_SELF 없음)
    t7371s 모자람 2 -> 실발화 2
    t7372  모자람 2 -> 실발화 0      (대조 팔 = 레버 OFF · 안 서는 것이 맞다)
    t7375  모자람 0 -> 실발화 0      자기완결 8
    t7376  모자람 0 -> 실발화 0      자기완결 13

=> **레버가 켜져 있고 술어가 성립한 8 자리에서 8/8 발화했다.** 죽은 배선이 아니다.
   t7375·t7376 에서 조용한 이유는 `T2_SG_CLOSE_SELF` 가 상류에서 결손을 막아
   **모자람 자체가 0/17** 이 됐기 때문이다. 둘은 직렬이고 ROW_COUNT 는 **그물**이다.
   => [[60]] 대로 끄지 않는다. 다만 *"이 레버가 총액을 고쳤다"* 는 귀속은 t7370 에서만 참이다.

## 남은 빈 칸

`초과`(sub > K)는 **아무도 안 잡는다** - t7372 8건 · t7376 3건. 다만 t7376 에서 초과가 남은
계좌들의 총액은 gold 와 일치했으므로 지금은 구속력이 없다. 검산을 새로 짓기 전에
**초과가 총액을 틀리게 만든 사례**를 먼저 찾아라([[62]] 순서).

사용: PYTHONPATH=. py -3 x586_row_count_liveness.py
"""
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))
RE_SIZE = re.compile(r"operand-size (\S+?)\.\w+: sub=(\d+) rows . source=(\d+) rows(?: . (\w+)=(\d+) rows)?")
RE_FIRE = re.compile(r"\[T2_SG_ROW_COUNT\]")
RE_SELF = re.compile(r"형태=자기완결")


def main(argv=None):
    rows = []
    for f in sorted(os.listdir(BASE)):
        if not (f.startswith("bank_") and f.endswith(".log.gz")):
            continue
        try:
            with gzip.open(os.path.join(BASE, f), "rt", encoding="utf-8", errors="replace") as fh:
                log = fh.read()
        except Exception:
            continue
        s = RE_SIZE.findall(log)
        kind = [m for m in s if m[4]]
        if not kind:
            continue
        rows.append({"tag": f.replace(".log.gz", "").replace("bank_", ""),
                     "obs": len(s), "kind": len(kind),
                     "short": sum(1 for m in kind if int(m[1]) < int(m[4])),
                     "over": sum(1 for m in kind if int(m[1]) > int(m[4])),
                     "fired": len(RE_FIRE.findall(log)),
                     "self_close": len(RE_SELF.findall(log))})
    if not rows:
        print("행 0 - 종류 선언이 있는 런이 없다(계기는 t7370 부터)"); return 1
    print("%-32s %-6s %-8s %-7s %-6s %-7s %s"
          % ("런", "관측", "종류선언", "모자람", "초과", "실발화", "자기완결"))
    for r in rows:
        print("%-32s %-6d %-8d %-7d %-6d %-7d %d"
              % (r["tag"], r["obs"], r["kind"], r["short"], r["over"], r["fired"], r["self_close"]))
    T = {k: sum(r[k] for r in rows) for k in ("obs", "kind", "short", "over", "fired", "self_close")}
    print("")
    print("합계  종류선언 %d · 모자람 %d · 초과 %d · 실발화 %d · 자기완결 %d"
          % (T["kind"], T["short"], T["over"], T["fired"], T["self_close"]))
    on = [r for r in rows if r["fired"] or (r["short"] and r["self_close"] == 0 and r["fired"])]
    print("")
    print("[판정] 레버가 켜진 런에서 모자람이 있던 자리의 발화율을 보라 -")
    print("       t7370/t7371s 는 모자람 == 실발화 다(8/8). 죽은 배선이 아니다.")
    print("       t7375/t7376 은 모자람 0 - `T2_SG_CLOSE_SELF` 가 상류에서 막았다. 그물로 남긴다([[60]]).")
    print("[빈칸] 초과(sub > K)는 아무도 안 잡는다. 짓기 전에 **초과가 총액을 틀리게 한 사례**를 찾아라.")
    dst = os.path.join(BASE, "..", "x586_row_count_liveness.json")
    with io.open(os.path.normpath(dst), "w", encoding="utf-8") as f:
        json.dump({"probe": "x586_row_count_liveness", "date": "2026-08-28",
                   "rows": rows, "totals": T,
                   "limits": ["종류 선언 계기는 t7370 부터라 그 전 런은 분모가 없다.",
                              "로그 축자 계수일 뿐 인과가 아니다([[08]]).",
                              "대조 팔은 레버가 OFF 라 실발화 0 이 맞다 - 결함이 아니다."]},
                  f, ensure_ascii=False, indent=1)
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
