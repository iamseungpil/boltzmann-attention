# -*- coding: utf-8 -*-
r"""x565 — env 가 **다른 필드로** 낸 값을 이 인자에 넣은 자리에서, 이름표를 말하면 고치는가.

## 결손 (x564 필드-이름표 계수 · 2026-08-27)

`[PROVENANCE]` 술어는 `_ctx_has` — *"문맥 어딘가에 있나"* 라서 **출처는 맞고 종류가 틀린** 값이
전부 통과한다. env 는 레코드를 `필드: 값` 으로 찍으므로 종류의 답은 이미 문맥에 있다.

    085  `user_id: f7d3a82c91`        → `account_id` 로  (11건)
    085  `account_id: chk_b4d92f7c28` → `card_id` 로     (5건)
    074  `ar72c5d8e3`(user 조각)      → `account_id` 로  (2건)
    057·079·040 도 같은 형태 · 다섯 태스크 12 sim · **reward 1.0 인 것 0**

잡음 둘은 선언으로 걷는다 — `Record ID:` 의 `ID`(덤프 머리) · 생산자 목록이 **같은 동의어**
(`phone`/`phone_number`).

## 팔 — 값은 지목하지 않는다

    A_asis   그대로                                   ← 재현 게이트(틀린 값이 나와야 한다)
    B_say    *"그 값은 기록에 `<F>` 로 나온다. `<K>` 의 값은 `<생산자>` 가 낸다"*  ← 값 0
    N_len    길이만 맞춘 무관 문장([[57]])

⛔B_say 는 **어느 값을 쓰라고 말하지 않는다** — 고르는 것은 끝까지 모델이다([[62]]③④).
문면의 재료는 전부 이미 있는 것이다: 필드 이름은 env 출력, 생산자는 A2 `arg_source_reads`.

## 채점 — 닫힌 술어 · gold 무참조

다음 호출이 그 인자에 **그 이름표로 문맥에 나온** 값을 넣는가. 엔진은 어느 계좌가 옳은지 모른다.

사용: PYTHONPATH=. py -3 x565_wrong_account_id_iso.py --port 8140 [--wiring-only]
"""
import argparse
import collections
import os
import re
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
ASK = (NL + NL + "What is the very next tool call you make? Reply with one line only, "
       "in the form `tool_name {\"arg\": \"value\"}`. Nothing else.")


def spots(a2, tag):
    """이름표가 어긋난 **첫** 호출 — 그 인자의 옳은 값이 문맥에 이미 있는 자리만."""
    asr = {k: v for k, v in (a2.get("arg_source_reads") or {}).items()
           if not k.startswith("_") and isinstance(v, list)}
    out = []
    for s in F.scored(tag):
        ms = s.get("messages") or []
        lab = {}
        for i, m in enumerate(ms):
            if m.get("role") == "tool":
                c = " ".join(str(m.get("content") or "").split())
                # ★**레코드 덤프에서만** 이름표를 읽는다. 도구 스키마 줄도 `이름: 값` 형태라
                #   (`account_id: string (required) - …`) 걸러내지 않으면 `string`·`The` 가
                #   그 인자의 "옳은 값" 으로 들어온다(배선 확인이 잡았다). 덤프의 표지는
                #   `Record ID:` 이고 `_parse_record_dump` 가 쓰는 것과 같은 표지다.
                k0 = c.find("Record ID:")
                if k0 < 0:
                    continue
                for f, v in re.findall(r"(\w+):\s*([^\s,;]+)", c[k0:]):
                    lab.setdefault(f, set()).add((i, v))
            for tc in (m.get("tool_calls") or ()):
                for k, v in (DOM._args_dict(X564._TC(tc)) or {}).items():
                    if k not in asr or not isinstance(v, str) or len(v.strip()) < 4:
                        continue
                    val = v.strip()
                    if any(j < i and w == val for j, w in lab.get(k, ())):
                        continue                     # 제 이름표로 나온 값 — 정상
                    src = [f for f, st in lab.items()
                           if f != k and f != X564.DUMP_HEAD
                           and not X564.same_axis(asr, f, k)
                           and any(j < i and w == val for j, w in st)]
                    if not src:
                        continue
                    # ★`good` 이 비어도 잰다 — 085 실측: 오배정은 **계좌 목록이 오기 전** 딱
                    #   한 번(msg[24])이고 그 뒤 열 호출은 전부 옳다. 즉 결손은 *"문맥의 옳은
                    #   값을 못 고른다"* 가 아니라 *"없는 값을 이웃 필드에서 빌린다"* 다.
                    #   그러면 옳은 다음 수는 둘 중 하나다 — 그 이름표의 값을 쓰거나,
                    #   **그 값을 낳는 read 를 부르거나**. 둘 다 채점한다.
                    good = sorted({w for j, w in lab.get(k, ()) if j < i})
                    out.append((s, i, k, val, src[0], good, asr.get(k) or []))
                    break
                if out and out[-1][0] is s:
                    break
            if out and out[-1][0] is s:
                break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tags", default="bank_t7363_hard0_20260827,bank_t7356_grpA2_20260826")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)
    a2 = GI.load_domain_a2("banking_knowledge") or {}

    found = []
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        for sp in spots(a2, tag)[:2]:
            found.append((tag,) + sp)
    if not found:
        print("그런 자리가 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2

    print("# x565 — 이름표가 어긋난 결속")
    for tag, s, i, k, val, src, good, prods in found:
        print("  %-26s %-9s msg[%d]  %s=%s  (기록에는 `%s` 로 나온 값) · 그 이름표로 나온 값 %d개: %s"
              % (tag[:26], F.task_id(s), i, k, val[:22], src, len(good), good[:3]))
    if a.wiring_only:
        return 0

    print()
    print("%-26s %-8s %-5s %-44s %s" % ("tag", "팔", "temp", "다음 호출", "판정"))
    print("-" * 112)
    tally = collections.defaultdict(lambda: [0, 0])
    for tag, s, i, k, val, src, good, prods in found:
        ms = s.get("messages") or []
        base = X559.render(ms, i)
        say = (NL + NL + "Error: [SOURCE] the value '%s' is what the records above give as `%s`, "
               "not as `%s`. The values for `%s` are the ones the records list under that name; "
               "%s is what produces them, and its output is above in this conversation. Re-issue "
               "the call with a value the records give under `%s`."
               % (val, src, k, k, (prods[0] if prods else "the lookup tool"), k))
        adds = {"A_asis": "", "B_say": say,
                "N_len": NL + NL + ("[note] " + "the records gathered so far in this conversation "
                                    "remain current and complete. " * 6)[:len(say)]}
        for nm in ("A_asis", "B_say", "N_len"):
            body = base + adds[nm] + ASK
            for tp, cnt in ((0.0, 1), (a.temp, a.n)):
                for _ in range(cnt):
                    try:
                        rep = " ".join(str(X559.gen(a.port, body, 96, tp)).split())
                    except Exception as e:
                        print("%-26s %-8s %-5s 호출 실패: %r" % (tag[:26], nm, tp, e))
                        continue
                    hit = [g for g in good if g in rep]
                    prod = [p for p in prods if p.split("_by_")[0] in rep or p in rep]
                    tally[(tag, nm)][1] += 1
                    if hit or prod:
                        tally[(tag, nm)][0] += 1
                    print("%-26s %-8s %-5s %-44s %s"
                          % (tag[:26], nm, tp, rep[:44],
                             ("이름표 맞는 값 %s" % hit[0][:18]) if hit else
                             ("생산자 read 호출" if prod else
                              ("옛 값 그대로" if val in rep else "-"))))
    print()
    print("## 판정 (이름표 맞는 값을 쓰거나 **생산자 read 를 부른** 비율)")
    for tag, s, i, k, val, src, good, prods in found:
        print("   %-26s %-9s %s" % (tag[:26], F.task_id(s), " · ".join(
            "%s %d/%d" % (nm, tally[(tag, nm)][0], tally[(tag, nm)][1])
            for nm in ("A_asis", "B_say", "N_len"))))
    print()
    print("⚠A_asis 가 이미 옳게 쓰면 결손이 아니다([[62]] 2b). N_len 이 같으면 길이다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
