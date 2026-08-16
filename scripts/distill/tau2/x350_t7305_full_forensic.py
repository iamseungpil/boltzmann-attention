# -*- coding: utf-8 -*-
r"""x350 — t7305 **전 sim 전수 포렌식**(사용자 지시 2026-08-17: *"1 sim 정독이 아니라 모든 sim"*).

집계가 감춘 것을 sim 단위로 되돌린다([[08]]). sim 마다 다음을 **한 화면**에 놓는다:

    ⑴ 결정점  로그 축자 `turn=N · group=G · 인용 n/m · [T2_DOCDECIDE] → '값'`
    ⑵ **그 turn 에 손님이 실제로 한 말**  — 메시지의 `turn_idx` 로 자른다(추정 0).
       ★이것이 이 파일의 핵심이다: 결정이 **요구보다 먼저** 나면 서브에 실을 요구가 없다.
    ⑶ 타임라인  손님 발화·어시스턴트 본문·호출 이름(도구 결과는 뺀다)
    ⑷ 최종 제출 vs gold · **서브 채택 여부**(DOCDECIDE 값 == 최종 제출인가)

판단은 인쇄하지 않는다 — 사람이 읽는다. 기계는 **가리키기**만 한다([[25]]).
gold 는 태스크 정의에서 읽는다(분석 전용·[[23]] 레버 무관·`x341` 규칙 재사용).

실행(로컬/리모트 공통):  py -3 x350_t7305_full_forensic.py [출력파일]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402
import x341_docbody_verdict as X                                   # noqa: E402
import x348_sub_requirement_verdict as V                           # noqa: E402

ARMS = [("ctl", ["bank_t7305_ctl_20260817a", "bank_t7305_ctlaux_20260817a"]),
        ("treat", ["bank_t7305_treat_20260817a", "bank_t7305_treataux_20260817a"])]
GROUP = r"\[T2_SEARCH_AGENT\] group=.*"
REQ = r"\[T2_SUB_REQUIREMENT\] 인용 .*"
DECIDE = r"\[T2_DOCDECIDE\] → .*"
CUT = 230                      # 메시지 축자 잘라내는 길이(정독용·너무 길면 못 읽는다)


def turn_of(line):
    """로그 줄의 `turn=N`(정규식 0 — 앵커 뒤 숫자만 읽는다). 없으면 None."""
    n = V.nums(line, "turn=")
    return n[0] if n else None


def group_of(line):
    """`group=savings_accounts · 문서 …` → `savings_accounts`(문자열 연산만)."""
    s = str(line or "")
    i = s.find("group=")
    if i < 0:
        return "?"
    return s[i + 6:].split(" ")[0].split("·")[0].strip()


def decisions(tag, sim):
    """sim 의 결정점 목록 [(turn, group, 인용 n/m, 결정값)] — 줄번호 순서로 묶는다."""
    key = F.simtag(sim)
    ltag = V.log_tag(tag)
    grp = (F.by_sim(ltag, GROUP, [sim]) or {}).get(key) or []
    req = (F.by_sim(ltag, REQ, [sim]) or {}).get(key) or []
    dec = (F.by_sim(ltag, DECIDE, [sim]) or {}).get(key) or []
    out = []
    for ln, line in sorted(grp):
        t, g = turn_of(line), group_of(line)
        # 이 group 줄과 다음 group 줄 사이에 있는 인용·결정 줄을 붙인다(순서만 씀)
        nxt = min([l for l, _x in sorted(grp) if l > ln] or [10 ** 9])
        q = [V.nums(x) for l, x in req if ln < l < nxt]
        d = [V.decided(x) for l, x in dec if ln < l < nxt]
        out.append((t, g, q, d))
    return out


def user_upto(sim, turn):
    """`turn` **이전에** 손님이 실제로 한 말(메시지 `turn_idx` 로 자른다·추정 0)."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "user":
            continue
        ti = m.get("turn_idx")
        if turn is None or (ti is not None and int(ti) < int(turn)):
            out.append((i, " ".join(str(m.get("content") or "").split())))
    return out


def block(arm, tag, sim, w):
    tid = F.task_id(sim)
    golds = X.gold_axes(tid)
    am = X.axis_match(tid, sim, golds)
    fin = X.final_axes(sim)
    decs = decisions(tag, sim)
    w("\n" + "=" * 104)
    w("### [%s] %s · R=%s · %s · dur=%.0fs · 호출 %d"
      % (arm, F.simtag(sim), (sim.get("reward_info") or {}).get("reward"),
         F.term_reason(sim), sim.get("duration") or 0, len(list(F.calls(sim)))))
    w("gold: %s" % {ax: g for ax, (_v, g, _m) in am.items()})
    w("최종 제출: %s" % {ax: (v[0] or "-") + (" =G" if v[2] else " ≠") for ax, v in am.items()})
    w("전 축 제출(참고): %s" % fin)

    w("--- 결정점(로그 축자) ---")
    if not decs:
        w("   (없음)")
    for t, g, q, d in decs:
        us = user_upto(sim, t)
        w("   turn=%s group=%-26s 인용%s 결정=%s" % (t, g, q or "없음", d or "없음"))
        w("        그 시점 손님 발화 %d개: %s"
          % (len(us), [i for i, _t in us] or "**없음 — 요구보다 먼저 결정**"))
        for i, txt in us[-2:]:
            w("          msg %-3d %s" % (i, txt[:CUT]))
        # 서브 채택 여부: 이 group 축의 최종 제출과 결정값이 같은가
        ax = "checking" if "checking" in g else ("savings" if "savings" in g else
                                                ("card" if "card" in g else g))
        sub = (d or [None])[-1]
        got = fin.get(ax)
        if sub is not None and got is not None:
            w("        서브 채택? 결정 %r vs 최종 %r → %s"
              % (sub, got, "같다" if X.norm(sub) == X.norm(got) else "**다르다(메인이 갈아탐)**"))

    w("--- 타임라인 ---")
    for i, m in enumerate(sim.get("messages") or []):
        r = m.get("role")
        if r not in ("user", "assistant"):
            continue
        txt = " ".join(str(m.get("content") or "").split())
        names = [F.label(F.nameof(tc), F.argsof(tc)) for tc in (m.get("tool_calls") or [])]
        if not txt and not names:
            continue
        w("  %3d %-9s t%-3s %s%s" % (i, r, m.get("turn_idx"), txt[:CUT],
                                     ("  CALLS=%s" % names[:6]) if names else ""))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "reports", "facet_rft_2026", "x350_t7305_full_forensic.txt")
    buf = []

    def w(s):
        buf.append(s)
    n = 0
    for arm, tags in ARMS:
        for tag in tags:
            for sim in sorted(F.sims(tag), key=lambda x: (F.task_id(x), str(x.get("seed")))):
                block(arm, tag, sim, w)
                n += 1
    txt = "\n".join(buf)
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(txt)
    print("sim %d · %d자 → %s" % (n, len(txt), os.path.normpath(out)))
    if n != 32:
        print("⚠sim 수가 32 가 아니다(%d) — 태그 누락 확인" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
