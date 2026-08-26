# -*- coding: utf-8 -*-
r"""x557 — t7362 세 레버가 **왜 안 들었나**: 발화 회계 + 스텝 궤적 3자 정렬

## 왜 (사용자 지시 2026-08-26 밤 · *"왜 레버가 동작하지 않은지 정밀하게 원인 추적하라"*)
A/B 결과만으로는 *"안 들었다"* 밖에 못 말한다. [[30]] 규율 축자 — **단위통과 ≠ 라이브 발화**이고
*"천장/결론 주장 전 레버 **실발화율** 전수확인"* 이다(calc 31/342 미발화를 천장으로 오인한 사고).
그래서 먼저 **발화했는가**를 세고, 발화했으면 **그 다음 스텝에 무엇이 일어났는가**를 본다.

## 세 레버가 남기는 자국 (정본 코드에서 읽어 온 것 · 이 파일에 추측 0)
    T2_PROCEDURE_LEFT       `[T2_PROCEDURE_LEFT] 종료 창에서 남은 칸 …` + `… regen tool_calls=…`
                            (`t2_gate_patch.py:13370,13383` · sim 당 **1회 상한**)
    T2_EPLAN_ENUM_SUBTRACT  `[T2_EPLAN] L1 deny (subtracted): …` / `[T2_EPLAN] L1 released: …`
                            (`t2_eplan_patch.py:919,922` · `_called` 가 비면 **애초에 안 탄다**)
    T2_SCOPE_ALL            ⚠**부호 주의** — ON = `_is_effective_write` 가드를 **건너뛴다**
                            (`t2_resolve.py:257`) ⇒ 읽기 오선택에도 `[OPERATOR-SCOPE]` 가
                            **더 많이** 난다(구판 복귀). OFF(기본) = 쓰기에만 발화.
                            자국은 `[T2_RESOLVE] operator-scope 침묵: …`(OFF 에서만) 과
                            사이드카의 `[OPERATOR-SCOPE]` 행 수.

## 무엇을 인쇄하나
  §1 팔×sim 별 **발화 회계** — 0 이면 그 레버는 *"안 들었다"* 가 아니라 **안 났다**.
  §2 표적 태스크의 **3자 스텝 정렬** — 세 팔의 궤적을 인덱스로 나란히 놓고 **첫 갈림**을 찍는다.
  §3 갈림 전후 우리 층 반려(사이드카·turn 포함)를 그 자리에 끼워 인쇄한다.
판단 0 — 결론은 이 재료 위에서 사람이 쓴다([[08]] 집계→결론 직행 금지).

사용: (리모트·cwd=scripts/distill/tau2) py -3 x557_lever_stepwise.py --task task_074
"""

import argparse
import collections
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

import t2_forensic as F                                              # noqa: E402

ARMS = (("A_ctl", "bank_t7362_A_ctl_20260826", "기본값"),
        ("B_say", "bank_t7362_B_say_20260826", "PROCEDURE_LEFT + EPLAN_ENUM_SUBTRACT"),
        ("C_scope", "bank_t7362_C_scope_20260826", "SCOPE_ALL=1 (구판 복귀)"))
# 세는 자국 — 전부 정본 코드의 축자 문자열이다.
COUNTS = (("PROCEDURE_LEFT 발화", r"\[T2_PROCEDURE_LEFT\] 종료 창에서"),
          ("PROCEDURE_LEFT regen", r"\[T2_PROCEDURE_LEFT\] regen"),
          ("PROCEDURE_LEFT 건너뜀", r"\[T2_PROCEDURE_LEFT\] 건너뜀"),
          ("EPLAN L1 subtracted", r"\[T2_EPLAN\] L1 deny \(subtracted\)"),
          ("EPLAN L1 released", r"\[T2_EPLAN\] L1 released"),
          ("EPLAN L1 (구판)", r"\[T2_EPLAN\] L1 deny: multi-entity"),
          ("scope 침묵(OFF 자국)", r"\[T2_RESOLVE\] operator-scope 침묵"),
          ("OPERATOR-SCOPE 반려", r"\[OPERATOR-SCOPE\]"))


def brief(m, n=86):
    """메시지 한 줄 요약 — 역할 + 도구 이름(있으면) + 본문 머리."""
    role = m.get("role")
    names = []
    for tc in (m.get("tool_calls") or ()):
        names.append(F.inner_name(F.argsof(tc)) or F.nameof(tc))
    c = " ".join(str(m.get("content") or "").split())
    tag = ("→" + ",".join(str(x) for x in names)) if names else ""
    return "%-9s %s %s" % (role, tag, c[:n])


def key(m):
    """정렬 비교용 지문 — 역할 + 도구 이름 + 본문 머리 40자."""
    names = tuple(sorted(str(F.inner_name(F.argsof(tc)) or F.nameof(tc))
                         for tc in (m.get("tool_calls") or ())))
    return (m.get("role"), names, " ".join(str(m.get("content") or "").split())[:40])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_074")
    ap.add_argument("--seed", default="")
    ap.add_argument("--span", type=int, default=8, help="갈림 전후 인쇄 폭")
    a = ap.parse_args(argv)

    got = {}
    for nm, tag, note in ARMS:
        try:
            sims = F.sims(tag)
        except Exception as e:
            print("[%s] 결과 없음: %r" % (nm, e))
            continue
        got[nm] = (tag, note, sims)

    print("=" * 104)
    print("# §1 발화 회계 — 레버가 **났는가** ([[30]] 단위통과 ≠ 라이브 발화)")
    for nm, (tag, note, sims) in got.items():
        try:
            txt = F.log_text(tag)
        except Exception:
            txt = ""
        print("\n-- %-8s %s · %s" % (nm, note, F.sidecar_note(tag).split(":")[0]))
        den = F.sidecar_denies(tag)
        for s in sims:
            st = F.simtag(s)
            rw = (s.get("reward_info") or {}).get("reward")
            cells = []
            for label, pat in COUNTS:
                rx = re.compile(pat)
                if label == "OPERATOR-SCOPE 반려":
                    rows = den["simtag"].get(st) or den["fp"].get(st) or []
                    n = sum(1 for r in rows if "[OPERATOR-SCOPE]" in str(r.get("text") or ""))
                else:
                    n = sum(1 for ln in (txt or "").splitlines()
                            if ("[sim=%s]" % st) in ln and rx.search(ln))
                if n:
                    cells.append("%s=%d" % (label, n))
            print("   %-20s reward %-5s %s" % (st, rw, " · ".join(cells) or "**자국 0**"))

    print()
    print("=" * 104)
    print("# §2 %s — 3자 스텝 정렬 (첫 갈림)" % a.task)
    tr = {}
    for nm, (tag, note, sims) in got.items():
        for s in sims:
            if F.task_id(s) != a.task:
                continue
            if a.seed and str(s.get("seed")) != a.seed:
                continue
            tr[nm] = (tag, s)
    if len(tr) < 2:
        print("  비교할 팔이 %d 개뿐이다 — 정렬하지 않는다([[25]])" % len(tr))
        return 0
    for nm, (tag, s) in tr.items():
        print("  %-8s msg %3d · reward %s · %ss"
              % (nm, len(s.get("messages") or []), (s.get("reward_info") or {}).get("reward"),
                 int(s.get("duration") or 0)))
    names = list(tr)
    base = names[0]
    bm = tr[base][1].get("messages") or []
    div = None
    for i in range(len(bm)):
        ks = set()
        for nm in names:
            ms = tr[nm][1].get("messages") or []
            ks.add(key(ms[i]) if i < len(ms) else ("<끝>",))
        if len(ks) > 1:
            div = i
            break
    print("\n  ★첫 갈림 = msg[%s]" % ("없음(공통 접두가 짧은 쪽 전체)" if div is None else div))
    lo = max(0, (div or 0) - 2)
    hi = (div or 0) + a.span
    for i in range(lo, hi):
        print("  --- msg[%d] ---" % i)
        for nm in names:
            ms = tr[nm][1].get("messages") or []
            print("    %-8s %s" % (nm, brief(ms[i]) if i < len(ms) else "(끝)"))

    print()
    print("=" * 104)
    print("# §3 갈림 부근 우리 층 반려 (사이드카 · turn)")
    for nm in names:
        tag, s = tr[nm]
        st = F.simtag(s)
        rows = (F.sidecar_denies(tag)["simtag"].get(st)
                or F.sidecar_denies(tag)["fp"].get(st) or [])
        near = [r for r in rows if isinstance(r.get("turn"), int)
                and lo - 6 <= int(r["turn"]) <= hi + 6]
        print("\n-- %-8s 그 창의 반려 %d / 전체 %d" % (nm, len(near), len(rows)))
        seen = set()
        for r in near:
            t = " ".join(str(r.get("text") or "").split())
            k = t[:70]
            if k in seen:
                continue
            seen.add(k)
            print("     turn=%-4s %s" % (r.get("turn"), t[:170]))
    print()
    print("⚠자국 0 인 레버는 *'안 들었다'* 가 아니라 **안 났다** — 원인은 술어이지 모델이 아니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
