# -*- coding: utf-8 -*-
"""레버 **생존 감사** 정본 — 켜졌나가 아니라 **전달했나**를 묻는다.

사용자 지시(2026-08-14 야간): *"지금까지 계속 이미 있던 원칙들을 안 쓰고 원점에서 다시 하고 있다.
원칙을 라이브러리화하고 검색도 하나로 고정하고, 다른 레버도 라이브러리로 고정해서 다시 하지
않게 하라."*

**왜 이 모듈이 필요한가.** `t2_levers` 는 레버가 *무엇인지* 안다. 런처는 레버를 *켠다*. 그런데
둘 다 **그 레버가 실제로 무언가를 전달했는지는 모른다**. 오늘 그 구멍에 하루를 태웠다:

  · `T2_SEARCH_AGENT` 는 t7290 072 에서 **10회 발화하고 10회 전부 침묵**했다
    (`now 미확정 · 원값 None`). 온톨로지 문서 색인은 695/698 을 덮고 있었고 필요한 문서
    (`bank_accounts_(general)_017`)도 색인 안에 있었다 — **레버는 있었고 배선이 죽어 있었다**.
  · 그걸 모른 채 "072 는 retrieval 문제인가 학습 문제인가"를 격리 프로브로 몇 시간 팠다.
    측정한 결손의 상당 부분이 **우리 죽은 레버가 만든 것**이었다.
  · 같은 부류의 선례가 이미 메모리에 있다([[55]]): `proc_fb` 死배선이 deny 11회를 인쇄로 만들었다.

⇒ 규칙: **결손을 모델에게 귀속하기 전에 이 감사를 돌린다.** [[55]](우리 배관 먼저)의 실행 형태다.

판정은 문면이 아니라 **우리 자신의 로그 프로토콜**로 한다(도메인 판단 0·[[59]]):
    DELIVERED  그 태그의 줄이 있고 침묵-표지가 없다
    SILENCED   침묵·건너뜀·무발화·미확정·실패 표지가 붙어 있다  ← **여기가 위험 구간**
    ABSENT     그 태그의 줄이 아예 없다(켜졌는데 도달조차 못 함 or 안 켬)

사용:
    py t2_liveness.py <log> [<log>...]          # 런 로그 감사
    from t2_liveness import audit               # 라이브러리로
"""
import collections
import glob
import io
import os
import re
import sys

# 우리 로그가 침묵을 알릴 때 쓰는 말들(엔진이 스스로 찍는 문면·도메인 어휘 0)
SILENT_MARKS = ("침묵", "건너뜀", "무발화", "미발화", "미확정", "실패", "폐기", "못 찾",
                "no-op", "skip")
TAG = re.compile(r"\[(T2_[A-Z0-9_]+)\]")
SIM = re.compile(r"\[sim=(task_\d+)")


def audit(paths):
    """로그들 → {태그: {"delivered": n, "silenced": n, "reasons": Counter, "sims": set}}"""
    out = collections.defaultdict(
        lambda: {"delivered": 0, "silenced": 0,
                 "reasons": collections.Counter(), "sims": set()})
    for p in paths:
        for ln in io.open(p, encoding="utf-8", errors="replace"):
            m = TAG.search(ln)
            if not m:
                continue
            tag = m.group(1)
            rec = out[tag]
            s = SIM.search(ln)
            if s:
                rec["sims"].add(s.group(1))
            body = ln[m.end():]
            hit = next((w for w in SILENT_MARKS if w in body), None)
            if hit:
                rec["silenced"] += 1
                # 사유는 그 줄에서 **엔진이 적은 문구**를 그대로 짧게 남긴다(해석 0)
                rec["reasons"][" ".join(body.split())[:70]] += 1
            else:
                rec["delivered"] += 1
    return out


def report(res, min_silent_ratio=0.5):
    rows = sorted(res.items(), key=lambda kv: -(kv[1]["silenced"]))
    print("%-28s %8s %8s %6s  %s" % ("lever", "delivered", "silenced", "sims", "판정"))
    print("-" * 104)
    dead = []
    for tag, r in rows:
        tot = r["delivered"] + r["silenced"]
        ratio = (r["silenced"] / tot) if tot else 0.0
        verdict = ("⚠DEAD" if r["delivered"] == 0 and r["silenced"] else
                   ("⚠주로 침묵" if ratio >= min_silent_ratio else "ok"))
        if verdict != "ok":
            dead.append((tag, r, ratio))
        print("%-28s %8d %8d %6d  %s" % (tag, r["delivered"], r["silenced"],
                                         len(r["sims"]), verdict))
    if dead:
        print("\n# 침묵 사유 (전달 0 이거나 침묵이 과반인 레버)")
        for tag, r, ratio in dead:
            print("  [%s] 침묵 %d/%d" % (tag, r["silenced"], r["delivered"] + r["silenced"]))
            for reason, n in r["reasons"].most_common(3):
                print("      %3d× %s" % (n, reason))
    print("\n※ DELIVERED 는 '전달했다'이지 '옳았다'가 아니다. 이 표는 **배선 생존**만 본다 — "
          "효과 판정은 격리 프로브와 라이브 대조가 한다([[57]]).")
    return dead


def main(argv):
    paths = []
    for a in argv:
        paths += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(__doc__)
        return 1
    print("감사 대상 %d 파일\n" % len(paths))
    dead = report(audit(paths))
    print("\n결과: 레버 %d종 · 위험(전달 0 or 침묵 과반) %d종" % (len(audit(paths)), len(dead)))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
