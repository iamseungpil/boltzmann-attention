# -*- coding: utf-8 -*-
r"""freeze.py — 유료 런 동안 엔진을 **동결**한다 (설계서 §7-0 · 원장 C423⒞).

## 왜

08-10 이후 `t2_gate_patch.py`/`a2` 에 **17커밋**이 들어갔고 그중 **3커밋은 유료 런이 도는
중**에 들어왔다. 그 결과 `bank_alllevers_20260810`(10/16) 과의 대조가 **원리적으로 귀속
불가**가 됐다 — 플래그 델타를 아무리 사전 등록해도 **바탕이 움직이면 실험이 성립하지 않는다.**

런 결과에 SHA 를 적는 것(오늘 신설)은 **사후 귀속**이다. 동결은 **사전 조건**이다.

## 어떻게

`FREEZE.json` 을 놓으면 `.claude/hooks/scaffold_guard.py` 가 그 경로들의 Edit/Write 를
**exit 2 로 막는다**. 훅은 같은 워크트리를 쓰는 **모든 세션**에 걸리므로 병렬 세션도 구속된다
([[07]]: 선언·메모리 같은 soft 수단으로는 통제되지 않는다 — enforced hook 이라야 한다).

거는 것은 사람이 아니라 **런처**다. 유료 런 스크립트가 발사 직전에 `--on` 을, 종료 뒤에
`--off` 를 부른다. 사람이 기억해서 거는 규율은 오늘 이미 실패했다.

## 쓰기

    python freeze.py --on  --tag bank_xx_20260812 --reason "CP2 라이브 판정"
    python freeze.py --off
    python freeze.py --status

`--off` 는 **동결 중 SHA 가 변했는지 함께 보고한다** — 변했으면 훅을 우회한 편집이 있었다는
뜻이고, 그 런의 결과는 버리는 것이 맞다.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PATH = os.path.join(HERE, "FREEZE.json")

# 무엇을 얼리나 — **런의 거동을 바꾸는 것**만. 프로브·설계서·원장은 얼지 않는다
# (얼리면 런 도중 관측·기록을 못 한다 = 오늘 필요했던 일을 금지하는 꼴).
DEFAULT_PATHS = [
    "/scripts/distill/tau2/t2_gate_patch.py",
    "/scripts/distill/tau2/t2_eplan_patch.py",
    "/scripts/distill/tau2/t2_dominance.py",
    "/scripts/distill/tau2/t2_search.py",
    "/scripts/distill/tau2/t2_precedence.py",
    "/scripts/distill/tau2/t2_source.py",
    "/scripts/distill/tau2/a2/",
    "/scripts/distill/tau2/go_stack.sh",
]


def sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO).decode().strip()
    except Exception:
        return "?"


def dirty():
    """추적 중인 파일의 미커밋 변경만 센다(`x*_out.json` 류 아티팩트는 제외)."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no",
                                       "--", "scripts/distill/tau2"], cwd=REPO).decode()
        return [l[3:] for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def now():
    """시각은 **git 이 준다** — 스크립트가 시계를 읽지 않는다(replay 가능성 유지)."""
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%cI"],
                                       cwd=REPO).decode().strip()
    except Exception:
        return "?"


def load():
    if not os.path.exists(PATH):
        return None
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", action="store_true")
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--tag", default="(unnamed)")
    ap.add_argument("--reason", default="(none)")
    a = ap.parse_args()
    cur = load()

    if a.status or not (a.on or a.off):
        if cur and cur.get("active"):
            d = dirty()
            print("FROZEN sha=%s tag=%s\n  이유: %s\n  건 시각: %s\n  지금 SHA: %s%s"
                  % (cur.get("sha"), cur.get("tag"), cur.get("reason"), cur.get("at"), sha(),
                     "" if sha() == cur.get("sha") else "  ⚠**동결 중 SHA 가 변했다**"))
            if d:
                print("  ⚠미커밋 변경 %d: %s" % (len(d), ", ".join(d[:5])))
        else:
            print("not frozen (sha=%s)" % sha())
        return 0

    if a.on:
        if cur and cur.get("active"):
            print("[freeze] 이미 동결 중이다 (sha=%s tag=%s) — 새로 걸지 않는다."
                  % (cur.get("sha"), cur.get("tag")), file=sys.stderr)
            return 1
        d = dirty()
        if d:
            # 미커밋 변경이 있으면 **얼려도 재현이 안 된다** — 얼리는 의미가 없다.
            print("[freeze] REFUSING: 추적 파일에 미커밋 변경 %d 개가 있다. 커밋하고 다시 걸라.\n  %s"
                  % (len(d), "\n  ".join(d)), file=sys.stderr)
            return 1
        z = {"active": True, "sha": sha(), "tag": a.tag, "reason": a.reason,
             "at": now(), "paths": DEFAULT_PATHS}
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(z, f, ensure_ascii=False, indent=1)
        print("[freeze] ON sha=%s tag=%s · %d 경로" % (z["sha"], a.tag, len(DEFAULT_PATHS)))
        return 0

    # --off
    if not (cur and cur.get("active")):
        print("[freeze] 동결 상태가 아니다.", file=sys.stderr)
        return 1
    moved = sha() != cur.get("sha")
    d = dirty()
    cur["active"] = False
    cur["released_sha"] = sha()
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=1)
    print("[freeze] OFF · 동결 SHA %s → 해제 SHA %s" % (cur.get("sha"), sha()))
    if moved or d:
        print("⚠**동결이 뚫렸다** — 이 런의 결과는 어떤 SHA 로도 재현되지 않는다.\n"
              "  SHA 변동: %s · 미커밋: %d\n  ⇒ 그 런은 [?] 로 기록하거나 버려라(C423⒞)."
              % (moved, len(d)), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
