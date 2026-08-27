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

`--off` 는 **동결된 경로의 내용이 변했는지** 함께 보고한다 — 변했으면 훅을 우회한 편집이
있었다는 뜻이고, 그 런의 결과는 버리는 것이 맞다.

## ⚠2026-08-23 수리 — HEAD SHA 로 재면 **모든 런이 "뚫렸다"로 기록된다**

구판은 `moved = sha() != 동결시 sha` 였다. 그런데 유료 런의 러너는 **자기 결과를 스스로
커밋한다**(스모크 gz → 본런 gz → push·[[30]] 영속 의무). 그래서 HEAD 는 런마다 반드시
움직이고, 경보는 **항상** 뜬다. t7346 실측: 동결 `d5ff0c10` → 해제 `29b3a283` 로 "뚫렸다"가
떴는데 그 구간 커밋 2개는 `t7346 smoke` · `t7346 all-on stage1 results` 뿐이고
`git diff --name-only` 는 **`reports/facet_rft_2026/sim_results` 8파일**만 냈다 — 엔진 diff 0.

경보가 늘 울리면 아무도 안 듣는다([[25]]: 우리 도구의 출력 결함이 유일한 근거원을 오염시킨다).
동결이 지키기로 한 것은 **`paths` 의 내용**이지 HEAD 가 아니다 ⇒ `--on` 이 그 경로들의
**git object hash 를 적어두고** `--off` 가 같은 해시를 다시 재서 비교한다. HEAD 변동은
정보로만 인쇄한다(그 자체는 위반이 아니다).
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


def rel(p):
    """`DEFAULT_PATHS` 표기(`/scripts/...`·디렉터리는 끝에 `/`) → git object 경로."""
    return p.strip("/")


def path_hashes(paths):
    """동결 경로 각각의 **git object hash**. 없는 경로는 `None`(나중에 생기면 그것도 변경이다).

    `git rev-parse HEAD:<path>` 는 파일이면 blob, 디렉터리면 tree 해시를 준다 — 디렉터리는
    그 아래 전부를 덮으므로 `a2/` 한 줄이 A2 전 파일을 지킨다.
    """
    out = {}
    for p in paths or []:
        r = rel(p)
        try:
            out[r] = subprocess.check_output(
                ["git", "rev-parse", "HEAD:%s" % r], cwd=REPO,
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            out[r] = None
    return out


def breached(cur):
    """동결이 실제로 뚫렸나 — **경로 내용 기준**. 반환 (뚫림?, 바뀐 경로 목록, 판정 근거).

    구판 freeze 파일(`path_hashes` 없음)은 잴 재료가 없으므로 종전대로 HEAD 비교로 떨어지되
    그 사실을 근거 문자열로 남긴다([[25]] 계기가 무엇을 쟀는지 말하게 한다).
    """
    before = cur.get("path_hashes")
    if not before:
        return (sha() != cur.get("sha"), [], "구판 freeze(경로 해시 없음) — HEAD 비교로 대체")
    after = path_hashes(cur.get("paths") or DEFAULT_PATHS)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    return (bool(changed), changed, "동결 경로 %d개 내용 비교" % len(before))


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
            # ★hold 마다 **자기 기준**으로 판정한다 — 동시 런 둘이 서로의 판정을 오염시키지 않게.
            hs = list(cur.get("holds") or []) or [cur]
            for h in hs:
                hit, changed, how = breached(h)
                print("FROZEN sha=%s tag=%s\n  이유: %s\n  건 시각: %s\n  지금 SHA: %s%s"
                      % (h.get("sha"), h.get("tag"), h.get("reason"), h.get("at"), sha(),
                         ("  ⚠**동결 경로가 변했다**: " + ", ".join(changed[:4])) if hit
                         else ("  (HEAD 만 움직였다 — 경로 무변)" if sha() != h.get("sha") else "")))
            if len(hs) > 1:
                print("  동시 hold %d 개" % len(hs))
            if d:
                print("  ⚠미커밋 변경 %d: %s" % (len(d), ", ".join(d[:5])))
        else:
            print("not frozen (sha=%s)" % sha())
        return 0

    if a.on:
        # ★다중 hold (2026-08-28): GPU 가 둘이 되면서 **런 두 개가 동시에** 돈다. 동결이 한 칸이면
        #   ⑴뒤에 뜬 런은 `--on` 이 거부돼 **동결 없이** 돌고 ⑵먼저 끝난 런의 `--off` 가 남의
        #   동결까지 풀어 버린다 ⇒ 어느 쪽이든 한 런이 조용히 [S] 를 잃는다.
        #   ⇒ hold 를 **목록**으로 두고 각자 자기 기준(`path_hashes`)으로 판정받는다.
        #   구판 파일(hold 목록 없음)은 로드 시 한 칸짜리 목록으로 **이관**한다(하위호환).
        holds = list(cur.get("holds") or []) if cur else []
        if cur and cur.get("active") and not holds:
            holds = [{k: cur.get(k) for k in
                      ("tag", "sha", "reason", "at", "paths", "path_hashes")}]
        if any(h.get("tag") == a.tag for h in holds):
            print("[freeze] 같은 태그가 이미 걸려 있다 (tag=%s) — 새로 걸지 않는다." % a.tag,
                  file=sys.stderr)
            return 1
        d = dirty()
        if d:
            # 미커밋 변경이 있으면 **얼려도 재현이 안 된다** — 얼리는 의미가 없다.
            print("[freeze] REFUSING: 추적 파일에 미커밋 변경 %d 개가 있다. 커밋하고 다시 걸라.\n  %s"
                  % (len(d), "\n  ".join(d)), file=sys.stderr)
            return 1
        # ★수리(2026-08-23): 판정 재료는 여기서 생긴다. 이것이 없으면 --off 는 HEAD 로
        #   떨어지고 러너 자신의 결과 커밋에 늘 걸린다(위 독스트링).
        holds.append({"tag": a.tag, "sha": sha(), "reason": a.reason, "at": now(),
                      "paths": DEFAULT_PATHS, "path_hashes": path_hashes(DEFAULT_PATHS)})
        z = dict(holds[0])                      # 최상위는 **가장 오래된 hold** 를 비춘다(구판 독자용)
        z.update({"active": True, "holds": holds})
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(z, f, ensure_ascii=False, indent=1)
        print("[freeze] ON sha=%s tag=%s · %d 경로%s"
              % (holds[-1]["sha"], a.tag, len(DEFAULT_PATHS),
                 ("  (동시 hold %d — 각자 자기 기준으로 판정된다)" % len(holds))
                 if len(holds) > 1 else ""))
        return 0

    # --off
    if not (cur and cur.get("active")):
        print("[freeze] 동결 상태가 아니다.", file=sys.stderr)
        return 1
    holds = list(cur.get("holds") or [])
    if not holds:
        holds = [{k: cur.get(k) for k in
                  ("tag", "sha", "reason", "at", "paths", "path_hashes")}]
    # 태그를 주면 그것을, 안 주면 **가장 오래된 것**을 푼다 — 러너들이 `--off` 만 부르던
    # 종전 문면에서 각자 자기 것을 푸는 순서가 된다(먼저 건 런이 먼저 끝난다는 보장은
    # 없지만, 어느 쪽이든 hold 하나만 풀리고 나머지는 살아 있다).
    idx = 0
    if a.tag and a.tag != "(unnamed)":
        idx = next((i for i, h in enumerate(holds) if h.get("tag") == a.tag), -1)
        if idx < 0:
            print("[freeze] 그 태그의 hold 가 없다 (tag=%s · 걸린 것: %s)"
                  % (a.tag, ", ".join(str(h.get("tag")) for h in holds)), file=sys.stderr)
            return 1
    mine = holds.pop(idx)
    hit, changed, how = breached(mine)
    d = dirty()
    rec = dict(mine)
    rec["released_sha"] = sha()
    rec["breach"] = {"changed_paths": changed, "basis": how, "uncommitted": len(d)}
    z = dict(holds[0]) if holds else dict(rec)
    z.update({"active": bool(holds), "holds": holds,
              "released": (cur.get("released") or [])[-9:] + [rec]})
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(z, f, ensure_ascii=False, indent=1)
    head_moved = sha() != mine.get("sha")
    print("[freeze] OFF tag=%s · 동결 SHA %s → 해제 SHA %s%s%s"
          % (mine.get("tag"), mine.get("sha"), sha(),
             "  (HEAD 는 움직였다 — 러너 자신의 결과 커밋이면 정상)" if head_moved else "",
             ("  · 남은 hold %d: %s" % (len(holds), ", ".join(str(h.get("tag")) for h in holds)))
             if holds else ""))
    if hit or d:
        print("⚠**동결이 뚫렸다** — 이 런의 결과는 어떤 SHA 로도 재현되지 않는다.\n"
              "  판정 근거: %s\n  바뀐 동결 경로 %d: %s\n  미커밋: %d\n"
              "  ⇒ 그 런은 [?] 로 기록하거나 버려라(C423⒞)."
              % (how, len(changed), ", ".join(changed[:6]) or "(없음)", len(d)), file=sys.stderr)
        return 2
    print("[freeze] 동결 경로 무변 — 런 유효(%s)" % how)
    return 0


if __name__ == "__main__":
    sys.exit(main())
