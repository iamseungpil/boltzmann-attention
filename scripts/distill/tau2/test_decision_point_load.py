#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""결정점 부하 상한 — write 한 자리에 얹히는 **재료 총량**을 묶는다.

## 왜 이 래칫이 생겼나 (2026-08-26 · t7361 실물)

`T2_ARG_POLICY_AT_WRITE` 를 격리(x551) 근거로 켰다. 격리 팔이 실은 것은 자격 기준 **827자**
였는데, 라이브 조인은 그 write 의 **선언 인자 15개 전부**를 실어 **3,033자**를 보냈다.
그리고 그 자리엔 이미 세 레버가 앉아 있었다. 040 의 **같은 결정점**을 두 런에서 재면:

    t7360  DECIDE 285 + SPEC 2137 + RULE 74             =  2,496자  → turn 79 · 완료
    t7361  DECIDE 6973 + SPEC 2137 + RULE 74 + AP 3033  = 12,217자  → `toolerr` 같은 지문
                                                                      **26회** 반복 · turn 98 중단

[[65]] 가 축자로 금지한 것이다 — *"재료를 메인에 올리는 것 자체가 부하다"*(x231 8/8→0/8 ·
x187 파레토 지배 · C397 4%↔100%). 레버 하나하나는 격리를 통과했는데 **합이 태스크를 죽였다**.
개별 격리로는 이걸 못 잡는다([[19]] *"간섭은 합성 런만이 드러낸다"*) ⇒ **합을 세는 칸**이 필요하다.

## 이 검정이 하는 일

`_decl_join` 이 한 메시지에 묶을 수 있는 재료 산출자들의 **합**을 정본 도메인에서 계산해
상한과 비교한다. 상한을 넘으면 붉어지고, **무엇이 얼마를 차지하는지** 인쇄한다.

⚠이 검정은 *"어느 레버가 옳은가"* 를 판정하지 않는다 — 합만 센다([[59]] 판단 0).
⚠상한은 **관측에서 왔다**: 완주한 판(2,496)과 루프에 빠진 판(12,217) 사이. 4,000 을 쓴다
  — 완주 판의 1.6배까지는 허용하고 루프 판의 1/3 에서 막는다. 근거 없는 예쁜 수가 아니라
  두 관측 사이의 자리다. 더 좋은 수치는 조인 부하를 **격리로 재면** 나온다(아직 미측정).

실행: PYTHONIOENCODING=utf-8 py -3 test_decision_point_load.py
"""
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import t2_gate_patch as G                                          # noqa: E402

DOMAIN = "banking_knowledge"
# 상한은 **관측 셋 사이**에 놓는다(예쁜 수 아님):
#     827자   x551 격리가 실제로 실어서 4/4 를 얻은 양(자격 기준 본문)
#   1,073자   같은 조인에서 **그 축 하나만** 실을 때의 양
#   3,033자   라이브가 실제로 보낸 양(선언 인자 15 전부) → 12,217자 결정점 → 루프
# ⇒ 축 하나는 통과하고 전체 조인은 막히는 자리. `T2_ARG_POLICY_CAP`(4000)보다 **낮아야**
#   의미가 있다 — 그것과 같으면 이 검정은 내부 상한의 복사본일 뿐이다(초판이 그랬다).
CAP = 1500
FAIL = []


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, ("  " + str(extra)) if extra else ""))


def ships_on(flag):
    """이 레버를 **켜서 내보내는 곳이 하나라도 있나** — go_stack 정본 + 런 드라이버 전부.

    ⚠초판이 여기서 틀렸다(2026-08-26): `enabled()` 로 go_stack 을 읽어 꺼져 있으면 합을 0 으로
      셌다. 그런데 **런 드라이버가 그 위에 `=1` 을 덮는다**(`run_t7361_smoke.sh` 가 정확히
      `T2_SPEC_AT_WRITE=1 T2_RULE_AT_WRITE=1` 을 export 한다) ⇒ 정본만 보면 라이브가 실제로
      보내는 양을 못 본다. 래칫이 **조용히 초록**이 되는 그 모양이다([[67]] 계기 함정).
    ⇒ 정본과 드라이버를 **함께** 본다. 어디서든 `=1` 이면 그 양은 언젠가 라이브로 나간다.
    """
    for p in ([os.path.join(HERE, "go_stack.sh")]
              + sorted(glob.glob(os.path.join(HERE, "run_*.sh")))):
        try:
            txt = io.open(p, encoding="utf-8").read()
        except Exception:
            continue
        for tok in txt.split():
            if tok.startswith("%s=" % flag) and tok[len(flag) + 1:].startswith("1"):
                return True
    return False


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    a2 = G._domain_a2(DOMAIN)
    surf = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"),
                             encoding="utf-8"))[DOMAIN]["tools"]

    print("결정점 부하 — write 도구마다 재료 산출자의 합 (상한 %d자)" % CAP)
    print("=" * 92)
    worst = (0, None, {})
    for tool, spec in sorted(surf.items()):
        if not spec.get("mutates"):
            continue
        params = set(spec.get("args") or ())   # env_surface 의 인자 키는 `args` 다
        parts = {}
        t = G._policy_rows_for(a2, params) if params else None
        parts["ARG_POLICY"] = len(t or "")

        class _TC:
            name = tool
            arguments = {}

        try:
            t = G._declared_rules_for(a2, _TC())
        except Exception:
            t = None
        parts["RULE"] = len(t or "")
        total = sum(parts.values())
        if total > worst[0]:
            worst = (total, tool, dict(parts))
    if worst[1]:
        print("  최대 = %-46s %d자  %s" % (worst[1][:46], worst[0], worst[2]))
    else:
        print("  (켜진 재료 레버가 없다 — 이 축에서 셀 것이 없다)")

    on = ships_on("T2_ARG_POLICY_AT_WRITE")
    print()
    print("[상한] — 이 레버를 켜서 내보내는 곳이 있을 때만 강제한다")
    print("     T2_ARG_POLICY_AT_WRITE 를 `=1` 로 내보내는 파일이 있나: %s" % ("**있다**" if on else "없다"))
    if on:
        chk(worst[0] <= CAP,
            "켜서 내보내는데 선언-유래 결정점 합이 상한 이하다",
            "%d/%d  %s" % (worst[0], CAP, worst[2]))
    else:
        print("  skip 꺼져 있다 — 최악치 %d자만 기록해 둔다(켜려면 이 수부터 줄여라)" % worst[0])

    print()
    print("[재료 레버 — 위 합은 **전부 켜진 최악**으로 셌다]")
    for f in ("T2_ARG_POLICY_AT_WRITE", "T2_RULE_AT_WRITE", "T2_SPEC_AT_WRITE",
              "T2_DECIDE_BEFORE_WRITE"):
        print("     %-28s %s" % (f, "내보냄" if ships_on(f) else "안 내보냄"))
    print("  ⚠`SPEC_AT_WRITE`(2,137자)·`DECIDE_BEFORE_WRITE`(런에서 285~6,973자)는 재료가")
    print("    **런타임 궤적**에서 오므로 여기서 못 센다 — 이 검정은 **선언에서 나오는 몫**만")
    print("    센다. 즉 이 상한을 통과해도 라이브 총량은 더 크다([[67]] 계기의 한계를 적는다).")

    print()
    print("RESULT: %s%s" % ("PASS" if not FAIL else "FAIL",
                            "" if not FAIL else " " + str(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
