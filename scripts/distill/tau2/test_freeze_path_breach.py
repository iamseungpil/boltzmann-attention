# -*- coding: utf-8 -*-
"""회귀 — 동결 판정은 **동결 경로의 내용**으로 하지 HEAD SHA 로 하지 않는다 (`freeze.py`).

★왜 (2026-08-23 · t7346 실측):
  유료 런의 러너는 **자기 결과를 스스로 커밋한다**(스모크 gz → 본런 gz → push·[[30]] 영속 의무).
  그래서 HEAD 는 런마다 반드시 움직이고, `moved = sha() != 동결시 sha` 로 재던 구판은
  **모든 런에 "동결이 뚫렸다"를 찍었다**. t7346 실물: 동결 `d5ff0c10` → 해제 `29b3a283` 인데
  그 구간 커밋은 `t7346 smoke` · `t7346 all-on stage1 results` 둘뿐이고 바뀐 파일은
  `reports/facet_rft_2026/sim_results` 8개 — **엔진 diff 0**.

  경보가 늘 울리면 아무도 안 듣는다. 그리고 이 경보의 출력은 *"이 런을 버려라"* 이므로
  오탐 하나가 멀쩡한 밤샘런을 폐기시킬 수 있다([[25]] 우리 도구의 출력 결함이 유일한 근거원을
  오염시킨다).

⇒ `--on` 이 `paths` 각각의 git object hash 를 적어두고 `--off` 가 다시 재서 비교한다.
  디렉터리(`a2/`)는 tree 해시라 그 아래 전부를 덮는다. HEAD 변동은 **정보로만** 인쇄한다.

오프라인 전용(모델 0·env 0·git 읽기만). 실행: py -3 test_freeze_path_breach.py
"""
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FZ = os.path.join(REPO, "reports", "facet_rft_2026", "freeze.py")

fails = []


def check(name, ok, detail=""):
    print("  %s %s%s" % ("✓" if ok else "✗", name, ("  — " + detail) if detail and not ok else ""))
    if not ok:
        fails.append(name)


def load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("freeze_mod", FZ)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    print("[freeze 경로-기준 판정]")
    src = io.open(FZ, encoding="utf-8").read()

    # ① 구판 술어가 **판정으로** 쓰이지 않는다 — 이것이 회귀의 몸통이다.
    #    (`head_moved` 는 정보 인쇄용으로 남아 있어도 된다. 판정에 쓰이는지를 본다.)
    check("--off 의 판정이 breached() 에서 나온다", "breached(cur)" in src)
    check("구판 판정식 `if moved or d:` 이 사라졌다", "if moved or d:" not in src,
          "구판 판정이 그대로 있다")

    m = load_module()
    check("path_hashes/breached 가 있다",
          hasattr(m, "path_hashes") and hasattr(m, "breached"))
    if fails:
        return 1

    # ② 실제 저장소에서 해시가 읽힌다(경로 표기 → git object 경로 변환 포함).
    ph = m.path_hashes(m.DEFAULT_PATHS)
    got = {k: v for k, v in ph.items() if v}
    check("동결 경로 해시를 실제로 읽는다 (%d/%d)" % (len(got), len(ph)), len(got) >= 5,
          "읽힌 것 %r" % (list(got)[:3],))
    check("a2 디렉터리는 tree 해시로 잡힌다",
          any(k.endswith("/a2") and v for k, v in ph.items()),
          "a2 항목 %r" % ({k: v for k, v in ph.items() if k.endswith("/a2")},))

    # ③ ★핵심: HEAD 가 움직여도 **동결 경로가 그대로면** 뚫린 것이 아니다.
    #    t7346 이 정확히 이 경우였다(러너 자신의 sim_results 커밋).
    cur = {"sha": "deadbee", "paths": m.DEFAULT_PATHS, "path_hashes": ph}
    hit, changed, how = m.breached(cur)
    check("HEAD 불일치만으로는 뚫림이 아니다", (not hit) and not changed,
          "hit=%r changed=%r (%s)" % (hit, changed, how))

    # ④ 반대로 동결 경로의 내용이 바뀌면 반드시 잡는다(부정통제·[[57]]).
    k0 = sorted(got)[0]
    tampered = dict(ph)
    tampered[k0] = "0" * 40
    hit2, changed2, _ = m.breached({"sha": m.sha(), "paths": m.DEFAULT_PATHS,
                                    "path_hashes": tampered})
    check("동결 경로 내용이 바뀌면 잡는다", hit2 and k0 in changed2,
          "hit=%r changed=%r" % (hit2, changed2))

    # ⑤ 구판 freeze 파일(해시 없음)은 종전 거동으로 떨어지되 근거를 말한다.
    hit3, _, how3 = m.breached({"sha": "deadbee", "paths": m.DEFAULT_PATHS})
    check("구판 파일은 HEAD 비교로 폴백하고 근거를 밝힌다", hit3 and "구판" in how3, how3)

    print("\nRESULT: %s" % ("ALL PASS" if not fails else "FAIL (%s)" % ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
