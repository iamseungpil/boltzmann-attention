# -*- coding: utf-8 -*-
"""레버가 **설치되는 경로**에 있는가 (2026-07-31 — V7 사망 사고의 재발 방지).

사고: V7(`T2_TOOL_SIGNATURE`)을 `gated`(=`BaseOrchestrator._execute_tool_calls`)에 구현했는데,
`go_stack`은 `T2_GATE_REGEN=1`이라 런처가 `_unified` 분기를 타서 **`t2_gate_patch.apply()`를 아예
호출하지 않는다**(`t2_run_gated.py:196`). 실행 훅은 `exec_augment`("deny 없음")가 차지한다.
⇒ V7은 selftest 4/4를 통과하면서도 **어떤 런에서도 발화할 수 없었다**. Z4·Z5·Y2에서 deny 0.

교훈: **selftest 통과 ≠ 라이브 도달.** 레버는 *설치되는 함수* 안에 있어야 한다.

이 테스트가 강제하는 것: go_stack이 켜는 on/off 레버 중 `t2_gate_patch.py`가 구현하는 것은
**`unified`(설치되는 생성-레벨 경로) 안에서 참조**되어야 한다. `gated`에만 있으면 실패한다.
(`gated`는 `T2_GATE_REGEN` 미사용 스택에서만 설치되므로 거기에만 있는 레버 = 사실상 죽은 코드.)
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
GO = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()


def body_of(func):
    """`def <func>(` 부터 같은 들여쓰기의 다음 def 전까지."""
    m = re.search(r"^(\s*)def %s\(" % re.escape(func), SRC, re.M)
    if not m:
        return ""
    ind, start = m.group(1), m.end()
    nxt = re.search(r"^%sdef " % ind, SRC[start:], re.M)
    return SRC[start:start + (nxt.start() if nxt else len(SRC) - start)]


UNIFIED = body_of("unified")
GATED = body_of("gated")
ON = set(re.findall(r"\b(T2_[A-Z0-9_]+)=1\b", GO))
IMPLEMENTED = set(re.findall(r"environ\.get\(\s*[\'\"](T2_[A-Z0-9_]+)[\'\"]", SRC))

# ★손으로 적은 예외 목록 폐기(2026-07-31 2차 교정).
#   초판은 "실행-레벨 read-augment는 정당"이라며 예외 목록을 **가정으로** 적었고, 거기에
#   `T2_WRITE_CAP`을 넣었다. 그런데 그 레버는 `gated` 안에서 **`_deny_msg`로 deny**한다 —
#   V7과 **동일한 죽은 경로**다. 즉 이 테스트가 잡으려던 바로 그 버그를 **내 예외 목록이 가렸다.**
#   ⇒ 기계 판정으로 바꾼다: `gated` 전용 레버라도 **deny를 만들면 죽은 코드**,
#      응답 가공만 하면 정당(설치되는 `exec_augment`가 같은 augment를 한다).
def denies_in_gated(flag):
    """그 레버의 `gated` 내 사용 지점 뒤 40줄 안에 `_deny_msg(`가 있나 = deny 레버인가."""
    body = GATED.splitlines()
    for i, l in enumerate(body):
        if flag in l:
            if any("_deny_msg(" in x for x in body[i:i + 40]):
                return True
    return False

OK = True
print("[경로 가드] go_stack ON 레버가 **설치되는 경로**(unified)에 있는가")
print("  unified 본문 %d자 · gated 본문 %d자" % (len(UNIFIED), len(GATED)))
dead, latent = [], []
# ★검사 대상 = go_stack이 켜는 것 + **엔진이 구현한 모든 on/off 레버**.
#   초판은 켜진 것만 봐서, 꺼져 있는 죽은 레버(`T2_WRITE_CAP`)를 놓쳤다 — 꺼져 있어도
#   **켜면 안 도는** 코드는 결함이다(승격 시점에 무음 실패한다).
for f in sorted(IMPLEMENTED):
    in_u, in_g = (f in UNIFIED), (f in GATED)
    if not (in_g and not in_u):
        continue
    dny = denies_in_gated(f)
    on = f in ON
    if dny and on:
        dead.append(f)          # ★활성 버그: 켜져 있는데 발화 불가(V7이 이 상태였다)
        OK = False
    elif dny:
        latent.append(f)        # 잠재 위험: 꺼져 있어 지금은 무해하나 **켜면 무음 실패**
    print("  %-26s gated 전용 · deny=%-5s ON=%-5s %s"
          % (f, dny, on, "★★활성 버그" if (dny and on) else
             ("⚠잠재(켜면 무음 실패)" if dny else "(augment만 — 정당)")))

if dead:
    print("\n✗ FAIL — 설치되지 않는 경로에만 있는 레버: %s" % ", ".join(dead))
    print("  (selftest는 통과하지만 라이브에서 절대 발화하지 않는다 — V7 사고와 동형)")
else:
    print("\n✓ PASS — gated 전용 레버 없음")

# ─────────────────────────────────────────────────────────────────────────────
# ★2번째 사망 유형 — **표적 집합이 공집합** (2026-07-31·C263)
#   `T2_WRITE_CAP`은 훅(위 검사)만이 아니라 *표적*으로도 죽어 있었다: 캡 대상 = `kind=='confirm'`
#   게이트의 `applies_to`인데 **banking_knowledge A2엔 confirm 게이트가 없다** ⇒ 훅을 고쳐도
#   0건 발화. 위 검사는 이걸 통과시킨다(경로만 보므로). ⇒ **A2에서 표적을 뽑는 레버는
#   설정 도메인에서 그 집합이 비었는지도 봐야 한다.**
#   판정: ON ∧ 공집합 = FAIL · OFF ∧ 공집합 = 경고(승격 시 무음 실패).
# ─────────────────────────────────────────────────────────────────────────────
print("\n[표적 가드] A2-도출 표적 집합이 설정 도메인에서 비어 있지 않은가")

# 소스에서 (플래그, A2 도출식) 쌍을 **기계 추출**한다 — 손목록 금지([[03b]]·C261 교훈).
#   형태 ①  x = (a2.get("KEY") or []) if os.environ.get("T2_F") == "1" else []
#   형태 ②  x = _helper(a2) if f_on else set()      (f_on = os.environ.get("T2_F") == "1")
PAIRS = {}
for m in re.finditer(r"a2\.get\(\s*[\'\"]([a-z0-9_]+)[\'\"]", SRC):
    key = m.group(1)
    seg = SRC[m.end():m.end() + 220]
    fm = re.search(r"environ\.get\(\s*[\'\"](T2_[A-Z0-9_]+)[\'\"]", seg)
    if fm:
        PAIRS.setdefault(fm.group(1), set()).add(("key", key))
for m in re.finditer(r"=\s*(_[a-z0-9_]+)\(a2\)\s+if\s+([a-z0-9_]+)\b", SRC):
    helper, var = m.group(1), m.group(2)
    vm = re.search(r"%s\s*=\s*os\.environ\.get\(\s*[\'\"](T2_[A-Z0-9_]+)[\'\"]" % re.escape(var), SRC)
    if vm:
        PAIRS.setdefault(vm.group(1), set()).add(("fn", helper))

DOMAINS = [d for d in re.findall(r"--domain\s+([a-z_]+)", GO)] or ["banking_knowledge"]
try:
    import t2_gate_patch as G
    empty_on, empty_off = [], []
    for flag in sorted(PAIRS):
        for kind, name in sorted(PAIRS[flag]):
            for dom in sorted(set(DOMAINS)):
                a2 = G._domain_a2(dom)
                if a2 is None:
                    continue
                got = (a2.get(name) or []) if kind == "key" else getattr(G, name)(a2)
                if got:
                    continue
                (empty_on if flag in ON else empty_off).append("%s→%s(%s)" % (flag, name, dom))
    for x in empty_on:
        print("  ★★활성 버그(ON인데 표적 0): %s" % x)
        OK = False
    for x in empty_off:
        print("  ⚠잠재(OFF·켜면 0건 발화): %s" % x)
    # 커버리지는 **항상** 출력한다 — 무엇을 검사했는지 안 보이면 "이상 없음"이 신뢰 불가다.
    print("  검사 대상 %d 레버(도메인 %s): %s"
          % (len(PAIRS), ",".join(sorted(set(DOMAINS))),
             ", ".join("%s→%s" % (f, "/".join(n for _, n in sorted(PAIRS[f]))) for f in sorted(PAIRS))))
    if not empty_on and not empty_off:
        print("  ✓ 공집합 표적 없음")
except Exception as e:      # A2 로더가 없는 환경에선 이 절만 건너뛴다(경로 가드는 유지)
    print("  ⚠표적 가드 미실행(%r) — A2 로더 사용 가능한 환경에서 돌릴 것" % (e,))

print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
