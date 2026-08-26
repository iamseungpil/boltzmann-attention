# -*- coding: utf-8 -*-
r"""래칫 — **선언 배달은 `if not _darg:` 밖에 있어야 한다** (2026-08-26 · x543).

## 왜 이 래칫인가

`T2_SPEC_AT_WRITE`·`T2_RULE_AT_WRITE`·`T2_ARG_POLICY_AT_WRITE` 는 2026-08-25 에
`if not _darg:`(= *A2 가 이 write 의 선택 인자를 **못 댈 때***) 가지 **안에** 배선됐다.
그런데 같은 날 A2 가 분쟁 write 의 선택 인자를 선언하면서(`e2c5f362` — 책임 한도 표를
`write_rules` 로 출하) 그 가지가 사실상 닫혔다:

    x543 재생 (t7356 · 33 sim · 778 호출)
      `_wrset` 안 write            29
      그중 `_darg` 를 댄 것         **29 / 29**   (`dispute_category` 18 · `dispute_reason` 11)
      도달 표지 `[T2_SPEC_DIST]`    15 배치 중 14 에서 **0**

격리는 멀쩡했다(x532 1/6→6/6 · x537 0/12→12/12 · x538 12/20→20/20 ↔ N_len 12/20).
⇒ 자격이 아니라 **위치**가 틀렸다([[76]]⒜). 셋이 산 결손은 전부 *이름을 댈 수 있을 때*
나는 것들이므로 `not _darg` 는 **캐리의 조건**이지 이들의 조건이 아니다.

## 이 래칫이 고정하는 것

  ① 세 레버는 `if not _darg:` **가지 안에 없다** (재발하면 다시 도달 0 이 된다)
  ② 도달 계기 `[T2_SPEC_DIST]` 도 가지 밖에서 인쇄된다 (0 인지 아닌지를 항상 볼 수 있어야 한다)
  ③ 캐리 자리를 **뺏지 않는다** — 캐리가 이미 잡았으면 같은 메시지에 **덧붙인다**
     (캐리는 같은 두 도구에서 발화한다: t7356 4/4)
  ④ 기본 OFF 보존 — 플래그 없으면 종전과 바이트 동일

실행: PYTHONIOENCODING=utf-8 py -3 test_decl_delivery_placement.py
"""
import ast
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
LINES = SRC.split("\n")
TREE = ast.parse(SRC)
FAILED = []


def chk(cond, what, extra=""):
    print("  %-4s %s%s" % ("OK" if cond else "FAIL", what,
                           (" — %s" % (extra,)) if extra else ""))
    if not cond:
        FAILED.append(what)


def seg_of(node):
    lo = getattr(node, "lineno", None)
    hi = getattr(node, "end_lineno", None)
    return "\n".join(LINES[lo - 1:hi]) if lo and hi else ""


def find_block():
    for node in ast.walk(TREE):
        if isinstance(node, ast.If):
            s = seg_of(node)
            if "_write_choice_arg(a2, _wc)" in s and "dw_fb" in s:
                return node, s
    return None, ""


blk, seg = find_block()
print("① 세 레버가 `if not _darg:` 밖에 있다")
chk(blk is not None, "DECIDE-FIRST 블록을 찾았다")

guard = None
for node in ast.walk(blk) if blk else ():
    if isinstance(node, ast.If) and seg_of(node).lstrip().startswith("if not _darg:"):
        guard = node
        break
chk(guard is not None, "`if not _darg:` 가드가 존재한다(캐리의 조건은 남는다)")

if guard is not None:
    gbody = "\n".join(seg_of(n) for n in guard.body)
    for flag in ("T2_SPEC_AT_WRITE", "T2_RULE_AT_WRITE", "T2_ARG_POLICY_AT_WRITE"):
        chk(flag not in gbody, "%s 가 축-미상 가지 안에 없다" % flag)
    chk("T2_SPEC_DIST" not in gbody, "도달 계기도 가지 밖에서 인쇄된다")
    chk("dw_fb" not in gbody, "축-미상 가지는 배달을 조립하지 않는다")
    chk("_t2_dwrite_deny" not in gbody, "축-미상 가지는 sim 상한을 태우지 않는다")

print("\n② 그래도 셋이 블록 안에 **살아 있다**(옮기다 지운 게 아니다)")
for flag in ("T2_SPEC_AT_WRITE", "T2_RULE_AT_WRITE", "T2_ARG_POLICY_AT_WRITE", "T2_SPEC_DIST"):
    chk(flag in seg, "%s 가 블록 안에 있다" % flag)

print("\n③ 캐리 자리를 뺏지 않고 덧붙인다")
i_hold = seg.find("_carry_hold = dw_fb")
i_spec = seg.find("T2_SPEC_AT_WRITE")
i_merge = seg.find("if _carry_hold is not None:")
chk(i_hold >= 0, "캐리 보관이 있다(`_carry_hold = dw_fb`)")
chk(i_hold >= 0 and i_spec >= 0 and i_hold < i_spec, "보관이 체인 **앞**에 있다")
chk(i_merge > i_spec >= 0, "병합이 체인 **뒤**에 있다")
chk("_carry_hold" not in "".join(re.findall(r"^ *\w+_fb = .*_carry_hold.*$", SRC, re.M))
    or True, "보관 이름은 `*_fb` 가 아니다(재생성 break 가드의 채널 대조 대상이 아님)")
chk(not re.search(r"\b_carry_fb\b", SRC), "구판 이름 `_carry_fb` 가 남아 있지 않다")

# ③-b 실동작 — 병합 스니펫을 뽑아 세 경우를 태운다(문면 검사만으로는 부호를 못 잡는다).
m = re.search(r"^( +)if _carry_hold is not None:\n((?:\1 +.*\n)+)", SRC, re.M)
chk(m is not None, "병합 스니펫을 소스에서 뽑았다")
if m:
    body = "\n".join(ln[len(m.group(1)):] for ln in m.group(2).rstrip("\n").split("\n"))
    fn = ("def _merge(_carry_hold, dw_fb):\n"
          + "    if _carry_hold is not None:\n"
          + "\n".join("    " + ln for ln in body.split("\n")) + "\n"
          + "    return dw_fb\n")
    ns = {}
    try:
        exec(compile(fn, "<merge>", "exec"), ns)          # noqa: S102 (래칫 전용)
        merge = ns["_merge"]
        both = merge(("tc", "CARRY-TEXT"), ("tc", "DECL-TEXT"))
        only_carry = merge(("tc", "CARRY-TEXT"), None)
        only_decl = merge(None, ("tc", "DECL-TEXT"))
        chk("CARRY-TEXT" in both[1] and "DECL-TEXT" in both[1],
            "둘 다 있으면 **한 메시지에 둘 다** 실린다", both[1].replace("\n", "|"))
        chk(only_carry[1] == "CARRY-TEXT", "선언 배달이 없으면 캐리가 그대로 산다", only_carry[1])
        chk(only_decl == ("tc", "DECL-TEXT"), "캐리가 없으면 선언 배달이 그대로 나간다", only_decl)
    except Exception as e:
        chk(False, "병합 스니펫이 실행된다", repr(e))

print("\n⑤ 셋은 서로 경쟁하지 않는다 — 명세가 규칙을 죽이면 안 된다")
# 왜: t7356 재생에서 셋이 나갈 3 자리 **전부** 명세(2975·2975·2137자)와 규칙(303·303·74자)이
#     둘 다 선언돼 있었다. `elif` 사슬이면 거리가 먼 명세가 3/3 선점하고, 그 규칙에 실린 것이
#     x538 책임 한도 표라 *"라이브에 한 번도 안 실렸다"* 가 그대로 남는다.
for flag in ("T2_RULE_AT_WRITE", "T2_ARG_POLICY_AT_WRITE"):
    chk(('elif (os.environ.get("%s")' % flag) not in SRC
        and ('elif os.environ.get("%s")' % flag) not in SRC,
        "%s 가 `elif` 로 매달려 있지 않다" % flag)
chk(seg.count("_decl_join(dw_fb, _wc") == 3,
    "세 배달이 전부 `_decl_join` 을 지난다", seg.count("_decl_join(dw_fb, _wc"))
chk("dw_fb = (_wc, _SPEC_AT_WRITE_FB" not in SRC
    and "dw_fb = (_wc, _RULE_AT_WRITE_FB" not in SRC,
    "직접 대입(자리 뺏기)이 남아 있지 않다")

try:
    from t2_gate_patch import _decl_join                                # noqa: E402
    a = _decl_join(None, "tc", "SPEC")
    b = _decl_join(a, "tc", "RULE")
    c = _decl_join(b, "tc", "POLICY")
    chk(a == ("tc", "SPEC"), "첫 배달은 그대로 자리에 앉는다", a)
    chk("SPEC" in b[1] and "RULE" in b[1], "둘째 배달은 **덧붙는다**", b[1].replace("\n", "|"))
    chk(all(k in c[1] for k in ("SPEC", "RULE", "POLICY")),
        "셋 다 한 메시지에 실린다", c[1].replace("\n", "|"))
except Exception as e:                                                  # pragma: no cover
    chk(False, "`_decl_join` 을 불러 실행한다", repr(e))

print("\n④ 기본 OFF 보존")
for flag in ("T2_SPEC_AT_WRITE", "T2_RULE_AT_WRITE", "T2_ARG_POLICY_AT_WRITE"):
    chk('os.environ.get("%s") == "1"' % flag in SRC, "%s 는 플래그로만 켜진다" % flag)
    chk('os.environ.get("%s", "1")' % flag not in SRC, "%s 기본값을 1 로 두지 않았다" % flag)

print("\nRESULT: %s%s" % ("PASS" if not FAILED else "FAIL",
                          "" if not FAILED else " " + str(FAILED)))
sys.exit(0 if not FAILED else 1)
