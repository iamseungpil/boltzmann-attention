# -*- coding: utf-8 -*-
"""회귀 — DECIDE-FIRST 캐리는 **어느 인자의 답인지** 대거나, 아니면 나가지 않는다 (R7·2026-08-23).

무엇을 막는 검정인가 (`refute_5.json` §surviving_our_layer⑵ · verdicts[3]) —

  구판 문면은 축자로 이랬다:
      "Error: [DECIDE-FIRST] … It has now been made:
       … It answers: General.
       If that answers the choice this call encodes, make the call again with that value."
  **어느 인자의 답인지 한 번도 말하지 않는다.** t7336_halfB `task_085#s373753` 에서 결정 서브
  (`decide_from_docs`)가 고른 것은 *문서 계열 라벨*(`General` — 형제 값이 'Blue Account'·
  'Gold Account'·'Sky Blue')이었는데, 축 이름이 없으니 그 값이 `dispute_category` 슬롯으로
  흘러들었고 그 sim 의 **11회 시도 전부**가 열거 밖 값으로 실패했다. 모델은 축자로
  *"Based on the error message, it appears that the dispute should be categorized as
  \"General.\""* 라며 **우리 배달을 지목**한다 ⇒ 우리 층 결함이다([[55]]·[[64]]).

  수리 = [[64]] 두 가지 중 하나를 강제한다:
    ⒜ 인자를 **댈 수 있으면** 문면이 그 인자를 지목한다(`'{arg}' argument of this call`),
    ⒝ **못 대면 배달하지 않는다**(축 미상 = 값의 자리를 모른다 = 아무 슬롯에나 놓으라는 초대).

  술어는 전부 닫혀 있다 — 이름 동등성 + **선언된** 접두 + dict 조회뿐(`_write_choice_arg`).
  출처는 이미 있는 A2 키 셋뿐(새 키·새 레버 0·[[62]]): `write_arg_enum[].arg` ·
  `choice_grounding[].arg` · `recommendation_verify.operand`.

검정 구성
  §1 계약(합성 A2·도메인 무관)   — 선언된 write 는 인자를 댄다 / 선언 없는 write 는 `None`
  §2 부정통제                     — 선언된 접두가 안 맞으면 지목하지 않는다(과폭 금지)
  §3 문면                         — 인자를 지목하고, 재료를 축자로 싣고, [[64]] 두 조각을 담는다
  §4 결함 재현(양성대조)          — 구판 문면 모양에는 인자 이름이 **없다**
  §5 배선                         — 축 미상이면 `dw_fb` 도 서브 호출도 없다(소스 구조)
  §6 실물 선언                    — 출하된 A2 로 085(분쟁 write)=미상 · 070(계좌 write)=지목

오프라인 전용(LLM·서버·env 0). 실행: py -3 test_decide_first_axis_named.py
"""
import ast
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G                                              # noqa: E402

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
FAILED = []


def chk(cond, label, extra=""):
    print(("  OK   " if cond else "  FAIL ") + label + ((" — " + str(extra)) if extra else ""))
    if not cond:
        FAILED.append(label)


class TC(object):
    """ToolCall 스텁 — 이름과 인자만 있으면 된다(엔진이 보는 것이 그 둘뿐이다)."""

    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


# ── 합성 A2: 도메인과 무관한 이름만 쓴다(전이 확인용·[[05]]) ────────────────────────
A2_SYN = {
    "write_arg_enum": [{
        "applies_to": "call_dispatch",
        "applies_when": {"arg": "inner_name", "prefix": "make_widget"},
        "arg": "widget_class",
        "group_arg": "widget_kind",
        "group_map": {"round": "round_widgets", "square": "square_widgets"},
    }],
    "choice_grounding": [{"tool": "pick_colour_77", "arg": "colour_name"}],
    "recommendation_verify": {"action_tool": "order_part", "operand": "part_type"},
}


def call(name, outer=None, inner=None):
    a = dict(outer or {})
    if inner is not None:
        a["arguments"] = json.dumps(inner)
    return TC(name, a)


print("\n§1 계약 — 선언이 지목하는 write 는 인자를 댄다")
arg, ax = G._write_choice_arg(A2_SYN, call("call_dispatch",
                                           {"inner_name": "make_widget_4821"},
                                           {"widget_kind": "round", "widget_class": "X"}))
chk(arg == "widget_class", "write_arg_enum: 인자 이름을 댄다", arg)
chk(ax == "round_widgets", "write_arg_enum: 축(group)도 종전대로 나온다", ax)

arg2, ax2 = G._write_choice_arg(A2_SYN, call("call_dispatch",
                                             {"inner_name": "make_widget_4821"},
                                             {"widget_kind": "hexagon"}))
chk(arg2 == "widget_class", "group 값이 선언 밖이어도 **인자 이름은** 나온다", arg2)
chk(ax2 is None, "그 때 축은 None (종전 `_dax` 의미 보존)", ax2)

arg3, _ = G._write_choice_arg(A2_SYN, call("pick_colour_77", {"colour_name": "Z"}))
chk(arg3 == "colour_name", "choice_grounding: 도구 이름 동등성으로 지목", arg3)

arg4, _ = G._write_choice_arg(A2_SYN, call("order_part", {"part_type": "Q"}))
chk(arg4 == "part_type", "recommendation_verify: action_tool → operand", arg4)

# 디스패처를 통해 불린 choice_grounding 도구도 같은 값으로 지목돼야 한다(unwrap 경로).
arg4b, _ = G._write_choice_arg(A2_SYN, call("call_discoverable_agent_tool",
                                            {"agent_tool_name": "pick_colour_77"},
                                            {"colour_name": "Z"}))
chk(arg4b == "colour_name", "디스패처로 불려도 env 축자 이름으로 지목된다", arg4b)

print("\n§2 부정통제 — 선언이 지목하지 않는 write 는 **미상**이어야 한다")
n1 = G._write_choice_arg(A2_SYN, call("call_dispatch",
                                      {"inner_name": "file_other_thing_6281"},
                                      {"some_category": "General"}))
chk(n1 == (None, None), "선언된 접두가 안 맞으면 지목 0 (085 모양)", n1)
n2 = G._write_choice_arg(A2_SYN, call("unrelated_write", {"a": 1}))
chk(n2 == (None, None), "선언에 없는 도구는 지목 0", n2)
n3 = G._write_choice_arg({}, call("call_dispatch", {"inner_name": "make_widget_1"}, {}))
chk(n3 == (None, None), "A2 가 비면 지목 0 (미선언 도메인 no-op)", n3)
n4 = G._write_choice_arg(A2_SYN, None)
chk(n4 == (None, None), "쓰레기 입력에도 예외 없이 (None, None)", n4)
# ★과폭 금지: group_map 값만 우연히 맞아도 **도구가 안 맞으면** 지목하지 않는다.
#   (구판 `_dax` 는 applies_to 를 안 보고 group_map 만 훑어서 여기서 축을 붙였다.)
n5 = G._write_choice_arg(A2_SYN, call("some_other_tool", {}, {"widget_kind": "round"}))
chk(n5 == (None, None), "group 값만 맞는 남의 도구에는 축을 붙이지 않는다", n5)

print("\n§3 문면 — 인자를 지목하고 재료를 축자로 싣는다 ([[64]])")
MAT = "\nA separate check was run on the documents on record. It answers: General."
body = G._DECIDE_FIRST_FB.format(arg="widget_class", material=MAT)
chk("'widget_class'" in body, "문면이 인자를 따옴표로 지목한다")
chk(body.count("widget_class") >= 3, "지목이 한 번으로 끝나지 않는다(조건절에도 있다)",
    body.count("widget_class"))
chk(MAT in body, "결정 재료가 **축자로** 실린다(엔진 편집 0)")
chk("held for one turn" in body and "had not been made" in body,
    "[[64]] ⓐ 무엇이 틀렸나 (유예 사유)")
chk("make the call again" in body, "[[64]] ⓑ 무엇을 하면 풀리나 (재호출)")
chk("not a value" in body and "not an answer for this call" in body,
    "값이 그 인자의 것이 아닐 때의 출구가 문면에 있다 (오배치 차단)")
chk("that value" not in body,
    "구판의 무지목 지시어('that value')가 남아 있지 않다")
# 엔진에 도메인 리터럴 0 — 인자 이름은 전부 format 인자에서 온다.
chk("{arg}" in G._DECIDE_FIRST_FB and "{material}" in G._DECIDE_FIRST_FB,
    "템플릿은 자리표시자만 갖는다(도메인 낱말 0·[[05]])")
other = G._DECIDE_FIRST_FB.format(arg="part_type", material=MAT)
chk("widget_class" not in other and "part_type" in other,
    "다른 도메인 인자로도 그대로 전이된다")
# 엔진은 고르지 않는다 — 문장은 전부 조건문이고 단정이 없다([[62]] ③④).
for bad in ("the correct value is", "you must use", "the answer is"):
    chk(bad not in body.lower(), "단정문이 없다: %r" % bad)

print("\n§4 결함 재현(양성대조) — 구판 문면에는 인자 이름이 없다")
OLD = ("Error: [DECIDE-FIRST] this write was held for one turn "
       "because the decision it encodes had not been made in this "
       "conversation yet. It has now been made:\n" + MAT +
       "\nIf that answers the choice this call encodes, make the "
       "call again with that value. Otherwise make it again as it was.")
chk("widget_class" not in OLD and "dispute_category" not in OLD,
    "구판은 어떤 인자도 지목하지 않는다 (결함이 실재했다)")
chk("with that value" in OLD, "구판의 무지목 지시어가 재현됐다 (통제)")
chk("with that value" not in G._DECIDE_FIRST_FB,
    "구판 지시어가 출하 템플릿에서 사라졌다")

print("\n§5 배선 — 축 미상이면 배달도 서브 호출도 없다")
tree = ast.parse(SRC)
LINES = SRC.split("\n")


def seg_of(node):
    """줄 번호로 잘라낸다 — `ast.get_source_segment` 는 노드마다 13k 줄을 다시 쪼갠다(느림)."""
    lo = getattr(node, "lineno", None)
    hi = getattr(node, "end_lineno", None)
    if not lo or not hi:
        return ""
    return "\n".join(LINES[lo - 1:hi])


def _find_block():
    """`_darg, _dax = _write_choice_arg(...)` 를 세우는 그 `if _wc is not None:` 블록."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        s = seg_of(node)
        if "_write_choice_arg(a2, _wc)" in s and "dw_fb" in s:
            return node, s
    return None, ""


blk, seg = _find_block()
chk(blk is not None, "DECIDE-FIRST 블록을 찾았다")
guard = None
for node in ast.walk(blk) if blk else ():
    if isinstance(node, ast.If) and seg_of(node).lstrip().startswith("if not _darg:"):
        guard = node
        break
chk(guard is not None, "`if not _darg:` 가드가 존재한다")
if guard is not None:
    body_txt = "\n".join(seg_of(n) for n in guard.body)
    else_txt = "\n".join(seg_of(n) for n in guard.orelse)
    chk("dw_fb" not in body_txt, "축 미상 가지에서 `dw_fb` 를 세우지 않는다")
    chk("_search_material" not in body_txt,
        "축 미상 가지에서 결정 서브를 돌리지 않는다(축 잠금 소모 방지)")
    chk("_t2_dwrite_deny" not in body_txt, "축 미상 가지는 sim 상한을 태우지 않는다")
    chk("무발화" in body_txt, "축 미상은 로그로 남는다([[25]] 계기)")
    chk("_DECIDE_FIRST_FB.format(arg=_darg" in else_txt,
        "배달은 `_darg` 를 실어서만 조립된다")
    chk("_search_material" in else_txt, "재료 조달은 지목 가지 안에 있다")
# ★2026-08-26 갱신 (x543 배치 수리): 종전엔 이 블록의 `dw_fb = (` 가 **한 자리**여야 한다고
#   봤다. 선언 배달(명세·규칙·인자-정책)이 같은 블록에서 **덧붙는** 구조가 되면서 대입은 둘이
#   된다 — ⑴캐리의 조립 ⑵캐리를 되살리는 병합. 지키려던 것은 *배달 문면이 여러 곳에서
#   조립되지 않는다* 이므로 그것을 직접 검정한다.
chk(seg.count("_DECIDE_FIRST_FB.format(") == 1, "캐리 문면 조립은 한 자리뿐",
    seg.count("_DECIDE_FIRST_FB.format("))
_assigns = [ln.strip() for ln in seg.split("\n") if ln.strip().startswith("dw_fb = (")]
chk(bool(_assigns) and all(("_DECIDE_FIRST_FB" in a) or ("_carry_hold" in a)
                           for a in _assigns),
    "직접 대입은 캐리 조립·병합뿐 (선언 배달은 `_decl_join` 을 지난다)", len(_assigns))
chk("arg=%s" in seg, "로그가 인자 이름을 병기한다(검산 가능)")
# 구판 문면(무지목)이 이 배달 자리에서 실제로 사라졌는지 — 주석·독스트링의 역사 인용은
# 남아 있어도 되지만 **배달 블록**에는 한 글자도 없어야 한다.
chk("with that value" not in seg and "It has now been made:" not in seg,
    "구판 무지목 문면이 배달 블록에서 사라졌다")

# 기본 OFF 보존 — 레버 자신의 플래그가 없으면 이 블록은 통째로 안 돈다.
chk('os.environ.get("T2_DECIDE_BEFORE_WRITE") == "1"' in SRC,
    "레버 기본 OFF 유지 (플래그 없으면 종전과 바이트 동일)")
chk('os.environ.get("T2_DECIDE_BEFORE_WRITE", "1")' not in SRC,
    "기본값을 1 로 두지 않았다")

print("\n§6 실물 선언 — 출하된 A2 로 085/070 을 가른다")
try:
    from gate_interpreter import load_domain_a2
    A2 = load_domain_a2("banking_knowledge")
except Exception as e:                                                 # pragma: no cover
    A2 = None
    print("  SKIP A2 로드 실패: %r" % (e,))
if A2:
    # ★2026-08-26 갱신 — **A2 가 바뀌었다**. 이 칸은 원래 *"085 분쟁 write 는 축 미상이라
    #   캐리가 안 나간다"* 를 고정했는데, 2026-08-25 `e2c5f362`(책임 한도 표를 write 자리에
    #   놓고 선언된 규칙이 출처를 대게 함)가 이 write 의 선택 인자를 **선언**했다. 그래서 지금
    #   옳은 계약은 *"선언이 있으니 인자를 대고 캐리가 나간다"* 이다.
    #   ⚠지키려던 것(= **이름을 못 대면 배달하지 않는다**)은 위 합성 A2 절(`n1`~`n5`)과
    #     `if not _darg:` 가드 검정이 그대로 들고 있다 — 여기서 푸는 것이 아니다.
    d085 = G._write_choice_arg(A2, call("call_discoverable_agent_tool",
                                        {"agent_tool_name":
                                         "file_debit_card_transaction_dispute_6281"},
                                        {"dispute_category": "General",
                                         "transaction_id": "t1"}))
    chk(d085[0] == "dispute_category",
        "085 분쟁 write = A2 선언대로 인자를 댄다 (e2c5f362 이후의 계약)", d085)
    # 070/071 계좌 개설 write — 선언이 있으므로 인자를 지목하고 계속 발화한다.
    d070 = G._write_choice_arg(A2, call("call_discoverable_agent_tool",
                                        {"agent_tool_name": "open_bank_account_4821"},
                                        {"account_type": "checking",
                                         "account_class": "Sky Blue"}))
    chk(d070[0] and d070[1],
        "070 계좌 write = 인자·축 둘 다 나온다 (기존 표적 보존)", d070)
    chk(G._DECIDE_FIRST_FB.format(arg=d070[0], material=MAT).count(d070[0]) >= 3,
        "그 문면이 실제로 그 인자를 지목한다")

print("\n%s (%d fail)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
for f in FAILED:
    print("  - " + f)
sys.exit(0 if not FAILED else 1)
