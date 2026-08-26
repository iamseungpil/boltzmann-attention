# -*- coding: utf-8 -*-
"""OPERATOR-SELECT 대칭 가드 — **이미 한 일을 틀렸다고 말하지 않는다**.

사건(2026-08-14 야간·t7292 073 t0·이 레버의 **세 번째** 같은 부류 결함):
  msg45  `apply_checking_account_credit_5829` **성공**(잔액 $5200.00 → $5209.50)
  그 뒤   우리 `[OPERATOR-SELECT]` 가 **5회 이상**(사이드카 arrived) —
         *"요청은 `apply_statement_credit_8472` 에 매핑된다"* ← **신용카드용 도구**·체킹 태스크서 오답
  msg49  모델이 재-unlock → 재호출 → **같은 계좌에 $9.50 두 번**(→ $5219.00)
  결과   gold 액션 8/11 로 **통과 때와 동일**한데 `db_match=False` → reward 0

되돌릴 수 없는 write 를 이미 한 자리에서 *"그건 틀린 도구다"* 라고 말하면, 모델이 할 수 있는 일은
**재시도뿐**이다 — 그 문구는 교정이 아니라 **중복 실행 지시**다.

기존 가드는 `want`(우리가 대신 지목하는 도구)만 봤다. 이 검정은 **`chosen`(모델이 이미 성공시킨
도구)** 쪽 대칭을 고정한다. 술어는 닫혀 있다(호출 이력·도메인 판단 0).

선례: C10(051 — 선언된 요구 집합 원소면 침묵) · 2026-08-12 070t0(`want` 완료면 침묵) · 이번(`chosen` 완료면 침묵).
"""
import io
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_resolve as R                                             # noqa: E402


class _Tool:
    """범위 표면화가 읽을 도구 설명(프레임워크 레지스트리 대역)."""

    def __init__(self, name, description):
        self.name = name
        self.description = description

FAIL = []
CHOSEN = "apply_checking_account_credit_5829"
WANT = "apply_statement_credit_8472"
DISP = "call_discoverable_agent_tool"
A2 = {"eplan": {"dispatch_tool": DISP}}
OPSPEC = {"operator_resolution": "discoverable", "arg": "agent_tool_name",
          "name_pattern": r"[a-z_]+_\d{4}", "find_intent": True, "getter": "KB_search_bm25"}


def chk(c, m):
    if not c:
        FAIL.append(m)
    print("  %s %s" % ("ok  " if c else "FAIL", m))


class TC:
    def __init__(self, name, args, tid="c1"):
        self.name = name
        self.arguments = args
        self.id = tid


class M:
    def __init__(self, role, content=None, calls=None, mid=None, error=False):
        self.role = role
        self.content = content
        self.tool_calls = calls
        self.id = mid
        self.error = error


def convo(executed):
    """발견 결과 + (선택) 그 도구의 **성공 실행** 이력."""
    # 발견은 **성공한** tool-result 에서만 모인다(`discovered_names` 계약) — error=True 로 두면
    # 후보가 비어 `find_intent` 분기 자체에 못 들어간다(초판 픽스처의 실수).
    ms = [M("tool", "docs mention %s and %s" % (CHOSEN, WANT))]
    if executed:
        ms += [M("assistant", None, [TC(DISP, {"agent_tool_name": CHOSEN}, "x1")]),
               M("tool", "Credit applied successfully! Amount: $9.50", mid="x1")]
    return ms


def run(executed, want):
    """`formalize_intent_tool` 을 우리가 원하는 답으로 고정해 판정부만 시험한다."""
    orig = R.formalize_intent_tool
    R.formalize_intent_tool = lambda *a, **k: want
    try:
        class _Ag:
            tools = [_Tool(CHOSEN, "Apply a credit to a customer's checking account."),
                     _Tool(WANT, "Apply a statement credit to a customer's credit card account.")]
        return R.resolve_operator(OPSPEC, {"agent_tool_name": CHOSEN}, convo(executed),
                                  agent=_Ag(), la=object(), UserMessage=object(), a2=A2)
    finally:
        R.formalize_intent_tool = orig


def main():
    print("[핵심 — 이미 성공한 chosen 을 부정하지 않는다]")
    r = run(executed=True, want=WANT)
    chk(r.get("status") == "ok",
        "chosen 이 이미 성공 실행 → 침묵(%s) ← 073 중복 적립의 자리" % r.get("status"))

    print("[지목 → 범위 표면화 (x322·기본 경로)]")
    r2 = run(executed=False, want=WANT)
    chk(r2.get("reason") == "operator-scope",
        "미실행이면 **범위 표면화**로 나간다(reason=%s)" % r2.get("reason"))
    fb = str(r2.get("feedback") or "")
    chk("OPERATOR-SCOPE" in fb, "문구가 범위 표면화 태그를 쓴다")
    chk("call that one" not in fb and "maps to" not in fb,
        "**지목 문구가 없다** — x322: 지목은 24/24 → 0/24 로 파괴한다")
    chk(CHOSEN in fb and WANT in fb, "두 후보의 범위가 나란히 제시된다(선택은 LLM)")

    print("[되돌릴 길 — 명시적으로 켤 때만 지목]")
    os.environ["T2_OPERATOR_PINPOINT"] = "1"
    try:
        r3 = run(executed=False, want=WANT)
        chk(r3.get("reason") == "operator-find", "플래그를 켜면 종전 지목(%s)" % r3.get("reason"))
    finally:
        os.environ.pop("T2_OPERATOR_PINPOINT", None)

    print("[기존 가드 회귀 없음]")
    chk(run(executed=True, want=CHOSEN).get("status") == "ok", "want==chosen 이면 침묵")
    chk(run(executed=False, want=None).get("status") == "ok", "formalize 실패면 침묵")

    # ─── ★수리 1: 검증한 뒤 참말 (2026-08-26·x550 §1) ───
    # 074 에서 우리 A2 스캐폴드 도구 `get_atm_fee_discrepancies` 를 **"네가 지어냈다"** 로
    # 6회 막았다(t7358·t7360 재현). 권위는 **`agent.tools`**(프레임워크)이지 A2 선언이 아니다.
    print("\n[수리 1 — 발견 안 된 이름: 우리가 **들고 있으면** 처방이 달라진다]")
    HELD = "get_atm_fee_discrepancies"          # 레지스트리에 있는 직접 호출 도구
    GHOST = "get_totally_made_up_thing_1111"    # 어디에도 없는 이름

    def run_prov(name, held):
        class _Ag:
            tools = ([_Tool(name, "A scaffold tool the agent already holds.")]
                     if held else [])
        return R.resolve_operator(OPSPEC, {"agent_tool_name": name},
                                  convo(False), agent=_Ag(), la=object(),
                                  UserMessage=object(), a2=A2)

    r3 = run_prov(HELD, held=True)
    chk(r3.get("reason") == "operator-direct",
        "레지스트리에 **있는** 이름 → operator-direct (reason=%s)" % r3.get("reason"))
    fb3 = str(r3.get("feedback") or "")
    chk("already have" in fb3 and "directly" in fb3,
        "문면이 **무엇을 하면 풀리는지**를 말한다 — '직접 불러라'([[64]])")
    chk("did not discover" not in fb3 and "invent" not in fb3,
        "**'지어냈다'고 말하지 않는다** — 우리가 준 도구다([[25]])")
    chk(r3.get("status") == "deny",
        "그래도 **거부는 유지**된다 — 감싼 호출은 env 가 어차피 죽인다(면제 아님)")

    r4 = run_prov(GHOST, held=False)
    chk(r4.get("reason") == "operator-fab",
        "레지스트리에 **없는** 이름 → 구판 문면 그대로 (reason=%s)" % r4.get("reason"))
    chk("invent" in str(r4.get("feedback") or ""),
        "진짜 날조에는 여전히 '발명하지 마라'라고 말한다")

    r5 = run_prov(HELD, held=False)
    chk(r5.get("reason") == "operator-fab",
        "**A2 가 뭐라 하든** `agent.tools` 에 없으면 날조 취급 — 선언은 권위가 아니다")

    # ─── ★수리 2: 읽기 선택에는 말하지 않는다 (2026-08-26·x550 §2) ───
    # 최근 12런 `[OPERATOR-SCOPE]` 61회 중 read 46 · **61 중 49 는 끝내 실행됐다**(턴만 태움).
    print("\n[수리 2 — 읽기 오선택에는 침묵, 쓰기에는 그대로]")
    READ, READ_W = "get_debit_cards_by_account_id_7823", "get_user_dispute_history_7291"

    def run_read(want, scope_all=False):
        if scope_all:
            os.environ["T2_SCOPE_ALL"] = "1"
        else:
            os.environ.pop("T2_SCOPE_ALL", None)
        orig = R.formalize_intent_tool
        R.formalize_intent_tool = lambda *a, **k: want
        try:
            class _Ag:
                tools = [_Tool(READ, "Retrieve all debit cards for a checking account."),
                         _Tool(READ_W, "Retrieve a user's dispute history.")]
            ms = [M("tool", "docs mention %s and %s" % (READ, READ_W))]
            return R.resolve_operator(OPSPEC, {"agent_tool_name": READ}, ms,
                                      agent=_Ag(), la=object(), UserMessage=object(), a2=A2)
        finally:
            R.formalize_intent_tool = orig
            os.environ.pop("T2_SCOPE_ALL", None)

    chk(run_read(READ_W).get("status") == "ok",
        "chosen 이 **read** 면 침묵한다 — 읽기 오선택은 회복 가능")
    chk(run_read(READ_W, scope_all=True).get("reason") == "operator-scope",
        "`T2_SCOPE_ALL=1` 이면 구판대로 발화한다 — 되돌릴 길을 남긴다([[60]])")
    chk(run(executed=False, want=WANT).get("reason") == "operator-scope",
        "**write 에는 그대로 발화**한다(회귀 없음)")

    print("\n%s (%d fail)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
