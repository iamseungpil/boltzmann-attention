# -*- coding: utf-8 -*-
"""`T2_NOW_SELFCALL` 검정 — 시계 자가호출이 (a) 열리고 (b) **아무것도 안 바꾸는지**.

왜 이 검정인가(t7295·071): 검색 에이전트의 창은 결정점에서만 열리는데 071 세 sim 통틀어
그 창이 **1번**, 그 한 번이 시계보다 **앞**이었다 → 만료 제거 기계가 재료를 한 번도 못 냈다
(`now 미확정` = arm b 침묵 80회 중 1위 사유). 수정은 "엔진이 A2 선언 도구를 직접 부른다"이고,
그 정당성은 **부작용 0** 에 달려 있다. 그래서 여기서 두 가지를 못 박는다:

  ⒜ `get_current_time` 은 **DB 해시를 바꾸지 않는다**(순수 읽기) — 이게 깨지면 fix 는 무효다.
  ⒝ 플래그 OFF 면 **호출 자체가 없다**(098·100 불변 의무·[[57]] Δspurious).

실행(리모트): PYTHONPATH=tau2-bench/src seka python test_now_selfcall.py
"""
import io
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    from tau2.registry import registry
    from t2_gate_patch import load_domain_a2

    a2 = load_domain_a2("banking_knowledge")
    spec = next((s for s in (a2.get("ledger_metrics") or ()) if s.get("now_prompt")), None)
    check(spec is not None, "A2 에 now_prompt 스펙이 있다")
    ntool = (spec or {}).get("now_tool")
    check(bool(ntool), "같은 스펙 안에 now_tool 이 있다 (%r)" % (ntool,))

    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="no_knowledge")
    before = env.tools.get_db_hash()
    res = env.make_tool_call(tool_name=ntool, requestor="assistant")
    after = env.tools.get_db_hash()

    check(bool(res), "엔진이 %s 를 직접 부를 수 있다" % ntool)
    # ⒜ 부작용 0 — 이 검정이 이 수정의 **유일한 정당화 근거**다.
    check(before == after, "호출 후 DB 해시 불변 (부작용 0)")
    # 내용은 판정하지 않는다(날짜 해석은 LLM 몫·[[59]]) — 문자열이 왔는지만 본다.
    check(isinstance(res, str) and len(res) > 0, "문자열을 돌려준다: %r" % (str(res)[:60],))

    # ⒝ 플래그 규약 — 켜지 않으면 자가호출 코드에 도달하지 않는다.
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "t2_gate_patch.py"), encoding="utf-8").read()
    check('os.environ.get("T2_NOW_SELFCALL") == "1"' in src, "자가호출은 플래그로 가려져 있다")
    check(src.count("[T2_NOW_SELFCALL]") >= 2, "성공·실패 둘 다 로그를 남긴다([[64]])")

    # ⒞ 두 번째 결함(창 자체가 안 열림) — 재료 배달이 `deny` 밖에서도 가능해야 한다.
    check('os.environ.get("T2_SEARCH_ON_PROCEED") == "1"' in src,
          "deny-밖 배달도 플래그로 가려져 있다")
    i_new = src.find('_ar.get("status") != "deny"')
    i_deny = src.find('if _ar.get("status") == "deny":\n                                _fb_ar')
    check(i_new != -1 and i_deny != -1 and i_new < i_deny,
          "deny-밖 배달은 deny 분기 **앞**에 있다 (분기 안이면 열리지 않는다)")
    check(src.count("[T2_SEARCH_ON_PROCEED]") >= 3,
          "배달·미배달·실패 셋 다 로그를 남긴다([[64]])")

    print("\n%s" % ("PASS" if not FAIL else "FAIL: " + " · ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
