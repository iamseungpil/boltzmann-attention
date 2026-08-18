# -*- coding: utf-8 -*-
"""VC **호출-트리거**(`T2_VERDICT_GATE`) 검정 — 설계 `VERDICT_CALL_TRIGGER_DESIGN_2026_08_18.md`.

핀으로 박는 것:
  ① **양성 대조** — VIOLATES + 근거검산 통과 + 제출값 일치 → 거부하고, 문면에 *LLM 이 쓴 판정 줄*과
     *충돌하지 않는 후보 명단*이 둘 다 있다([[64]]).
  ② `OK` 판정이면 침묵(엔진은 고르지 않는다).
  ③ **근거 미검산이면 침묵** — 인용이 문서에 없으면 막지 않는다([[25]]).
  ④ 요구 인용이 0건이면 침묵(판별자가 없으면 판정도 없다).
  ⑤ 대안이 0개면 침묵 — 이름 없는 거부는 창 순환을 만든다(C536ⓑ).
  ⑥ **표기 왕복** — `_slug_disp`(하이픈 뒤 대문자) ↔ `by_name` 슬러그 키가 어긋나지 않는다(FIX-6 가족).
  ⑦ 배선 — 플래그·상한·호출이 소스에 실재한다(死배선 방지·[[67]] 0단계).

⚠전부 오프라인이다. 서브콜은 대역으로 갈아 끼우고 모델을 부르지 않는다.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as TS
import t2_gate_patch as GP

# ★스텁 주입 — 로컬엔 `tau2` 가 없다(리모트 전용). 없는 채로 두면 헬퍼가 첫 줄에서
#   `skip=import` 로 빠져나가고 **부정 대조 ②~⑤ 가 전부 거짓 통과**한다(C543ⓕ 계기 실패와
#   같은 형태 = 신호가 100%/0% 로 붙는 실행). 서브콜은 대역이라 이 두 이름은 쓰이지 않는다.
#   ⚠리모트에는 진짜 모듈이 있으므로 그때는 이 블록이 아무 일도 하지 않는다.
try:
    import tau2.agent.llm_agent  # noqa: F401
except Exception:
    import types
    for _n in ("tau2", "tau2.agent", "tau2.agent.llm_agent",
               "tau2.data_model", "tau2.data_model.message"):
        sys.modules.setdefault(_n, types.ModuleType(_n))
    sys.modules["tau2.data_model.message"].UserMessage = object
    sys.modules["tau2"].agent = sys.modules["tau2.agent"]
    sys.modules["tau2.agent"].llm_agent = sys.modules["tau2.agent.llm_agent"]

HERE = os.path.dirname(os.path.abspath(__file__))
DOC_A = "Silver Plus Account. Up to 15 free withdrawals per month. No monthly fee."
DOC_B = "Green Fee-Free Account. Monthly fee applies after 3 withdrawals."
SUBS = {"silver_plus_account": ["doc_a"], "green_fee-free_account": ["doc_b"]}
PO = {"doc_index": {"g": SUBS},
      "verdict_prompt": "{req}\n{doc}",
      "verdict_line_template": "- {name}: {verdict}{why}",
      "verdict_max_candidates": 12,
      "requirement_prompt": "{messages}"}
A2 = {"policy_ontology": PO}
USER_TEXT = "I want an account with no monthly fee."


class _Msg(object):
    def __init__(self, role, content):
        self.role, self.content = role, content


class _Env(object):
    def __init__(self):
        self.tools = {"doc_a": DOC_A, "doc_b": DOC_B}


class _Orch(object):
    def __init__(self):
        self.environment = _Env()


class _Agent(object):
    def __init__(self):
        self._t2_orch = _Orch()


class _FakeSC(object):
    """서브콜 대역 — 태그로 갈라 정해진 답을 준다(모델 호출 0)."""
    def __init__(self, reqs, verdicts):
        self.reqs, self.verdicts, self.i = reqs, list(verdicts), 0

    def sub_generate(self, agent, la, UM, body, tag, temperature=None):
        if tag == "sub_requirement":
            return self.reqs
        a = self.verdicts[min(self.i, len(self.verdicts) - 1)]
        self.i += 1
        return a


def run(reqs, verdicts, val, spec=None):
    """헬퍼 1회 실행 — (반환 문면 | None)."""
    real = TS.SC
    TS.SC = _FakeSC(reqs, verdicts)
    try:
        return GP._verdict_gate_fb(_Agent(), [_Msg("user", USER_TEXT)], A2, "g", val,
                                   SUBS, spec or {})
    finally:
        TS.SC = real


# 후보는 **슬러그 정렬**로 돈다: green_fee-free_account → silver_plus_account.
REQ_OK = '["no monthly fee"]'
V_GREEN_BAD = "VIOLATES\nMonthly fee applies after 3 withdrawals."
V_SILVER_OK = "OK\nNo monthly fee."


def main():
    bad = 0

    fb = run(REQ_OK, [V_GREEN_BAD, V_SILVER_OK], "Green Fee-Free Account")
    print("① 양성 대조 → %s" % (repr(fb)[:220],))
    if not fb:
        print("   FAIL — 위반 판정인데 침묵했다(죽은 레버)"); bad += 1
    else:
        if "Monthly fee applies after 3 withdrawals." not in fb:
            print("   FAIL — LLM 판정 줄이 문면에 없다"); bad += 1
        if "Silver Plus Account" not in fb:
            print("   FAIL — 충돌하지 않는 대안이 없다([[64]] 위반)"); bad += 1
        if "Fee-free" in fb:
            # 한 문면에 'Green Fee-Free' 와 'Green Fee-free' 가 섞이면 우리 도구가 오표기를
            # 가르친다 — FIX-6 이 실제로 채점 칸을 죽인 그 결함이다([[25]]).
            print("   FAIL — 표기가 두 갈래로 섞였다(FIX-6 가족)"); bad += 1

    fb = run(REQ_OK, [V_GREEN_BAD, V_SILVER_OK], "Silver Plus Account")
    print("② OK 판정 → 침묵: %s" % (fb is None))
    if fb is not None:
        print("   FAIL — 엔진이 옳은 선택을 막았다(over-block)"); bad += 1

    fb = run(REQ_OK, ["VIOLATES\n(문서에 없는 문장)", V_SILVER_OK], "Green Fee-Free Account")
    print("③ 근거 미검산 → 침묵: %s" % (fb is None))
    if fb is not None:
        print("   FAIL — 검산 안 된 근거로 막았다([[25]] 위반)"); bad += 1

    fb = run('["a requirement the customer never said"]', [V_GREEN_BAD, V_SILVER_OK],
             "Green Fee-Free Account")
    print("④ 요구 인용 0건(원문 미실재) → 침묵: %s" % (fb is None))
    if fb is not None:
        print("   FAIL — 검산 안 된 요구로 판정했다"); bad += 1

    fb = run(REQ_OK, [V_GREEN_BAD, "VIOLATES\nUp to 15 free withdrawals per month."],
             "Green Fee-Free Account")
    print("⑤ 대안 0 → 침묵: %s" % (fb is None))
    if fb is not None:
        print("   FAIL — 무엇을 하면 풀리는지 못 대면서 막았다(C536ⓑ)"); bad += 1

    disp = GP._slug_disp("green_fee-free_account")
    print("⑥ 표기 왕복: %r" % disp)
    if disp != "Green Fee-Free Account":
        print("   FAIL — 표기 규약이 어긋났다(FIX-6 가족 회귀)"); bad += 1
    real = TS.SC
    TS.SC = _FakeSC(REQ_OK, [V_GREEN_BAD, V_SILVER_OK])
    try:
        _, st = TS.verdict_lines("a", "la", "UM", PO, "req", "g",
                                 corpus={"doc_a": DOC_A, "doc_b": DOC_B})
    finally:
        TS.SC = real
    if set(st.get("by_name") or {}) != set(SUBS):
        print("   FAIL — by_name 키가 슬러그가 아니다(표시명 조회는 조용히 빗나간다)"); bad += 1

    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    wired = ('T2_VERDICT_GATE"' in src and "_verdict_gate_fb(self," in src
             and "T2_VERDICT_GATE_CAP" in src and "_t2_vgate_deny" in src)
    print("⑦ 배선(플래그·호출·상한): %s" % wired)
    if not wired:
        print("   FAIL — 배선 없음"); bad += 1

    print("\n%s" % ("test_verdict_gate PASS" if not bad else "test_verdict_gate FAIL %d건" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
