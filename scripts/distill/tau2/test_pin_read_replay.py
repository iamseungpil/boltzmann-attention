"""Does the shipped predicate fire where the design said it would?

x74 measured a rule over the N97 transcripts and the design pre-registered its numbers
(36 firings, every one aimed at the gateway). Those numbers were produced by a *model* of
the rule written inside the instrument. This replays the same transcripts through the
real `t2_pin_read.pin_for` — the function the run will actually call — and requires the
two to agree.

Unit tests cannot catch this class of drift: they check the predicate against situations
the author imagined, while the pre-registration is a claim about a population. If the
shipped code fires 12 times where the design said 36, the design's cost/benefit numbers
belong to something that is not being shipped.

Free: reads persisted trajectories, calls no model.
"""

import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["T2_PIN_READ"] = "1"

import t2_pin_read as PR                                            # noqa: E402
from x50_says_not_does import ARMS, SIM                             # noqa: E402

EXPECTED_FIRES = 46            # 설계서 §1.7 `demand2h` 전수(계기 버그 수정 후)
EXPECTED_TARGET = "get_all_user_accounts_by_user_id_3847"
A2DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")


class Msg:
    """전사 dict를 pin_for가 기대하는 속성 접근 형태로 감싼다."""

    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.tool_calls = [Call(c) for c in (d.get("tool_calls") or [])]


class Call:
    def __init__(self, d):
        self.name = d.get("name") or (d.get("function") or {}).get("name")
        a = d.get("arguments")
        if a is None:
            a = (d.get("function") or {}).get("arguments")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                a = {}
        self.arguments = a if isinstance(a, dict) else {}


def load_a2():
    """정본 로더로 읽는다 — 층을 병합해야 `eplan.unlock_tool`이 채워진다([[24]]).

    specific.json만 직접 읽으면 `eplan`이 비어 있어 레버가 조용히 죽는다. 이 테스트가
    처음 잡은 것이 그것이다.
    """
    from gate_interpreter import load_domain_a2
    return load_domain_a2("banking_knowledge")


class Registry:
    """레지스트리 해소만 흉내낸다 — env가 없는 오프라인이므로 이름 목록을 직접 준다.

    실 런에서는 `t2_axis_levers.registry_from_env`가 같은 집합을 env에서 뽑는다.
    """

    NAMES = {"get_all_user_accounts_by_user_id_3847", "get_bank_account_transactions_9173",
             "get_credit_limit_increase_history_4829", "get_payment_history_6183",
             "check_card_application_fit", "get_credit_card_accounts_by_user"}


def main():
    import t2_callable_hint as CH
    CH.registry = lambda orch: Registry.NAMES

    a2 = load_a2()
    files = sorted(glob.glob(f"{SIM}/{ARMS['N97']}.results.json.gz"))
    if not files:
        sys.exit("N97 궤적 없음 — 경로 확인")

    fires, targets, per_trial = 0, {}, {0: 0, 1: 0}
    for p in files:
        for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
            orch = type("O", (), {})()
            msgs = [Msg(m) for m in (s.get("messages") or [])]
            hist = []
            for m in msgs:
                # 재생성은 **assistant 턴**에서만 일어난다. 역할을 안 보면 tool/user 메시지까지
                # 재생성 지점으로 세어 발화가 부풀려진다(이 테스트의 첫 오답 62 = 그 원인).
                if m.role == "assistant" and not m.tool_calls:
                    pin = PR.pin_for(orch, m, a2, hist)
                    if pin:
                        PR.mark_pinned(orch)
                        fires += 1
                        targets[pin[2]] = targets.get(pin[2], 0) + 1
                        per_trial[s.get("trial", 0)] = per_trial.get(s.get("trial", 0), 0) + 1
                hist.append(m)

    print(f"발화 {fires} sim · trial0 {per_trial.get(0)} · trial1 {per_trial.get(1)}")
    for t, n in sorted(targets.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<46} {n}")

    ok = True
    if abs(fires - EXPECTED_FIRES) > 3:
        print(f"FAIL 발화 {fires} ≠ 사전등록 {EXPECTED_FIRES}(±3) — 설계서 §1.7과 구현이 어긋난다")
        ok = False
    if set(targets) - {EXPECTED_TARGET}:
        print(f"FAIL 관문 외 표적: {set(targets) - {EXPECTED_TARGET}}")
        ok = False
    print("PASS — 구현이 사전등록과 일치" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
