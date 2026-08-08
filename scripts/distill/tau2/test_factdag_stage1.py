# -*- coding: utf-8 -*-
"""단계 1 게이트⒜ — **오늘 세 버그가 이 구조에서 재현에 실패하는가** (유료 0·모델 없이 돈다).

정본 = `FACT_DAG_DESIGN_2026_08_08.md` §7 단계 1. 세 버그(§0 표):

    1 오늘 날짜   대화 **첫머리**에만 있는데 꼬리 8개만 발췌해서 못 읽었다
    2 유형별 상한 문서가 **나중에** 도착하는데 원장 read 시점에 물었고, **빈손을 메모**해 끝났다
    3 관계기간 문턱 두 선언이 **한 슬롯**을 공유해 마지막 것이 덮었다(추출 0회)

각 검정은 **버그를 만드는 조건을 그대로 재현**하고, 노드 구조에서 값이 나오는 것을 본다.
`ask`는 고정 함수다 — 모델 없이 돌아야 회귀 검정이다.

실행: `python test_factdag_stage1.py`
"""

import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")

import t2_factdag as F   # noqa: E402

DF = ["%m/%d/%Y", "%Y-%m-%d"]

NODES = [
    {"out": "today", "inputs": ["corpus"], "op": "formalize", "shape": "scalar",
     "prompt": "now_prompt", "params": {"date_formats": DF}},
    {"out": "rows:ref", "inputs": ["tool:get_referrals"], "op": "formalize", "shape": "rows",
     "prompt": "formalize_prompt", "params": {"row_keys": ["date", "type"]}},
    {"out": "rows:acct", "inputs": ["tool:get_accounts"], "op": "formalize", "shape": "rows",
     "prompt": "formalize_prompt", "params": {"row_keys": ["level", "opened"]}},
    {"out": "usage", "inputs": ["rows:ref"], "op": "tally", "params": {"group_field": "type"}},
    {"out": "tenure", "inputs": ["rows:acct", "today"], "op": "days_since_earliest",
     "params": {"age_field": "opened", "date_formats": DF}},
    {"out": "limits", "inputs": ["corpus"], "op": "formalize", "shape": "pairs",
     "prompt": "limit_prompt", "params": {"field": "limit"}},
    {"out": "minimums", "inputs": ["corpus"], "op": "formalize", "shape": "pairs",
     "prompt": "threshold_prompt", "params": {"field": "min_days"}},
    {"out": "left", "inputs": ["usage", "limits"], "op": "subtract_by_group"},
    {"out": "ok", "inputs": ["tenure", "minimums"], "op": "compare_ge"},
]

HEAD = "The current time is 11/14/2025. How can I help?"
LIMIT_DOC = 'Annual limit: 3 referral bonuses per year for the Light Green Account.'
MIN_DOC = 'A minimum relationship duration of 60 days is required for Hunter Green.'
REF_BODY = "…원장 본문…"
ACCT_BODY = "…계좌 본문…"


def answer(node, text):
    """고정 응답 — **모델이 볼 수 있는 것만** 답한다(발췌에 안 들어오면 못 답한다).

    이것이 검정의 핵심이다: 발췌가 문장을 잘라내면 이 함수도 빈손을 돌려주므로,
    *"발췌 때문에 못 읽었다"* 는 버그가 그대로 재현된다.
    """
    p = node["prompt"]
    if p == "now_prompt":
        return "11/14/2025" if "current time is" in text else "모르겠다"
    if p == "formalize_prompt":
        if "원장" in text:
            return '[{"date":"11/10/2025","type":"Light Green"},{"date":"11/12/2025","type":"Light Green"}]'
        if "계좌" in text:
            return '[{"level":"Hunter Green","opened":"09/10/2025"}]'
        return "[]"
    if p == "limit_prompt":
        return ('{"Light Green": {"limit": 3, "quote": "%s"}}' % LIMIT_DOC) \
            if LIMIT_DOC in text else "{}"
    return ('{"Hunter Green": {"min_days": 60, "quote": "%s"}}' % MIN_DOC) \
        if MIN_DOC in text else "{}"


def _sched():
    return F.Scheduler(NODES)


def _old_head_tail(items, head=3, tail=8, per=1500):
    """구판 발췌(`formalize_now`·`t2_ledger.py:118`) — **판별력 확인용**으로만 둔다."""
    sel, seen = [], set()
    for t in list(items[:head]) + list(items[-tail:]):
        if id(t) not in seen:
            seen.add(id(t))
            sel.append(str(t)[:per])
    return sel


def test_bug1_date_in_the_middle():
    """날짜가 **머리도 꼬리도 아닌 자리**에 있어도 읽어야 한다.

    ⚠판별력: 인덱스 0에 두면 구판(head 3 + tail 8)도 통과하므로 회귀 검정이 못 된다.
    아침에 고친 것이 *"머리를 추가한 것"* 이었기 때문이다 — 그 수리는 날짜가 **가운데** 있으면
    다시 실패한다. 그래서 여기서는 인덱스 5/30에 두고, 구판 규칙이 실제로 그것을 잃는지
    같은 검정 안에서 확인한다(잃지 않으면 이 검정은 아무것도 증명하지 않는다).
    """
    corpus = ["대화 %d" % i for i in range(5)] + [HEAD] + ["대화 %d" % i for i in range(5, 30)]
    assert not any("current time is" in t for t in _old_head_tail(corpus)), \
        "구판이 이 배치를 통과한다 = 이 검정엔 판별력이 없다"

    s = _sched()
    s.update(F.Inputs(corpus=corpus, tools={}), ask=answer)
    assert s.vals["today"] == "11/14/2025", ("구판 발췌 형태가 재현됐다", s.vals["today"], s.trace)


def test_bug2_documents_arrive_later():
    """원장을 **먼저** 읽고 문서가 **나중에** 와도 상한이 채워져야 한다(빈손 메모 금지)."""
    s = _sched()
    early = F.Inputs(corpus=[HEAD], tools={"get_referrals": REF_BODY})
    s.update(early, ask=answer)
    assert s.vals["limits"] in (None, {}), "이 시점엔 문서가 없다"
    assert s.vals["usage"] == {"Light Green": 2}, s.vals["usage"]

    late = F.Inputs(corpus=[HEAD, LIMIT_DOC], tools={"get_referrals": REF_BODY})
    changed = s.update(late, ask=answer)
    assert s.vals["limits"] == {"Light Green": (3, LIMIT_DOC)}, (s.vals["limits"], s.trace)
    assert "limits" in changed and s.vals["left"] == {"Light Green": 1}, s.vals["left"]


def test_bug3_two_declarations_do_not_overwrite():
    """두 프롬프트가 **각자 노드**를 가지므로 하나가 다른 하나를 덮을 수 없다."""
    s = _sched()
    s.update(F.Inputs(corpus=[HEAD, LIMIT_DOC, MIN_DOC],
                      tools={"get_referrals": REF_BODY, "get_accounts": ACCT_BODY}), ask=answer)
    assert s.vals["limits"] == {"Light Green": (3, LIMIT_DOC)}, s.vals["limits"]
    assert s.vals["minimums"] == {"Hunter Green": (60, MIN_DOC)}, s.vals["minimums"]
    assert s.vals["tenure"]["days"] == 65, s.vals["tenure"]        # 09/10 → 11/14
    assert s.vals["ok"] == {"Hunter Green": True}, s.vals["ok"]


def test_reread_updates_downstream():
    """재리뷰 B1 — **새 반환이 오면** 하류가 옛 수를 말하면 안 된다."""
    s = _sched()
    base = {"get_referrals": REF_BODY}
    s.update(F.Inputs(corpus=[HEAD], tools=base), ask=answer)
    assert s.vals["usage"] == {"Light Green": 2}

    def answer2(node, text):
        if node["prompt"] == "formalize_prompt" and "원장" in text:
            return ('[{"date":"11/10/2025","type":"Light Green"},'
                    '{"date":"11/12/2025","type":"Light Green"},'
                    '{"date":"11/13/2025","type":"Sky Blue"}]')
        return answer(node, text)

    changed = s.update(F.Inputs(corpus=[HEAD], tools={"get_referrals": REF_BODY + " 갱신"}),
                       ask=answer2)
    assert s.vals["usage"] == {"Light Green": 2, "Sky Blue": 1}, (s.vals["usage"], s.trace)
    assert "usage" in changed


def test_pairs_asks_each_item_and_unions():
    """★C313 — `pairs`는 **항목마다 따로** 묻고 결과를 합친다(덩어리 질의 금지).

    두 문서가 각각 다른 상품을 말할 때, 한 덩어리로 물으면 실측 재현율이 12~44%였다.
    항목별로 물으면 둘 다 나와야 한다.
    """
    calls = []

    def counting(node, text):
        calls.append((node["out"], text[:20]))
        return answer(node, text)

    s = _sched()
    s.update(F.Inputs(corpus=[HEAD, LIMIT_DOC, MIN_DOC], tools={}), ask=counting)
    assert s.vals["limits"] == {"Light Green": (3, LIMIT_DOC)}, s.vals["limits"]
    assert s.vals["minimums"] == {"Hunter Green": (60, MIN_DOC)}, s.vals["minimums"]
    per_node = [c for c in calls if c[0] == "limits"]
    assert len(per_node) == 3, ("항목 3개를 각각 물어야 한다", per_node)


def test_pairs_do_not_reask_an_item_already_asked():
    """비용은 **항목 단위 메모이즈**로 유계다 — 회수된 항목은 불변이므로 다시 묻지 않는다."""
    calls = []

    def counting(node, text):
        calls.append(node["out"])
        return answer(node, text)

    s = _sched()
    s.update(F.Inputs(corpus=[HEAD, MIN_DOC], tools={}), ask=counting)
    first = calls.count("limits")
    s.update(F.Inputs(corpus=[HEAD, MIN_DOC, "새 문서 하나"], tools={}), ask=counting)
    added = calls.count("limits") - first
    assert added == 1, ("새로 들어온 항목 1개만 물어야 한다", added, first)


def test_same_document_retrieved_twice_is_asked_once():
    """★같은 문서를 두 번 회수하면 **한 번만** 묻는다.

    구판 식별자(위치+길이)는 인덱스가 다르면 다른 항목으로 봐서 **두 번 물었다** — 궤적에
    중복 회수가 실재한다(문턱 문장이 오프셋 5,901·9,147에 각각 두 번). 비용을 줄이려고 고른
    식별자가 비용을 늘리고 있었다. 식별자를 내용 다이제스트로 바꾼 이유가 이것이다.
    """
    calls = []

    def counting(node, text):
        calls.append(node["out"])
        return answer(node, text)

    s = _sched()
    s.update(F.Inputs(corpus=[HEAD, MIN_DOC, "사이 문서", MIN_DOC], tools={}), ask=counting)
    assert s.vals["minimums"] == {"Hunter Green": (60, MIN_DOC)}, s.vals["minimums"]
    assert calls.count("minimums") == 3, ("중복 문서를 두 번 물었다", calls.count("minimums"))


def test_silence_when_nothing_changed():
    """아무것도 안 바뀌면 **아무 말도 안 한다**(§6 위험 3 — 말이 늘면 과행동이 는다)."""
    s = _sched()
    inp = F.Inputs(corpus=[HEAD, LIMIT_DOC], tools={"get_referrals": REF_BODY})
    s.update(inp, ask=answer)
    again = s.update(F.Inputs(corpus=[HEAD, LIMIT_DOC], tools={"get_referrals": REF_BODY}),
                     ask=answer)
    assert again == {} and s.trace == [], (again, s.trace)


def test_ask_budget_is_capped():
    """모르는 것을 **무한히 다시 묻지 않는다**([[09]]) — 상한 뒤에는 이유가 남는다.

    ⚠표적은 **같은 내용을 다시 묻는 노드**(`today`=scalar)다. 항목별 노드(`pairs`)에는 이 상한을
    걸지 않는다 — 각 항목을 한 번씩만 묻는 것이 memo로 보장돼 이미 유계이고, 상한을 또 걸면
    늦게 도착한 문서를 영영 안 묻게 된다(아래 회귀가 그 케이스다).
    """
    def never(node, _t):
        return "" if node["prompt"] == "now_prompt" else answer(node, _t)

    s = F.Scheduler(NODES, cap=2)
    for i in range(5):
        s.update(F.Inputs(corpus=["잡담 %d" % j for j in range(i + 1)], tools={}), ask=never)
    assert s.asked["today"] == 2, s.asked
    assert any(w.startswith("sim 상한") for _o, _st, w in s.trace), s.trace


def test_a_document_arriving_after_a_value_is_still_asked():
    """★늦게 온 문서를 **영영 안 묻는** 버그의 회귀 (2026-08-08 실측·사용자 질문에서 발견).

    항목별 질의로 바꾼 뒤에도 재평가 조건이 *"값이 비었나"* 로 남아 있었다. 그러면 첫 문서에서
    상한이 하나 서는 순간 그 노드는 더 이상 stale이 아니고, **나중에 도착한 문서의 상한이 영영
    안 들어온다** — §0 버그 ②가 형태만 바꿔 되살아난 것이다. 올바른 조건은 **"아직 안 물어본
    항목이 있나"** 이고, 그 조건을 쓸 수 있게 해 주는 것이 **항목 동일성**이다(memo의 본래 목적).
    """
    A = "Account A allows up to 3 referral bonuses per year"
    B = "Account B allows up to 9 referral bonuses per year"

    def ask(node, text):
        if node["prompt"] != "limit_prompt":
            return answer(node, text)
        if A in text:
            return '{"A": {"limit": 3, "quote": "%s"}}' % A
        if B in text:
            return '{"B": {"limit": 9, "quote": "%s"}}' % B
        return "{}"

    s = _sched()
    s.update(F.Inputs(corpus=[A], tools={}), ask=ask)
    assert sorted(s.vals["limits"]) == ["A"], s.vals["limits"]
    s.update(F.Inputs(corpus=[A, B], tools={}), ask=ask)
    assert sorted(s.vals["limits"]) == ["A", "B"], ("늦게 온 문서를 안 물었다", s.vals["limits"])


def test_exception_is_not_swallowed():
    """§2b — 한 노드가 터져도 **다른 노드는 계속** 평가되고, 터진 것은 `오류`로 남는다."""
    def boom(node, _t):
        if node["out"] == "limits":
            raise RuntimeError("모델 호출 실패")
        return answer(node, _t)

    s = _sched()
    s.update(F.Inputs(corpus=[HEAD, LIMIT_DOC, MIN_DOC], tools={"get_accounts": ACCT_BODY}),
             ask=boom)
    assert any(st == "오류" for _o, st, _w in s.trace), s.trace
    assert s.vals["minimums"] == {"Hunter Green": (60, MIN_DOC)}, "한 노드 실패가 다른 노드를 삼켰다"
    assert s.vals["today"] == "11/14/2025"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as e:
            bad += 1
            print("  FAIL  %s\n        %s" % (name, e))
        except Exception as e:
            bad += 1
            print("  ERROR %s\n        %r" % (name, e))
    print("\n%s %d/%d" % ("[단계 1 게이트⒜ PASS]" if not bad else "[FAIL]",
                          len(tests) - bad, len(tests)))
    sys.exit(1 if bad else 0)
