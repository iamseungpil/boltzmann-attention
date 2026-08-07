# -*- coding: utf-8 -*-
"""t2_factdag — 파생 사실의 DAG. **언제 계산하는지를 코드가 아니라 선언이 정한다.**

정본 설계 = `reports/facet_rft_2026/FACT_DAG_DESIGN_2026_08_08.md` (rev4).
이 파일 = **단계 0**(현행 계산을 노드로 재서술·거동 동일). 스케줄러·계기·라이브 배선은 단계 1 이후.

왜 (§0): 하루에 세 번 같은 실패를 했고 셋 다 침묵이었다 — *"파생 사실을 그 입력이 갖춰지지 않은
지점에서 계산했다"*. 지금 구조에서는 *"이 계산을 어느 훅에 넣을까"* 를 사람이 매번 고르고, 그
선택이 암묵적이고 검증되지 않는다.

분담([[52]]·[[10]]·[[59]]): **엔진 = 스케줄링 + 닫힌 산수**. 도메인 텍스트를 읽지 않는다.
어느 문장이 상한/날짜인지 판정하는 것은 `formalize` 노드가 **모델에게** 묻고, 엔진은 형식과
인용 실재만 본다(파서는 `t2_ledger`에 한 벌만 둔다 — 두 벌이면 갈린다).

계약(§1) — 필드 일곱, 그것이 전부:
    out · inputs · op · params · prompt(LLM 노드) · shape(LLM 노드) · text
입력 3종:  corpus · tool:<name> · <노드 이름>          (`read:` 개념 없음)
금지(§6 위험 1): **제어 구성물 — 조건 분기 · 루프 · 우선순위 · 실행자.**

[[05]] 3질문은 설계서 §4에 답이 있다(요지: A2 재배치·판정은 모델 몫·도구를 부르지 않는다).
"""

import sys

# ★파서는 **한 벌**만 둔다. 새로 쓰면 같은 술어가 두 벌이 되고 갈린다 — `t2_precedence` 첫 주석이
#   같은 병을 적어 두었다. 날짜 변환도 같은 이유로 빌려 쓴다(값 변환이지 도메인 파싱이 아니다).
from t2_ledger import parse_rows, parse_pairs, parse_scalar, _date

__all__ = ["FactDagError", "load", "validate", "evaluate", "Inputs", "OPS", "SHAPES"]


class FactDagError(Exception):
    """선언이 계약을 어겼다 — **로드 시점에** 죽는다(§2a). 조용히 건너뛰는 것이 오늘의 실패 형태."""


# ── 닫힌 op 집합 (§2a 동결) ────────────────────────────────────────────────────
# 새 규칙형은 새 op가 아니라 `formalize`(LLM 노드)가 받는다. 여기에 op를 더하는 것은 설계 변경이다.
OPS = ("tally", "window_remaining", "days_since_earliest",
       "subtract_by_group", "compare_ge", "formalize")
SHAPES = ("rows", "pairs", "scalar")

_REQUIRED = {                       # op → 필수 params (로드 시점 검사)
    "tally": ("group_field",),
    "window_remaining": ("date_field", "date_formats", "window_days", "window_max"),
    "days_since_earliest": ("age_field", "date_formats"),
    "subtract_by_group": (),
    "compare_ge": (),
}
_REQUIRED_SHAPE = {"rows": ("row_keys",), "pairs": ("field",), "scalar": ("date_formats",)}
_ARITY = {                          # op → 입력 개수(제어 구성물을 안 넣기 위해 고정한다)
    "tally": 1, "window_remaining": 2, "days_since_earliest": 2,
    "subtract_by_group": 2, "compare_ge": 2, "formalize": 1,
}


def load(a2):
    """A2의 `derived` 선언 → 검증된 노드 목록. 선언이 없으면 `[]`(=이 도메인은 비활성)."""
    nodes = list(((a2 or {}).get("derived") or []))
    validate(nodes)
    return nodes


def validate(nodes):
    """계약 위반이면 **여기서** 죽는다 — 미지 op/shape · 필수 params 누락 · 미지 입력 · 순환."""
    names = [n.get("out") for n in nodes]
    for n in nodes:
        out, op = n.get("out"), n.get("op")
        if not out:
            raise FactDagError("노드에 out 이 없다: %r" % (n,))
        if names.count(out) > 1:
            raise FactDagError("%s: out 이 중복 선언됐다" % out)
        if op not in OPS:
            raise FactDagError("%s: 미지의 op %r (동결 집합=%s)" % (out, op, ",".join(OPS)))
        ins = list(n.get("inputs") or ())
        if len(ins) != _ARITY[op]:
            raise FactDagError("%s: op %s 는 입력 %d개인데 %d개를 받았다"
                               % (out, op, _ARITY[op], len(ins)))
        params = dict(n.get("params") or {})
        for k in _REQUIRED[op] if op != "formalize" else ():
            if k not in params:
                raise FactDagError("%s: op %s 에 필수 params %r 가 없다" % (out, op, k))
        if op == "formalize":
            shape = n.get("shape")
            if shape not in SHAPES:
                raise FactDagError("%s: formalize 에는 shape 이 필수다 (%s)" % (out, "|".join(SHAPES)))
            if not n.get("prompt"):
                raise FactDagError("%s: formalize 에는 prompt 이름이 필요하다" % out)
            for k in _REQUIRED_SHAPE[shape]:
                if k not in params:
                    raise FactDagError("%s: shape %s 에 필수 params %r 가 없다" % (out, shape, k))
        for i in ins:
            if i == "corpus" or str(i).startswith("tool:"):
                continue
            if i not in names:
                raise FactDagError("%s: 미지의 입력 %r (corpus · tool:<name> · 노드 이름만)" % (out, i))
    _order(nodes)            # 순환이면 여기서 던진다
    return True


def _order(nodes):
    """위상 정렬 — 순환은 계약 위반이다(제어 구성물 금지의 자연스러운 귀결)."""
    by = {n["out"]: n for n in nodes}
    seen, done, out = set(), set(), []

    def walk(name, stack):
        if name in done:
            return
        if name in stack:
            raise FactDagError("순환: %s" % " -> ".join(list(stack) + [name]))
        n = by.get(name)
        if n is None:
            return
        stack.append(name)
        for i in (n.get("inputs") or ()):
            if i in by:
                walk(i, stack)
        stack.pop()
        done.add(name)
        out.append(n)

    for n in nodes:
        walk(n["out"], [])
        seen.add(n["out"])
    return out


# ── 입력 ──────────────────────────────────────────────────────────────────────
class Inputs(object):
    """노드가 볼 수 있는 것 전부. **엔진은 여기 담긴 것 밖을 보지 않는다.**

    corpus : role ∈ {tool, user} 텍스트 목록(시간순)
    tools  : {가족 이름: 가장 최근 반환 본문}  ← "성공"을 묻지 않는다([[25]]·§1)
    """

    def __init__(self, corpus=(), tools=None):
        self.corpus = list(corpus or ())
        self.tools = dict(tools or {})


def excerpt(items, per=3000, budget=90000):
    """§1a **엔진의 단일 발췌 규칙** — 위치로만 고르고 내용은 보지 않는다([[59]]).

    최신부터 담고 총량 예산을 지키며, **항목이 둘 이상일 때만** 항목당 상한을 건다
    (하나뿐인데 자르면 원장이 조용히 잘린다). 반환 `(선택 텍스트, 탈락 항목 수)` —
    떨어진 것이 조용하면 발췌-범위 버그가 다시 조용해진다.
    """
    items = list(items or ())
    cap = per if len(items) > 1 else budget
    sel, used, dropped = [], 0, 0
    for t in reversed(items):
        s = str(t)[:cap]
        if used + len(s) > budget:
            dropped += 1
            continue                      # 잘려 나가는 쪽은 가장 오래된 것이어야 한다
        sel.append(s)
        used += len(s)
    sel.reverse()
    return sel, dropped


# ── 닫힌 산수 ─────────────────────────────────────────────────────────────────
def _tally(rows, p):
    gf = p["group_field"]
    out = {}
    for r in rows or ():
        g = r.get(gf)
        if g:
            out[g] = out.get(g, 0) + 1
    return out


def _window_remaining(rows, now, p):
    """(잔여, 창 안 건수). 기준일을 모르면 None = 미개입 — 현행과 같은 규약."""
    fmts = p["date_formats"]
    ref = _date(now, fmts) if now else None
    if ref is None:
        return None
    inwin = 0
    for r in rows or ():
        d = _date(r.get(p["date_field"]), fmts)
        if d is not None and 0 <= (ref - d).days <= int(p["window_days"]):
            inwin += 1
    return {"remaining": max(0, int(p["window_max"]) - inwin), "used": inwin}


def _days_since_earliest(rows, now, p):
    fmts = p["date_formats"]
    ref = _date(now, fmts) if now else None
    ds = [_date(r.get(p["age_field"]), fmts) for r in (rows or ())]
    ds = [d for d in ds if d is not None]
    if ref is None or not ds:
        return None
    first = min(ds)
    return {"since": first, "days": (ref - first).days}


def _subtract_by_group(usage, limits, _p):
    """상한 − 누계. 상한이 **말해진 그룹만** 답한다 — 모르는 유형은 아무 말도 안 한다."""
    return {g: int(lim[0] if isinstance(lim, (tuple, list)) else lim) - int((usage or {}).get(g, 0))
            for g, lim in (limits or {}).items()}


def _compare_ge(days, minimums, _p):
    if days is None:
        return None
    d = int(days["days"] if isinstance(days, dict) else days)
    return {g: d >= int(need[0] if isinstance(need, (tuple, list)) else need)
            for g, need in (minimums or {}).items()}


def _formalize(node, text, hay, ask):
    """유일한 LLM 노드. `ask(node, text)` 가 **모델 응답 문자열**을 돌려준다(엔진이 부르지 않는다).

    형태별 검증은 `t2_ledger`의 파서 한 벌이 한다(§2c). 엔진은 인용의 **뜻을 읽지 않는다**.
    """
    if ask is None:
        return None, "ask 없음(오프라인)"
    raw = ask(node, text)
    if raw is None:
        return None, "모델 응답 없음"
    shape, p = node.get("shape"), dict(node.get("params") or {})
    if shape == "rows":
        rows = parse_rows(raw, list(p["row_keys"]))
        return (rows or None), ("" if rows else "전사 0행")
    if shape == "pairs":
        got, rej, given = parse_pairs(raw, p["field"], hay)
        return (got or None), ("" if got else "모델 %d종 중 채택 0 (거절 %d)" % (given, rej))
    val = parse_scalar(raw, list(p["date_formats"]))
    return val, ("" if val else "형식 불일치")


def evaluate(nodes, inputs, ask=None, excerpt_args=None):
    """위상순 1회 평가. 반환 `(값, trace)`.

    **예외를 삼키지 않는다**(§2b): 노드마다 잡고 `오류`로 남기고, 한 노드의 실패가 다른 노드의
    평가를 막지 않는다. 계산 못 한 노드는 **이유와 함께** 남는다 — 침묵이 보이게 하는 것이
    이 설계의 절반이다(§5).
    """
    ordered = _order(nodes)
    ex = dict(excerpt_args or {})
    vals, trace = {}, []
    hay = " ".join("\n".join(str(t) for t in inputs.corpus).split())

    for n in ordered:
        out, op = n["out"], n["op"]
        ins = list(n.get("inputs") or ())
        p = dict(n.get("params") or {})
        try:
            missing = [i for i in ins
                       if (i.startswith("tool:") and i[5:] not in inputs.tools)
                       or (i in vals and vals[i] is None)
                       or (not i.startswith("tool:") and i != "corpus" and i not in vals)]
            if missing:
                trace.append((out, "미계산", "입력 없음: %s" % ", ".join(missing)))
                vals[out] = None
                continue
            if op == "formalize":
                src = ins[0]
                items = inputs.corpus if src == "corpus" else [inputs.tools[src[5:]]]
                sel, dropped = excerpt(items, **ex)
                v, why = _formalize(n, "\n---\n".join(sel), hay, ask)
                vals[out] = v
                trace.append((out, "계산" if v is not None else "미계산",
                              why + (" · 예산 탈락 %d" % dropped if dropped else "")))
                continue
            a = vals.get(ins[0]) if ins[0] in vals else None
            if op == "tally":
                v = _tally(a, p)
            elif op == "window_remaining":
                v = _window_remaining(a, vals.get(ins[1]), p)
            elif op == "days_since_earliest":
                v = _days_since_earliest(a, vals.get(ins[1]), p)
            elif op == "subtract_by_group":
                v = _subtract_by_group(a, vals.get(ins[1]), p)
            else:
                v = _compare_ge(a, vals.get(ins[1]), p)
            vals[out] = v
            trace.append((out, "계산" if v is not None else "미계산",
                          "" if v is not None else "산수 불가(기준값 없음)"))
        except Exception as e:                        # 삼키지 않는다 — 오류로 남긴다(§2b)
            vals[out] = None
            trace.append((out, "오류", repr(e)))
    return vals, trace


def format_trace(trace):
    """계기 한 줄씩 — sim 종료 요약도 이 형태를 쓴다(§5)."""
    return "\n".join("[T2_DAG] %-16s = %-6s %s" % (o, s, w) for o, s, w in trace)


if __name__ == "__main__":                            # 자기검정 — 산수와 계약만(모델 없이 돈다)
    N = [
        {"out": "today", "inputs": ["corpus"], "op": "formalize", "shape": "scalar",
         "prompt": "now_prompt", "params": {"date_formats": ["%m/%d/%Y"]}},
        {"out": "rows:x", "inputs": ["tool:T"], "op": "formalize", "shape": "rows",
         "prompt": "formalize_prompt", "params": {"row_keys": ["d", "g"]}},
        {"out": "usage", "inputs": ["rows:x"], "op": "tally", "params": {"group_field": "g"}},
        {"out": "win", "inputs": ["rows:x", "today"], "op": "window_remaining",
         "params": {"date_field": "d", "date_formats": ["%m/%d/%Y"],
                    "window_days": 9, "window_max": 2}},
    ]
    validate(N)

    # 계약 위반은 로드 시점에 죽는다
    for bad, why in (({"out": "a", "inputs": ["corpus"], "op": "nope"}, "미지 op"),
                     ({"out": "a", "inputs": ["corpus"], "op": "formalize",
                       "prompt": "p"}, "shape 누락"),
                     ({"out": "a", "inputs": ["corpus"], "op": "formalize", "shape": "scalar",
                       "prompt": "p"}, "scalar 인데 date_formats 없음"),
                     ({"out": "a", "inputs": ["b"], "op": "tally",
                       "params": {"group_field": "g"}}, "미지 입력")):
        try:
            validate([bad])
            raise AssertionError("통과하면 안 된다: %s" % why)
        except FactDagError:
            pass
    try:
        validate([{"out": "a", "inputs": ["b"], "op": "tally", "params": {"group_field": "g"}},
                  {"out": "b", "inputs": ["a"], "op": "tally", "params": {"group_field": "g"}}])
        raise AssertionError("순환이 통과하면 안 된다")
    except FactDagError:
        pass

    rows = [{"d": "11/%02d/2025" % (i + 1), "g": "G%d" % (i % 2)} for i in range(1, 6)]
    _ans = {"now_prompt": "11/14/2025",
            "formalize_prompt": '[%s]' % ",".join(
                '{"d":"%s","g":"%s"}' % (r["d"], r["g"]) for r in rows)}
    v, tr = evaluate(N, Inputs(corpus=["The current time is 11/14/2025"], tools={"T": "…"}),
                     ask=lambda n, t: _ans[n["prompt"]])
    assert v["today"] == "11/14/2025", v["today"]
    assert v["usage"] == {"G1": 3, "G0": 2}, v["usage"]
    assert v["win"] == {"remaining": 0, "used": 2}, v["win"]       # 경계(정확히 9일)는 포함

    # 입력이 없으면 하류는 **조용히 틀리지 않고** 미계산으로 남는다
    v2, tr2 = evaluate(N, Inputs(corpus=["오늘이 언제인지 아무도 말하지 않았다"], tools={}),
                       ask=lambda n, t: "" if n["prompt"] == "now_prompt" else "[]")
    assert v2["today"] is None and v2["usage"] is None and v2["win"] is None
    assert any(s == "미계산" for _o, s, _w in tr2)

    # 항목이 하나면 항목당 상한을 걸지 않는다(§1a) — 원장이 조용히 잘리지 않게
    assert len(excerpt(["x" * 50000])[0][0]) == 50000
    assert len(excerpt(["x" * 50000, "y" * 50000])[0][0]) == 3000

    print("t2_factdag self-test OK · 노드 %d · trace:\n%s" % (len(N), format_trace(tr)),
          file=sys.stderr)
