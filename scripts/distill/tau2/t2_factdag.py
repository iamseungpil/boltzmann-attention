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

import hashlib
import sys

# ★파서는 **한 벌**만 둔다. 새로 쓰면 같은 술어가 두 벌이 되고 갈린다 — `t2_precedence` 첫 주석이
#   같은 병을 적어 두었다. 날짜 변환도 같은 이유로 빌려 쓴다(값 변환이지 도메인 파싱이 아니다).
from t2_ledger import parse_rows, parse_pairs, parse_scalar, _date

__all__ = ["FactDagError", "load", "validate", "evaluate", "Inputs", "OPS", "SHAPES",
           "Scheduler", "unmatched_groups"]


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
    """상한 − 누계. 상한이 **말해진 그룹만** 답한다 — 모르는 유형은 아무 말도 안 한다.

    ⚠**이름이 안 맞으면 조용히 틀린다**(2026-08-08·x135 limit 축에서 실물 발견): 상한의 키는
    **문서에서 모델이 읽은 이름**이고 누계의 키는 **DB 행이 말한 이름**이다. 둘이 다르면
    `usage.get(g, 0)`이 0으로 떨어져 *"자리가 다 남았다"* 는 **틀린 수**가 나온다 —
    실제로 오라클에 `World Blue International Checking`과 `World Blue`가 **따로** 잡혔다.
    엔진은 이름을 맞출 수 없고([[22]] 의미 판단은 LLM 몫) 맞춰서도 안 되지만,
    **안 맞았다는 사실은 반드시 표면화한다**(`unmatched_groups`) — 침묵이 오늘의 병이다.
    """
    return {g: int(lim[0] if isinstance(lim, (tuple, list)) else lim) - int((usage or {}).get(g, 0))
            for g, lim in (limits or {}).items()}


def unmatched_groups(usage, limits):
    """상한은 말해졌는데 **누계 쪽에 그 이름이 없는** 그룹 — 0으로 셌는지 이름이 어긋났는지 모른다."""
    return sorted(g for g in (limits or {}) if g not in (usage or {}))


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


def _formalize_pairs_per_item(node, items, hay, ask, memo, budget):
    """★`pairs`는 **항목마다 따로** 묻는다 (2026-08-08·x135·원장 C313).

    실측(threshold 축·5 sim·부정통제 변동 0): 한 덩어리로 물으면 재현율 **12~44%**, 항목별로
    물으면 **100%**. 기전은 둘이고 둘 다 실재한다 — 잘려서 못 본 것(4)과 **입력에 있는데도 못
    뽑은 것**(5). 항목별 질의는 둘 다 없앤다. 이것은 등대 §1.4의 부하 정의를 **우리 자신의
    서브콜**에 적용한 결과이고, 처방도 같다(분해).

    비용은 **항목 단위 메모이즈**로 유계다: 같은 내용은 한 번만 묻는다 ⇒ sim당 호출 ≈ **서로 다른
    항목 수**(재평가 횟수에도, 같은 문서의 재회수에도 곱해지지 않는다).

    ★식별자 = **내용 다이제스트**. 처음엔 위치+길이로 썼고 그 근거를 [[59]]로 댔는데 **틀렸다**:
    [[59]]가 금지하는 것은 엔진이 도메인 텍스트를 **패턴매칭으로 뜯어 내용을 알아내는 것**이고
    (`parse_records`류), 해시는 정규식도 어휘도 필드 추출도 없이 *같은 바이트인가*만 답한다 —
    이미 쓰고 있는 `quote not in hay`(부분문자열 멤버십)가 오히려 텍스트에 더 가깝게 닿는다.
    위치+길이가 실제로 나쁜 이유는 둘이다: ⓐ**같은 문서가 두 번 회수되면 두 번 묻는다**(궤적에
    실재 — 문턱 문장이 오프셋 5,901·9,147에 각각 두 번) ⓑ같은 인덱스의 내용이 바뀌면 **조용히
    옛 답을 쓴다**(덧붙기 가정이 깨지는 순간 침묵으로 틀린다 = 이 설계가 없애려는 그 형태).

    반환 `(값, 사유, 이번에 실제로 물은 횟수)`.
    """
    p = dict(node.get("params") or {})
    out, asked = {}, 0
    for t in items:
        s = str(t)[:budget]
        if not s.strip():
            continue
        key = (node["out"], hashlib.sha1(s.encode("utf-8")).hexdigest())
        if key not in memo:
            raw = ask(node, s)
            asked += 1
            memo[key] = parse_pairs(raw, p["field"], hay)[0] if raw is not None else {}
        out.update(memo[key])
    if out:
        return out, ("" if not asked else "항목 %d개 중 %d개 신규 질의" % (len(items), asked)), asked
    return None, "항목 %d개 전부에서 채택 0 (신규 질의 %d)" % (len(items), asked), asked


def evaluate(nodes, inputs, ask=None, excerpt_args=None, seed=None, memo=None):
    """위상순 1회 평가. 반환 `(값, trace)`.

    **예외를 삼키지 않는다**(§2b): 노드마다 잡고 `오류`로 남기고, 한 노드의 실패가 다른 노드의
    평가를 막지 않는다. 계산 못 한 노드는 **이유와 함께** 남는다 — 침묵이 보이게 하는 것이
    이 설계의 절반이다(§5).
    """
    ordered = _order(nodes)
    ex = dict(excerpt_args or {})
    # `seed` = 이미 서 있는 값(스케줄러가 다시 안 본 노드들). 없으면 1회 평가와 같다.
    vals, trace = dict(seed or {}), []
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
                if n.get("shape") == "pairs" and ask is not None:
                    # 항목별 질의 — 발췌 예산은 **항목 하나에** 걸린다(자를 이유가 없다).
                    v, why, _a = _formalize_pairs_per_item(
                        n, items, hay, ask, memo if memo is not None else {},
                        int(ex.get("budget", 90000)))
                else:
                    sel, dropped = excerpt(items, **ex)
                    v, why = _formalize(n, "\n---\n".join(sel), hay, ask)
                    why += (" · 예산 탈락 %d" % dropped if dropped else "")
                vals[out] = v
                trace.append((out, "계산" if v is not None else "미계산", why))
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
                _um = unmatched_groups(a, vals.get(ins[1]))
                if _um:      # 이름이 어긋난 것을 **말한다** — 조용히 0으로 세지 않는다
                    trace.append((out, "주의",
                                  "상한은 있는데 누계 쪽에 이름이 없다(0으로 셌다): %s"
                                  % ", ".join(_um)))
            else:
                v = _compare_ge(a, vals.get(ins[1]), p)
            vals[out] = v
            trace.append((out, "계산" if v is not None else "미계산",
                          "" if v is not None else "산수 불가(기준값 없음)"))
        except Exception as e:                        # 삼키지 않는다 — 오류로 남긴다(§2b)
            vals[out] = None
            trace.append((out, "오류", repr(e)))
    return vals, trace


class Scheduler(object):
    """단계 1 — **갱신된 입력이 있을 때만** 다시 평가한다(§1b). 값과 계기를 sim 수명으로 들고 있다.

    재평가 규칙(§1b·전부 실측에서 나온 것):
      · `corpus` 입력 노드 : **개수가 바뀌었고** ∧ **현재 값이 비어 있을 때만** 다시 묻는다
        (내용 해시가 아니다 — [[59]]) · **노드당 sim `cap`회 상한**([[09]])
      · `tool:` 입력 노드  : **새 반환이 오면 무효화하고 다시 평가**한다(하류까지).
        구판 규칙을 그대로 곱하면 *첫 반환이 이기고 재호출이 무시된다* — 제출 후 원장을 다시
        읽어도 하류가 옛 수를 말한다(재리뷰 B1).
      · 값이 **안 바뀐** 노드는 표면화하지 않는다(침묵) — 말이 느는 것이 과행동을 부른다(§6 위험 3).
    """

    def __init__(self, nodes, cap=3, excerpt_args=None):
        validate(nodes)
        self.nodes = list(nodes)
        self.cap = int(cap)
        self.ex = dict(excerpt_args or {})
        self.vals, self.asked, self.trace = {}, {}, []
        self.memo = {}          # (노드, 항목 내용 다이제스트) → 그 항목에서 뽑힌 쌍 (C313)
        self._corpus_n, self._tool_sig = None, {}

    def _stale(self, inputs):
        """이번 갱신으로 **다시 봐야 하는** 노드 이름 집합. 아무것도 안 바뀌면 빈 집합."""
        stale = set()
        if len(inputs.corpus) != self._corpus_n:
            self._corpus_n = len(inputs.corpus)
            for n in self.nodes:
                if "corpus" in (n.get("inputs") or ()) and self.vals.get(n["out"]) in (None, {}, []):
                    stale.add(n["out"])          # 값이 서 있으면 다시 묻지 않는다(§1b 절약)
        for n in self.nodes:
            for i in (n.get("inputs") or ()):
                if not str(i).startswith("tool:"):
                    continue
                fam = i[5:]
                cur = inputs.tools.get(fam)
                if cur is None:
                    continue
                sig = (len(cur), hash(cur))      # **교체 감지**(내용을 읽는 것이 아니라 동일성만)
                if self._tool_sig.get(fam) != sig:
                    self._tool_sig[fam] = sig
                    stale.add(n["out"])
        # 하류 전파 — 상류가 다시 계산되면 그 아래도 다시 계산돼야 한다
        changed = True
        while changed:
            changed = False
            for n in self.nodes:
                if n["out"] in stale:
                    continue
                if any(i in stale for i in (n.get("inputs") or ())):
                    stale.add(n["out"])
                    changed = True
        return stale

    def update(self, inputs, ask=None):
        """입력이 바뀐 만큼만 다시 평가하고 **값이 바뀐 노드**만 돌려준다 `{이름: 값}`."""
        stale = self._stale(inputs)
        if not stale:
            self.trace = []
            return {}
        budget_hit = set()
        for n in self.nodes:                     # 상한을 넘긴 LLM 노드는 이번 라운드에서 뺀다
            if n["op"] == "formalize" and self.asked.get(n["out"], 0) >= self.cap:
                budget_hit.add(n["out"])
        run = [n for n in self.nodes if n["out"] in stale and n["out"] not in budget_hit]
        for n in run:
            if n["op"] == "formalize":
                self.asked[n["out"]] = self.asked.get(n["out"], 0) + 1
        keep = {k: v for k, v in self.vals.items() if k not in stale}
        fresh, trace = evaluate(run, inputs, ask=ask, excerpt_args=self.ex, seed=keep,
                                memo=self.memo)
        for name in budget_hit & stale:
            trace.append((name, "미계산", "sim 상한 %d회 소진" % self.cap))
        changed = {k: v for k, v in fresh.items()
                   if v is not None and self.vals.get(k) != v}
        self.vals.update(fresh)
        self.trace = trace
        return changed

    def summary(self):
        """sim 종료 한 줄 요약 — 포렌식이 로그를 전수로 읽지 않게(§5·재리뷰 C)."""
        got = [k for k, v in sorted(self.vals.items()) if v is not None]
        miss = [k for k, v in sorted(self.vals.items()) if v is None]
        return ("[T2_DAG] sim 종료 · 계산 %d/%d · 미계산 %s · LLM 질의 %s"
                % (len(got), len(self.vals), ",".join(miss) or "(없음)",
                   ",".join("%s=%d" % kv for kv in sorted(self.asked.items())) or "(없음)"))


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
