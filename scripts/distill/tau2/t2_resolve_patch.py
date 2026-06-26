#!/usr/bin/env python
"""resolve_selection live wiring — A2 grounding-spec 구동 (도메인-일반·A2_GROUNDING_WIRING_DESIGN).

오프라인 op-eval 폐기. 실 τ² user-sim e2e 안에서 모델이 emit하는 native `resolve_selection(op,...)`을
(1) 실제 도구로 노출(environment.get_tools)
(2) 실행을 가로채 결정론 엔진(ma/tau2_op_resolver.resolve_op_tau2)으로 실행 —
    catalog/anchor를 *A2 grounding-spec*에 따라 대화 trace(직전 fetch한 tool 출력)서 grounding.

★grounding은 더 이상 retail 하드코딩(_ground_retail 폐기)이 아니라 spec-driven 투영(관계대수 π/unnest/⋈/σ):
  candidate_source: producer 출력 컨테이너(map|list) → rows {item_id, options, available}
    - options_path(평면 dict copy) ∪ fields(project) ∪ explode(unnest: nested enum-keyed dict→행)
    - available = spec 술어({path}|{attr,pred,value}|true) — 엔진 기본값 금지(위험4)
  anchor_source(선택·substitute/comparative): 후보 producer-field ⋈ anchor item(공유 키)
엔진 코드 `grep "if domain|retail|airline|variants"` = 0. 도메인 차이 = spec 파일 차이뿐.

★ground 실패 라우팅(§5a·위험B): ground_OK=0 → 후보 컨테이너가 trace에 있나?
  없음(fetchable) → FETCH 신호(ask 아님·P2b) / 있음+resolve 실패 → ASK/refine(비유일·clarify).
계측(§7·위험2): 매 emission을 JSONL(T2_GROUND_LOG)에 기록 → P(ground_OK|emitted∧called)·분모율 분해.

활성화: t2_resolve_patch.apply(spec_path)  또는  T2_GROUNDING_SPEC=<path> 후 apply().
"""
import json
import os
import sys

_MA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ma")
sys.path.insert(0, _MA)
from tau2_op_resolver import resolve_op_tau2  # noqa: E402

_GSPEC = None  # apply()가 로드하는 도메인 grounding-spec(고정 PROVIDE)


def resolve_selection(op: str, attr: str = None, among: dict = None,
                      dir: str = None, k: int = None, set: dict = None) -> str:
    """Identify the candidate the user wants by NAMING the selection operation only.

    Works for any catalog of candidates (retail product variants, airline flights, ...). Do NOT
    invent any id. Name the operation and its operands (attribute, constraints) as references; the
    engine resolves the concrete id over the candidates you already fetched (e.g. via a product /
    flight search). For a modification, name only the changed attribute(s) in `set`; the engine
    keeps every other attribute of the current item.

    IMPORTANT — use ONLY attributes that DIFFER between the candidate items you fetched (their
    per-item option fields, e.g. color/size/cabin/price). Do NOT use the search/query parameters or
    category labels you used to FIND the candidates (such as the product type/category, or the
    origin/destination/date you searched) — those already located the candidate set and are NOT
    selection attributes. Pick the ONE item by its differentiating attributes.

    Args:
        op: one of filter, argmax, argmin, rank, comparative, substitute, create.
        attr: the ordinal/numeric attribute (for argmax/argmin/rank/comparative), e.g. price.
        among: categorical filter constraints as {attribute: value}, e.g. {"cabin": "economy"}.
        dir: comparative direction, "greater" or "less".
        k: rank position (1 = top).
        set: for substitute/create, the changed attribute values as {attribute: value}.

    Returns:
        The resolved concrete id.
    """
    return ""  # 실제 실행은 _execute_tool_calls 인터셉트(아래)·이 본문은 호출되지 않음.


# ───────────────────────────── helpers (spec-driven projection) ─────────────────────────────
def _args_dict(tc):
    a = getattr(tc, "arguments", None)
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return {}


def _callid_to_name(msgs):
    """assistant tool_call id → tool 이름. tool 출력에 producer명을 붙이기 위함(which-producer)."""
    m2 = {}
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            cid, nm = getattr(tc, "id", None), getattr(tc, "name", None)
            if cid and nm:
                m2[cid] = nm
    return m2


def _tool_outputs(orch):
    """[(parsed_json, producer_name)] — role==tool·비-error, 최근→과거 순. producer명은 id→name."""
    outs = []
    try:
        msgs = orch.get_messages()
    except Exception:
        return outs
    id2name = _callid_to_name(msgs)
    for m in reversed(msgs):
        if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        try:
            parsed = json.loads(c)
        except Exception:
            continue
        outs.append((parsed, id2name.get(getattr(m, "id", None))))
    return outs


def _tools_called(orch):
    """assistant 메시지가 발행한 tool_call 이름 집합(producer-called의 직접 신호)."""
    names = set()
    try:
        msgs = orch.get_messages()
    except Exception:
        return names
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            n = getattr(tc, "name", None)
            if n:
                names.add(n)
    return names


def _get_path(obj, path):
    """dotted-path getter. path ''/None → obj 자체."""
    if path is None or path == "":
        return obj
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur


def _container_elems(out, cs):
    """out에서 candidate 컨테이너를 (eid, elem) 리스트로. 없으면 None(이 out엔 후보 없음)."""
    cont = _get_path(out, cs.get("field", ""))
    container = cs.get("container", "list")
    if container == "map":
        if not isinstance(cont, dict):
            return None
        return [(eid, elem) for eid, elem in cont.items()]
    # list
    if not isinstance(cont, list):
        return None
    idk = cs.get("id_key")
    return [((elem.get(idk) if (isinstance(elem, dict) and idk) else None), elem) for elem in cont]


def _avail(opts, elem, cs):
    """가용성 = spec 술어 평가(엔진 기본값 금지·위험4)."""
    av = cs.get("available", True)
    if av is True or av is None:
        return True
    if isinstance(av, dict):
        if "path" in av:
            v = _get_path(elem, av["path"])
            return True if v is None else bool(v)
        if "attr" in av:
            v = opts.get(av["attr"])
            pred = av.get("pred", "truthy")
            val = av.get("value")
            if v is None:
                return False
            if pred == "gt":
                return v > val
            if pred == "ge":
                return v >= val
            if pred == "truthy":
                return bool(v)
    return True


def _project_rows(eid, elem, cs):
    """한 컨테이너 원소 → 0+ relational rows. options_path(평면)∪fields(project)∪explode(unnest)."""
    base = {}
    op = cs.get("options_path")
    if op:
        d = _get_path(elem, op)
        if isinstance(d, dict):
            base.update(d)
    for attr, p in (cs.get("fields") or {}).items():
        base[attr] = _get_path(elem, p)
    explode = cs.get("explode")
    rows = []
    if explode:
        froms = explode["from"]  # {attr: nested-dict-path}
        first_path = next(iter(froms.values()))
        keyset = _get_path(elem, first_path)
        if not isinstance(keyset, dict):
            return rows
        for key in keyset.keys():
            opts = dict(base)
            opts[explode["key_attr"]] = key
            for attr, dpath in froms.items():
                dd = _get_path(elem, dpath)
                opts[attr] = dd.get(key) if isinstance(dd, dict) else None
            rows.append((eid, opts, elem))
    else:
        rows.append((eid, base, elem))
    return rows


def _build_catalog(out, cs):
    """한 출력 → relational rows(없으면 None)."""
    elems = _container_elems(out, cs)
    if elems is None:
        return None
    cat = []
    for eid, elem in elems:
        for rid, opts, src in _project_rows(eid, elem, cs):
            cat.append({"item_id": rid, "options": opts, "available": _avail(opts, src, cs)})
    return cat


def _ground(outs, gspec, want_keys=None):
    """(catalog, anchor, producer_present) — A2 spec 구동 투영. retail/airline 무관·동일 코드.

    which-output 일반화(§7 NO-GO-a): 여러 후보 출력 중 (1)spec producer명 일치 우선
    (2)among/attr 키-커버리지 최대(want_keys ∩ optkeys) (3)최근 순으로 *올바른* 출력 선택.
    tool 출력에 producer명이 붙으면(id→name) 그걸로 좁히고, 없으면 구조-매칭으로 폴백.
    """
    cs = gspec["candidate_source"]
    producer = cs.get("producer")
    want = set(want_keys or [])
    # 후보 출력 = 컨테이너 추출 가능한 것 전부 (producer명 일치 우선)
    cands = []  # (catalog, cand_out, name, recency_idx)
    for idx, item in enumerate(outs):
        out, name = item if isinstance(item, tuple) else (item, None)
        cat = _build_catalog(out, cs)
        if cat is None:
            continue
        cands.append((cat, out, name, idx))
    catalog, cand_out, producer_present = None, None, False
    if cands:
        producer_present = True
        named = [c for c in cands if c[2] == producer]
        pool = named or cands  # producer명 일치분 우선·없으면 전체
        def _score(c):
            cat = c[0]
            optkeys = set()
            for r in cat:
                optkeys |= set((r.get("options") or {}).keys())
            cover = len(want & optkeys) if want else 0
            return (cover, -c[3])  # 키-커버리지↑, 그다음 최근(idx작음)
        best = max(pool, key=_score)
        catalog, cand_out = best[0], best[1]
    anchor = None
    asrc = gspec.get("anchor_source")
    if asrc and cand_out is not None:
        m = asrc.get("match", {})
        pid = _get_path(cand_out, m.get("producer_field"))
        if pid is not None:
            a_producer = asrc.get("producer")
            for item in outs:
                out, name = item if isinstance(item, tuple) else (item, None)
                if a_producer and name is not None and name != a_producer:
                    continue
                cont = _get_path(out, asrc.get("field", ""))
                if not isinstance(cont, list):
                    continue
                hit = next((it for it in cont
                            if isinstance(it, dict) and it.get(m.get("anchor_field")) == pid), None)
                if hit is not None:
                    anchor = hit.get(asrc.get("id_key"))
                    break
    return catalog, anchor, producer_present


def _log_event(orch, op_ir, producer_present, producer_called, n_cand, ground_ok, routed, anchor, err,
               dropped_nav=None):
    """§7 계측: 매 resolve_selection emission을 JSONL로(T2_GROUND_LOG). 미설정이면 무동작."""
    path = os.environ.get("T2_GROUND_LOG")
    if not path:
        return
    ev = {
        "task_id": getattr(getattr(orch, "task", None), "id", None),
        "domain": getattr(orch, "domain", None),
        "op": op_ir.get("op"),
        "producer_present": producer_present,      # 후보 컨테이너가 trace에 존재(구조 신호)
        "producer_called": producer_called,        # assistant가 producer 도구를 호출(직접 신호)
        "n_candidates": n_cand,
        "ground_OK": ground_ok,
        "routed": routed,                          # ok | fetch | ask_refine
        "anchor": anchor,
        "err": err,
        "dropped_nav": dropped_nav or [],          # T2_GROUND_DROP_NAVKEYS arm: 행-필터서 제외한 비스키마 키
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(ev) + "\n")
    except Exception:
        pass


def _resolve(orch, tc):
    from tau2.data_model.message import ToolMessage
    args = _args_dict(tc)
    op_ir = {k: v for k, v in args.items() if v is not None and k != "anchor_id"}
    outs = _tool_outputs(orch)
    want_keys = set((op_ir.get("among") or {}).keys())
    if isinstance(op_ir.get("attr"), str):
        want_keys.add(op_ir["attr"])
    catalog, anchor, producer_present = _ground(outs, _GSPEC, want_keys=want_keys)
    producer = _GSPEC["candidate_source"]["producer"]
    producer_called = producer in _tools_called(orch)
    # ablation(T2_GROUND_DROP_NAVKEYS=1): among의 비-스키마 키=네비게이션 힌트(which-output에 이미 사용·
    # 카탈로그가 그 entity로 scoped)로 보고 행-필터서 제외. 관계대수상 존재안하는 속성 제약=vacuous.
    dropped_nav = []
    if catalog and os.environ.get("T2_GROUND_DROP_NAVKEYS") == "1":
        optkeys = set()
        for r in catalog:
            optkeys |= set((r.get("options") or {}).keys())
        among = op_ir.get("among")
        if isinstance(among, dict):
            dropped_nav = [k for k in among if k not in optkeys]
            if dropped_nav:
                op_ir = dict(op_ir)
                op_ir["among"] = {k: v for k, v in among.items() if k in optkeys}
    rid, err = None, None
    if catalog:
        try:
            rid = resolve_op_tau2(op_ir, catalog, anchor_id=anchor)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    n_cand = len(catalog) if catalog else 0

    if rid:
        routed = "ok"
        content = json.dumps({"status": "ok", "tool": "resolve_selection", "item_id": rid})
        msg = ToolMessage(id=tc.id, role="tool", requestor="assistant", error=False, content=content)
    elif not producer_present:
        # (2) fetchable-but-not-fetched → FETCH (ask 아님·P2b). bare ask면 user-sim이 모름→거짓실패.
        routed = "fetch"
        content = (f"Error: no candidates in context yet. First call {producer} to fetch the "
                   f"candidates, then name the selection again. Do NOT guess an id.")
        msg = ToolMessage(id=tc.id, role="tool", requestor="assistant", error=True, content=content)
    else:
        # 후보는 있으나 resolve 실패: 비유일/anchor 불명/among 과다 → clarify/refine (ASK 갈래)
        routed = "ask_refine"
        content = ("Error: could not uniquely resolve the selection from the fetched candidates "
                   "(ambiguous, over-constrained, or missing the item to modify). Refine the "
                   "operation/constraints, or ask the user to clarify. Do NOT guess an id.")
        msg = ToolMessage(id=tc.id, role="tool", requestor="assistant", error=True, content=content)

    _log_event(orch, op_ir, producer_present, producer_called, n_cand, bool(rid), routed, anchor, err,
               dropped_nav=dropped_nav)
    return msg


def apply(spec_path=None):
    global _GSPEC
    spec_path = spec_path or os.environ.get("T2_GROUNDING_SPEC")
    if not spec_path:
        raise SystemExit("[t2_resolve_patch] grounding spec 미지정 — apply(spec_path) 또는 "
                         "T2_GROUNDING_SPEC 환경변수 필요 (a2/<domain>.grounding.json).")
    with open(spec_path, encoding="utf-8") as f:
        _GSPEC = json.load(f)

    from tau2.environment.environment import Environment
    from tau2.environment.tool import as_tool
    from tau2.orchestrator.orchestrator import BaseOrchestrator

    rtool = as_tool(resolve_selection)
    _orig_get_tools = Environment.get_tools

    def _get_tools(self):
        tools = list(_orig_get_tools(self))
        if not any(getattr(t, "name", None) == "resolve_selection" for t in tools):
            tools.append(rtool)
        return tools

    Environment.get_tools = _get_tools

    _orig_exec = BaseOrchestrator._execute_tool_calls

    def _exec(self, tool_calls):
        out = []
        for tc in tool_calls:
            if getattr(tc, "name", None) == "resolve_selection" \
                    and getattr(tc, "requestor", "assistant") == "assistant":
                out.append(_resolve(self, tc))
            else:
                out.extend(_orig_exec(self, [tc]))
        return out

    BaseOrchestrator._execute_tool_calls = _exec
    print(f"[t2_resolve_patch] resolve_selection wired · spec={spec_path} "
          f"(producer={_GSPEC['candidate_source']['producer']})")
    return _orig_exec


if __name__ == "__main__":
    apply(sys.argv[1] if len(sys.argv) > 1 else None)
