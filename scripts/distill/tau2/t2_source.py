# -*- coding: utf-8 -*-
"""C1 — a factual claim that drives an action has to name where it came from.

This replaces seven keys that were all asking the same question about different nouns. A value
had to be in the records (`write_arg_grounding`), a row had to match the dump
(`transcription_check`), a name had to have been discovered (`discoverable_name_check`), an
asserted action had to be in the ledger (`claim_prov`), a transfer reason had to be in a document
(`transfer_reason_policy`), a document had to have been read (`require_doc_before`), a choice had
to be grounded (`choice_grounding`). Seven declarations, one sentence: **is it in the corpus.**

The corpus has four parts and they are not interchangeable:

  · **ledger**    — output of tool calls that returned without error
  · **documents** — what the knowledge base actually handed back in this conversation
  · **registry**  — the tools the environment really has, on either side
  · **ours**      — what our own scaffold said, **restricted by type**

That last restriction is the load-bearing one. Our scaffold is canonical, so what it says has a
source — but only where a *closed re-check* exists for that type. A name we stated can be checked
against the registry. A value we stated can be checked against a ledger row. A quantity, a
verdict or a summary cannot be re-checked against anything, so it is not a source, and the four
output defects already on record (a card's fx shown one way and judged another, an exclusion list
read as candidates, an abstention list read as an instruction list, a real tool reported missing)
would otherwise become quotable evidence. task_102 turns on a quantity claim, so without the type
limit the protection would be missing exactly where it is needed.

The same corpus read backwards is the second half of this contract. If C1 refuses claims from
outside the corpus, the mirror points at what is *inside* it and still unused — a tool a retrieved
document named that nobody called, or one unlocked and never invoked. Twelve of the twenty-three
gold tools nobody called were sitting in that set. The boundary travels with it and is not
negotiable: **registry ∩ text actually retrieved**. Widening it to "the search came back without a
tool name" was measured and discarded — 26% on failures against 32% on passes discriminates
nothing.

No domain vocabulary appears here. The corpus is built from message roles, the registry, and A2's
declared dispatcher argument names.
"""

import os
import re

__all__ = ["build_corpus", "has_source", "uncalled_in_corpus", "SOURCE_KINDS"]

# 유형 → 재검증 가능한가. 재검증이 없는 유형은 '우리 층'을 출처로 인정하지 않는다.
SOURCE_KINDS = {
    "name": "registry",     # 레지스트리 소속으로 재검증
    "value": "ledger",      # 원장 행으로 재검증
    "document": "docs",     # 회수 문서 id로 재검증
    "quantity": None,       # ★재검증 없음 — 우리 층 불인정
    "verdict": None,        # ★재검증 없음
    "summary": None,        # ★재검증 없음
}


def _name_args(a2):
    """디스패처의 '내부 도구 이름' 인자 키들(A2 선언·엔진 리터럴 0)."""
    drc = ((a2 or {}).get("dispatcher_role_check") or {})
    return sorted({v for v in (drc.get("name_args") or {}).values() if v})


def _registry(env, agent=None):
    out = {getattr(t, "name", None) for t in (getattr(agent, "tools", None) or [])}
    for holder in ("tools", "user_tools", "agent_tools"):
        tk = getattr(env, holder, None)
        if tk is None:
            continue
        try:
            out |= set(getattr(tk, "tools", {}) or {})
        except Exception:
            pass
        f = getattr(tk, "get_discoverable_tools", None)
        if callable(f):
            try:
                out |= set(f() or {})
            except Exception:
                pass
    return {n for n in out if n}


def build_corpus(messages, env=None, agent=None, a2=None):
    """이 대화의 출처집합. 네 부분을 **섞지 않고** 따로 돌려준다."""
    ledger, ours, docs = [], [], set()
    for m in (messages or []):
        role = getattr(m, "role", None)
        c = getattr(m, "content", None)
        c = c if isinstance(c, str) else ("" if c is None else str(c))
        if role != "tool":
            continue
        if getattr(m, "error", False):
            ours.append(c)          # 우리 deny·피드백이 나가는 채널
        else:
            ledger.append(c)
            docs |= set(re.findall(r"\bdoc_[A-Za-z0-9_()\-]+", c))
    reg = _registry(env, agent)
    # ★레지스트리 소속과 **모델이 볼 수 있었는가**는 다른 질문이다(자기검정이 잡은 결함).
    #   레지스트리는 프롬프트에 없다 — 에이전트 자기 도구 목록만 보인다. 그래서 이름의 출처는
    #   `본 적 있다 ∩ 실재한다` 이고, 레지스트리는 **출처가 아니라 재검증 필터**다.
    own = {getattr(t, "name", None) for t in (getattr(agent, "tools", None) or [])}
    _atk = getattr(env, "tools", None)          # env의 에이전트-측 툴킷 = 프롬프트에 실리는 집합
    try:
        own |= set(getattr(_atk, "tools", {}) or {})
    except Exception:
        pass
    return {
        "ledger": "\n".join(ledger),
        "ours": "\n".join(ours),
        "docs": docs,
        "registry": reg,
        "own": {n for n in own if n},      # 프롬프트에 실린 = 항상 가시
    }


def has_source(claim, kind, corpus, allow_ours=True):
    """`claim`이 `kind`의 출처집합 안에 있는가.

    ⚠**유형 제한**: 우리 층은 `SOURCE_KINDS[kind]`가 있는 유형에서만 출처로 쓰인다. 재검증이 없는
    유형(quantity·verdict·summary)에서는 우리 층 텍스트를 보지 않는다 — 우리가 한 번 틀리면 그것이
    스스로를 뒷받침하는 근거가 되기 때문이다.
    """
    s = str(claim or "").strip()
    if not s:
        return False
    recheck = SOURCE_KINDS.get(kind)
    if kind == "name":
        # 실재(레지스트리)는 **필요조건**이지 출처가 아니다. 출처는 '모델이 볼 수 있었는가'다.
        if s not in corpus.get("registry", ()):
            return False
        if s in corpus.get("own", ()):
            return True                       # 프롬프트 도구 목록 = 항상 가시
        if s in (corpus.get("ledger") or ""):
            return True                       # 회수 텍스트가 이름을 말함
        return bool(allow_ours and s in (corpus.get("ours") or ""))
    if kind == "document":
        if s in corpus.get("docs", ()):
            return True
    elif s in (corpus.get("ledger") or ""):
        return True
    if kind == "document" and not (allow_ours and recheck):
        return False
    if not (allow_ours and recheck):
        return False
    # 우리 층은 **재검증을 통과한 뒤에만** 출처가 된다.
    if s not in (corpus.get("ours") or ""):
        return False
    if recheck == "registry":
        return s in corpus.get("registry", ())
    if recheck == "ledger":
        return s in (corpus.get("ledger") or "")
    if recheck == "docs":
        return s in corpus.get("docs", ())
    return False


def uncalled_in_corpus(messages, corpus, a2=None, called=None):
    """거울상 — **출처집합 안에 있는데 아직 안 부른** 도구 이름.

    경계 = `레지스트리 ∩ 회수 텍스트`. 새 정보 0·추측 0. 이 교집합 밖은 말하지 않는다(기각된
    일반 넛지가 부활하지 않도록).
    """
    reg = corpus.get("registry") or set()
    seen_text = corpus.get("ledger") or ""
    named = {n for n in reg if n and n in seen_text}
    if called is None:
        called = set()
        keys = _name_args(a2)
        for m in (messages or []):
            for tc in (getattr(m, "tool_calls", None) or []):
                n = getattr(tc, "name", None)
                if n:
                    called.add(n)
                ar = getattr(tc, "arguments", None)
                if isinstance(ar, dict):
                    for k in keys:
                        if ar.get(k):
                            called.add(str(ar[k]))
    return sorted(named - called)


if __name__ == "__main__":                                   # 자기검정 (오프라인)
    class _C:
        def __init__(self, name, args=None):
            self.name = name
            self.arguments = args or {}

    class _M:
        def __init__(self, role, content="", error=False, calls=None):
            self.role, self.content, self.error = role, content, error
            self.tool_calls = [_C(*c) if isinstance(c, tuple) else _C(c) for c in (calls or [])]

    class _TK:
        def __init__(self, names, disc=()):
            self.tools = {n: None for n in names}
            self._d = {n: None for n in disc}

        def get_discoverable_tools(self):
            return self._d

    class _ENV:
        tools = _TK(["get_referrals_by_user", "log_verification"])
        user_tools = _TK(["submit_referral"], disc=["get_referral_link"])

    A2 = {"dispatcher_role_check": {"name_args": {"give_discoverable_user_tool": "discoverable_tool_name"}}}
    msgs = [
        _M("tool", "Found 32 record(s)\n  referred_account_type: Sky Blue Account\n  doc_business_x"),
        _M("tool", "Error: [ORDER] call get_referral_link and the count is 8", error=True),
        _M("assistant", "", calls=["get_referrals_by_user"]),
    ]
    cp = build_corpus(msgs, env=_ENV(), a2=A2)
    assert "Found 32" in cp["ledger"] and "[ORDER]" in cp["ours"]
    assert "doc_business_x" in cp["docs"]
    assert "submit_referral" in cp["registry"] and "get_referral_link" in cp["registry"]

    assert has_source("Sky Blue Account", "value", cp)          # 원장에 실재
    assert not has_source("Hunter Green Account", "value", cp)  # 어디에도 없음
    assert has_source("get_referrals_by_user", "name", cp)      # 프롬프트 도구 목록 = 가시
    assert has_source("doc_business_x", "document", cp)

    # ★실재는 필요조건일 뿐 출처가 아니다 — 레지스트리에만 있고 본 적 없는 이름은 불인정.
    assert not has_source("submit_referral", "name", cp)        # 레지스트리 O · 가시 X

    # ★유형 제한: 우리 층이 말한 것이라도 재검증이 없는 유형은 출처가 아니다.
    assert has_source("get_referral_link", "name", cp)          # 우리 층 ∩ 레지스트리 = 통과
    assert not has_source("8", "quantity", cp)                  # 우리 층에만 있음 = 불인정
    assert not has_source("8", "verdict", cp)
    assert not has_source("get_referral_link", "name", cp, allow_ours=False)
    assert not has_source("no_such_tool_9999", "name", cp)      # 실재 X = 항상 불인정

    # 거울상: 원장 텍스트가 이름을 말했는데 안 부른 것만.
    msgs2 = msgs + [_M("tool", "see get_referral_link and submit_referral for next steps")]
    cp2 = build_corpus(msgs2, env=_ENV(), a2=A2)
    un = uncalled_in_corpus(msgs2, cp2, A2)
    assert "get_referral_link" in un and "submit_referral" in un
    assert "get_referrals_by_user" not in un                    # 이미 호출됨
    assert "log_verification" not in un                         # 회수 텍스트에 없음(레지스트리에만)
    print("t2_source self-check OK")
