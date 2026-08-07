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


CLAIM_PROMPT = (
    "Below is the last thing a customer-service agent said. List only the statements in it that "
    "assert a LIMIT, a MAXIMUM, an ALLOWANCE, a COUNT, or WHETHER SOMEONE OR SOMETHING QUALIFIES "
    "- anything the company's own policy documents or records decide, not things the customer "
    "alone would know, and not the agent's plans.\n"
    "For each one, give the claim and the document id it rests on, or \"\" if none was cited.\n"
    "Agent said:\n{text}\n"
    'Reply JSON only: {"claims": [{"claim": "<short>", "doc": "<doc id or empty>"}]}'
)

UNSOURCED_FB = (
    "Error: [SOURCE] you stated {n} thing(s) as fact that the policy documents decide, without "
    "having the document: {claims}. A limit or an allowance is not something the records show - "
    "the records show how many were used, the document says how many are allowed. Search the "
    "knowledge base for the document that states it, quote the figure it gives, and only then say "
    "whether the limit is reached. If the document does not support what you said, correct it."
)


def formalize_claims(agent, la, UserMessage, messages, cap_attr="_t2_source_calls", cap=1,
                     text=None):
    """에이전트의 **직전 발화**에서 정책이 정하는 수량 주장을 뽑는다 (LLM 몫·[[52]]).

    ⚠**호출 순증 0이 아니다.** `formalize_intent_tool`은 user 메시지만 보므로 얹을 수 없다 —
    별도 서브콜이고, 그 비용을 인정하는 대신 **sim당 `cap`회**로 묶는다.

    엔진은 뽑지 않고 **검증만** 한다. 반환 = [{claim, doc}].
    """
    if agent is None or la is None:
        return []
    if getattr(agent, cap_attr, 0) >= cap:
        return []
    # ★`text`가 우선이다. 검사 대상은 **지금 생성 중인 메시지**이고 그것은 아직 `messages`에 없다
    #   — 라이브 첫 발사에서 이 함수가 조용히 빈손이었던 이유가 정확히 그것이었다.
    last = text if isinstance(text, str) else ""
    if not last.strip():
        for m in reversed(messages or []):
            if getattr(m, "role", None) == "assistant":
                c = getattr(m, "content", None)
                if isinstance(c, str) and c.strip():
                    last = c
                    break
    if not last.strip():
        return []
    setattr(agent, cap_attr, getattr(agent, cap_attr, 0) + 1)
    try:
        p = CLAIM_PROMPT.replace("{text}", last[:2500])
        try:
            um = UserMessage(role="user", content=p)
        except TypeError:
            um = UserMessage(content=p)
        kw = {k: v for k, v in dict(getattr(agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um],
                          call_name="source_claim_formalize", **kw)
        txt = getattr(sub, "content", None) or ""
        out = []
        for m2 in re.finditer(r'\{[^{}]*"claim"\s*:\s*"([^"]{1,200})"[^{}]*\}', txt):
            blob = m2.group(0)
            doc = re.search(r'"doc"\s*:\s*"([^"]*)"', blob)
            out.append({"claim": m2.group(1).strip(), "doc": (doc.group(1).strip() if doc else "")})
        return out[:5]
    except Exception:
        return []


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _figures(text):
    return {m.group(0).replace(",", "") for m in _NUM.finditer(str(text or ""))}


def _anchors(text):
    """수치에 붙은 명사 — 같은 숫자가 다른 항목으로 우연히 통과하는 것을 막는 최소 단서."""
    words = re.findall(r"[A-Za-z][A-Za-z_\-]{3,}", str(text or ""))
    return {w.lower() for w in words}


def unsourced_claims(claims, corpus):
    """근거를 못 댄 주장만.

    ★구판은 *"모델이 doc id를 인용했는가"* 를 물었다. 라이브가 그것을 기각했다 — 모델은 doc id를
    쓰지 않는다. 그래서 KB에서 **실제로 회수해 요약한** 수치까지 전부 무근거로 셌다(claims=5 unsourced=5).
    모델의 협조가 있어야 성립하는 술어였다.

    이제 **우리가 이미 쥔 것으로 판정한다**: 그 주장의 수치가 회수된 문서 텍스트에 실재하는가.
    doc id를 댔고 그 문서가 회수됐으면 그것도 인정한다(더 강한 근거이므로).

    ⚠약점을 남겨 둔다: 숫자만 대조하면 같은 숫자가 **다른 항목**으로 코퍼스에 있어 거짓 통과할 수 있다.
    그래서 수치와 함께 **주장의 단어 하나 이상**이 같은 줄에 있을 것을 요구한다. 완전하지 않다 —
    한 줄 단위 근접성이지 의미 대조가 아니다.
    """
    ledger = corpus.get("ledger") or ""
    lines = [ln for ln in ledger.splitlines() if ln.strip()]
    bad = []
    for c in (claims or []):
        claim = (c.get("claim") or "").strip()
        if not claim:
            continue
        doc = (c.get("doc") or "").strip()
        if doc and has_source(doc, "document", corpus):
            continue                                  # 문서를 대고 그 문서가 회수됨 = 최강
        figs = _figures(claim)
        if not figs:
            # ★수치 없는 갈래(=자격 판정)를 잡으려던 시도는 **라이브에서 회귀를 냈다**(2026-08-07).
            #   102는 직전 구성에서 db_match 2/2였는데, 이 갈래를 켠 arm에서 0/2로 떨어졌고 제출도
            #   1·1 → 5·3으로 늘었다. 대조할 수치가 없어 **거의 모든 자격 문장이 무근거로 분류**되고,
            #   그만큼 합병 메시지가 길어져 직전에 얻은 이득을 밀어낸 것으로 보인다(귀속 미확정 —
            #   같은 arm의 다른 레버는 발화 0이라 후보에서 제외됨).
            #   ⇒ **대조할 수 없는 것은 판정하지 않는다.** 자격 판정을 닫으려면 수치 대조가 아니라
            #      별도 형식(모델이 요건 문장을 인용하게 하는 quote-back)이 필요하고, 그건 격리 arm에서.
            if os.environ.get("T2_SOURCE_QUALIFY") != "1":
                continue
            bad.append(claim)
            continue
        anch = _anchors(claim)
        # 공유 단어 **2개 이상**을 요구한다. 1개면 'bonuses' 같은 공용어 하나로 다른 항목 줄이
        # 통과해 버린다(검정이 잡았다) — 그러면 틀린 한도를 놓친다.
        need = min(2, len(anch)) if anch else 0
        hit = False
        for ln in lines:
            if not (figs & _figures(ln)):
                continue
            if len(anch & _anchors(ln)) >= need:
                hit = True
                break
        if not hit:
            bad.append(claim)
    return bad


def unsourced_text(a2, bad):
    tpl = str((((a2 or {}).get("arbitration") or {}).get("unsourced_claim_feedback"))
              or UNSOURCED_FB)
    return tpl.replace("{n}", str(len(bad))).replace(
        "{claims}", "; ".join('"%s"' % b for b in bad))


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

    # ── 주장-측: 인용한 doc이 이 대화에서 회수됐는가 ──────────────────────────
    # 회수 텍스트에 수치가 실재하면 doc id가 없어도 근거가 있다(라이브가 구판 술어를 기각했다).
    cpk = build_corpus(
        [_M("tool", "Sky Blue: you can earn up to 8 referral bonuses per calendar year\n"
                    "Gold Years: annual maximum 6 referrals")],
        env=_ENV(), a2=A2)
    claims = [
        {"claim": "Sky Blue allows up to 8 referral bonuses per year", "doc": ""},   # 코퍼스 실재
        {"claim": "Sky Blue allows up to 7 referral bonuses per year", "doc": ""},   # ★7은 없다
        {"claim": "Gold Years annual maximum 6 referrals", "doc": ""},               # 실재
        {"claim": "annual max is 8", "doc": "doc_never_retrieved"},                  # 문서 미회수·8은 실재
        {"claim": "both companies qualify for Sky Blue", "doc": ""},                 # ★자격·수치없음
        {"claim": "TechFlow qualifies for Sky Blue", "doc": "doc_business_x"},        # 문서 미회수
    ]
    bad = unsourced_claims(claims, cpk)
    # ★트레이드오프를 검정에 박아 둔다. 단어 근접을 요구하면 "annual max is 8"처럼 **패러프레이즈된
    #   정당한 주장도 잡힌다**(코퍼스는 "up to 8 ... per calendar year"라 공유 단어가 없다).
    #   그래도 요구를 유지하는 이유: 빼면 102 표적을 놓친다 — 틀린 숫자 7이 **다른 카드 줄**에 흔히
    #   있어서 그냥 통과해 버린다. 표적을 지키고 이 오탐을 비용으로 인정한다(§6b-c의 ⓒ 계수 대상).
    # 기본(플래그 OFF) = 수치 없는 자격 문장은 **판정하지 않는다**(라이브 회귀로 되돌린 갈래).
    assert bad == ["Sky Blue allows up to 7 referral bonuses per year",
                   "annual max is 8"], bad
    os.environ["T2_SOURCE_QUALIFY"] = "1"
    bad_q = unsourced_claims(claims, cpk)
    assert bad_q[-2:] == ["both companies qualify for Sky Blue",
                          "TechFlow qualifies for Sky Blue"], bad_q
    os.environ.pop("T2_SOURCE_QUALIFY", None)

    # 같은 숫자가 **다른 줄**에 있어도 단어가 안 맞으면 통과시키지 않는다.
    cpn = build_corpus([_M("tool", "Platinum card: up to 7 bonuses per calendar year")],
                       env=_ENV(), a2=A2)
    assert unsourced_claims([{"claim": "Sky Blue allows up to 7 referral bonuses", "doc": ""}],
                            cpn) == ["Sky Blue allows up to 7 referral bonuses"]

    txt = unsourced_text({}, bad)
    assert "up to 7" in txt and "2 thing" in txt

    # 서브콜 캡: sim당 1회
    class _AG:
        llm = None
        llm_args = {}
    ag = _AG()
    assert formalize_claims(ag, None, None, []) == []            # la 없음 = 안전 실패
    # ★`text` 우선 — 지금 생성 중인 메시지를 검사한다(라이브 첫 발사가 여기서 조용히 빈손이었다).
    seen = {}

    class _LA:
        def generate(self, **kw):
            seen["p"] = kw["messages"][0].content

            class _R:
                content = '{"claims": [{"claim": "limit reached", "doc": ""}]}'
            return _R()

    class _UM:
        def __init__(self, role=None, content=""):
            self.content = content
    ag2 = _AG()
    got = formalize_claims(ag2, _LA(), _UM, [_M("assistant", "OLD MESSAGE")],
                           text="NEW MESSAGE limit reached")
    assert "NEW MESSAGE" in seen["p"] and "OLD MESSAGE" not in seen["p"], seen["p"][:200]
    assert got == [{"claim": "limit reached", "doc": ""}], got

    ag._t2_source_calls = 1
    assert formalize_claims(ag, object(), object(), [_M("assistant", "x")]) == []   # 캡 소진
    print("t2_source self-check OK")
