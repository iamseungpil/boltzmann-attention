# -*- coding: utf-8 -*-
"""R5 회귀 — `T2_ARG_EMPTY` 가 **디스패처 경유 write** 에도 발화하는가 (오프라인·모델 0·env 0).

결함(refute_4 claim 1 · CONFIRMED): 구판은 필수 인자 목록을 `agent.tools`(에이전트에게 **노출된**
목록)에서만 찾고, 조회 이름은 `_eff_tool_name` 이 `_\\d+$` 를 지운 철자였다. 발견형 도구는 노출
목록에 없고 레지스트리에는 **접미사째** 있으므로 두 겹 모두 빗나가 `req=[]` → 무발화였다.
세 번째 다리: 배치 페이로드(`{"disputes":[{…},{…}]}`)는 구판 unwrap 이 한 겹 못 들어갔다.

이 파일이 지키는 것:
  ⑴ **양성대조(결함 재현)** — 구판 술어를 축자 재현한 `_legacy_deny` 는 실물 모양에 무발화다.
  ⑵ **수리** — 같은 입력에서 현행 `_arg_empty_deny` 가 빈 필수 인자를 **이름으로** 짚는다.
  ⑶ **부정통제** — 채워짐·키 부재·`None`·`0`/`False`·미등록 도구·접미사 쌍둥이 모호성은 침묵.
  ⑷ **불변식** — 값 0(엔진이 값을 주지 않는다) · 도메인 리터럴 0 · 플래그 OFF 경로 보존.

실물 모양의 출처(축자 재현): `bank_t7335_halfB_20260821 | task_040` 의 호출 #18(배치) 과 #19~
(단건) — `call_discoverable_agent_tool(agent_tool_name="file_credit_card_transaction_dispute_4829",
arguments="<JSON 문자열>")` 이고 `address` 가 `""` 다. 도구 이름·인자 이름은 **환경 선언**
(`a2/env_surface.json` 레지스트리)에서 읽어 온 것이지 gold 가 아니다([[23]]).
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G

HERE = os.path.dirname(os.path.abspath(__file__))
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


# ── 환경 선언에서 이름을 읽는다(엔진에도 테스트에도 도메인 리터럴을 박지 않는다) ──
_SURF = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
_B = _SURF["banking_knowledge"]
_REG, _EXPOSED = _B["tools"], list(_B["exposed"])
# 표적 = **노출 목록에 없고 접미사를 단** 변이 도구 중 인자가 가장 많은 것(선택 규칙은 닫혀 있다)
_DISPATCHED = sorted(
    (n for n, v in _REG.items()
     if v.get("mutates") and re.search(r"_\d+$", n) and n not in _EXPOSED),
    key=lambda n: (-len(_REG[n].get("args") or []), n))
TOOL = _DISPATCHED[0]
ARGS = list(_REG[TOOL].get("args") or [])
SLOT = ARGS[-1] if len(ARGS) > 1 else ARGS[0]          # 비울 슬롯 하나(어느 것이든 술어는 같다)
CALLA = next(n for n in _EXPOSED if n.startswith("call_") and "agent" in n)
EXPOSED_WRITE = next(n for n in _EXPOSED
                     if (_REG.get(n) or {}).get("mutates") and (_REG[n].get("args") or []))


class _T(object):
    """Tool 객체 — `openai_schema` 하나로 이름·required 를 말한다(프레임워크 형태)."""

    def __init__(self, name, args, required=None):
        self.name = name
        self.openai_schema = {"function": {
            "name": name,
            "parameters": {"type": "object",
                           "properties": dict((a, {"type": "string"}) for a in args),
                           "required": list(args if required is None else required)}}}


class _Toolkit(object):
    """`get_tools()` = {이름: Tool} · `get_discoverable_tools()` = 그 부분집합 · `.tools` = {이름: 함수}."""

    def __init__(self, tools, disc=()):
        self._t = dict((t.name, t) for t in tools)
        self._d = set(disc)

    def get_tools(self):
        return dict(self._t)

    def get_discoverable_tools(self):
        return dict((k, v) for k, v in self._t.items() if k in self._d)


class _Env(object):
    def __init__(self, tools=None, user_tools=None):
        self.tools, self.user_tools = tools, user_tools


class _Orch(object):
    def __init__(self, env):
        self.environment = env


class _Agent(object):
    """에이전트에게 **노출된** 목록 + (있으면) 환경 레지스트리."""

    def __init__(self, exposed, env=None):
        self.tools = list(exposed)
        if env is not None:
            self._t2_orch = _Orch(env)


class _TC(object):
    def __init__(self, name, args):
        self.name, self.arguments = name, args


def _legacy_deny(agent, tc):
    """★양성대조 — **구판 술어의 축자 재현**(agent.tools 만 · `_eff_tool_name` · 평평한 unwrap)."""
    name = G._eff_tool_name(tc)
    cache = {}
    for t in (getattr(agent, "tools", None) or []):
        try:
            sc = t.openai_schema
            fn = sc.get("function") if isinstance(sc.get("function"), dict) else sc
            cache[fn.get("name")] = [str(x) for x in ((fn.get("parameters") or {}).get("required") or [])]
        except Exception:
            pass
    req = cache.get(name) or []
    if not req:
        return None
    args = G._args_dict(tc)
    inner = {}
    for vv in args.values():
        if isinstance(vv, str) and vv.strip().startswith("{"):
            try:
                j = json.loads(vv)
                if isinstance(j, dict):
                    inner.update(j)
            except Exception:
                pass
    for k2, v2 in args.items():
        if not isinstance(v2, (dict, list)):
            inner.setdefault(k2, v2)
    bad = [k for k in req if isinstance(inner.get(k), str) and not inner[k].strip()]
    if not bad:
        return None
    return G.ARG_EMPTY_FEEDBACK.replace("{tool}", str(name)).replace(
        "{args}", ", ".join("'%s'" % b for b in bad))


# ── 실행 모형: 노출 목록에는 디스패처만, 레지스트리에는 접미사 실명이 있다(실물 배치와 동형) ──
EXPOSED_TOOLS = [_T(CALLA, ["agent_tool_name", "arguments"], ["agent_tool_name"]),
                 _T(EXPOSED_WRITE, list(_REG[EXPOSED_WRITE]["args"]))]
REGISTRY = _Toolkit(EXPOSED_TOOLS + [_T(TOOL, ARGS)], disc=[TOOL])
AGENT = _Agent(EXPOSED_TOOLS, _Env(REGISTRY, None))
AGENT_NO_ENV = _Agent(EXPOSED_TOOLS)

FULL = dict((a, "v%d" % i) for i, a in enumerate(ARGS))
EMPTY = dict(FULL, **{SLOT: ""})


def _call(payload):
    return _TC(CALLA, {"agent_tool_name": TOOL, "arguments": json.dumps(payload)})


print("표적(환경 선언에서 도출): tool=%s · 빈 슬롯=%s · 디스패처=%s" % (TOOL, SLOT, CALLA))

print("\n── ⑴ 양성대조: 구판 술어는 이 모양에 **구조적으로** 무발화 ──")
chk("구판 = 단건 디스패처 발행에 무발화(결함 재현)", _legacy_deny(AGENT, _call(EMPTY)) is None)
chk("구판 = 배치 발행에도 무발화", _legacy_deny(AGENT, _call({"items": [dict(EMPTY)]})) is None)
chk("구판이 죽는 이유는 이름이다(접미사 제거 철자는 레지스트리에 없다)",
    G._eff_tool_name(_call(EMPTY)) != TOOL and G._exact_tool_name(_call(EMPTY)) == TOOL,
    (G._eff_tool_name(_call(EMPTY)), G._exact_tool_name(_call(EMPTY))))

print("\n── ⑵ 수리: 같은 입력에서 발화하고, 이름을 댄다 ──")
d = G._arg_empty_deny(AGENT, _call(EMPTY))
chk("단건 디스패처 발행에서 빈 필수 인자를 짚는다", d and ("'%s'" % SLOT) in d, d)
chk("호출 가능한 **실명**(접미사 포함)으로 도구를 부른다", d and TOOL in d, d)
chk("무엇을 하면 풀리는지 함께 말한다([[64]])",
    d and ("filled in" in d or "{args}" not in G.ARG_EMPTY_FEEDBACK), d)
chk("값을 주지 않는다(엔진 출력에 대화 값 0)",
    d and not any(v in d for v in FULL.values()), d)

db = G._arg_empty_deny(AGENT, _call({"items": [dict(FULL), dict(EMPTY)]}))
chk("배치 페이로드의 원소도 본다(t7335 #18 모양)", db and ("'%s'" % SLOT) in db, db)

dn = G._arg_empty_deny(AGENT, _TC(TOOL, dict(EMPTY)))
chk("접미사 실명 직접 호출도 본다", dn and ("'%s'" % SLOT) in dn, dn)

print("\n── ⑶ 부정통제: 아래는 전부 침묵이어야 한다 ──")
chk("채워져 있으면 침묵", G._arg_empty_deny(AGENT, _call(FULL)) is None)
chk("키 부재는 건드리지 않는다",
    G._arg_empty_deny(AGENT, _call(dict((k, v) for k, v in FULL.items() if k != SLOT))) is None)
chk("`None` 은 빈 문자열이 아니다(측정 후 기각한 축·false-block 회피)",
    G._arg_empty_deny(AGENT, _call(dict(FULL, **{SLOT: None}))) is None)
chk("`0`/`False` 도 건드리지 않는다",
    G._arg_empty_deny(AGENT, _call(dict(FULL, **{SLOT: 0}))) is None
    and G._arg_empty_deny(AGENT, _call(dict(FULL, **{SLOT: False}))) is None)
chk("빈 목록·빈 사전도 침묵(같은 축)",
    G._arg_empty_deny(AGENT, _call(dict(FULL, **{SLOT: []}))) is None
    and G._arg_empty_deny(AGENT, _call(dict(FULL, **{SLOT: {}}))) is None)
chk("레지스트리에 없는 도구는 침묵",
    G._arg_empty_deny(AGENT, _TC(CALLA, {"agent_tool_name": "no_such_tool_9999",
                                         "arguments": json.dumps({"x": ""})})) is None)
chk("required 선언이 비면 침묵",
    G._arg_empty_deny(_Agent([_T("f", ["a"], [])], _Env(_Toolkit([_T("f", ["a"], [])]))),
                      _TC("f", {"a": ""})) is None)
chk("환경을 못 잡으면 종전 거동(노출 목록만) — 발견형은 침묵",
    G._arg_empty_deny(AGENT_NO_ENV, _call(EMPTY)) is None)

_twin_a, _twin_b = _T("t_1111", ["a"]), _T("t_2222", ["a"])
_amb = _Agent([], _Env(_Toolkit([_twin_a, _twin_b])))
chk("접미사 쌍둥이가 있으면 bare 이름은 추정하지 않는다(C279 철자 규칙 금지)",
    G._arg_empty_deny(_amb, _TC("t", {"a": ""})) is None)
chk("쌍둥이라도 실명은 본다", (G._arg_empty_deny(_amb, _TC("t_1111", {"a": ""})) or "").find("'a'") > 0)

_uniq = _Agent([], _Env(_Toolkit([_T("u_9", ["a"])])))
chk("접히는 도구가 유일하면 bare 이름 폴백은 산다(구판 거동 보존)",
    (G._arg_empty_deny(_uniq, _TC("u", {"a": ""})) or "").find("'a'") > 0)

print("\n── ⑷ 불변식 ──")
chk("applies_to 밖이면 침묵", G._arg_empty_deny(AGENT, _call(EMPTY), None, {"other_tool"}) is None)
chk("applies_to 는 실명으로도 걸린다", G._arg_empty_deny(AGENT, _call(EMPTY), None, {TOOL}))
chk("문구는 A2 가 갈아 끼울 수 있다(슬롯 둘)",
    G._arg_empty_deny(AGENT, _call(EMPTY), {"arg_empty": {"feedback": "x {tool}/{args}"}})
    == "x %s/'%s'" % (TOOL, SLOT))


def _sig_fn(transaction_id, note=""):        # 기본값 없는 파라미터 = required (프레임워크 규약)
    return None


class _FnToolkit(object):
    def __init__(self, m):
        self.tools = dict(m)


chk("스키마가 없는 레지스트리 형태(`.tools`={이름: 함수})는 서명에서 required 를 읽는다",
    ("'transaction_id'" in (G._arg_empty_deny(
        _Agent([], _Env(_FnToolkit({"sig_tool": _sig_fn}))),
        _TC("sig_tool", {"transaction_id": "  ", "note": ""})) or "")))
chk("그 형태에서도 기본값 있는 인자는 required 가 아니다",
    "'note'" not in (G._arg_empty_deny(
        _Agent([], _Env(_FnToolkit({"sig_tool": _sig_fn}))),
        _TC("sig_tool", {"transaction_id": "  ", "note": ""})) or ""))

_src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
chk("호출 자리의 플래그 가드가 그대로다(OFF = 무발화)",
    "if not wd and ae_on and not _fab_only:" in _src)



def _code_only(path):
    """주석·문자열을 뺀 **실행 토큰**만 — 리터럴 감사는 코드에서 하고 주석은 세지 않는다."""
    import tokenize
    out = []
    with io.open(path, "rb") as f:
        for tok in tokenize.tokenize(f.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


_code = _code_only(os.path.join(HERE, "t2_gate_patch.py"))
chk("필수 목록의 출처는 여전히 환경뿐(**실행 코드**에 도구·인자 리터럴 0)",
    "_decl_tool_collections" in _src and TOOL not in _code and SLOT not in _code)

_ex, _st = G._schema_required_index(AGENT)
chk("색인 캐시는 환경 신원으로 무효화된다(sim 교차 오염 방지)",
    isinstance(getattr(AGENT, "_t2_schema_req", None), tuple)
    and AGENT._t2_schema_req[0] == id(AGENT._t2_orch.environment))
chk("색인이 노출 목록과 레지스트리를 **둘 다** 담는다",
    TOOL in _ex and EXPOSED_WRITE in _ex, sorted(_ex)[:4])

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
