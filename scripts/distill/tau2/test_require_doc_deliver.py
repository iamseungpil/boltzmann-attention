# -*- coding: utf-8 -*-
r"""검정: **033형 정의 문서 전달** (`T2_REQUIRE_DOC_DELIVER`·2026-08-22·정본
`T7336_FORENSIC_033_2026_08_22.md`·격리 C592 `x465_transfer_doc_iso.py`).

[[71]] 계약 4문 답([[17]] 동형·파일 머리 의무):
  ①기능 하나 — 서브 없음. 배달을 받은 **메인 턴 하나**가 다음 행동을 고른다(x465 B 팔 인터페이스).
  ②재료는 선언에서 — 도구 집합은 A3 `require_doc_before.tools`(검정 ⑩은 **실제 A3 정본**을 읽어
    엔진 소스에 그 이름이 없음을 확인), 문서 집합은 정본 `_docs_naming`(코퍼스 도출). 이 검정의
    픽스처는 중립 어휘(zeta_*·doc_test_*)라 도메인 리터럴 0.
  ③전달 = 선언된 id 정확 집기 — 검정 ①이 코퍼스 축자 본문 전부가 실렸는지 본다(검색 0).
  ④엔진 해석 0 — 도출 집합 전부·순위 0·지목 문장 0(검정 ① "선택 없음").

무엇을 막는 검정인가 —
 ① 트리거→배달: 선언 도구 시도 ∧ 정의 문서 미열람 → 헤더 첫 줄 = x465 `DELIVER_HEAD` 축자 ·
    지시가 재료 **앞**(C578) · 도출 문서 **전부** 축자 · 미열람 id 를 이름으로 댐([[64]]) · 차단 없음
    문구 · `_isolate_trace` 기록 · 부기(fired/turn)
 ② OFF = 무발화·무로그·무부기(바이트 동일)
 ③ 이미 읽었으면 무발화(술어 불성립)
 ④ 정의 문서 도출 0 → 무발화 + 폴백 로그("도출 0편")
 ⑤ 같은 턴 재배달 생략 · 다음 턴 재배달 · sim당 CAP
 ⑥ 생성 창 초과 → 건너뜀 + 로그(축약·선별 0)
 ⑦ 미선언 도구 → 무발화·무로그 / 디스패치 호출(내부 이름)도 선언 멤버십으로 본다
 ⑧ `_docs_naming` json 디렉터리 경로 불변 + 코퍼스 폴백 · `_ctx_fits` 산식
 ⑨ 배선(소스): `unified` 호출부·`rdd_fb` 가드 변수·표면화 조건의 `rdd_fb is None`·부착 마크
 ⑩ 엔진 소스에 A3 선언 도구명 리터럴 0 · go_stack 등재(REQUIRE_DOC_DELIVER·SEARCH_REARM 둘 다 ON)
⚠️단위통과≠라이브발화([[30]]) — 배선만 본다. 실제 선언×실제 코퍼스×모델은 x465 가 쟀다.
실행: py -3 test_require_doc_deliver.py
"""
import ast
import contextlib
import inspect
import io
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── tau2 하네스 스텁 (오프라인 전용·test_search_rearm/test_sg_docs_delivery 관용구) ──────────
if "tau2" not in sys.modules:
    _m = types.ModuleType("tau2.data_model.message")

    class UserMessage(object):
        def __init__(self, role="user", content=""):
            self.role, self.content = role, content

    class ToolMessage(object):
        def __init__(self, id=None, role="tool", requestor="assistant", content="", error=False):
            self.id, self.role, self.requestor, self.content, self.error = id, role, requestor, content, error

    _m.UserMessage, _m.ToolMessage = UserMessage, ToolMessage
    _m.MultiToolMessage = type("MultiToolMessage", (), {})
    sys.modules["tau2"] = types.ModuleType("tau2")
    sys.modules["tau2.data_model"] = types.ModuleType("tau2.data_model")
    sys.modules["tau2.data_model.message"] = _m
    sys.modules["tau2.agent"] = types.ModuleType("tau2.agent")
    sys.modules["tau2.agent.llm_agent"] = types.ModuleType("tau2.agent.llm_agent")

import t2_gate_patch as GP   # noqa: E402

FAILED = []


def chk(c, label, detail=""):
    print(("  OK   " if c else "  FAIL ") + label + ((" — " + detail) if (detail and not c) else ""))
    if not c:
        FAILED.append(label)


# ── 픽스처 (중립 어휘) ────────────────────────────────────────────────────────────────
T_GEN, T_PRI, T_ORPHAN = "zeta_handoff_tool", "zeta_priority_handoff_tool", "zeta_orphan_tool"
A2 = {"require_doc_before": {"tools": [T_GEN, T_PRI, T_ORPHAN], "feedback": "[PROTOCOL] {tool}"}}
CORPUS = {
    "doc_test_alpha_001": "Alpha policy. Use zeta_handoff_tool only after the checklist is done.",
    "doc_test_alpha_002": ("Alpha incident. First call zeta_priority_handoff_tool, then "
                           "zeta_handoff_tool. Do not skip steps."),
    "doc_test_beta_001": "Beta: unrelated product terms. Nothing about handoffs here.",
}


class _Pipe(object):
    def __init__(self, corpus):
        self.state = {"doc_content_map": dict(corpus)}


class _Tools(object):
    def __init__(self, corpus):
        self._kb_pipeline = _Pipe(corpus)


class _Env(object):
    def __init__(self, corpus):
        self.tools = _Tools(corpus)


def _agent(corpus=CORPUS):
    return types.SimpleNamespace(_t2_orch=types.SimpleNamespace(environment=_Env(corpus)))


def _msg(role, content):
    return types.SimpleNamespace(role=role, content=content, tool_calls=None, id=None)


def _tc(name, args=None):
    return types.SimpleNamespace(name=name, arguments=json.dumps(args or {}), id="tc1")


BASE = [_msg("user", "I need to hand this off."),
        _msg("assistant", "Let me look at the knowledge base."),
        _msg("tool", "grep: ... proceed with the handoff protocol when symptoms match ...")]


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run(agent, msgs, calls, a2=A2):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        out = GP._require_doc_deliver(agent, a2, msgs, calls)
    return out, err.getvalue()


def _x465_head():
    """x465 의 `DELIVER_HEAD` 문자열을 **소스에서** 읽는다(모듈 import 0 — 하네스 의존 회피)."""
    src = io.open(os.path.join(HERE, "x465_transfer_doc_iso.py"), encoding="utf-8").read()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "DELIVER_HEAD" for t in n.targets):
            v = n.value
            return v.value if isinstance(v, ast.Constant) else None
    return None


TRACE = os.path.join(tempfile.mkdtemp(prefix="rdd_"), "trace.jsonl")
COMMON = dict(T2_REQUIRE_DOC_DELIVER="1", T2_KB_DOCS_DIR=None, T2_SG_ISOLATE_TRACE=TRACE,
              T2_REQUIRE_DOC_DELIVER_CAP=None, T2_REQUIRE_DOC_DELIVER_MAX=None)

print("\n§1 트리거 → 배달 (x465 B 팔 동형·[[64]]·전부·축자·지시-앞)")
with _env(**COMMON):
    ag = _agent()
    out, err = _run(ag, BASE, [_tc(T_GEN)])
chk(out is not None, "선언 도구 시도 ∧ 미열람 → 배달")
if out:
    want = sorted(GP._docs_naming(T_GEN, None, corpus=CORPUS))
    chk(out["ids"] == want == ["doc_test_alpha_001", "doc_test_alpha_002"],
        "도출 집합 **전부**·선택 0 (beta 제외는 술어지 선택이 아니다)", repr(out["ids"]))
    head = _x465_head()
    chk(bool(head) and out["text"].split("\n")[0] == (head % T_GEN),
        "헤더 첫 줄 = x465 `DELIVER_HEAD` 축자", repr(out["text"].split("\n")[0])[:120])
    chk(out["text"].index("[KB DELIVERY]") < out["text"].index("### doc_test_alpha_001"),
        "지시가 재료보다 **앞**(C578)")
    chk(all(CORPUS[i] in out["text"] for i in out["ids"]), "문서 본문 전부 축자(전문·절단 0)")
    chk(CORPUS["doc_test_beta_001"] not in out["text"], "도출 밖 문서는 싣지 않는다")
    why = out["text"].split("\n")[1]
    chk(all(i in why for i in out["ids"]) and T_GEN in why, "[[64]] 무엇이 틀렸나 — 미열람 id·도구 이름")
    chk("Nothing is blocked" in why, "[[64]] 무엇을 하면 풀리나 — 차단 없음·읽고 고르기")
    chk(getattr(ag, "_t2_rdd_fired", 0) == 1 and getattr(ag, "_t2_rdd_turn", None) == len(BASE),
        "부기: fired=1·turn=len(messages)")
    chk("[T2_REQUIRE_DOC_DELIVER] deliver tool=%s docs=2" % T_GEN in err, "발화 마크(조립)")
    rows = [json.loads(l) for l in io.open(TRACE, encoding="utf-8") if l.strip()]
    chk(len(rows) == 1 and rows[0].get("mode") == "require_doc_deliver"
        and rows[0].get("ids") == out["ids"] and rows[0].get("tool") == T_GEN,
        "_isolate_trace 기록(mode·ids·tool)", repr(rows)[:200])
    chk(not out["truncated"] and not out["missing"], "절단 0·누락 0")

print("\n§2 OFF = 바이트 동일 (무발화·무로그·무부기)")
with _env(**dict(COMMON, T2_REQUIRE_DOC_DELIVER=None)):
    ag = _agent()
    out, err = _run(ag, BASE, [_tc(T_GEN)])
chk(out is None and err == "" and not hasattr(ag, "_t2_rdd_fired"), "OFF → None·stderr ''·속성 없음")

print("\n§3 이미 읽었으면 무발화 (술어 불성립)")
with _env(**COMMON):
    ag = _agent()
    read = BASE + [_msg("tool", "### doc_test_alpha_001\n" + CORPUS["doc_test_alpha_001"])]
    out, err = _run(ag, read, [_tc(T_GEN)])
chk(out is None and "deliver" not in err, "정의 문서 하나라도 도구 출력에 등장 → None")

print("\n§4 정의 문서 도출 0 → 무발화 + 폴백 로그")
with _env(**COMMON):
    ag = _agent()
    out, err = _run(ag, BASE, [_tc(T_ORPHAN)])
chk(out is None and "도출 0편" in err and T_ORPHAN in err, "코퍼스에 도구명 0편 → None·'도출 0편' 로그")

print("\n§5 반복 규율 — 같은 턴 생략 · 다음 턴 재배달 · sim당 CAP")
with _env(**dict(COMMON, T2_REQUIRE_DOC_DELIVER_CAP="2")):
    ag = _agent()
    o1, e1 = _run(ag, BASE, [_tc(T_GEN)])
    o2, e2 = _run(ag, BASE, [_tc(T_GEN)])                       # 같은 턴(메시지 수 동일)
    nxt = BASE + [_msg("assistant", "..."), _msg("user", "please, again")]
    o3, e3 = _run(ag, nxt, [_tc(T_GEN)])                        # 다음 턴
    nxt2 = nxt + [_msg("assistant", "..."), _msg("user", "third time")]
    o4, e4 = _run(ag, nxt2, [_tc(T_GEN)])                       # CAP=2 소진
chk(o1 is not None and o2 is None and "같은 턴 재배달 생략" in e2, "같은 턴 → 생략(버퍼에 이미 실림)")
chk(o3 is not None and getattr(ag, "_t2_rdd_fired", 0) == 2, "다음 턴·여전히 미열람 → 재배달(fired=2)")
chk(o4 is None and "cap 2 reached" in e4, "CAP 소진 → None·로그")

print("\n§6 생성 창 초과 → 건너뜀 + 로그 (축약·선별 0)")
BIG = dict(CORPUS)
BIG["doc_test_alpha_009"] = "zeta_handoff_tool appears here. " + ("lorem ipsum " * 600)   # ≈7.2k자
with _env(**COMMON):
    ag = _agent(BIG)
    huge = BASE + [_msg("tool", "x" * 90000)]
    out, err = _run(ag, huge, [_tc(T_GEN)])
chk(out is None and "skipped: est" in err and not hasattr(ag, "_t2_rdd_fired"),
    "히스토리 90k + 배달 ≥5k → skipped·부기 없음")
with _env(**COMMON):
    ag = _agent(BIG)
    out, err = _run(ag, BASE, [_tc(T_GEN)])
chk(out is not None and "doc_test_alpha_009" in out["ids"] and len(out["text"]) > 5000,
    "같은 배달이 작은 히스토리에서는 들어간다(가드는 크기만 본다)")

print("\n§7 미선언 도구 무발화 · 디스패치 호출(내부 이름)은 선언 멤버십으로")
with _env(**COMMON):
    ag = _agent()
    out, err = _run(ag, BASE, [_tc("some_other_tool")])
chk(out is None and err == "", "선언 밖 도구 → None·로그 0")
with _env(**COMMON):
    ag = _agent()
    out, err = _run(ag, BASE, [_tc("call_zeta_wrapper", {"agent_tool_name": T_PRI})])
chk(out is not None and out["tool"] == T_PRI and out["ids"] == ["doc_test_alpha_002"],
    "`call_*(agent_tool_name=선언 도구)` → 내부 이름으로 판정(정본 `_exact_tool_name`)")

print("\n§8 `_docs_naming` json 경로 불변 + 코퍼스 폴백 · `_ctx_fits` 산식")
d = tempfile.mkdtemp(prefix="rdd_docs_")
for i, (k, v) in enumerate(CORPUS.items()):
    io.open(os.path.join(d, "%s.json" % k), "w", encoding="utf-8").write(
        json.dumps({"id": k, "title": "t%d" % i, "content": v}))
chk(sorted(GP._docs_naming(T_GEN, d)) == ["doc_test_alpha_001", "doc_test_alpha_002"],
    "json 디렉터리 도출(종전 경로·corpus 인자 없음)")
chk(sorted(GP._docs_naming(T_GEN, d, corpus={"doc_zzz": T_GEN})) == ["doc_test_alpha_001", "doc_test_alpha_002"],
    "디렉터리 도출이 있으면 코퍼스는 보지 않는다(폴백은 0편일 때만)")
chk(sorted(GP._docs_naming(T_GEN, os.path.join(d, "nope"), corpus=CORPUS)) == ["doc_test_alpha_001", "doc_test_alpha_002"],
    "디렉터리 없음 → 코퍼스 폴백(같은 술어)")
chk(GP._docs_naming(T_GEN, os.path.join(d, "nope2")) == set(), "디렉터리 없음·코퍼스 없음 → 빈 집합(종전 동일)")
chk(GP._ctx_fits([], "x" * 4999) == (True, 0), "_ctx_fits: 5k 미만은 검사 없이 통과")
ok_big, hist_big = GP._ctx_fits([_msg("tool", "x" * 90000)], "x" * 6000)
ok_small, hist_small = GP._ctx_fits([_msg("tool", "x" * 1000)], "x" * 6000)
chk((not ok_big) and hist_big == 90000 and ok_small and hist_small == 1000,
    "_ctx_fits: (hist+len)/3.5 > 24456 이면 불합격 (cp2 가드와 같은 산식)")

print("\n§9 배선(소스) — unified 호출부·가드 변수·표면화 조건·부착 마크")
SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
chk("_rdd = _require_doc_deliver(self, a2, state.messages, am.tool_calls or [])" in SRC,
    "unified 안에서 `_require_doc_deliver(self, a2, state.messages, am.tool_calls or [])` 호출")
chk("rdd_fb = None" in SRC and 'os.environ.get("T2_REQUIRE_DOC_DELIVER") == "1"' in SRC,
    "루프 변수 `rdd_fb` + 플래그 가드")
chk("and rdd_fb is None\n" in SRC and 'os.environ.get("T2_REQUIRE_DOC") == "1"' in SRC,
    "표면화(T2_REQUIRE_DOC) 조건에 `rdd_fb is None` — 같은 턴 문구 모순 방지")
chk("fb.append(UserMessage(role=\"user\", content=rdd_fb))" in SRC
    and "[T2_REQUIRE_DOC_DELIVER] 이 턴 재생성 버퍼에 부착" in SRC, "부착 자리(비커밋 fb 채널) + 부착 마크")
chk("_fit2, _hist = _ctx_fits(work, _cp2)" in SRC, "cp2 가드가 같은 `_ctx_fits` 를 쓴다(사본 0)")
chk("argmax" not in inspect.getsource(GP._require_doc_deliver), "엔진 함수에 선택·순위 코드 없음")

print("\n§10 리터럴 0 · go_stack 등재")
A3P = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
a3_tools = list(((json.load(io.open(A3P, encoding="utf-8")) or {}).get("require_doc_before") or {}).get("tools") or [])
fn_src = inspect.getsource(GP._require_doc_deliver) + inspect.getsource(GP._docs_naming_fallback)
chk(bool(a3_tools) and not any(t in fn_src for t in a3_tools),
    "엔진 함수 소스에 A3 선언 도구명 %d종 리터럴 0" % len(a3_tools))
GS = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
chk("export T2_REQUIRE_DOC_DELIVER=1" in GS, "go_stack: T2_REQUIRE_DOC_DELIVER=1 등재")
chk("export T2_SEARCH_REARM=1" in GS, "go_stack: T2_SEARCH_REARM=1 등재")
chk("export T2_REQUIRE_DOC_DELIVER_CAP=" in GS and "export T2_REQUIRE_DOC_DELIVER_MAX=" in GS,
    "go_stack: CAP·MAX 노브 선언(래칫 미선언 0)")

print("\n%s" % ("test_require_doc_deliver PASS" if not FAILED else
                "test_require_doc_deliver FAIL: %s" % FAILED))
sys.exit(1 if FAILED else 0)
