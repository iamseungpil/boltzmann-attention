#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3 읽기-명세 전달(`T2_SG_DOCS=1`·`_docs_delivery`) 오프라인 배선 테스트 (2026-08-21·C582 처방).

[[71]] 계약 4문 답([[17]] 동형·파일 머리 의무):
  ①기능 하나 — 클래스-선택 서브(닫힌 목록 선택 하나)와 형식화 서브(components 하나)는 **다른 서브**다.
  ②재료는 선언에서 — 검정 ①은 **실제 A3 정본**의 docs 선언 형상을 검사하고, 검정 ②~⑥은 합성
    선언(테스트가 양쪽을 다 통제)으로 **기구**를 검사한다. 문서 id 는 코드에 없다.
  ③전달 = 선언된 id 정확 집기 — bm25 미노출을 ②에서 직접 검정한다.
  ④엔진 해석 0 — 소속 검산·자르기·앵커 비교만. 검정 ④가 소속 검산을 직접 본다.

검정: ①A3 정본 선언 형상(두 층 동일·클래스 38·앵커 실재)
      ②docs 모드 — 도구 미노출·지시가 재료보다 앞(C578)·always+선택 클래스 범위 전달·operand 파싱
      ③앵커 불일치 → 문서 전량 폴백(밀린 조각 배달 금지)
      ④목록 밖 클래스 → 소속 검산이 버림·유효분만
      ⑤코퍼스 0 → 검색 경로 폴백(거동보존)
      ⑥T2_SG_DOCS 미설정 → 종전 경로 불변(도구 노출·검색)
⚠️단위통과≠라이브발화([[30]]) — 배선만 본다. 실제 선언×실제 코퍼스는 x456 이 잰다.
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_msg = types.ModuleType("tau2.data_model.message")


class UserMessage:
    def __init__(self, role="user", content=""):
        self.role, self.content = role, content


class ToolMessage:
    def __init__(self, id=None, role="tool", requestor="assistant", content="", error=False):
        self.id, self.role, self.requestor, self.content, self.error = id, role, requestor, content, error


_msg.UserMessage, _msg.ToolMessage = UserMessage, ToolMessage
_msg.MultiToolMessage = type("MultiToolMessage", (), {})
sys.modules.setdefault("tau2", types.ModuleType("tau2"))
sys.modules.setdefault("tau2.data_model", types.ModuleType("tau2.data_model"))
sys.modules["tau2.data_model.message"] = _msg
_la = types.ModuleType("tau2.agent.llm_agent")
sys.modules.setdefault("tau2.agent", types.ModuleType("tau2.agent"))
sys.modules["tau2.agent.llm_agent"] = _la

import t2_scaffold_get as SG  # noqa: E402

# ── 합성 선언·코퍼스 (검정 ②~⑥ · 테스트가 양쪽을 다 통제한다) ─────────────────────────
DOC_A = "doc_test_alpha_001"      # always
DOC_B = "doc_test_beta_001"       # by_class[클래스1]·범위 2개
DOC_C = "doc_test_beta_002"       # by_class[클래스1]·앵커 불일치 검정용
TXT_A = "Common policy. Relationship bonuses stack. Tier bonuses stack."
TXT_B = "Header line first. BASE APY: 4.0% on all balances. Filler text. BOOST: +0.25% with pairing."
TXT_C = "Shifted content here. CARD BONUS: +0.6% for EcoCard holders. Tail."


def _rg(txt, off, ln):
    return [off, ln, " ".join(txt[off:off + 40].split())]


def synth_decl(c_anchor_ok=True):
    b_r1 = _rg(TXT_B, 19, 31)      # "BASE APY: 4.0% on all balances."
    b_r2 = _rg(TXT_B, 64, 26)      # "BOOST: +0.25% with pairing"
    c_r1 = _rg(TXT_C, 22, 33) if c_anchor_ok else [22, 33, "WRONG ANCHOR THAT WILL NOT MATCH XX"]
    return {
        "name": "synth_apy_tool",
        "op": {"op": "group_reduce", "over": "components"},
        "isolate": {
            "mode": "fetch_formalize",
            "ref_params": ["savings_account_type", "customer_products"],
            "getter_tools": ["KB_search_bm25"],
            "operand_keys": ["components"],
            "max_rounds": 4,
            "instructions": "legacy search instructions (should NOT appear in docs mode)",
            "answer_format": 'Reply with exactly one JSON object: {"components": [...]}',
            "docs": {
                "instructions": "DOCS-MODE INSTRUCTIONS: extract components from DOCUMENTS only.",
                "axes": ["apy"],
                "always": [DOC_A],
                "by_class": {
                    "beta_account": [
                        {"doc": DOC_B, "ranges": [b_r1, b_r2], "basis": "content"},
                        {"doc": DOC_C, "ranges": [c_r1], "basis": "content"},
                    ],
                    "other_account": [{"doc": "doc_test_other_001", "ranges": [[0, 10, "no"]]}],
                },
            },
        },
    }


CORPUS = {DOC_A: TXT_A, DOC_B: TXT_B, DOC_C: TXT_C, "doc_test_other_001": "no material here"}

CALLS = []


class _Resp:
    def __init__(self, content=None, tool_calls=None):
        self.role, self.content, self.tool_calls = "assistant", content, tool_calls


def fake_generate(model=None, tools=None, messages=None, call_name=None, **kw):
    CALLS.append({"call_name": call_name, "tool_choice": kw.get("tool_choice"),
                  "tools": [getattr(t, "name", None) for t in (tools or [])],
                  "prompt": (messages[-1].content if messages else "")})
    if call_name == "sg_docs_class":
        # 클래스-선택 서브: 유효 1 + 목록 밖 1 (소속 검산 검정)
        return _Resp(content='["beta_account", "not_a_class"]')
    # 형식화 서브: 값이 배달 재료에 실재(4.0·0.25)
    return _Resp(content='{"components": [{"kind": "base", "value": 4.0, '
                         '"source": "BASE APY: 4.0% on all balances."}, '
                         '{"kind": "checking", "value": 0.25, '
                         '"source": "BOOST: +0.25% with pairing"}]}')


_la.generate = fake_generate


def mk_orch(with_corpus=True):
    tool = types.SimpleNamespace(name="KB_search_bm25")
    env = types.SimpleNamespace(db=(dict(CORPUS) if with_corpus else {}))
    return types.SimpleNamespace(
        agent=types.SimpleNamespace(tools=[tool], llm="fake-model",
                                    llm_args={"temperature": 0.0}),
        environment=env)


def run_env(tcs):
    return [ToolMessage(id=getattr(t, "id", "c1"), content="Found 0 record(s).") for t in tcs]


def main():
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  ✓ " if cond else "  ✗ ") + msg)

    print("① A3 정본 선언 형상:")
    layers = []
    for lay in ("specific", "gate"):
        d = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.%s.json" % lay),
                           encoding="utf-8"))
        sg = next(x for x in d["scaffold_get_tools"] if x["name"] == "get_correct_savings_apy")
        layers.append(sg["isolate"]["docs"])
    da, dg = layers
    chk(json.dumps(da, sort_keys=True, ensure_ascii=False)
        == json.dumps(dg, sort_keys=True, ensure_ascii=False), "두 층 docs 블록 동일([[24]])")
    chk(len(da.get("by_class") or {}) == 38, "by_class 38 클래스")
    chk(len(da.get("always") or []) == 3, "always 3편(공용 정책)")
    n_rg = sum(len(e.get("ranges") or []) for v in da["by_class"].values() for e in v)
    n_anc = sum(1 for v in da["by_class"].values() for e in v
                for rg in (e.get("ranges") or []) if len(rg) > 2 and rg[2])
    chk(n_rg == 152 and n_anc == 152, "범위 152 · 전부 앵커 실재 (%d/%d)" % (n_anc, n_rg))
    chk(bool(da.get("instructions")), "docs.instructions(검색 문구 없는 판) 선언됨")
    chk("search tool" not in da.get("instructions", "").lower()
        or "no search tool" in da.get("instructions", "").lower(),
        "docs.instructions 에 검색 지시 없음")

    print("② docs 모드 (T2_SG_DOCS=1):")
    os.environ["T2_SG_DOCS"] = "1"
    os.environ.pop("T2_SG_ISOFB", None)
    os.environ.pop("T2_SG_GROUND", None)
    del CALLS[:]
    decl = synth_decl()
    iso = decl["isolate"]
    orch = mk_orch()
    ctx = {"savings_account_type": "Beta Account", "customer_products": "beta checking"}
    sub = SG._sub_fetch_formalize(orch, decl, iso, ctx, run_env)
    chk(len(CALLS) == 2 and CALLS[0]["call_name"] == "sg_docs_class",
        "서브 둘: 클래스-선택 → 형식화 (결정 하나당 서브 하나·[[65]])")
    chk(CALLS[0]["tools"] == [] and CALLS[1]["tools"] == [],
        "두 서브 모두 getter 미노출(검색 0·[[71]] 계약 3항)")
    p = CALLS[1]["prompt"]
    chk(p.find("DOCS-MODE INSTRUCTIONS") == 0, "docs 지시판 사용 + 지시가 맨 앞(C578)")
    chk(p.find("DOCS-MODE INSTRUCTIONS") < p.find("=== DOCUMENTS ==="),
        "지시 < 재료 순서(C578)")
    chk("legacy search instructions" not in p, "검색 지시판은 미사용")
    chk(TXT_A in p, "always 문서 전량 전달")
    chk("BASE APY: 4.0% on all balances." in p and "BOOST: +0.25% with pairing" in p,
        "선택 클래스 범위 2개 전달(앵커 일치)")
    chk("Filler text" not in p, "범위 밖 바이트는 미전달(자르기 작동)")
    chk("no material here" not in p, "미선택 클래스 문서 미전달")
    chk(isinstance(sub, dict) and len(sub.get("components") or []) == 2,
        "operand 파싱(components 2건)")
    chk((getattr(orch, "_t2_docs_mat", None) or {}).get("picked") == ["beta_account"],
        "발화 계측 마크(_t2_docs_mat) 기록")

    print("③ 앵커 불일치 → 문서 전량 폴백:")
    del CALLS[:]
    decl3 = synth_decl(c_anchor_ok=False)
    orch3 = mk_orch()
    SG._sub_fetch_formalize(orch3, decl3, decl3["isolate"], ctx, run_env)
    p3 = CALLS[1]["prompt"]
    chk(TXT_C in p3, "불일치 문서는 전량 배달(밀린 조각 배달 금지)")
    chk((getattr(orch3, "_t2_docs_mat", None) or {}).get("anchor_fallback") == 1,
        "anchor_fallback=1 계측")

    print("④ 소속 검산:")
    chk((getattr(orch, "_t2_docs_mat", None) or {}).get("alien") == ["not_a_class"],
        "목록 밖 이름은 버리고 기록(엔진=집합 검산만·[[59]])")

    print("⑤ 코퍼스 0 → 검색 폴백:")
    del CALLS[:]
    orch5 = mk_orch(with_corpus=False)
    SG._sub_fetch_formalize(orch5, decl, iso, ctx, run_env)
    chk(CALLS and CALLS[0]["call_name"] == "sg_fetch_iso"
        and CALLS[0]["tools"] == ["KB_search_bm25"],
        "docs 실패 시 종전 검색 경로(도구 노출·거동보존)")
    chk(getattr(orch5, "_t2_docs_mat", "MISS") is None, "발화 마크 = None(미발화 가시화)")

    print("⑥ T2_SG_DOCS 미설정 → 종전 경로 불변:")
    os.environ.pop("T2_SG_DOCS", None)
    del CALLS[:]
    orch6 = mk_orch()
    SG._sub_fetch_formalize(orch6, decl, iso, ctx, run_env)
    chk(CALLS and CALLS[0]["call_name"] == "sg_fetch_iso"
        and CALLS[0]["tools"] == ["KB_search_bm25"]
        and CALLS[0]["tool_choice"] == "required",
        "미설정 = 검색 경로 그대로(1라운드 required 포함)")
    chk("=== DOCUMENTS ===" not in CALLS[0]["prompt"], "docs 재료 미주입")

    print("\n%s" % ("PASS — A3 읽기-명세 전달 배선 정상 (실선언×실코퍼스는 x456·[[30]])"
                    if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
