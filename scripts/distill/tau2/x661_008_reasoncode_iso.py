# -*- coding: utf-8 -*-
r"""x661 - 008: **표를 결정점에 놓으면** 옳은 reason 코드를 고르는가 (격리 · Q38 · [[76]]·[[78]]).

## 왜 (사용자 지시 2026-08-30: *"008 도 격리하고 문서를 잘 전달하면 풀리지 않을까?"*)

내가 008 을 *"재료가 도착했으니 LLM 몫"* 으로 성급히 닫았다. 그러나 **문맥에 도착했다 !=
결정점에 전달됐다** 이다([[65]] 부하). 실측: 표는 메시지 **[21]** 에 왔고, 모델은 그 뒤 검색을
**21회 더** 하며 문맥을 키운 끝에 골랐다. [[62]] 규칙 - **격리에서 되면 레버는 전달뿐**이다.

## 결손 (x644 실측)
```
gold reason = customer_demands_after_unavailable_offer_refusal
실제 reason = unconfirmed_external_communication      (둘 다 유효 코드 · db_match true)
action_match false · compare_args = ["reason"]
```

## 팔 - 바뀌는 것은 **한 칸**
    A_asis   회수된 라이브 문맥 그대로                       <- 재현 게이트(틀린 코드가 나와야 한다)
    B_table  + A3 `action_index` 가 그 도구에 대해 지목한 **문서 축자**  <- 엔진 빌더가 읽어 놓는다
    N_len    같은 길이의 무관 문장                           <- 길이 통제([[57]])

⛔이 파일은 프롬프트를 쓰지 않는다([[78]]). B 팔 재료는 **선언(action_index) -> 문서 축자** 로만 온다.
⛔엔진은 어느 코드인지 고르지 않는다 - 표를 놓을 뿐이다([[62]] (3)(4)).

## 채점 - 닫힌 술어 · gold 는 **요구 목록으로만** 쓴다([[23]])
    호출 인자의 `reason` 이 닫힌 19개 중 무엇인가. gold 코드와 일치하는 비율.

사용: PYTHONPATH=. python x661_008_reasoncode_iso.py --port 8141 [--n 8] [--wiring-only]
"""
import argparse
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import load_domain_a2                     # noqa: E402
import t2_search as SRCH                                        # noqa: E402
import t2_gate_patch as G                                       # noqa: E402

RES = "/home/woori/iso_tau3/tau2-bench/data/simulations/bank_x644_q38base_bank78_20260830/results.json"
DOCS_DIR = "/home/woori/iso_tau3/tau2-bench/data/tau2/domains/banking_knowledge/documents"
TARGET = "task_008"
TOOL = "transfer_to_human_agents"
MODEL = "Qwen/Qwen3.8-27B-FP8"


def declared_docs(a2, tool):
    """A3 `action_index` 가 그 도구에 대해 지목한 문서 id. 코드에 문서 id 를 적지 않는다."""
    out = []
    for e in (((a2 or {}).get("policy_ontology") or {}).get("action_index") or []):
        if tool in (e.get("tools") or []):
            d = e.get("doc")
            if d and d not in out:
                out.append(d)
    return out


def cut_before(sim, tool):
    """그 도구 호출을 담은 어시스턴트 턴 **직전**까지 = 결정점."""
    msgs = sim.get("messages") or []
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []) or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            if n == tool:
                return msgs[:i]
    return None


def gold_reason(d, tid, tool):
    for t in (d.get("tasks") or []):
        if str(t.get("id")) != tid:
            continue
        for a in ((t.get("evaluation_criteria") or {}).get("actions") or []):
            if str(a.get("name")) == tool:
                return (a.get("arguments") or {}).get("reason")
    return None


def to_openai(msgs):
    out = []
    for m in msgs:
        role = str(m.get("role") or "")
        c = str(m.get("content") or "")
        if role in ("system", "user"):
            out.append({"role": role, "content": c})
        elif role == "assistant":
            d = {"role": "assistant", "content": c or None}
            tcs = []
            for tc in (m.get("tool_calls") or []) or []:
                a = tc.get("arguments")
                nm = tc.get("name") or (tc.get("function") or {}).get("name") or "unknown"
                tcs.append({"id": tc.get("id") or "x", "type": "function",
                            "function": {"name": nm, "arguments": a if isinstance(a, str)
                                         else json.dumps(a or {}, ensure_ascii=False)}})
            if tcs:
                d["tool_calls"] = tcs
            out.append(d)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id") or "x", "content": c[:6000]})
    return out


def env_tools():
    sys.path.insert(0, "/home/woori/iso_tau3/tau2-bench/src")
    from tau2.registry import registry
    env = registry.get_env_constructor("banking_knowledge")()
    out = []
    for t in env.get_tools():
        sch = None
        for attr in ("openai_schema", "as_openai_tool", "to_openai"):
            v = getattr(t, attr, None)
            if callable(v):
                try:
                    sch = v()
                except Exception:
                    sch = None
            elif isinstance(v, dict):
                sch = v
            if sch:
                break
        if isinstance(sch, dict):
            out.append(sch if sch.get("type") == "function"
                       else {"type": "function", "function": sch})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args()

    a2 = load_domain_a2("banking_knowledge")
    docs = declared_docs(a2, TOOL)
    print("A3 action_index 가 `%s` 에 지목한 문서: %s" % (TOOL, docs))
    if not docs:
        print("REFUSING - 선언이 없다(그 행을 저작해야 한다)")
        return

    d = json.load(io.open(RES, encoding="utf-8"))
    sim = next((s for s in (d.get("simulations") or [])
                if str(s.get("task_id")) == TARGET), None)
    if sim is None:
        print("SKIP - %s sim 없음" % TARGET)
        return
    gold = gold_reason(d, TARGET, TOOL)
    pre = cut_before(sim, TOOL)
    if not pre:
        print("REFUSING - %s 호출 자리를 못 찾았다" % TOOL)
        return

    corpus = {}
    for f in os.listdir(DOCS_DIR):
        if f.endswith(".json"):
            j = json.load(io.open(os.path.join(DOCS_DIR, f), encoding="utf-8"))
            corpus[j.get("id")] = j.get("content") or ""
    read, missing = SRCH.read_docs(docs, doc_dir=None, corpus=corpus)
    note = None
    if read:
        parts = ["[%s]" % k + chr(10) + " ".join(str(v).split())[:3000] for k, v in read.items()]
        note = chr(10) + chr(10).join(parts)
    print("문맥 %d 메시지 · gold reason = %s" % (len(pre), gold))
    print("배달 문서 %d개 · 미독 %s · 문면 %d B" % (len(read), missing, len(note or "")))
    if note:
        print("--- B 팔 문면 앞 500자 ---")
        print(note[:500])
    if a.wiring_only:
        return
    if not note:
        print("REFUSING - 문서를 못 읽었다")
        return

    codes = []
    tools = env_tools()
    for t in tools:
        f = t.get("function") or {}
        if f.get("name") != TOOL:
            continue
        pr = (f.get("parameters") or {}).get("properties") or {}
        en = (pr.get("reason") or {}).get("enum")
        if en:
            codes = list(en)
    print("도구 스키마가 노출한 reason enum: %d개" % len(codes))
    if not codes:
        print("REFUSING - enum 을 스키마에서 못 얻었다")
        return

    import urllib.request
    url = "http://localhost:%d/v1/chat/completions" % a.port

    def ask(msgs):
        body = json.dumps({"model": MODEL, "messages": msgs, "temperature": a.temp,
                           "max_tokens": 500, "tools": tools}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.loads(r.read().decode())
        return j["choices"][0]["message"]

    base = to_openai(pre)
    filler = "Please continue helping the customer. " * max(1, len(note) // 38)
    print()
    print("%-10s %-4s %-8s %-8s %s" % ("arm", "n", "호출", "**정답**", "고른 코드 분포"))
    print("-" * 88)
    for arm, extra in (("A_asis", None), ("B_table", note), ("N_len", filler)):
        hit = called = 0
        picks = collections.Counter()
        for _ in range(a.n):
            msgs = list(base) + ([{"role": "user", "content": extra}] if extra else [])
            try:
                m = ask(msgs)
            except Exception as e:
                print("  %s ERR %s" % (arm, e))
                continue
            raw = json.dumps(m.get("tool_calls") or [], ensure_ascii=False)
            if TOOL in raw:
                called += 1
            got = [c for c in codes if '"%s"' % c in raw]
            picks[",".join(got) if got else "(코드 없음)"] += 1
            if gold and gold in got:
                hit += 1
        print("%-10s %-4d %-8d %-8d %s" % (arm, a.n, called, hit,
              " · ".join("%s=%d" % (k[:40], v) for k, v in picks.most_common(3))))


main()
