# -*- coding: utf-8 -*-
r"""x649 - 033: **받은 문서가 선언하는 도구를 라벨로 붙이면** 다음 수를 두는가 (격리 · Q38 · [[76]]·[[78]]).

## 왜 (사용자 지시 2026-08-30: *"실패 7개를 retrieval 먼저 수리해서 pass 시켜라" · "033 부터 하라"*)

`x642` per-step 이 결손을 특정했다. 재료는 왔다 - 그런데 **문서 안의 도구 이름 둘 중 하나만** 건졌다:

    A3 action_index :  doc_credit_cards_credit_cards_(general)_011 -> ['_0218', '_1822']
    Q38 이 받은 것  :  그 문서 (x642 확인: 네 팔 모두 도착)
    Q38 이 쓴 것    :  `_1822` 만.  `_1822` 가 "아직 단계가 아니다" 라고 답하자
                       "Let me initiate that process for you now." 라고 **말만 하고 멈춤**
    32B(통과)       :  같은 자리에서 `_0218` 을 unlock/call -> reward 1.0

⇒ retrieval 측 수리 지점이 실재한다: **받은 문서가 선언하는 도구를 라벨로 붙인다.**
   배달을 늘리지 않는다(문서는 이미 와 있다). 엔진은 `action_index` 를 **조회해 이름만** 붙인다.

## 팔 - 바뀌는 것은 **한 칸**(문면)

    A_asis     회수된 라이브 문맥 그대로            <- 재현 게이트(도구 호출 0 이 나와야 한다)
    B_toolidx  + **받은 문서가 선언한 도구 목록**   <- 엔진이 action_index 에서 조회
    N_len      같은 길이의 무관 문장                <- 길이 통제([[57]])

⛔`tool_choice` 를 쓰지 않는다(사용자: *"tool_choice 는 역효과만 있다"*).
⛔엔진은 어느 도구를 부르라고 고르지 않는다 - **받은 문서에 딸린 선언을 나열**할 뿐이다([[62]]④).

## 채점 - 닫힌 술어 · gold 미접촉([[23]])
    called   도구를 불렀나
    target   **아직 안 부른 체인 도구**를 지목했나 (그 이름은 action_index 선언에서 파생 · [[63]] 빼기)

사용: PYTHONPATH=. python x649_033_toolindex_iso.py --port 8141 [--n 8] [--wiring-only]
"""
import argparse
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_interpreter import load_domain_a2                     # noqa: E402
import t2_gate_patch as G                                       # noqa: E402

RES = "/home/woori/iso_tau3/tau2-bench/data/simulations/bank_x617_iso_q38_bank20_20260830/results.json"
TARGET = "task_033"
MODEL = "Qwen/Qwen3.8-27B-FP8"
DOC = re.compile(r"doc_[a-z0-9_()\-]{6,90}", re.I)


def cut_at_stall(sim, chain):
    """체인 도구 호출의 **결과 메시지 직후**까지 자른다 = 말만 하고 멈춘 자리 직전."""
    msgs = sim.get("messages") or []
    cut = None
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []) or []:
            a = tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            nm = str((a or {}).get("agent_tool_name") or "")
            if nm in chain:
                tid = tc.get("id")
                for j in range(i + 1, len(msgs)):
                    if msgs[j].get("role") == "tool" and msgs[j].get("id") == tid:
                        cut = j + 1
                        break
    return msgs[:cut] if cut else None


def tool_index(a2):
    """A3 `action_index`: 문서 -> 그 문서가 선언한 도구. 정규화 키로."""
    out = {}
    for e in (((a2 or {}).get("policy_ontology") or {}).get("action_index") or []):
        d = G._a3_norm_doc(e.get("doc"))
        ts = [str(t) for t in (e.get("tools") or [])]
        if d and ts:
            out[d] = ts
    return out


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
    tidx = tool_index(a2)
    chain = sorted({t for ts in tidx.values() for t in ts if "transfer_to_human" in t})
    print("A3 action_index: 문서 %d개 · 이관 체인 도구 %s" % (len(tidx), chain))

    d = json.load(io.open(RES, encoding="utf-8"))
    sim = next((s for s in (d.get("simulations") or [])
                if str(s.get("task_id")) == TARGET
                and (s.get("reward_info") or {}).get("reward") == 0.0), None)
    if sim is None:
        print("SKIP - 실패 sim 없음")
        return
    pre = cut_at_stall(sim, set(chain))
    if not pre:
        print("REFUSING - 체인 호출 자리를 못 찾았다")
        return

    blob = json.dumps(pre, ensure_ascii=False)
    called = [t for t in chain if ('"agent_tool_name": "%s"' % t) in blob]
    rest = [t for t in chain if t not in called]
    print("이미 부른 체인 도구: %s · **표적(빼기의 나머지)**: %s" % (called, rest))
    if len(rest) != 1:
        print("REFUSING - 남은 체인 도구가 %d개" % len(rest))
        return
    WANT = rest[0]

    # ★B 팔 문면 = **받은 문서**에 딸린 선언을 엔진이 조회해 나열 (선택 0 · 순위 0)
    got = {G._a3_norm_doc(h) for h in DOC.findall(blob)}
    lines = []
    for dnorm in sorted(got & set(tidx)):
        lines.append("- %s declares: %s" % (dnorm, ", ".join(tidx[dnorm])))
    note = None
    if lines:
        note = (chr(10) + "[Agent tools declared by the knowledge-base documents you have "
                "already received. This list is complete for those documents.]"
                + chr(10) + chr(10).join(lines))
    print()
    print("문맥 %d 메시지 · 받은 문서 %d · action_index 적중 %d · 문면 %d B"
          % (len(pre), len(got), len(got & set(tidx)), len(note or "")))
    print("마지막 도구 응답: %s" % " ".join(str(pre[-1].get("content") or "").split())[:180])
    if note:
        print("--- B 팔 문면 ---")
        print(note)
    if a.wiring_only:
        return
    if not note:
        print("REFUSING - 문면이 비었다(받은 문서 중 action_index 적중 0)")
        return

    tools = env_tools()
    if not tools:
        print("REFUSING - 환경 도구 스키마 없음")
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
    print("%-11s %-4s %-10s %-12s %s" % ("arm", "n", "도구호출", "표적지목", "부른 도구"))
    print("-" * 76)
    for arm, extra in (("A_asis", None), ("B_toolidx", note), ("N_len", filler)):
        called_n = hit = 0
        names = collections.Counter()
        for _ in range(a.n):
            msgs = list(base) + ([{"role": "user", "content": extra}] if extra else [])
            try:
                m = ask(msgs)
            except Exception as e:
                print("  %s ERR %s" % (arm, e))
                continue
            tcs = m.get("tool_calls") or []
            raw = json.dumps(tcs, ensure_ascii=False)
            called_n += 1 if tcs else 0
            hit += 1 if WANT in raw else 0
            for tc in tcs:
                nm = (tc.get("function") or {}).get("name") or "?"
                names[nm + ("(표적)" if WANT in json.dumps(tc, ensure_ascii=False) else "")] += 1
            if not tcs:
                names["(말만 함)"] += 1
        print("%-11s %-4d %-10d %-12d %s" % (arm, a.n, called_n, hit,
              " · ".join("%s=%d" % (k[:32], v) for k, v in names.most_common(3))))


main()
