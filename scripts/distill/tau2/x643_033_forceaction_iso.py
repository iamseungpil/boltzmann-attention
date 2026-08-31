# -*- coding: utf-8 -*-
r"""x643 - 033 의 **say-don't-do** 를 `T2_FORCE_ACTION` 이 닫는가 (격리 · Qwen3.8 · [[76]]·[[78]]).

## 왜 (사용자 지시 2026-08-30: *"gpu 1 에 033 격리해서 레버 켜고 해결되나 실험하라"*)

`x642` per-step 대조가 결손을 특정했다. **재료 결손이 아니다** - 필요한 도구 이름이
**네 팔 모두에** 도착해 있었다(`doc_credit_cards_credit_cards_(general)_011` 이 `_0218`·`_1822`
둘을 함께 선언하고, 그 문서가 전 팔에 실렸다). 갈린 곳은 **행동**이다:

    [12] tool : "the conversation has not reached the necessary stages to transfer yet.
                 Try to help them by asking them about basic scenarios..."
    [13] Q38  : "Let me initiate that process for you now."   <- 말만 하고 도구 호출 0
    [14] user : ###TRANSFER###                                 <- 손님이 대화를 끝낸다
    (32B 는 같은 자리에서 `_0218` 을 unlock/call 하고 계속 갔다 -> reward 1.0)

## 팔 - 바뀌는 것은 **한 칸**(생성 제약)

    A_asis    그대로 생성                      <- 재현 게이트(도구 호출 0 이 나와야 한다)
    B_force   `tool_choice="required"`         <- `T2_FORCE_ACTION` 의 격리 등가물
    C_forced_after_say  A 가 말만 했을 때 **그 직후** 강제 재생성  <- 라이브 레버와 같은 순서

## 채점 - 닫힌 술어 · gold 미접촉([[23]])
    called      도구를 불렀나
    names_0218  호출 인자에 `initial_transfer_to_human_agent_0218` 이 있나
      (그 이름은 A3 `action_index` 가 `..._(general)_010/011` 에 대해 **선언**해 둔 것이고,
       그 문서는 이 문맥에 이미 실려 있다. 엔진은 무엇이 옳은지 모르고 **이름 동일성만** 본다.)

⛔이 파일은 프롬프트를 쓰지 않는다([[78]]). 바뀌는 것은 생성 제약 한 칸뿐이다.

사용: PYTHONPATH=. python x643_033_forceaction_iso.py --port 8141 [--n 8] [--wiring-only]
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

RES = "/home/woori/iso_tau3/tau2-bench/data/simulations/bank_x617_iso_q38_bank20_20260830/results.json"
TARGET = "task_033"
MODEL = "Qwen/Qwen3.8-27B-FP8"
# ★표적 도구 이름은 **선언에서 파생**한다 (코드에 도메인 리터럴 0 · [[05]]·[[23]]).
#   = A3 `action_index` 의 이관 체인 중 **이미 부른 것을 뺀 나머지**([[63]] 빼기).
WANT = None


def cut_at_saydo(sim):
    """도구가 *'아직 단계가 아니다'* 라고 답한 **그 직후**까지 자른다 = 결정점.

    자르는 기준은 **도구 응답의 축자**가 아니라 위치다: `_1822` 호출의 결과 메시지까지.
    (그 도구 이름은 A3 `action_index` 선언에 있는 것을 쓴다 - 코드에 도메인 리터럴을 적지 않기
     위해 아래 `chain_tools` 로 선언에서 읽어 온다.)
    """
    msgs = sim.get("messages") or []
    cutpoint = None
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []) or []:
            a = tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            if not isinstance(a, dict):
                continue
            nm = str(a.get("agent_tool_name") or "")
            if nm and nm.endswith("_1822"):
                tid = tc.get("id")
                for j in range(i + 1, len(msgs)):
                    if msgs[j].get("role") == "tool" and msgs[j].get("id") == tid:
                        cutpoint = j + 1
                        break
    return msgs[:cutpoint] if cutpoint else None


def chain_tools(a2):
    """A3 `action_index` 가 선언한 **이관 체인 도구 이름**. 코드에 도메인 리터럴 0."""
    out = set()
    for e in (((a2 or {}).get("policy_ontology") or {}).get("action_index") or []):
        for t in (e.get("tools") or []):
            if "transfer_to_human" in str(t):
                out.add(str(t))
    return sorted(out)


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
    """격리 clone 의 **환경이 든 도구 스키마**. 우리가 만들지 않는다."""
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
    global WANT
    chain = chain_tools(a2)
    print("A3 `action_index` 가 선언한 이관 체인 도구: %s" % chain)

    d = json.load(io.open(RES, encoding="utf-8"))
    sim = next((s for s in (d.get("simulations") or [])
                if str(s.get("task_id")) == TARGET
                and (s.get("reward_info") or {}).get("reward") == 0.0), None)
    if sim is None:
        print("SKIP - %s 실패 sim 없음" % TARGET)
        return
    pre = cut_at_saydo(sim)
    if not pre:
        print("REFUSING - 이미 부른 체인 도구의 응답 자리를 못 찾았다")
        return
    # 이미 부른 체인 도구를 문맥에서 읽고, **나머지**를 표적으로 삼는다(빼기·선언 파생)
    _blob0 = json.dumps(pre, ensure_ascii=False)
    _called = [t for t in chain if ('"agent_tool_name": "%s"' % t) in _blob0]
    _rest = [t for t in chain if t not in _called]
    if len(_rest) != 1:
        print("REFUSING - 남은 체인 도구가 %d개(1개여야 표적이 유일하다): %s" % (len(_rest), _rest))
        return
    WANT = _rest[0]
    print("이미 부른 체인 도구: %s · **표적(나머지)**: %s" % (_called, WANT))
    tools = env_tools()
    blob = json.dumps(pre, ensure_ascii=False)
    print("문맥 %d 메시지 (마지막 = `_1822` 응답) · 환경 도구 %d개" % (len(pre), len(tools)))
    print("이 문맥에 `%s` 가 이미 등장하나: %s" % (WANT, "예" if WANT in blob else "아니오"))
    print("마지막 도구 응답: %s" % " ".join(str(pre[-1].get("content") or "").split())[:200])
    if a.wiring_only:
        return
    if not tools:
        print("REFUSING - 환경 도구 스키마를 못 얻었다")
        return

    import urllib.request
    url = "http://localhost:%d/v1/chat/completions" % a.port

    def ask(msgs, forced=False):
        body = {"model": MODEL, "messages": msgs, "temperature": a.temp,
                "max_tokens": 600, "tools": tools}
        if forced:
            body["tool_choice"] = "required"
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.loads(r.read().decode())
        return j["choices"][0]["message"]

    base = to_openai(pre)

    def score(msg):
        tcs = msg.get("tool_calls") or []
        raw = json.dumps(tcs, ensure_ascii=False)
        return bool(tcs), (WANT in raw)

    print()
    print("%-20s %-4s %-10s %-12s %s" % ("arm", "n", "도구호출", "_0218 지목", "부른 도구"))
    print("-" * 80)
    for arm, forced in (("A_asis", False), ("B_force", True)):
        called = named = 0
        names = collections.Counter()
        for _ in range(a.n):
            try:
                m = ask(base, forced=forced)
            except Exception as e:
                print("  %s ERR %s" % (arm, e))
                continue
            c, w = score(m)
            called += 1 if c else 0
            named += 1 if w else 0
            for tc in (m.get("tool_calls") or []):
                nm = (tc.get("function") or {}).get("name") or tc.get("name") or "?"
                arg = json.dumps(tc.get("function", {}).get("arguments", ""), ensure_ascii=False)
                names[nm if WANT not in arg else nm + "(_0218)"] += 1
        print("%-20s %-4d %-10d %-12d %s" % (arm, a.n, called, named,
              " · ".join("%s=%d" % (k[:30], v) for k, v in names.most_common(3))))

    # C: A 가 말만 했을 때 그 직후 강제 재생성 (라이브 레버와 같은 순서)
    print()
    print("=== C_forced_after_say — 라이브 순서(말만 하면 그때 강제) ===")
    called = named = 0
    names = collections.Counter()
    for _ in range(a.n):
        try:
            m1 = ask(base, forced=False)
            if m1.get("tool_calls"):
                c, w = score(m1)
            else:
                m2 = ask(base + [{"role": "assistant",
                                  "content": m1.get("content") or ""}], forced=True)
                c, w = score(m2)
                m1 = m2
        except Exception as e:
            print("  ERR %s" % e)
            continue
        called += 1 if c else 0
        named += 1 if w else 0
        for tc in (m1.get("tool_calls") or []):
            nm = (tc.get("function") or {}).get("name") or tc.get("name") or "?"
            arg = json.dumps(tc.get("function", {}).get("arguments", ""), ensure_ascii=False)
            names[nm if WANT not in arg else nm + "(_0218)"] += 1
    print("%-20s %-4d %-10d %-12d %s" % ("C_forced_after_say", a.n, called, named,
          " · ".join("%s=%d" % (k[:30], v) for k, v in names.most_common(3))))


main()
