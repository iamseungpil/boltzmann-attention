#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op / operand 격리 프로브 — 부하(load)인가 능력인가 (무료·단일턴·user-sim 0).

정본 doc: reports/facet_rft_2026/ASSERTION_PROVENANCE_ARMS_DESIGN_2026_07_16.md §10
근거: 등대 §1.4 — load(s) = p_iso(s) − p_traj(s) > 0. ★**정보-맞춘 격리**여야 한다:
      임의 프롬프트가 아니라 **실제 궤적의 그 지점 접두 문맥 그대로** 재생한다.

★op와 operand는 순차 결정이므로 조건부로 분해한다(사용자 지적·2026-07-16):
    p(op 맞음)                     ← ARM op   : tool_choice=auto, 자유 선택
    p(operand 맞음 | op 맞음)      ← ARM operand: tool_choice=그 도구로 고정(측정용 조건화이지
                                     라이브 레버 아님 — 조건부 능력을 재려면 op를 통제해야 함)

측정 지점 (둘 다 실제 궤적서 추출):
  VERIFY : 사용자가 dob+phone을 준 직후. 올바른 행동 = ASK(name) 또는 by_name/by_email/by_id.
           날조 = get_user_information_by_phone* / by_date_of_birth (도구 목록에 없음).
  REWARD : 거래 23건 tool 출력 직후. 올바른 op = get_reward_discrepancies.
           operand = 그 23건을 읽어 transactions 배열로 formalize([[10]] LLM 몫).
           ★operand 정답 판정 = LLM formalize → **엔진 apply_op** → gold 4건과 일치하는가(end-to-end).

Run (리모트):
  python3 bank_op_operand_probe.py --base http://localhost:8141/v1 \
      --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --n 20
"""
import argparse
import gzip
import json
import os
import sys
import urllib.request

SIMDIR = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results"
A2P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "banking_knowledge.gate.json")

# tau2 src/tau2/agent/llm_agent.py 와 동일 (e11a_isolated_probe.py 선례와 동일 문안)
AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n- Send a message to the user.\n- Make a tool call.\n"
    "You cannot do both at the same time.\n\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."
)

GOLD_DISCREPANT = {"txn_f093f96e2001", "txn_580773a8649e", "txn_d398545ca1a2", "txn_37b5b8e67a5e"}
FAB_PREFIXES = ("get_user_information_by_phone", "get_user_information_by_date_of_birth")
VALID_LOOKUP = {"get_user_information_by_name", "get_user_information_by_email",
                "get_user_information_by_id"}


def post(base, payload):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def load_env_tools():
    """tau2 실물 스키마 + 우리 A2 주입 도구 (라이브 에이전트가 본 것과 동일 집합)."""
    from tau2.registry import registry
    env = registry.get_env_constructor("banking_knowledge")()
    tools = [t.openai_schema for t in env.get_tools()]
    policy = env.get_policy()
    a2 = json.load(open(A2P, encoding="utf-8"))
    for d in (a2.get("scaffold_get_tools") or []):
        tools.append({"type": "function", "function": {
            "name": d["name"], "description": d.get("description", ""),
            "parameters": {"type": "object",
                           "properties": {k: {"type": "string", "description": v}
                                          for k, v in (d.get("params") or {}).items()},
                           "required": list(d.get("params") or {})}}})
    return tools, policy, a2


def to_openai(msgs):
    """tau2 궤적 메시지 → OpenAI chat 포맷 (정보 보존·가공 0)."""
    out = []
    for m in msgs:
        role, content = m.get("role"), m.get("content")
        tcs = m.get("tool_calls") or []
        if role == "assistant":
            e = {"role": "assistant", "content": content or ""}
            if tcs:
                e["tool_calls"] = [{"id": tc.get("id") or f"c{i}", "type": "function",
                                    "function": {"name": tc.get("name"),
                                                 "arguments": json.dumps(tc.get("arguments") or {})}}
                                   for i, tc in enumerate(tcs)]
                e["content"] = e["content"] or None
            out.append(e)
        elif role == "user":
            if tcs:            # user-실행 도구 호출은 에이전트 시점에 존재하지 않음 → 텍스트만
                if content:
                    out.append({"role": "user", "content": content})
            else:
                out.append({"role": "user", "content": content or ""})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id") or "c0", "content": content or ""})
    return out


def cut_at(msgs, pred):
    """pred(msg)가 참인 첫 메시지까지 포함해 자른다."""
    for i, m in enumerate(msgs):
        if pred(m):
            return msgs[:i + 1]
    return None


def probe(base, model, msgs, tools, n, tool_choice=None, temp=1.0):
    calls = []
    for _ in range(n):
        p = {"model": model, "messages": msgs, "tools": tools, "temperature": temp, "max_tokens": 900}
        p["tool_choice"] = tool_choice or "auto"
        try:
            r = post(base, p)
            m = r["choices"][0]["message"]
            tcs = m.get("tool_calls") or []
            if tcs:
                f = tcs[0]["function"]
                try:
                    args = json.loads(f.get("arguments") or "{}")
                except Exception:
                    args = {"__raw": f.get("arguments")}
                calls.append((f["name"], args))
            else:
                calls.append((None, m.get("content") or ""))
        except Exception as e:
            calls.append(("__ERR__", repr(e)))
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sims", default=SIMDIR + "/bank_dreq_20260716_2140.results.json.gz")
    a = ap.parse_args()

    tools, policy, a2 = load_env_tools()
    sysmsg = [{"role": "system", "content": AGENT_INSTRUCTION + "\n\n<policy>\n" + policy + "\n</policy>"}]
    print("도구 %d개 주입 (env %d + A2 %d)" % (len(tools), len(tools) - len(a2.get("scaffold_get_tools") or []),
                                              len(a2.get("scaffold_get_tools") or [])))

    op = gzip.open if a.sims.endswith(".gz") else open
    with op(a.sims, "rt", encoding="utf-8") as f:
        sims = json.load(f)["simulations"]

    # ── 지점 1: VERIFY (사용자가 dob+phone 준 직후)
    for s in sims:
        pre = cut_at(s["messages"], lambda m: m.get("role") == "user" and isinstance(m.get("content"), str)
                     and "312-555-0481" in m["content"])
        if pre:
            break
    msgs = sysmsg + to_openai(pre)
    print("\n" + "=" * 66)
    print("지점 VERIFY (문맥 %d msg) — 사용자가 dob+phone 제공 직후" % len(pre))
    res = probe(a.base, a.model, msgs, tools, a.n)
    fab = sum(1 for n_, _ in res if n_ and n_.startswith(FAB_PREFIXES))
    good = sum(1 for n_, _ in res if n_ in VALID_LOOKUP)
    ask = sum(1 for n_, _ in res if n_ is None)
    other = a.n - fab - good - ask
    print("  ★op 분포 (n=%d): 날조 %d (%.0f%%) | 유효조회 %d (%.0f%%) | 텍스트/ASK %d (%.0f%%) | 기타 %d"
          % (a.n, fab, 100 * fab / a.n, good, 100 * good / a.n, ask, 100 * ask / a.n, other))
    from collections import Counter
    print("  이름별:", Counter(n_ for n_, _ in res).most_common())
    for n_, c in res[:3]:
        if n_ is None:
            print("   [텍스트 예시]", str(c)[:180].replace("\n", " "))

    # ── 지점 2: REWARD (거래 23건 tool 출력 직후)
    pre2 = None
    for s in sims:
        p = cut_at(s["messages"], lambda m: m.get("role") == "tool" and isinstance(m.get("content"), str)
                   and "credit_card_transaction_history" in m["content"])
        if p:
            pre2 = p
            break
    if not pre2:
        print("\n(REWARD 지점 없음 — 거래 도달 sim 부재)")
        return
    msgs2 = sysmsg + to_openai(pre2)
    print("\n" + "=" * 66)
    print("지점 REWARD (문맥 %d msg) — 거래 레코드 tool 출력 직후" % len(pre2))
    res2 = probe(a.base, a.model, msgs2, tools, a.n)
    okop = sum(1 for n_, _ in res2 if n_ == "get_reward_discrepancies")
    print("  ★op 분포 (n=%d): get_reward_discrepancies %d (%.0f%%)" % (a.n, okop, 100 * okop / a.n))
    print("  이름별:", Counter(n_ for n_, _ in res2).most_common())

    # ── 지점 2b: OPERAND | op 고정 (조건부 능력)
    print("\n" + "-" * 66)
    print("지점 REWARD / ARM operand — op를 get_reward_discrepancies로 고정(측정용 조건화)")
    tc = {"type": "function", "function": {"name": "get_reward_discrepancies"}}
    res3 = probe(a.base, a.model, msgs2, tools, a.n, tool_choice=tc)
    ctx = "\n".join(str(m.get("content") or "") for m in pre2)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import t2_compute as C
    decl = {d["name"]: d for d in a2["scaffold_get_tools"]}["get_reward_discrepancies"]
    nrec = badid = gold_hit = 0
    for n_, args in res3:
        if n_ != "get_reward_discrepancies" or not isinstance(args, dict):
            continue
        v = args.get("transactions")
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                v = None
        if not isinstance(v, list) or not v:
            continue
        nrec += 1
        ids = [str(r.get("transaction_id")) for r in v if isinstance(r, dict)]
        if any(i not in ctx for i in ids):
            badid += 1
        try:                       # ★end-to-end: LLM formalize → 엔진 결정론 계산 → gold 대조
            out = C.apply_op(decl.get("op"), dict(args, transactions=v))
            if isinstance(out, list) and set(map(str, out)) == GOLD_DISCREPANT:
                gold_hit += 1
        except Exception:
            pass
    print("  transactions 배열 채움: %d/%d | 문맥에 없는 id 포함: %d | ★엔진 통과 후 gold 4건 정확 일치: %d/%d"
          % (nrec, a.n, badid, gold_hit, a.n))
    print("\n판정 규칙: 격리서 맞으면 **부하**(→결정론 controller가 산다) / 격리서도 틀리면 **능력·prior**"
          "(→프롬프트로 못 닫음·learn/scale). op와 operand를 각각 판정할 것.")


if __name__ == "__main__":
    main()
