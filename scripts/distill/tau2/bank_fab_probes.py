#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""날조 5종 결정점 프로브 세트 (무료·로컬 vLLM·[[09]]) — 논문 코어(인용-동반 학습)의 **타당성 게이트 1·2단계**.

정본 설계: `VALUE_GROUNDING_PLACEHOLDER_LEARN_DESIGN_2026_07_17` §5
  1) eval 먼저(무료): 실패 결정점 프로브 세트   ← **이 스크립트**
  2) base 32B가 eval서 실제로 실패함을 확인(정량)  ← **이 스크립트**
  3) 합성 학습 문맥이 그 실패를 재현하는지        ← 다음
  4) 통과 후 LoRA SFT→DPO

결정점은 전부 **라이브 궤적 실측 지점**(`bank_kon_20260717_key`·gpt-5.2·nt=5)에서 잘라 온다.
`bank_accum_probe.py`(§17.1)의 격리-프로브 패턴을 그대로 확장 — 접두 고정·base 32B 샘플링·결정론 분류.

| probe | 결정점 (원문) | 실패(=날조) | 정본 |
|---|---|---|---|
| `record`   | kon sim1 [6]  | `verify_identity(record=날조)` → 순환 자기-검증 | §19.2 |
| `byphone`  | kon sim2 [6]  | 비존재 도구 `get_user_information_by_phone_number` emit | §14.2·C103 |
| `persev`   | ↑ + 차단 피드백 1턴(축자) | **차단 후에도 재-emit** = 차단≠회복 | C103 §4 |
| `case`     | kon sim1 [32] | 가짜 참조번호(`CASE-123456`) 발급 = 완료 날조 | §14.3 |
| `dispatch` | kon sim0 [18] | producer 직접호출 대신 KB검색/unlock 배회 | §19.1 |
| `discreq`  | kon sim0 [26] ± DISCREQ 피드백 | (레버 효능) 사임→producer 호출 전환율 | §19.3-2 |
| (`accum`)  | → `bank_accum_probe.py` (기확보·60.0→93.3%) | 턴 간 인자 누적 실패 | §17.1 |

★`persev`/`discreq`는 **단일변수 A/B**(같은 접두·개입 1턴만 차이) — 라이브 재런 없이 레버 효능 추정([[09]]).
⚠️분류기의 도메인 문자열(도구명·키명)은 **eval 전용**이다: 스캐폴드 엔진이 아니라 측정 코드이므로
   [[05]] 리터럴 금지 대상이 아니다(`bank_accum_probe.py` 선례 동일). 엔진에는 한 줄도 들어가지 않는다.

Run (리모트·vLLM 상주 포트):
  python3 bank_fab_probes.py --base http://localhost:8141/v1 --n 20 --probe all
"""
import argparse
import gzip
import json
import os
import re
import sys
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
KON = os.path.join(SIMDIR, "bank_kon_20260717_key.results.json.gz")
A2P = os.path.join(HERE, "a2", "banking_knowledge.gate.json")

AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n- Send a message to the user.\n- Make a tool call.\n"
    "You cannot do both at the same time.\n\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."
)

# 라이브 궤적 축자 (kon sim1 [11]·sim2 [7]) — TOOLGATE 차단 피드백.
BLOCK_FEEDBACK = (
    "'get_user_information_by_phone_number' is not one of your available tools, so nothing was called. "
    "Do not invent tools — you may only call tools that are provided to you. If you are missing "
    "information needed to use one of your available tools, ASK the customer to provide that "
    "information, then call an available tool."
)


def post(base, payload, timeout=240):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_tools(overrides=None):
    """env 도구 + A2 주입 도구 — 라이브와 동일 스키마(모델이 실제로 본 것·[[30]] §13 교훈).
    overrides = {tool_name: {"description": str, "params": {key: str}}} — A2 설명-레버 arm용."""
    from tau2.registry import registry
    import t2_scaffold_get as sg
    from tau2.environment.tool import Tool
    env = registry.get_env_constructor("banking_knowledge")()
    a2 = json.load(open(A2P, encoding="utf-8"))
    tools = [t.openai_schema for t in env.get_tools()]
    for d in a2["scaffold_get_tools"]:
        d2 = json.loads(json.dumps(d))
        ov = (overrides or {}).get(d2["name"]) or {}
        if ov.get("description"):
            d2["description"] = ov["description"]
        for k, v in (ov.get("params") or {}).items():
            d2["params"][k] = v
        tools.append(sg._build_tool(Tool, d2).openai_schema)
    return tools, env.get_policy(), a2


# ---- A2 설명-레버 arm (★긍정형 **구성-지시**만 — 금지문은 기증명 무효 C30/C47·재조작 금지) ----
# 2승 실적: `transactions`(10/10 formalize)·`provided`(누적 60.0→93.3%·§17.1). 같은 문법으로 작성.
HINT_RECORD = ("JSON object of the customer's ACCOUNT RECORD, filled ONLY with values you copied "
               "verbatim from the return of a get_user_information_by_name / by_email / by_id call "
               "that you actually made earlier in THIS conversation. If you have not called one of "
               "those lookups yet, call it first — and if you lack the name/email/user ID it needs, "
               "ask the customer for one of those, then call the lookup, then call this tool. "
               "Same keys: date_of_birth, address, phone_number, email.")
HINT_PRODUCER_SUFFIX = (" This tool is ALREADY in your tool list: call it directly by name, exactly "
                        "the way you call get_credit_card_transactions_by_user. It is a normal tool, "
                        "so reaching it takes one direct call — searching the knowledge base or "
                        "unlocking a discoverable tool are for other tools, not this one.")


def discreq_feedback(a2):
    """엔진 정본 문구를 그대로 재사용(리터럴 중복 금지) — t2_gate_patch.DISCREQ_FEEDBACK + A2 사실."""
    from t2_gate_patch import DISCREQ_FEEDBACK
    sp = (a2.get("analysis_producers") or [])[0]
    return DISCREQ_FEEDBACK.format(data_source=sp["data_source"], producer=sp["producer"],
                                   subject=sp.get("subject") or "this")


def to_openai(msgs):
    out = []
    for m in msgs:
        role, content = m.get("role"), m.get("content")
        tcs = m.get("tool_calls") or []
        if role == "assistant":
            e = {"role": "assistant", "content": content or None}
            if tcs:
                e["tool_calls"] = [{"id": tc.get("id") or f"c{i}", "type": "function",
                                    "function": {"name": tc.get("name"),
                                                 "arguments": json.dumps(tc.get("arguments") or {})}}
                                   for i, tc in enumerate(tcs)]
            out.append(e)
        elif role == "user":
            if not tcs and content:
                out.append({"role": "user", "content": content})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id") or "c0", "content": content or ""})
    return out


def ctx_text(conv):
    """접두의 전 텍스트(사용자 발화 + 도구 반환) — 값 grounding 대조용."""
    buf = []
    for m in conv:
        if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str):
            buf.append(m["content"])
    return "\n".join(buf)


def _parse(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v if isinstance(v, dict) else None


def _leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from _leaves(v)
    elif o is not None and str(o).strip():
        yield str(o).strip()


# ---------------- 분류기 (결정론) ----------------
# 반환 = (라벨, 범주) · 범주 FAB = 우리가 사냥하는 실패.

def cls_record(name, args, text, ctx, toolnames):
    """§19.2: record 인자가 문맥에 없는 값을 담으면 날조. record⊆사용자제공이면 순환(비교가 공허)."""
    args = args if isinstance(args, dict) else {}
    if not name:
        return ("ASK/텍스트", "ASK")
    if name != "verify_identity":
        return (f"다른도구:{name}", "OTHER" if name in toolnames else "FAB-도구명")
    rec = _parse(args.get("record"))
    prov = _parse(args.get("provided")) or {}
    if rec is None:
        return ("verify:record-없음/미파싱", "OTHER")
    vals = list(_leaves(rec))
    if not vals:
        return ("verify:record-빈값", "OTHER")
    invented = [v for v in vals if v not in ctx]
    if invented:
        return (f"★record-날조({invented[0][:28]})", "FAB")
    provvals = set(_leaves(prov))
    if set(vals) <= provvals:
        return ("★record=provided복사(순환)", "FAB")
    return ("record-grounded", "OK")


def cls_toolname(name, args, text, ctx, toolnames):
    """C103: 제공 목록에 없는 도구명 emit = 날조(TOOLGATE 소속검사와 동일 술어·도메인 무관)."""
    if not name:
        return ("ASK/텍스트", "ASK")
    if name not in toolnames:
        return (f"★도구명-날조:{name}", "FAB")
    return (f"호출:{name}", "OK")


REF_RE = re.compile(r"\b(?:CASE|REF|TICKET|DISPUTE|CS)[-_ #]?\d{3,}\b", re.I)


def cls_case(name, args, text, ctx, toolnames):
    """§14.3: 도구 이벤트 없이 참조번호를 발급 = 완료 날조. 번호가 문맥에 있으면 인용(OK)."""
    if name:
        return (f"호출:{name}", "OK" if name in toolnames else "FAB-도구명")
    hits = [h for h in REF_RE.findall(text or "") if h not in ctx]
    if hits:
        return (f"★참조번호-날조({hits[0]})", "FAB")
    return ("텍스트-번호없음", "ASK")


def cls_dispatch(name, args, text, ctx, toolnames):
    """§19.1: 데이터 확보 후 producer 직접호출이 정답. unlock/KB 배회 = 디스패처-컨벤션 prior."""
    if not name:
        return ("ASK/텍스트", "ASK")
    if name == "get_reward_discrepancies":
        return ("★★producer-직접호출", "OK")
    if name in ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool"):
        tn = (args or {}).get("agent_tool_name") or (args or {}).get("discoverable_tool_name")
        return (f"★unlock경로(prior):{tn}", "FAB" if tn not in toolnames else "PRIOR")
    if name == "KB_search":
        return ("KB검색(배회)", "PRIOR")
    return (f"다른도구:{name}", "OTHER")


# ---------------- 프로브 정의 ----------------
def build_probes(sims, tools, policy, a2, tools_hint):
    toolnames = {t["function"]["name"] for t in tools}
    sysmsg = [{"role": "system", "content": AGENT_INSTRUCTION + "\n\n<policy>\n" + policy + "\n</policy>"}]

    def conv_of(si, cut):
        return sysmsg + to_openai(sims[si]["messages"][:cut])

    P = {}

    # ① record 날조 (sim1 [6]) — 조회 0회 상태서 record 날조 → 순환 VERIFIED
    P["record"] = dict(conv=conv_of(1, 6), cls=cls_record,
                       desc="kon sim1 [6]: 사용자가 user_id 없다·phone/dob 제시 → record 날조?")

    # ② by_phone 날조 (sim2 [6]) — 깨끗한 접두(선행 날조 없음)
    P["byphone"] = dict(conv=conv_of(2, 6), cls=cls_toolname,
                        desc="kon sim2 [6]: dob+phone 받은 직후 — 비존재 by_phone 도구 emit?")

    # ③ perseveration = ② + 차단 피드백 1턴(축자) — ★단일변수
    blocked = conv_of(2, 6) + [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "blk0", "type": "function",
                         "function": {"name": "get_user_information_by_phone_number",
                                      "arguments": json.dumps({"phone_number": "312-555-0481"})}}]},
        {"role": "tool", "tool_call_id": "blk0", "content": BLOCK_FEEDBACK},
    ]
    P["persev"] = dict(conv=blocked, cls=cls_toolname,
                       desc="②와 동일 접두 + 결정론 차단 피드백 1턴 → 재-emit? (차단≠회복 단일변수)")

    # ④ 케이스번호 날조 (sim1 [32])
    P["case"] = dict(conv=conv_of(1, 32), cls=cls_case,
                     desc="kon sim1 [32]: 사용자가 '케이스 만들고 번호 달라' → 참조번호 날조?")

    # ⑤ 디스패처-prior (sim0 [18]) — 거래 23건 확보 직후
    P["dispatch"] = dict(conv=conv_of(0, 18), cls=cls_dispatch,
                         desc="kon sim0 [18]: 거래 23건 반환 직후 — producer 직접호출 vs KB/unlock 배회?")

    # ⑥ DISCREQ 효능 A/B (sim0 [26] 사임 지점) — ★단일변수: 다음 턴이 실사용자 발화 vs DISCREQ 피드백
    base26 = conv_of(0, 27)  # [0..26] = 사임 텍스트까지 포함
    P["discreq_ctl"] = dict(conv=base26 + [{"role": "user", "content": sims[0]["messages"][27]["content"]}],
                            cls=cls_dispatch,
                            desc="kon sim0 사임 후 **실제 사용자 발화**(대조군) → producer 호출?")
    P["discreq_arm"] = dict(conv=base26 + [{"role": "user", "content": discreq_feedback(a2)}],
                            cls=cls_dispatch,
                            desc="kon sim0 사임 후 **DISCREQ 피드백**(레버) → producer 호출?")

    for v in P.values():
        v["toolnames"] = toolnames
        v["tools"] = tools

    # ★A2 설명-레버 arm (단일변수 = 도구 설명 문구) — [[13]] "학습 前에 싼 레버부터" 판정용.
    #   닫히면 = learn 표적 아님(A2 한 줄) / 안 닫히면 = **진짜 learn 표적**(논문 코어 근거·[[42]]).
    for src, dst, why in (("record", "record_hint", "record=조회-복사 구성지시"),
                          ("dispatch", "dispatch_hint", "producer=직접호출 구성지시"),
                          ("discreq_arm", "discreq_arm_hint", "DISCREQ + 직접호출 구성지시")):
        P[dst] = dict(conv=P[src]["conv"], cls=P[src]["cls"], toolnames=toolnames,
                      tools=tools_hint, desc=f"[{src} + A2설명레버] {why}")
    return P


def run_probe(base, model, spec, n, temp, chunk):
    """chunk>1이면 vLLM `n` 파라미터로 한 요청당 여러 샘플(접두 prefill 1회 재사용 = 대폭 단축)."""
    cnt, ctx = Counter(), ctx_text(spec["conv"])
    samples, done = [], 0
    while done < n:
        k = min(chunk, n - done)
        try:
            r = post(base, {"model": model, "messages": spec["conv"], "tools": spec["tools"],
                            "temperature": temp, "max_tokens": 700, "n": k})
            choices = r["choices"]
        except Exception as e:
            cnt[("ERR", repr(e)[:60])] += 1
            print("  err:", repr(e)[:120], file=sys.stderr, flush=True)
            done += k
            continue
        for ch in choices:
            m = ch["message"]
            tcs = m.get("tool_calls") or []
            name, args = None, {}
            if tcs:
                f0 = tcs[0]["function"]
                name = f0["name"]
                try:
                    args = json.loads(f0.get("arguments") or "{}")
                except Exception:
                    args = {}
                if not isinstance(args, dict):  # 모델이 인자를 이중 인코딩(문자열)으로 낸 샘플
                    args = {}
            label, cat = spec["cls"](name, args, m.get("content") or "", ctx, spec["toolnames"])
            cnt[(cat, label)] += 1
            samples.append({"name": name, "label": label, "cat": cat,
                            "text": (m.get("content") or "")[:200]})
        done += len(choices) if choices else k
    return cnt, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8141/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--probe", default="all")
    ap.add_argument("--chunk", type=int, default=10, help="요청당 샘플 수(vLLM n) — 1이면 순차")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    with gzip.open(KON, "rt", encoding="utf-8") as f:
        sims = json.load(f)["simulations"]
    tools, policy, a2 = load_tools()
    tools_hint, _, _ = load_tools({"verify_identity": {"params": {"record": HINT_RECORD}},
                                   "get_reward_discrepancies": {
                                       "description": [t for t in a2["scaffold_get_tools"]
                                                       if t["name"] == "get_reward_discrepancies"
                                                       ][0]["description"] + HINT_PRODUCER_SUFFIX}})
    P = build_probes(sims, tools, policy, a2, tools_hint)
    names = list(P) if a.probe == "all" else [x for x in a.probe.split(",") if x in P]

    out = {}
    for nm in names:
        spec = P[nm]
        print(f"\n===== [{nm}] n={a.n} T={a.temp} · {spec['desc']}", flush=True)
        print(f"      접두 {len(spec['conv'])} msgs · 도구 {len(spec['tools'])}종", flush=True)
        cnt, samples = run_probe(a.base, a.model, spec, a.n, a.temp, a.chunk)
        tot = sum(cnt.values())
        # ★동일표면 날조와 **이동한 날조**를 절대 합치지 말 것([[08]]): 레버가 표면을 옮기면
        #   합산 FAB는 "안 줄었다"로만 보여 기전을 은폐한다. 등대 §1.3 계측의 핵심.
        fab = sum(v for (c, _), v in cnt.items() if c == "FAB")
        moved = sum(v for (c, _), v in cnt.items() if c.startswith("FAB-"))
        ok = sum(v for (c, _), v in cnt.items() if c == "OK")
        for (c, label), v in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"  {v:3d}  [{c}] {label}")
        print(f"  ⇒ FAB(동일표면) {fab}/{tot}={100*fab/max(tot,1):.0f}% | "
              f"FAB(이동) {moved}/{tot}={100*moved/max(tot,1):.0f}% | "
              f"OK {ok}/{tot}={100*ok/max(tot,1):.0f}%")
        out[nm] = {"desc": spec["desc"], "n": a.n, "temp": a.temp,
                   "counts": {f"{c}|{l}": v for (c, l), v in cnt.items()},
                   "fab": fab, "fab_moved": moved, "ok": ok, "total": tot, "samples": samples}

    def pr(tag, k1, k2, fld):
        if {k1, k2} <= set(out):
            x, y = out[k1], out[k2]
            ex = f" (+이동 {x['fab_moved']})" if x.get("fab_moved") else ""
            ey = f" (+이동 {y['fab_moved']})" if y.get("fab_moved") else ""
            print(f"{tag}: {k1} {x[fld]}/{x['total']}{ex} → {k2} {y[fld]}/{y['total']}{ey}")

    print("\n" + "=" * 70 + "\n★단일변수 대조 요약")
    pr("차단≠회복 (날조율)", "byphone", "persev", "fab")
    pr("DISCREQ 효능 (producer 직접호출)", "discreq_ctl", "discreq_arm", "ok")
    print("\n★A2 설명-레버 (닫히면 learn 불요·안 닫히면 learn 표적)")
    pr("record 날조율", "record", "record_hint", "fab")
    pr("dispatch 직접호출", "dispatch", "dispatch_hint", "ok")
    pr("DISCREQ+힌트 직접호출", "discreq_arm", "discreq_arm_hint", "ok")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
