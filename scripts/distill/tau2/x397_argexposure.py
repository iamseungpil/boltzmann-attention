# -*- coding: utf-8 -*-
r"""x397 — G1-c: 인수 축자 노출률 · 팔 정보량 대조 (LLM 0 · 결정론 · 프롬프트 재구성만)

x395 의 케이스 선별·프롬프트 조립을 **그대로 재사용**([[67]] 사본 금지)하고,
gold 인자값(계측용)이 각 팔 프롬프트 본문에 축자로 있는지를 센다.
"""
import io, json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import t2_forensic as F
import x395_compliance_iso as X

ARMS = ["A_min", "B_full", "B_tail4", "B_tail8", "B_tail16", "B_tail32", "C_neg"]
IDRE = re.compile(r"\b((?:chk|sav|dbc|txn|cc|acc)_[A-Za-z0-9_]+)\b")


def gold_args_for(sim, tool):
    """그 표적 도구의 gold 호출 인자(계측용). action_checks 원형 그대로 반환."""
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        nm = str(ar.get("agent_tool_name") or ar.get("user_tool_name")
                 or ar.get("discoverable_tool_name") or a.get("name") or "")
        if nm != tool:
            continue
        inner = ar.get("arguments") if isinstance(ar.get("arguments"), dict) else None
        out.append({"raw_name": a.get("name"), "outer": ar, "inner": inner,
                    "match": ck.get("action_match")})
    return out


def flatten(v, prefix=""):
    """인수값을 (경로, 문자열값) 목록으로 평탄화."""
    res = []
    if isinstance(v, dict):
        for k, vv in v.items():
            res += flatten(vv, prefix + ("." if prefix else "") + str(k))
    elif isinstance(v, (list, tuple)):
        for i, vv in enumerate(v):
            res += flatten(vv, prefix + "[%d]" % i)
    elif isinstance(v, bool) or v is None:
        pass
    elif isinstance(v, (int, float)):
        res.append((prefix, repr(v) if isinstance(v, float) else str(v), "num"))
    else:
        s = str(v)
        if s.strip():
            res.append((prefix, s, "str"))
    return res


def build_cases(maxcases=14):
    docs = X.load_docs()
    TOOLS = X.tool_universe(docs)
    cases = []
    for tag in X.TAGS:
        for sim in F.scored(tag, X.SUF):
            rw = (sim.get("reward_info") or {}).get("reward")
            if (rw or 0) >= 1.0:
                continue
            gn, cn = X.gold_names(sim), X.called_names(sim)
            for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
                if ck.get("action_match"):
                    continue
                aa = ck.get("action") or {}
                ar = aa.get("arguments") or {}
                nm = str(ar.get("agent_tool_name") or ar.get("user_tool_name")
                         or ar.get("discoverable_tool_name") or aa.get("name") or "")
                if not nm or cn.get(nm):
                    continue
                pl = X.proc_lines(docs, nm)
                if not pl:
                    continue
                body = " ".join(" ".join(str(m.get("content") or "").split())
                                for m in (sim.get("messages") or []) if m.get("role") == "tool")
                reached = [s for s in pl if s.split("] ", 1)[-1][:55] in body]
                if not reached:
                    continue
                cases.append({"tag": tag, "task": F.task_id(sim), "trial": sim.get("trial"),
                              "tool": nm, "lines": pl, "reached": len(reached),
                              "gold": sorted(gn), "sim": sim})
    seen, uniq = set(), []
    for c in cases:
        k = (c["task"], c["tool"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return docs, TOOLS, uniq[:maxcases]


def build_prompts(c, TOOLS):
    """x395.main 과 **동일한** 조립(축자 복제 — 원본이 함수화돼 있지 않음)."""
    sim = c["sim"]
    calls_, ents = X.ledger_of(sim)
    ask = X.user_ask(sim)
    led = ("지금까지 호출한 도구: %s\n대화에 등장한 레코드 id: %s"
           % (", ".join(calls_[:25]) or "(없음)", ", ".join(ents[:25]) or "(없음)"))
    proc = "\n".join("- " + s for s in c["lines"])
    neg = X.FILLER * max(1, len(proc) // len(X.FILLER) + 1)
    neg = neg[:len(proc)]
    tools = "# 호출 가능한 도구(정책 문서가 정의한 것)\n" + ", ".join(TOOLS) + "\n\n"
    base = tools + ("# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n%s\n\n" % (ask, led))
    q = ("\n\n# 질문\n지금 시점에서 **다음에 호출할 도구 하나**를 정하라. "
         "JSON 하나로만 답하라: {\"tool\": \"<이름>\", \"arguments\": {…}}")
    P = {"A_min": base + "# 정책 절차(축자)\n" + proc + q,
         "C_neg": base + "# 안내\n" + neg + q,
         "B_full": tools + "# 대화 전문\n" + X.convo(sim) + "\n\n# 정책 절차(축자)\n" + proc + q}
    for nn in (4, 8, 16, 32, 64):
        P["B_tail%d" % nn] = (tools + "# 대화(마지막 %d 메시지)\n" % nn
                              + X.convo(sim, tail=nn) + "\n\n# 정책 절차(축자)\n" + proc + q)
    P["_parts"] = {"tools": tools, "ask": ask, "led": led, "proc": proc, "neg": neg,
                   "convo_full": X.convo(sim),
                   "convo_tail": {n: X.convo(sim, tail=n) for n in (4, 8, 16, 32, 64)}}
    return P


def main():
    docs, TOOLS, cases = build_cases()
    print("도구 우주 %d · 표적 %d" % (len(TOOLS), len(cases)))
    dump = []
    for c in cases:
        ga = gold_args_for(c["sim"], c["tool"])
        P = build_prompts(c, TOOLS)
        dump.append({"task": c["task"], "trial": c["trial"], "tool": c["tool"],
                     "gold_args": ga, "prompts": {a: P[a] for a in ARMS},
                     "parts": P["_parts"], "lines": c["lines"]})
        print("\n=== %s t%s %s" % (c["task"], c["trial"], c["tool"]))
        for g in ga:
            print("   raw_name=%s match=%s" % (g["raw_name"], g["match"]))
            print("   outer=%s" % json.dumps(g["outer"], ensure_ascii=False)[:600])
    io.open("/home/woori/scratch/x397_dump.json", "w", encoding="utf-8").write(
        json.dumps(dump, ensure_ascii=False, indent=1))
    print("\ndump: /home/woori/scratch/x397_dump.json")


if __name__ == "__main__":
    main()
