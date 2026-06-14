#!/usr/bin/env python
"""P1c: ComplexFuncBench -> native OpenAI function-calling 궤적 (observe-then-use 2-hop 소스, 2026-06-15).

v7 진짜 gap = multi-turn observe-then-use(호출→*실제 응답 관찰*→출력서 값 추출→하류 arg).
ComplexFuncBench(zai-org·Booking.com API via RapidAPI·녹화 응답 포함)가 정확히 이 구조.
  예: Search_Car_Location → observation{latitude:32.873055} → Search_Car_Rentals(pick_up_latitude=32.873055)
= grounded fetch-then-use(Seal-Tools/TaskBench 단발-심볼형과 다름·v6이 불충분 입증한 그 갭).

소스 포맷: {id, conversations[{role:user|assistant(function_call[])|observation|assistant(content)}], functions[]}.
변환: functions→tools / assistant.function_call→tool_calls / observation→tool(녹화응답) / 최종 assistant.content→assistant.
  - parallel function_call: observation이 list-of-N이면 분할, 아니면 blob→첫 call·나머지 stub(native-FC 페어링 유지).
  - system 추가(소스에 없음). loss-mask=assistant 턴(_supervise).
★ToU: 논문(연구)용만. 특허/프로덕션은 clean 소스 재생성(설계서 §8c-BLOCKING#2).
★변환 후 fc_randomize_fetchable로 observe→arg 값 randomize(copy 강제·grounded fetch 학습).

Usage: fc_convert_complexfuncbench.py --in ComplexFuncBench.jsonl --out cfb.jsonl [--exclude_domain Flights] [--sample N]
"""
import argparse, json, re

SYS = "You are a tool-using assistant. Call the appropriate functions to fulfill the user request."


def to_tools(functions):
    out = []
    for f in functions:
        out.append({"type": "function", "function": {
            "name": f.get("name"), "description": f.get("description", ""),
            "parameters": f.get("parameters", {"type": "object", "properties": {}})}})
    return out


def convert(ex):
    convs = ex.get("conversations") or []
    funcs = ex.get("functions") or []
    if not convs or not funcs:
        return None
    tools = to_tools(funcs)
    msgs = [{"role": "system", "content": SYS}]
    cid = 0
    i = 0
    while i < len(convs):
        c = convs[i]
        role = c.get("role")
        if role == "user":
            msgs.append({"role": "user", "content": c.get("content", "") or ""})
            i += 1
        elif role == "assistant" and c.get("function_call"):
            calls = c["function_call"]
            tcs, ids = [], []
            for fc in calls:
                cid += 1
                tid = "call_%d" % cid
                ids.append(tid)
                args = fc.get("arguments", {})
                argstr = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                tcs.append({"id": tid, "type": "function",
                            "function": {"name": fc.get("name"), "arguments": argstr}})
            msgs.append({"role": "assistant", "content": c.get("content", "") or "",
                         "tool_calls": tcs, "_supervise": True})
            # 다음 observation 페어링
            obs = None
            if i + 1 < len(convs) and convs[i + 1].get("role") == "observation":
                obs = convs[i + 1].get("content")
                i += 2
            else:
                i += 1
            results = None
            if obs is not None:
                try:
                    parsed = json.loads(obs) if isinstance(obs, str) else obs
                    if isinstance(parsed, list) and len(parsed) == len(ids):
                        results = [json.dumps(x, ensure_ascii=False) for x in parsed]
                except Exception:
                    results = None
            obs_str = obs if isinstance(obs, str) else (json.dumps(obs, ensure_ascii=False) if obs is not None else None)
            for k, tid in enumerate(ids):
                if results:
                    content = results[k]
                elif k == 0 and obs_str is not None:
                    content = obs_str
                else:
                    content = "{}"
                msgs.append({"role": "tool", "tool_call_id": tid, "content": content})
        elif role == "assistant":
            cont = c.get("content", "") or ""
            if cont.strip():
                msgs.append({"role": "assistant", "content": cont, "_supervise": True})
            i += 1
        elif role == "observation":
            i += 1  # orphan(앞에 call 없음) 스킵
        else:
            i += 1
    if not any(m.get("tool_calls") for m in msgs):
        return None
    dom = re.sub(r"-\d+$", "", ex.get("id", "unk"))
    return {"tools": tools, "messages": msgs,
            "_meta": {"bench": "complexfuncbench", "id": ex.get("id"), "domain": dom}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_domain", default="Flights",
                    help="콤마구분 도메인 제외(기본 Flights=τ² airline 근접·BLOCKING#4). 빈문자=제외없음")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    excl = {d.strip() for d in a.exclude_domain.split(",") if d.strip()}
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    out, skip, exc = [], 0, 0
    for ex in rows:
        dom = re.sub(r"-\d+$", "", ex.get("id", "unk"))
        if dom in excl:
            exc += 1
            continue
        c = convert(ex)
        if c is None:
            skip += 1
        else:
            out.append(c)
    from collections import Counter
    dc = Counter(c["_meta"]["domain"] for c in out)
    calls = [sum(len(m.get("tool_calls", [])) for m in c["messages"] if m["role"] == "assistant") for c in out]
    print("input=%d  converted=%d  skip(no-call)=%d  excluded(%s)=%d" % (len(rows), len(out), skip, ",".join(excl), exc))
    print("domain mix:", dict(dc))
    print("avg calls/traj: %.2f" % (sum(calls) / max(len(out), 1)))
    if a.sample:
        for c in out[:a.sample]:
            print("\n=== %s (domain=%s) ===" % (c["_meta"]["id"], c["_meta"]["domain"]))
            for m in c["messages"][:8]:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    print("  [A] CALL", [(t["function"]["name"], t["function"]["arguments"][:70]) for t in m["tool_calls"]])
                elif m["role"] == "tool":
                    print("    [T]", str(m.get("content"))[:80])
                else:
                    print("  [%s] %s" % (m["role"], str(m.get("content"))[:80]))
    if not a.dry:
        with open(a.out, "w", encoding="utf-8") as f:
            for c in out:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
