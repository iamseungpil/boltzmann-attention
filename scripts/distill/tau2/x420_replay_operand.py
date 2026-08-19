# -*- coding: utf-8 -*-
r"""x420 - **재생(replay) operand 프로브**: 격리를 구성하지 말고 잘라낸다

## 왜 (x417·x419 연속 무효 후 · 2026-08-19)
두 번 다 [[18]] 정보-맞춘 격리를 어겼다. x417 은 **도구 스키마**를, x419 는 **대화 원문**을 뺐다.
둘 다 요약(`ledger_of`)으로 원문을 대신하려다 *"내 프롬프트에 값이 있었나"* 를 쟀다.
⇒ 이번엔 **그 sim 의 메시지를 표적 호출 직전까지 그대로 잘라** 준다. 정보 일치가 **구성상 보장**된다.

## 자름 지점
- 표적 도구를 **실제로 호출한 경우**: 그 호출 메시지 **직전**까지.
- **한 번도 호출 안 한 경우**: 종료마커 앞까지(대화 끝).

## 스모크의 판정선 (재현 검정)
`R_asis` 팔에서 모델의 답이 **라이브가 실제로 낸 인자**를 얼마나 재현하는가.
재현이 0에 가까우면 내 재생이 깨진 것이다 — 결손이 아니라 계기 결함. 전량 실행 금지.

## 팔
    R_asis     재생 그대로                     <- 라이브 재현(기준선)
    R_doc      + 정답 값이 든 문서 조각(오라클) <- 천장
    R_neg      + 무관 문서 조각                <- 부정통제 [[57]]

사용: py -3 x420_replay_operand.py --smoke 3 --port 8141
"""
import argparse
import io
import json
import os
import re
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402
import x395_compliance_iso as X  # noqa: E402
import x396_saying_vs_doing as C  # noqa: E402
import x397_pvi_channel as P  # noqa: E402

PAD = 700
MSG_CAP = 3000          # 메시지 1건 최대 문자 (잘리면 표시하고 계수한다)
TOTAL_CAP = 60000       # 프롬프트 총 문자 상한

SYS_ARG = ("You fill in the arguments for ONE named bank tool call. "
           "Reply with ONE JSON object only: {\"arguments\": {…}}. "
           "Use ONLY the parameter names given in the tool schema. No prose, no markdown fence.")


def env_tools():
    d = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
    return d["banking_knowledge"]["tools"]


def flat(a):
    a = F.norm_args(a)
    if isinstance(a, dict) and isinstance(a.get("arguments"), dict):
        a = a["arguments"]
    if isinstance(a, dict) and isinstance(a.get("arguments"), str):
        try:
            a = json.loads(a["arguments"])
        except Exception:
            pass
    return a if isinstance(a, dict) else {}


def render(msgs):
    """재생 렌더 — 메시지당 MSG_CAP, 총 TOTAL_CAP. 자른 건 세어서 돌려준다."""
    out, cut_msgs = [], 0
    for m in msgs:
        role = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        if len(c) > MSG_CAP:
            c = c[:MSG_CAP] + " …[TRUNCATED]"
            cut_msgs += 1
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if c:
                out.append("ASSISTANT: " + c)
            for tc in tcs:
                out.append("ASSISTANT_TOOL_CALL: %s %s"
                           % (F.nameof(tc),
                              json.dumps(F.argsof(tc), ensure_ascii=False, default=str)[:1200]))
        elif role == "user":
            out.append("CUSTOMER: " + c)
        elif role == "tool":
            out.append("TOOL_RESULT: " + c)
    txt = "\n".join(out)
    trimmed = False
    if len(txt) > TOTAL_CAP:
        txt = "…[EARLIER CONVERSATION OMITTED]…\n" + txt[-TOTAL_CAP:]
        trimmed = True
    return txt, cut_msgs, trimmed


def doc_slice(docs, needles):
    for _id, d in docs.items():
        t = d.get("content") or ""
        for n in needles:
            i = t.find(n)
            if i >= 0:
                return "[%s] %s" % (d.get("title", "")[:60],
                                    " ".join(t[max(0, i - PAD):i + PAD].split()))
    return None


def other_slice(docs, avoid):
    for _id, d in docs.items():
        t = d.get("content") or ""
        if len(t) > 900 and (d.get("title") or "")[:60] != avoid:
            return "[%s] %s" % (d["title"][:60], " ".join(t[:2 * PAD].split()))
    return ""


def parse_args_obj(raw):
    t = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    a = d.get("arguments", d)
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return None
    return a if isinstance(a, dict) else None


def norm(v):
    s = str(v).strip()
    try:
        return "%g" % float(s)
    except Exception:
        return s.lower()


def score(pred, ref):
    if pred is None or not ref:
        return 0.0, False
    same = [k for k in ref if k in pred and norm(pred[k]) == norm(ref[k])]
    return len(same) / float(len(ref)), len(same) == len(ref)


def build_cases(docs, TOOLS, MUT):
    cases = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            msgs = sim.get("messages") or []
            # 표적 도구 -> 실제 호출이 있던 메시지 index + 그 인자
            live = {}
            for i, m in enumerate(msgs):
                for tc in (m.get("tool_calls") or []):
                    a = F.argsof(tc)
                    nm = str(F.inner_name(a) or F.nameof(tc))
                    fa = flat(a)
                    if nm in MUT and fa and nm not in live:
                        live[nm] = (i, fa)
            for g in C.gold_rows(sim):
                nm = g["name"]
                if nm not in MUT or nm not in TOOLS:
                    continue
                ga = flat(g["args"])
                if not ga or set(ga.keys()) <= {"agent_tool_name", "discoverable_tool_name"}:
                    continue
                if nm in live:
                    cut, liveargs, kind = live[nm][0], live[nm][1], "CALLED"
                else:
                    cut, liveargs, kind = X.close_index(sim), None, "NEVER"
                cases.append({"task": F.task_id(sim), "trial": sim.get("trial"), "tool": nm,
                              "gold": ga, "live": liveargs, "kind": kind, "cut": cut,
                              "sim": sim})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", type=int, default=0, help=">0 이면 CALLED 표적 N개만")
    a = ap.parse_args()

    TOOLS = env_tools()
    MUT = {k for k, v in TOOLS.items() if v.get("mutates")}
    docs = X.load_docs()
    cases = build_cases(docs, TOOLS, MUT)
    called = [c for c in cases if c["kind"] == "CALLED"]
    print("표적 %d건 (CALLED %d · NEVER %d)" % (len(cases), len(called), len(cases) - len(called)),
          flush=True)
    if a.smoke:
        seen, pick = set(), []
        for c in called:
            k = (c["task"], c["tool"])
            if k in seen:
                continue
            seen.add(k)
            pick.append(c)
            if len(pick) >= a.smoke:
                break
        cases = pick
        print("★스모크: CALLED 표적 %d개만 — 판정선 = R_asis 가 **라이브 실제 인자**를 재현하는가"
              % len(cases), flush=True)

    jobs = []
    for c in cases:
        sim = c["sim"]
        spec = TOOLS[c["tool"]]
        body, ncut, trimmed = render((sim.get("messages") or [])[:c["cut"]])
        c["_ncut"], c["_trim"], c["_chars"] = ncut, trimmed, len(body)
        schema = ("# 도구 스키마(환경 선언)\n%s\n  파라미터: %s\n  설명: %s\n"
                  % (c["tool"], ", ".join(spec.get("args") or []),
                     " ".join(str(spec.get("desc") or "").split())[:400]))
        needles = [str(v) for v in c["gold"].values()
                   if not re.match(r"^(?:chk|sav|dbc|txn|cc|acc|btxn)_", str(v)) and len(str(v)) > 2]
        sl = doc_slice(docs, needles)
        head = "# 지금까지의 대화(원문 재생)\n" + body + "\n\n" + schema
        q = ("\n\n# 질문\n바로 지금 `%s` 를 호출한다. **인자를 정확한 값으로** 채워라. "
             "스키마의 파라미터 이름만 쓴다. JSON 하나로만: {\"arguments\": {…}}" % c["tool"])
        arms = {"R_asis": head + q,
                "R_neg": head + "\n# 정책 문서(축자)\n" + other_slice(docs, "") + "\n" + q}
        if sl:
            arms["R_doc"] = head + "\n# 정책 문서(축자)\n" + sl + "\n" + q
        for an, bd in arms.items():
            for k in range(a.n):
                jobs.append({"c": c, "arm": an, "k": k, "temp": (0.0 if k == 0 else a.temp),
                             "msgs": [{"role": "system", "content": SYS_ARG},
                                      {"role": "user", "content": bd}]})
    print("작업 %d건 · 프롬프트 중앙 %d자 · 메시지 절단 총 %d · 앞부분 생략 %d건"
          % (len(jobs), sorted(c["_chars"] for c in cases)[len(cases) // 2],
             sum(c["_ncut"] for c in cases), sum(1 for c in cases if c["_trim"])), flush=True)

    lock, out = threading.Lock(), []

    def work(_i):
        while True:
            with lock:
                if not jobs:
                    return
                j = jobs.pop(0)
            try:
                d = P.post(a.port, "/v1/chat/completions",
                           {"model": X.MODEL, "messages": j["msgs"],
                            "temperature": j["temp"], "max_tokens": 400})
                raw = d["choices"][0]["message"]["content"]
            except Exception as e:
                raw = "ERROR " + str(e)[:160]
            pr = parse_args_obj(raw)
            gp, ge = score(pr, j["c"]["gold"])
            lp, le = score(pr, j["c"]["live"] or {})
            with lock:
                out.append({"task": j["c"]["task"], "trial": j["c"]["trial"], "tool": j["c"]["tool"],
                            "kind": j["c"]["kind"], "arm": j["arm"], "k": j["k"],
                            "parsed": pr is not None,
                            "gold_part": gp, "gold_exact": ge,
                            "live_part": lp, "live_exact": le,
                            "raw": raw[:240]})
                if len(out) % 30 == 0:
                    print("  ... %d/%d" % (len(out), len(out) + len(jobs)), flush=True)

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 팔별")
    print("%-8s %6s %10s %10s %10s %10s %8s"
          % ("arm", "n", "GOLD_EXACT", "GOLD_PART", "LIVE_EXACT", "LIVE_PART", "PARSED"))
    for arm in ("R_asis", "R_neg", "R_doc"):
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-8s %6d %10.2f %10.2f %10.2f %10.2f %8.2f"
              % (arm, len(r), sum(x["gold_exact"] for x in r) / n,
                 sum(x["gold_part"] for x in r) / n, sum(x["live_exact"] for x in r) / n,
                 sum(x["live_part"] for x in r) / n, sum(x["parsed"] for x in r) / n))

    print("\n## 표적별")
    for k in sorted(set((x["task"], x["trial"], x["tool"]) for x in out)):
        r = [x for x in out if (x["task"], x["trial"], x["tool"]) == k]
        asis = [x for x in r if x["arm"] == "R_asis"]
        print("  %-9s t%-2s %-40s LIVE재현 %.2f · GOLD %.2f"
              % (k[0], k[1], k[2][:40],
                 (sum(x["live_part"] for x in asis) / len(asis)) if asis else 0.0,
                 (sum(x["gold_part"] for x in asis) / len(asis)) if asis else 0.0))
    print("\n## 실물 (최대 4)")
    for x in out[:4]:
        print("  [%s] %s" % (x["arm"], x["raw"][:200]))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x420_replay_operand%s.json" % ("_smoke" if a.smoke else "")))
    io.open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
