# -*- coding: utf-8 -*-
r"""x419 - **operand 선택만** 격리한다 (x417 계기 결함 수리판)

## x417 이 왜 무효였나 (2026-08-19 · 자기 감사)
1. **도구 스키마를 안 줬다** — 이름 목록만 줬더니 모델이 `user_income`·`required_credit_limit`
   같은 **없는 파라미터를 지어냈다**. 라이브 에이전트는 API 로 스키마를 받는다 ⇒ 정보 불일치([[18]]).
2. **한 답 vs N개 gold** — gold 쓰기가 5개인 sim 에서 "다음 호출 하나"를 물어 구조적으로 최대 1/N.
⇒ ARGS_HIT 0.00 은 결손이 아니라 내 결함이었다. 인용 금지.

## 이번 설계
표적 도구를 **지정하고 인자만** 묻는다. op 선택은 빼고 **operand 선택만** 잰다.
스키마(`env_surface.json` 의 `args`)를 준다 — 환경 선언이지 gold 가 아니다.

## 팔
    A_min    손님 요청 + 원장 + 스키마
    B_doc    + **정답 값이 든 KB 문서 슬라이스**(오라클 = 천장)
    C_neg    + 같은 길이의 무관 문서 슬라이스 (부정통제 [[57]])
    D_live   + 라이브 꼬리 창

## 지표
    EXACT     gold 인자 전 키가 값까지 일치
    PART      gold 키 중 값이 일치한 비율
    SCHEMA_OK 예측 인자 키가 **스키마 안**에 있나 (지어내기 여부)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tail", type=int, default=8)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    TOOLS = env_tools()
    MUT = {k for k, v in TOOLS.items() if v.get("mutates")}
    docs = X.load_docs()

    cases = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            for g in C.gold_rows(sim):
                nm = g["name"]
                if nm not in MUT or nm not in TOOLS:
                    continue
                ga = flat(g["args"])
                if not ga or set(ga.keys()) <= {"agent_tool_name", "discoverable_tool_name"}:
                    continue
                needles = [str(v) for v in ga.values()
                           if not re.match(r"^(?:chk|sav|dbc|txn|cc|acc|btxn)_", str(v))
                           and len(str(v)) > 2]
                cases.append({"task": F.task_id(sim), "trial": sim.get("trial"), "tool": nm,
                              "gold": ga, "slice": doc_slice(docs, needles), "sim": sim})
    print("표적 %d건 · 고유 task/tool %d · 오라클 문서 %d건"
          % (len(cases), len(set((c["task"], c["tool"]) for c in cases)),
             sum(1 for c in cases if c["slice"])), flush=True)

    jobs = []
    for c in cases:
        sim = c["sim"]
        spec = TOOLS[c["tool"]]
        schema = ("# 도구 스키마(환경 선언)\n%s\n  파라미터: %s\n  설명: %s\n"
                  % (c["tool"], ", ".join(spec.get("args") or []),
                     " ".join(str(spec.get("desc") or "").split())[:400]))
        calls_, ents = X.ledger_of(sim)
        led = ("지금까지 호출한 도구: %s\n대화에 등장한 레코드 id: %s"
               % (", ".join(calls_[:25]) or "(none)", ", ".join(ents[:30]) or "(none)"))
        head = ("# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n%s\n\n%s" % (X.user_ask(sim), led, schema))
        ci = X.close_index(sim)
        win = (sim.get("messages") or [])[:ci][-a.tail:]
        convo = ("\n# 대화(꼬리 %d메시지)\n" % len(win)
                 + X.convo({"messages": win}, respect_close=False))
        doc = ("\n# 정책 문서(축자)\n" + c["slice"] + "\n") if c["slice"] else ""
        neg = "\n# 정책 문서(축자)\n" + other_slice(docs, (c["slice"] or "")[1:61]) + "\n"
        q = ("\n\n# 질문\n위 도구 `%s` 를 지금 호출한다. **인자를 정확한 값으로** 채워라. "
             "스키마에 있는 파라미터 이름만 쓴다. JSON 하나로만: {\"arguments\": {…}}" % c["tool"])
        arms = {"A_min": head + q, "C_neg": head + neg + q, "D_live": head + convo + q}
        if doc:
            arms["B_doc"] = head + doc + q
        for an, body in arms.items():
            for k in range(a.n):
                jobs.append({"c": c, "arm": an, "k": k, "temp": (0.0 if k == 0 else a.temp),
                             "msgs": [{"role": "system", "content": SYS_ARG},
                                      {"role": "user", "content": body}]})
    print("작업 %d건" % len(jobs), flush=True)

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
            ar = parse_args_obj(raw)
            gold = j["c"]["gold"]
            keys = list(gold.keys())
            allowed = set(TOOLS[j["c"]["tool"]].get("args") or [])
            same = [k for k in keys if ar is not None and k in ar and norm(ar[k]) == norm(gold[k])]
            rec = {"task": j["c"]["task"], "trial": j["c"]["trial"], "tool": j["c"]["tool"],
                   "arm": j["arm"], "k": j["k"], "parsed": ar is not None,
                   "exact": bool(ar is not None and len(same) == len(keys)),
                   "part": (len(same) / float(len(keys)) if keys else 0.0),
                   "schema_ok": bool(ar is not None and set(ar.keys()) <= allowed),
                   "bad_keys": sorted(set((ar or {}).keys()) - allowed)[:5],
                   "miss": [k for k in keys if k not in same][:6],
                   "raw": raw[:220]}
            with lock:
                out.append(rec)
                if len(out) % 60 == 0:
                    print("  ... %d/%d" % (len(out), len(out) + len(jobs)), flush=True)

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 팔별")
    print("%-8s %6s %8s %8s %10s %10s" % ("arm", "n", "EXACT", "PART", "SCHEMA_OK", "PARSED"))
    for arm in ("A_min", "C_neg", "B_doc", "D_live"):
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-8s %6d %8.2f %8.2f %10.2f %10.2f"
              % (arm, len(r), sum(x["exact"] for x in r) / n, sum(x["part"] for x in r) / n,
                 sum(x["schema_ok"] for x in r) / n, sum(x["parsed"] for x in r) / n))

    print("\n## 태스크×도구별 EXACT (A_min / C_neg / B_doc / D_live)  · 최다 실패키")
    for k in sorted(set((x["task"], x["tool"]) for x in out)):
        cells = []
        for arm in ("A_min", "C_neg", "B_doc", "D_live"):
            r = [x for x in out if (x["task"], x["tool"]) == k and x["arm"] == arm]
            cells.append("%.2f" % (sum(x["exact"] for x in r) / float(len(r))) if r else " -  ")
        mk = {}
        for x in out:
            if (x["task"], x["tool"]) == k and x["arm"] in ("B_doc", "A_min"):
                for m_ in x["miss"]:
                    mk[m_] = mk.get(m_, 0) + 1
        print("  %-9s %-42s %s   실패키: %s"
              % (k[0], k[1][:42], "  ".join(cells),
                 ", ".join("%s×%d" % (p, q_) for p, q_ in sorted(mk.items(), key=lambda z: -z[1])[:4])))

    bad = [x for x in out if not x["schema_ok"] and x["parsed"]]
    print("\n## 스키마 밖 인자를 지어낸 응답 %d/%d" % (len(bad), len(out)))
    import collections
    for k, v in collections.Counter(b for x in bad for b in x["bad_keys"]).most_common(12):
        print("   %-34s %d" % (k[:34], v))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x419_operand_isolation.json"))
    io.open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
