# -*- coding: utf-8 -*-
r"""x417 - WRONGARG(값 오선택) / MISSING(미실행) 이 **격리로 넘어가나** (사용자 지시 2026-08-19)

## 왜 ([[69]] §3 4단계 중 3단계 → [[62]] 0단계)
`x416` 로 실패 단위를 바로잡으니 18 실패 태스크가 **WRONGARG 7 · MISSING 7 · EXTRA 2 · ACTION 2** 였다.
KB 압축(wrap)은 그 중 033 하나에만 닿는다. 나머지 14(WRONGARG+MISSING)가 무엇으로 닫히는지를
**레버 짓기 전에** 격리로 잰다.

## 표적
실패 sim 의 gold **변이(mutating)** 액션 전량. 각 표적마다 gold 도구 이름과 gold 인자가 정답이다.

## 팔 (사전 고정)
    A_min    손님 요청 + 원장(호출 이력·등장 id)                  <- 재료 없음
    B_doc    + **정답 값이 든 KB 문서 슬라이스**(오라클 선택)      <- 상한
    C_neg    + 같은 길이의 **무관 KB 문서 슬라이스**               <- 부정통제 [[57]]
    D_live   + 라이브 꼬리 창(종료마커 앞)                        <- 라이브 부하 재현

⚠B_doc 는 **오라클**이다 — gold 값 문자열로 KB 를 찾아 잘랐다. 회수가 그걸 스스로 할 수 있다는 뜻이
아니라 *"재료가 코앞에 있을 때의 천장"* 이다(x408 A_slice 와 같은 지위).

## 지표 (두 개를 따로 센다 — 미실행과 값 오선택은 다른 결손이다)
    NAME_HIT  예측 도구 이름 == gold 도구 이름          <- MISSING 이 넘어가나
    ARGS_HIT  gold 인자 전 키가 값까지 일치             <- WRONGARG 이 넘어가나
    ARGS_PART gold 키 중 값이 일치한 비율               <- 어느 인자에서 갈리는지

## 사전 고정 해석
- `B_doc` 에서 ARGS_HIT 이 오르면 => 값은 **전달(문서 제시)** 로 닫힌다.
- `B_doc` 에서도 ARGS_HIT 이 낮으면 => 문서를 코앞에 줘도 못 만든다 = **계산·집계 결손**
  (093/094 형: 최고값 선택 + 종류 간 합산) => 그 단계에만 결정론 후보.
- `A_min ≈ C_neg` 여야 부정통제가 성립한다. `C_neg` 가 `B_doc` 만큼 오르면 문서가 원인이 아니다.
- NAME_HIT 이 A_min 에서 이미 높으면 => MISSING 은 지식 결손이 아니라 **국면/이행** 결손이다.
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

ENVERR = ("Error:", "NOT_VERIFIED", "not been given", "Unknown", "Invalid", "cannot be",
          "[READ-FIRST]", "blocked by a policy gate")
PAD = 700

SYS_ONE = ("You are the tool-calling module of a Rho-Bank support agent. "
           "Reply with ONE JSON object only: {\"tool\": \"<tool name>\", \"arguments\": {…}}. "
           "No prose, no markdown fence. Give the exact arguments.")


def mutating_set():
    d = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
    return {k for k, v in d["banking_knowledge"]["tools"].items() if v.get("mutates")}


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
    """gold 값 문자열이 들어 있는 KB 문서 조각(오라클)."""
    for _id, d in docs.items():
        t = d.get("content") or ""
        for n in needles:
            i = t.find(n)
            if i >= 0:
                return "[%s] %s" % (d.get("title", "")[:60], " ".join(
                    t[max(0, i - PAD):i + PAD].split()))
    return None


def other_slice(docs, avoid_title):
    for _id, d in docs.items():
        t = d.get("content") or ""
        if len(t) > 800 and d.get("title") and d["title"][:60] != avoid_title:
            return "[%s] %s" % (d["title"][:60], " ".join(t[:2 * PAD].split()))
    return ""


def parse_one(raw):
    t = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None, {}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None, {}
    if not isinstance(d, dict):
        return None, {}
    a = d.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return (str(d.get("tool") or "") or None), (a if isinstance(a, dict) else {})


def norm(v):
    s = str(v).strip()
    try:
        f = float(s)
        return ("%g" % f)
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

    MUT = mutating_set()
    docs = X.load_docs()
    TOOLS = X.tool_universe(docs)

    cases = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            ri = sim.get("reward_info") or {}
            if (ri.get("reward") or 0) >= 1.0:
                continue
            for g in C.gold_rows(sim):
                nm = g["name"]
                if nm not in MUT:
                    continue
                ga = flat(g["args"])
                if not ga or set(ga.keys()) <= {"agent_tool_name", "discoverable_tool_name"}:
                    continue
                needles = [str(v) for v in ga.values()
                           if not re.match(r"^(?:chk|sav|dbc|txn|cc|acc|btxn)_", str(v))
                           and len(str(v)) > 2]
                sl = doc_slice(docs, needles)
                cases.append({"task": F.task_id(sim), "trial": sim.get("trial"), "tool": nm,
                              "gold": ga, "slice": sl, "sim": sim, "matched": g["match"]})
    print("표적(변이 gold) %d건 · 고유 task/tool %d · 오라클 문서 확보 %d건"
          % (len(cases), len(set((c["task"], c["tool"]) for c in cases)),
             sum(1 for c in cases if c["slice"])))
    print("팔 4종 · n=%d · tail=%d · port %d" % (a.n, a.tail, a.port))

    jobs = []
    for c in cases:
        sim = c["sim"]
        calls_, ents = X.ledger_of(sim)
        led = ("지금까지 호출한 도구: %s\n대화에 등장한 레코드 id: %s"
               % (", ".join(calls_[:25]) or "(none)", ", ".join(ents[:30]) or "(none)"))
        head = ("# 호출 가능한 도구(정책 문서가 정의한 것)\n" + ", ".join(TOOLS) + "\n\n"
                + "# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n%s\n" % (X.user_ask(sim), led))
        ci = X.close_index(sim)
        win = (sim.get("messages") or [])[:ci][-a.tail:]
        convo = ("\n# 대화(꼬리 %d메시지)\n" % len(win)
                 + X.convo({"messages": win}, respect_close=False))
        doc = ("\n# 정책 문서(축자)\n" + c["slice"] + "\n") if c["slice"] else ""
        neg = "\n# 정책 문서(축자)\n" + other_slice(docs, (c["slice"] or "")[1:61]) + "\n"
        q = ("\n\n# 질문\n지금 시점에서 실행해야 할 **변이 도구 호출 하나**를 정하라. "
             "인자는 정확한 값으로 채워라. JSON 하나로만 답하라: "
             "{\"tool\": \"<name>\", \"arguments\": {…}}")
        arms = {"A_min": head + q, "C_neg": head + neg + q, "D_live": head + convo + q}
        if doc:
            arms["B_doc"] = head + doc + q
        for an, body in arms.items():
            for k in range(a.n):
                jobs.append({"c": c, "arm": an, "k": k, "temp": (0.0 if k == 0 else a.temp),
                             "msgs": [{"role": "system", "content": SYS_ONE},
                                      {"role": "user", "content": body}]})
    print("작업 %d건" % len(jobs))

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
            nm, ar = parse_one(raw)
            gold = j["c"]["gold"]
            keys = list(gold.keys())
            same = [k for k in keys if k in ar and norm(ar[k]) == norm(gold[k])]
            rec = {"task": j["c"]["task"], "trial": j["c"]["trial"], "tool": j["c"]["tool"],
                   "arm": j["arm"], "k": j["k"], "pred": nm,
                   "name_hit": bool(nm and nm == j["c"]["tool"]),
                   "args_hit": bool(nm == j["c"]["tool"] and len(same) == len(keys)),
                   "args_part": (len(same) / float(len(keys)) if keys else 0.0),
                   "missing_keys": [k for k in keys if k not in same][:6],
                   "raw": raw[:200]}
            with lock:
                out.append(rec)
                if len(out) % 50 == 0:
                    print("  ... %d/%d" % (len(out), len(out) + len(jobs)), flush=True)

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 팔별")
    print("%-8s %6s %10s %10s %10s" % ("arm", "n", "NAME_HIT", "ARGS_HIT", "ARGS_PART"))
    for arm in ("A_min", "C_neg", "B_doc", "D_live"):
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-8s %6d %10.2f %10.2f %10.2f"
              % (arm, len(r), sum(x["name_hit"] for x in r) / n,
                 sum(x["args_hit"] for x in r) / n, sum(x["args_part"] for x in r) / n))

    print("\n## 태스크별 ARGS_HIT (A_min / C_neg / B_doc / D_live)")
    for k in sorted(set((x["task"], x["tool"]) for x in out)):
        cells = []
        for arm in ("A_min", "C_neg", "B_doc", "D_live"):
            r = [x for x in out if (x["task"], x["tool"]) == k and x["arm"] == arm]
            cells.append("%.2f" % (sum(x["args_hit"] for x in r) / float(len(r))) if r else " -  ")
        nh = [x for x in out if (x["task"], x["tool"]) == k and x["arm"] == "B_doc"]
        miss = {}
        for x in nh:
            for mk in x["missing_keys"]:
                miss[mk] = miss.get(mk, 0) + 1
        print("  %-9s %-42s %s   B_doc 실패키: %s"
              % (k[0], k[1][:42], "  ".join(cells),
                 ", ".join("%s×%d" % (a_, b_) for a_, b_ in sorted(miss.items(), key=lambda z: -z[1])[:4])))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x417_write_isolation.json"))
    io.open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
