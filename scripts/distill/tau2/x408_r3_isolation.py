# -*- coding: utf-8 -*-
r"""x408 - R3 격리 프로브: 이미 배달된 재료를 결정점에 세우면 그 단계가 계획에 뜨나

## 왜 ([[62]] 0단계 - 레버 짓기 전에 결손을 격리로 잰다)
x406 실측: 완전 무언급 65건 중 48건(74%)은 그 도구 이름이 이미 도구 결과 본문에 배달돼 있었고
48/48 전부 마지막 assistant 턴보다 앞서 도착했다. 그런데 대화에 그 단계가 등장하지 않았다.
결손이 (1)부하 (2)전달 위치 (3)이행 중 무엇인지를 격리로 갈라야 한다.

## 계기 선택
결손 정의가 "그 단계가 대화에 등장한 적이 없다" 이므로 단일 다음-수가 아니라
x395.SYS_PLAN(남은 호출 계획)이 맞는 계기다 - 표적이 계획 안에 있나를 센다.

## 팔 (전부 SYS_PLAN · 기권 선택지 없음 - x398 이 프로브 산물로 확정)
    C_neg    요청 + 원장                              <- 부정통제 [[57]]
    A_slice  요청 + 원장 + 배달된 슬라이스
    B_live   요청 + 원장 + 라이브 꼬리 창(종료마커 앞)  <- 라이브 부하 재현
    D_both   요청 + 원장 + 꼬리 창 + 슬라이스           <- 부하 두고 재제시만

## 사전 고정 해석 (결과 보기 전에 못 박는다)
- A_slice ~ C_neg 둘 다 높으면 => 슬라이스가 원인 아님. 라이브 결손은 이행/국면.
- A_slice >> C_neg 이고 A_slice >> B_live => 부하다. 레버는 전달뿐.
- D_both ~ A_slice => 부하를 안 줄여도 재제시만으로 닫힌다.
- A_slice 자체가 낮으면 => 코앞에 줘도 못 한다 => 그 단계에만 결정론 후보.
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

SLICE_PAD = 600


def r3_cases():
    """R3 = 호출 0회 · 이름/엔티티 무언급 · 그 이름이 도구 결과 본문에 배달된 미매치 gold."""
    out = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            msgs = sim.get("messages") or []
            body = " ".join(" ".join(str(m.get("content") or "").split())
                            for m in msgs if m.get("role") == "assistant" and m.get("content"))
            calls = C.called(sim)
            cks = ((sim.get("reward_info") or {}).get("action_checks") or [])
            for g in C.gold_rows(sim):
                if g["match"] or calls.get(g["name"]):
                    continue
                ops = C.operand_tokens(g["args"])
                if g["name"] in body or (ops and any(o in body for o in ops)):
                    continue
                sl = None
                for m in msgs:
                    if m.get("role") != "tool":
                        continue
                    c = " ".join(str(m.get("content") or "").split())
                    i = c.find(g["name"])
                    if i >= 0:
                        sl = c[max(0, i - SLICE_PAD):i + SLICE_PAD]
                        break
                if sl is None:
                    continue
                acc = set()
                for ck in cks:
                    a = ck.get("action") or {}
                    ar = a.get("arguments") or {}
                    nm = (ar.get("agent_tool_name") or ar.get("user_tool_name")
                          or ar.get("discoverable_tool_name") or a.get("name"))
                    if str(nm) == g["name"]:
                        acc |= X.accept_names(ck)
                out.append({"task": F.task_id(sim), "trial": sim.get("trial"), "tool": g["name"],
                            "type": g["type"], "accept": sorted(acc or set([g["name"]])),
                            "slice": sl, "sim": sim})
    return out


def parse_plan(raw):
    t = (raw or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t).strip()
    d = None
    try:
        d = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = None
    if not isinstance(d, dict):
        return None
    pl = d.get("plan")
    if not isinstance(pl, list):
        return None
    out = []
    for e in pl:
        if isinstance(e, dict):
            out.append(str(e.get("tool") or e.get("name") or ""))
        elif isinstance(e, str):
            out.append(e)
    return [x for x in out if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tail", type=int, default=8)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    docs = X.load_docs()
    TOOLS = X.tool_universe(docs)
    cases = r3_cases()
    print("R3 표적 %d건 (고유 task/tool %d) · 팔 4종 · n=%d · tail=%d · 포트 %d"
          % (len(cases), len(set((c["task"], c["tool"]) for c in cases)), a.n, a.tail, a.port))
    print("슬라이스 규칙: 그 도구 이름이 처음 등장한 도구 결과 본문 +-%d자 (축자)" % SLICE_PAD)
    print("시스템: x395.SYS_PLAN · 채점 = 표적 이름이 계획 리스트에 있나")

    jobs = []
    for c in cases:
        sim = c["sim"]
        calls_, ents = X.ledger_of(sim)
        led = ("지금까지 호출한 도구: %s\n대화에 등장한 레코드 id: %s"
               % (", ".join(calls_[:25]) or "(none)", ", ".join(ents[:25]) or "(none)"))
        head = ("# 호출 가능한 도구(정책 문서가 정의한 것)\n" + ", ".join(TOOLS) + "\n\n"
                + "# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n%s\n" % (X.user_ask(sim), led))
        ci = X.close_index(sim)
        win = (sim.get("messages") or [])[:ci][-a.tail:]
        convo = ("\n# 대화(꼬리 %d메시지)\n" % len(win)
                 + X.convo({"messages": win}, respect_close=False))
        slc = "\n# 이미 회수된 자료(축자)\n" + c["slice"] + "\n"
        q = ("\n\n# 질문\n남은 호출을 순서대로 계획하라. JSON 하나로만 답하라: "
             "{\"plan\": [{\"tool\": \"<name>\"}, ...]} (최대 8개)")
        arms = {"C_neg": head + q,
                "A_slice": head + slc + q,
                "B_live": head + convo + q,
                "D_both": head + convo + slc + q}
        for an, body in arms.items():
            for k in range(a.n):
                jobs.append({"c": c, "arm": an, "k": k, "temp": (0.0 if k == 0 else a.temp),
                             "msgs": [{"role": "system", "content": X.SYS_PLAN},
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
            pl = parse_plan(raw)
            acc = set(j["c"]["accept"])
            rec = {"task": j["c"]["task"], "trial": j["c"]["trial"], "tool": j["c"]["tool"],
                   "type": j["c"]["type"], "arm": j["arm"], "k": j["k"],
                   "parsed": pl is not None, "n_plan": len(pl or []),
                   "in_plan": bool(pl and (set(pl) & acc)),
                   "first_hit": bool(pl and pl[0] in acc), "raw": raw[:240]}
            with lock:
                out.append(rec)
                if len(out) % 40 == 0:
                    print("  ... %d/%d" % (len(out), len(out) + len(jobs)))

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 팔별 (계획에 표적 포함 / 계획 첫 원소 / 파싱 / 계획길이)")
    print("%-10s %6s %10s %10s %10s %8s" % ("arm", "n", "IN_PLAN", "FIRST", "PARSED", "LEN"))
    for arm in ("C_neg", "A_slice", "B_live", "D_both"):
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-10s %6d %10.2f %10.2f %10.2f %8.1f"
              % (arm, len(r), sum(x["in_plan"] for x in r) / n,
                 sum(x["first_hit"] for x in r) / n, sum(x["parsed"] for x in r) / n,
                 sum(x["n_plan"] for x in r) / n))

    def rt(arm):
        r = [x for x in out if x["arm"] == arm]
        return (sum(x["in_plan"] for x in r) / float(len(r)) if r else 0.0)

    print("\n## 사전 고정 대비")
    print("  A_slice - C_neg (슬라이스 효과) %+.2f" % (rt("A_slice") - rt("C_neg")))
    print("  A_slice - B_live (부하 효과)    %+.2f" % (rt("A_slice") - rt("B_live")))
    print("  D_both  - B_live (재제시 효과)  %+.2f" % (rt("D_both") - rt("B_live")))

    print("\n## 표적별 IN_PLAN (C_neg / A_slice / B_live / D_both)")
    for k in sorted(set((x["task"], x["trial"], x["tool"]) for x in out)):
        cells = []
        for arm in ("C_neg", "A_slice", "B_live", "D_both"):
            r = [x for x in out if (x["task"], x["trial"], x["tool"]) == k and x["arm"] == arm]
            cells.append("%.2f" % (sum(x["in_plan"] for x in r) / float(len(r))) if r else "-")
        print("  %-9s t%-2s %-44s %s" % (k[0], k[1], k[2][:44], "  ".join(cells)))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x408_r3_isolation.json"))
    io.open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
