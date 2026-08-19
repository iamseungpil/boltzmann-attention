# -*- coding: utf-8 -*-
r"""x398 — **조기 종결의 교락 배제**: 완료 신념이 어디서 오는지 요인 설계로 가른다.

## 왜 (사용자 지시 2026-08-19: "교락배제 3팔 돌려라")
`x395` 에서 도구를 안 낸 87건은 전부 **형식 정상 기권**(`{"tool": null, "reason": …}`)이었고
이유 문면은 *"have already been applied"* 였다. 그런데 그 기권을 만든 후보가 셋이고 **서로 섞여 있다**:

  ⒜ **우리가 기권을 허가했다** — 시스템 프롬프트가 `{"tool": null}` 을 명시 제공했다(프로브 산물 가능성)
  ⒝ **손님이 대화를 닫았다** — 창에 종결 마커(`###STOP###` 등)와 감사 발화가 들어 있었다(게이트: 12/12)
  ⒞ **모델이 자기 문장을 사실로 삼았다** — 앞 턴에서 *"적용했다"* 고 쓴 것이 창 안에 있다

세 팔을 따로 돌리면 상호작용을 못 본다 ⇒ **창 4종 × 시스템 2종 = 8팔 요인 설계**.

    창  W0_raw   원본 꼬리 8메시지(종결 포함) ← 재현 대조
        W1_noclose  종결 마커·감사 발화 메시지 제거
        W2_noclaim  어시스턴트 본문의 **완료 주장 문장**만 제거(나머지 문장은 유지)
        W3_both     둘 다 제거
    시스템  S_abst  기권 선택지 있음(원본) · S_noabst 기권 선택지 없음

## 사전 고정 해석 (결과 보기 전에 못 박는다)
- `S_noabst` 에서 기권이 사라지는데 **EMIT 이 안 오르면** ⇒ 기권은 프로브 산물이고 결손은 그대로다.
- `W1` 에서 EMIT 이 오르면 ⇒ **손님 종결 발화**가 원인(문맥 위생으로 닫힌다).
- `W2` 에서 오르면 ⇒ **자기 사실화**가 원인(자기 발화를 근거로 쓰는 것).
- 아무 데서도 안 오르면 ⇒ 세 교락 다 아니고 **이행/계획 결손**이 남는다.

⚠채점은 `x395.accept_names`(래퍼 정답 인정) 사용. 문장 삭제 규칙은 **축자 목록**이고 인쇄한다([[59]] 해석 0).

사용: py -3 x398_closure_confound.py [--n 3] [--tail 8] [--port 8141]
"""
import argparse
import collections
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
import x396_saying_vs_doing as C  # noqa: E402  (완료 주장 문형 정본 — 사본 금지 [[67]])
import x397_pvi_channel as P  # noqa: E402  (post/first_token_dist 재사용)

# 손님이 대화를 닫는 발화 — 축자 목록(해석 0). 종결 마커는 x395.CLOSE_MARKS.
THANKS = ("thank you", "thanks", "that's all", "that is all", "appreciate it",
          "perfect", "great, thanks", "no further", "nothing else", "we're done",
          "that works", "sounds good", "all set")

SYS_ABST = X.SYS                      # 원본: 기권 선택지 포함
SYS_NOABST = ("You are the tool-calling module of a Rho-Bank support agent. "
              "Reply with ONE JSON object only: {\"tool\": \"<tool name>\", \"arguments\": {…}}. "
              "No prose, no markdown fence. Name the single next tool call.")


def strip_closing(msgs):
    """종결 마커 메시지와 손님 종결 발화 메시지를 통째로 뺀다."""
    out = []
    for m in msgs:
        c = str(m.get("content") or "")
        low = " ".join(c.split()).lower()
        if any(k in c for k in X.CLOSE_MARKS):
            continue
        if m.get("role") == "user" and any(k in low for k in THANKS):
            continue
        out.append(m)
    return out


def strip_claims(msgs):
    """어시스턴트 본문에서 **완료 주장 문장만** 지운다(문장 단위·나머지 유지)."""
    out = []
    for m in msgs:
        if m.get("role") != "assistant" or not (m.get("content") or ""):
            out.append(m)
            continue
        keep = [s for s in re.split(r"(?<=[.!?])\s+", str(m["content"]))
                if not C.DONE_RE.search(" ".join(s.split()))]
        mm = dict(m)
        mm["content"] = " ".join(keep)
        out.append(mm)
    return out


def windows(sim, tail):
    msgs = sim.get("messages") or []
    return {
        "W0_raw": msgs[-tail:],
        "W1_noclose": strip_closing(msgs)[-tail:],
        "W2_noclaim": strip_claims(msgs)[-tail:],
        "W3_both": strip_claims(strip_closing(msgs))[-tail:],
    }


def classify(raw, accept):
    """EMIT / ABSTAIN / PROSE / MALFORMED — 규칙을 먼저 인쇄하고 이 규칙으로만 센다."""
    t = (raw or "").strip()
    if re.search(r'"tool"\s*:\s*null', t):
        return "ABSTAIN", None
    nm, _o = X.parse_tool(t)
    if nm:
        return ("EMIT_HIT" if nm in accept else "EMIT_MISS"), nm
    if t.startswith("{") or t.startswith("```"):
        return "MALFORMED", None
    return "PROSE", None


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
    cases = P.build_cases(docs)
    print("표적 %d · 창 4종 × 시스템 2종 = 8팔 · n=%d · tail=%d · 포트 %d"
          % (len(cases), a.n, a.tail, a.port))
    print("종결 마커(축자): %s" % ", ".join(X.CLOSE_MARKS))
    print("손님 종결 발화(축자 %d개): %s …" % (len(THANKS), ", ".join(THANKS[:6])))
    print("완료 주장 문형: x396.DONE_PAT %d개 (정본 재사용)\n" % len(C.DONE_PAT))

    jobs = []
    for c in cases:
        sim = c["sim"]
        calls_, ents = X.ledger_of(sim)
        led = ("지금까지 호출한 도구: %s\n대화에 등장한 레코드 id: %s"
               % (", ".join(calls_[:25]) or "(없음)", ", ".join(ents[:25]) or "(없음)"))
        proc = "\n".join("- " + s for s in c["lines"])
        tools = "# 호출 가능한 도구(정책 문서가 정의한 것)\n" + ", ".join(TOOLS) + "\n\n"
        q = ("\n\n# 질문\n지금 시점에서 **다음에 호출할 도구 하나**를 정하라. "
             "JSON 하나로만 답하라: {\"tool\": \"<이름>\", \"arguments\": {…}}")
        for wname, sub in windows(sim, a.tail).items():
            body = (tools + "# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n%s\n\n" % (X.user_ask(sim), led)
                    + "# 대화(%s · %d메시지)\n" % (wname, len(sub))
                    + X.convo({"messages": sub}, respect_close=False)
                    + "\n\n# 정책 절차(축자)\n" + proc + q)
            for sname, sysmsg in (("S_abst", SYS_ABST), ("S_noabst", SYS_NOABST)):
                for k in range(a.n):
                    jobs.append({"case": c, "w": wname, "s": sname, "k": k,
                                 "temp": (0.0 if k == 0 else a.temp),
                                 "msgs": [{"role": "system", "content": sysmsg},
                                          {"role": "user", "content": body}]})

    print("작업 %d건\n" % len(jobs))
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
            acc = set(j["case"].get("accept") or [j["case"]["tool"]])
            cls, nm = classify(raw, acc)
            with lock:
                out.append({"task": j["case"]["task"], "tool": j["case"]["tool"],
                            "w": j["w"], "s": j["s"], "k": j["k"], "cls": cls,
                            "pred": nm, "raw": raw[:300]})
                if len(out) % 24 == 0:
                    print("  … %d/%d" % (len(out), len(out) + len(jobs)))

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 창 × 시스템 (EMIT_HIT / ABSTAIN / PROSE 비율)")
    print("%-12s %-10s %6s %8s %8s %8s %8s" % ("창", "시스템", "n", "HIT", "MISS", "ABSTAIN", "PROSE"))
    for w in ("W0_raw", "W1_noclose", "W2_noclaim", "W3_both"):
        for s in ("S_abst", "S_noabst"):
            r = [x for x in out if x["w"] == w and x["s"] == s]
            if not r:
                continue
            n = float(len(r))
            cc = collections.Counter(x["cls"] for x in r)
            print("%-12s %-10s %6d %8.2f %8.2f %8.2f %8.2f"
                  % (w, s, len(r), cc["EMIT_HIT"] / n, cc["EMIT_MISS"] / n,
                     cc["ABSTAIN"] / n, cc["PROSE"] / n))

    print("\n## 주효과 (사전 고정 대비)")
    def rate(pred, cls="EMIT_HIT"):
        r = [x for x in out if pred(x)]
        return (sum(1 for x in r if x["cls"] == cls) / float(len(r)), len(r)) if r else (0.0, 0)
    for label, pa, pb in (
            ("기권 선택지 제거", lambda x: x["s"] == "S_noabst", lambda x: x["s"] == "S_abst"),
            ("손님 종결 제거", lambda x: x["w"] in ("W1_noclose", "W3_both"),
             lambda x: x["w"] in ("W0_raw", "W2_noclaim")),
            ("자기 완료문 제거", lambda x: x["w"] in ("W2_noclaim", "W3_both"),
             lambda x: x["w"] in ("W0_raw", "W1_noclose"))):
        (ha, na), (hb, nb) = rate(pa), rate(pb)
        (aa, _), (ab, _) = rate(pa, "ABSTAIN"), rate(pb, "ABSTAIN")
        print("  %-14s HIT %.2f(n=%d) ↔ %.2f(n=%d)  Δ=%+.2f | ABSTAIN %.2f ↔ %.2f  Δ=%+.2f"
              % (label, ha, na, hb, nb, ha - hb, aa, ab, aa - ab))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x398_closure_confound.json"))
    io.open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
