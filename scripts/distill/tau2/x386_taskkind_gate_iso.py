# -*- coding: utf-8 -*-
r"""x386 — **"선택 태스크인가"를 LLM이 가를 수 있는가**(VC 발화 범위 게이트의 분리력 측정).

## 왜 (오늘의 사슬)

`VERDICT_CARRY`(VC)는 **빼기 도구**다 — 후보 중 탈락시킬 것을 골라 선택을 돕는다. 그런데
C535 가 잰 대로 **고를 것이 없는 태스크**에서도 발화해 `후보 10 · OK 10 · VIOLATES 0`(무정보)을
낸다. 그리고 073 에서 그 무정보 발화가 **pass 를 죽였다**(같은 시드·노브 하나 차이:
ctl VC0 **1.0** ↔ vconly VC1 **0.0**).

⇒ 처방 후보 = **발화 범위 게이트**. 단 *"지금이 고르는 자리인가"* 를 **엔진이** 판단하면
[[66]] 위반이고, LLM 에게 **라벨만** 물으면 검산할 수가 없다(x378: 모델은 심은 이름을 20/25 로
그냥 따른다). ⇒ 허용된 형태 = **라벨 + 근거 인용**, 엔진은 ⑴닫힌 집합 소속 ⑵인용의 원문 실재만
본다(C45 동형·정규식 0·[[22]]·[[52]]).

## 이 프로브가 재는 것

그 게이트가 **두 무리를 실제로 가르는가**. 정답은 이미 붙어 있다 — 라이브가 낸 판정 줄이다:

    갈림(VIOLATES ≥1)  → 기대 라벨 **선택**       (후보가 실제로 걸러진 자리)
    무정보(VIOLATES 0) → 기대 라벨 **비선택**     (걸러진 게 없는 자리)

⚠이 정답은 **gold 가 아니라 라이브 산출물**이다 — gold 무참조([[23]]).

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    갈림에서 '선택' ≥10/12 ∧ 무정보에서 '비선택' ≥7/9 ∧ 인용 검산 ≥80%  → **게이트 설계 진행**
    한쪽만 충족                                                        → 편향된 게이트(그 방향 손실 계측 필요)
    둘 다 미달                                                          → 라벨 축 폐기 ⇒ A3 절차 선언(③)으로
    두 무리 라벨이 같음                                                 → **계기 무효**(결과 인용 금지)

사용: T2_PROBE_URL=http://localhost:8140/v1/chat/completions python x386_taskkind_gate_iso.py
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from x216_read_and_offset import chat                      # noqa: E402
import t2_search as TS                                     # noqa: E402

ROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGS = "/home/woori/scratch/logs"
TAGS = ["bank_t7313_treat_20260818h", "bank_t7312_treat_20260818g",
        "bank_t7310_treat_20260818e", "bank_t7314_treat_20260818j"]

# ⚠문면에 **도메인 어휘 0** — 상품군·클래스 이름은 한 글자도 안 넣는다([[05]]).
#   묻는 것은 손님 요청의 성격 하나이고, 근거는 손님 말에서만 온다.
PROMPT = (
    "Below are the customer's own messages in this conversation.\n\n{msgs}\n\n"
    "Question: is the customer asking the agent to CHOOSE one option from a set of offerings "
    "on file (for example, to pick which one to open, apply for, or recommend)?\n"
    "Answer with JSON only: {{\"kind\": \"CHOOSE\" or \"NOT_CHOOSE\", \"quote\": \"...\"}}\n"
    "The quote must be copied VERBATIM from the customer's messages above and must be the "
    "sentence that shows it. If the customer is not asking for a choice, quote the sentence "
    "that shows what they are asking for instead."
)


def sims_of(tag):
    p = os.path.join(ROOT, tag, "results.json")
    if not os.path.exists(p):
        return {}
    doc = json.load(io.open(p, encoding="utf-8"))
    return {str(s.get("task_id")): s for s in (doc.get("simulations") or doc.get("results") or [])}


def verdict_cuts(tag):
    """라이브 판정 줄 → (task, turn, VIOLATES 수). 사이드카 축자."""
    p = os.path.join(LOGS, "fb_%s.jsonl" % tag)
    out = []
    if not os.path.exists(p):
        return out
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("kind") != "verdict-lines":
            continue
        txt = str(r.get("text") or "")
        vio = sum(1 for l in txt.splitlines() if ": VIOLATES" in l)
        out.append({"tag": tag, "task": str(r.get("simtag", "")).split("#")[0],
                    "turn": r.get("turn"), "vio": vio})
    return out


def user_text(sim, turn):
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "user":
            continue
        ti = m.get("turn_idx", i)
        if turn is not None and ti is not None and int(ti) >= int(turn):
            break
        t = " ".join(str(m.get("content") or "").split())
        if t:
            out.append(t)
    return "\n\n".join(out)


def ask(msgs):
    raw = str((chat(PROMPT.format(msgs=msgs[:7000]), None, 0.0, 300) or {}).get("content") or "")
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        return None, ""
    try:
        d = json.loads(raw[i:j + 1])
    except Exception:
        return None, ""
    k = str(d.get("kind") or "").strip().upper()
    return (k if k in ("CHOOSE", "NOT_CHOOSE") else None), str(d.get("quote") or "")


def main():
    print("=" * 100)
    print("x386 · 선택/비선택 라벨의 분리력 (LLM 판단 + 인용 검산 · 엔진은 소속·실재만)")
    print("판정(사전 고정): 갈림 CHOOSE ≥10/12 ∧ 무정보 NOT_CHOOSE ≥7/9 ∧ 인용검산 ≥80% → 게이트 진행 · "
          "둘 다 미달 → 라벨 축 폐기(A3 선언으로) · 두 무리 같음 → 계기 무효")
    print("=" * 100)

    cuts, seen = [], set()
    for tag in TAGS:
        sims = sims_of(tag)
        for c in verdict_cuts(tag):
            k = (c["task"], c["turn"])
            if c["task"] not in sims or k in seen:
                continue
            seen.add(k)
            c["sim"] = sims[c["task"]]
            cuts.append(c)
    print("컷 %d개 (갈림 %d · 무정보 %d)"
          % (len(cuts), sum(1 for c in cuts if c["vio"] > 0), sum(1 for c in cuts if c["vio"] == 0)))
    print("")

    agg = collections.Counter()
    rows = []
    for c in sorted(cuts, key=lambda x: (x["task"], x["turn"] or 0)):
        msgs = user_text(c["sim"], c["turn"])
        if not msgs:
            continue
        kind, quote = ask(msgs)
        ok_q = bool(quote) and TS.quote_in(quote, msgs)
        want = "CHOOSE" if c["vio"] > 0 else "NOT_CHOOSE"
        hit = int(kind == want)
        agg[(want, "n")] += 1
        agg[(want, "hit")] += hit
        agg[("quote", "ok")] += int(ok_q)
        agg[("quote", "n")] += 1
        rows.append({"task": c["task"], "tag": c["tag"].split("_")[1], "turn": c["turn"],
                     "vio": c["vio"], "want": want, "kind": kind, "quote_ok": ok_q,
                     "quote": quote[:90]})
        print("  %-9s %-6s t%-3s VIO=%-2d 기대=%-10s 답=%-10s 인용검산=%s | %s"
              % (c["task"], c["tag"].split("_")[1], c["turn"], c["vio"], want,
                 kind or "(파싱실패)", "OK" if ok_q else "✗", quote[:52]))

    print("")
    print("## 집계")
    for w in ("CHOOSE", "NOT_CHOOSE"):
        print("  %-11s %d/%d" % (w, agg[(w, "hit")], agg[(w, "n")]))
    print("  인용검산      %d/%d" % (agg[("quote", "ok")], agg[("quote", "n")]))
    a, an = agg[("CHOOSE", "hit")], agg[("CHOOSE", "n")]
    b, bn = agg[("NOT_CHOOSE", "hit")], agg[("NOT_CHOOSE", "n")]
    qok = agg[("quote", "ok")] / max(1, agg[("quote", "n")])
    kinds = {r["kind"] for r in rows}
    if len(kinds - {None}) <= 1:
        v = "⛔**계기 무효** — 두 무리가 같은 라벨(결과 인용 금지)"
    elif an and bn and a >= 10 * an / 12.0 and b >= 7 * bn / 9.0 and qok >= 0.8:
        v = "**게이트 설계 진행** — 라벨이 두 무리를 가르고 인용도 검산된다"
    elif (an and a >= 10 * an / 12.0) or (bn and b >= 7 * bn / 9.0):
        v = "편향된 게이트 — 한쪽만 충족. 손실 방향을 계측해야 한다(거짓 비선택 = 055/063 상실)"
    else:
        v = "라벨 축 폐기 ⇒ **A3 절차 선언**(③)으로 간다"
    print("판정: %s" % v)
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "..", "reports", "facet_rft_2026",
                                        "x386_taskkind_gate.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"rows": rows, "verdict": v}, ensure_ascii=False, indent=1))
    print("원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
