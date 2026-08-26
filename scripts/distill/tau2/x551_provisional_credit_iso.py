# -*- coding: utf-8 -*-
r"""x551 - 040 격리: **가불 자격 규칙**을 결정점에 되놓으면 모델이 그 규칙을 적용하는가

## 왜 이 칸인가 (2026-08-26 · t7360 포렌식)

040 은 gold 9행을 **전부 실행**했고 6행이 **정확히 한 칸**만 다르다:

    eligible_for_provisional_credit    gold False  <->  제출 True   (6/6)

나머지 인자(transaction_id·card_action·card_last_4_digits·full_name…)는 전부 일치한다.
`[DECIDE-FIRST]` 축자가 이유를 말한다 - *"it answers exactly one thing - the 'dispute_reason'
argument of this call, **and no other argument**"* ⇒ 우리 결정 배달이 이 칸을 안 덮는다.

⛔**PM 핸드오프 §8-4 의 *"문서가 궤적에 안 온다"* 는 틀렸다.** 문서는 온다 -
`Provisional Credit Eligibility Guidelines (Internal)` 이 **msg3 에 전문으로** 실린다.
다만 그 메시지는 **31,591자**이고 규칙은 그 **6,819번째 글자**에 있으며, write 는 40 메시지
뒤다. 큐가 한 문장으로 적어 둔 그 축이다 - *"재료는 상류에 있고 결정점에 없다."*

## 이 프로브가 재는 것

*그 규칙이 결정점에 있으면 모델이 적용하는가.* 규칙은 **궤적의 도구 결과에서 찾아** 쓴다 -
코드에 문서 id 도 조항도 적지 않는다([[71]](2)·[[05]]). 못 찾으면 그 sim 은 건너뛰고 남긴다([[25]]).

## 채점 - 닫힌 술어 · **gold 미접촉**([[23]])

규칙의 기준2 는 자격 있는 분쟁 사유를 **열거**한다. 그 목록을 문서에서 파싱해서, 사유가
**목록 밖**인 행만 채점한다 - 그런 행의 정답은 문서만으로 `False` 로 닫힌다.
목록 **안**인 행은 나머지 기준(60일·$25·직전 분쟁 수)이 필요한데 그 재료가 창에 없으므로
**채점하지 않고 세어만 둔다**. gold 는 입력에도 채점에도 쓰지 않는다.

⇒ 이 설계에서 "정답"은 정책 문서의 논리적 귀결이지 gold 사본이 아니다(x538 과 같은 규율).

## 팔 ([[57]] 부정통제 포함)

    A_asis   분쟁 write 직전 창 그대로            <- 라이브 오답(True 일색)을 재현해야 한다
    B_rule   창 + **궤적에서 찾은 규칙 축자**       <- 수리 후보
    N_len    창 + 궤적의 **다른 문장**(같은 길이대) <- 길이가 아니라 내용임을 가른다

A_asis 가 오답을 재현 못 하면 격리가 불공정하고 **판정하지 않는다**([[62]] 2b).

사용: (리모트·cwd=scripts/distill/tau2) py -3 x551_provisional_credit_iso.py --port 8140
"""
import argparse
import gzip
import io
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TASK = "task_040"
TOOL = "file_credit_card_transaction_dispute"
RUNS = ("bank_t7360_smoke_20260826", "bank_t7356_grpB1_20260826",
        "bank_t7356_grpB3_20260826")

NL = chr(10)
# 닻은 상품명이 아니라 이 결정의 **술어 이름**이다: 가불 자격.
RE_DOC = re.compile(r"Provisional Credit Eligibility Guidelines.*?(?=\n\s*\d+\.\s+[A-Z]|\Z)",
                    re.S)
RE_REASON_ITEM = re.compile(r"^\s*[-*]\s*'([a-z_]+)'\s*$", re.M)
RE_CRIT2 = re.compile(r"Dispute Reason Category.*?(?=^\s*\d+\.\s)", re.S | re.M)
ASK = (NL + NL + "You are filing this credit card dispute now." + NL +
       "dispute_reason for this transaction is: {reason}" + NL +
       "Reply with ONLY one word - true or false - the value you will pass as "
       "`eligible_for_provisional_credit`. No explanation.")


def gen(port, body, maxtok=8):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def find_rule(msgs):
    """궤적의 **도구 결과**에서 자격 규칙 본문을 찾는다(코드에 조항 0)."""
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        c = str(m.get("content") or "")
        hit = RE_DOC.search(c)
        if hit and "Eligibility Criteria" in hit.group(0):
            return " ".join(hit.group(0).split())[:2600]
    return None


def eligible_reasons(rule):
    """기준2 가 **열거한** 자격 사유 집합 — 문서에서 파싱한다(상수 아님)."""
    if not rule:
        return set()
    seg = RE_CRIT2.search(rule)
    src = seg.group(0) if seg else rule
    out = set(re.findall(r"'([a-z_]{4,})'", src))
    return {x for x in out if x.endswith(("charge", "received", "charging",
                                          "described", "processed", "amount")) or "_" in x}


def find_filler(msgs, want_len):
    """같은 길이대의 **무관한** 궤적 문장 — 부정통제([[57]])."""
    best = None
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        c = " ".join(str(m.get("content") or "").split())
        if "provisional" in c.lower() or "eligib" in c.lower():
            continue
        for k in range(0, max(1, len(c) - want_len), max(1, want_len // 2)):
            seg = c[k:k + want_len]
            if len(seg) < want_len * 0.8:
                continue
            if best is None or abs(len(seg) - want_len) < abs(len(best) - want_len):
                best = seg
        if best is not None:
            break
    return best


def rows_of(sim):
    """gold 행의 (transaction_id, dispute_reason) — **사유만** 읽는다. 자격 칸은 안 본다."""
    out = []
    for c in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = (c.get("action") or {}).get("arguments") or {}
        if isinstance(a, dict) and isinstance(a.get("arguments"), str):
            try:
                a = json.loads(a["arguments"])
            except Exception:
                pass
        if not isinstance(a, dict) or "eligible_for_provisional_credit" not in a:
            continue
        out.append((str(a.get("transaction_id")), str(a.get("dispute_reason"))))
    return out


def windows():
    cases, skipped = [], []
    W = 12
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or d.get("results") or []):
            if s.get("task_id") != TASK:
                continue
            ms = s.get("messages") or []
            sim = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            rule = find_rule(ms)
            cut = None
            for i, m in enumerate(ms):
                blob = json.dumps(m.get("tool_calls") or [], ensure_ascii=False)
                if TOOL in blob and "unlock" not in blob:
                    cut = i
                    break
            why = None
            if rule is None:
                why = "자격 규칙이 이 궤적의 도구 결과에 **없다**"
            elif cut is None:
                why = "분쟁 호출이 없다"
            if why:
                skipped.append({"sim": sim, "run": tag, "why": why})
                continue
            txt = []
            for m in ms[max(0, cut - W):cut]:
                c = str(m.get("content") or "").strip()
                if c:
                    txt.append("[%s] %s" % (m.get("role"), c[:1500]))
            if not txt:
                skipped.append({"sim": sim, "run": tag, "why": "창이 비었다"})
                continue
            cases.append({"run": tag, "sim": sim, "win": (NL + NL).join(txt),
                          "rule": rule, "filler": find_filler(ms, len(rule)),
                          "rows": rows_of(s)})
    return cases, skipped


def parse(ans):
    a = str(ans or "").strip().lower()
    if a.startswith("true"):
        return True
    if a.startswith("false"):
        return False
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    a = ap.parse_args(argv)
    cases, skipped = windows()
    print("=" * 100)
    print("x551 — 040 가불 자격: 규칙을 결정점에 놓으면 적용하는가")
    print("=" * 100)
    print("창 %d개 · 건너뜀 %d개" % (len(cases), len(skipped)))
    for s in skipped:
        print("   skip %-28s %s" % (s["sim"], s["why"]))
    if not cases:
        print("⛔창이 없다 — 판정하지 않는다([[25]] 없음은 못 찾음과 다르다)")
        return 1

    tally = {}
    detail = []
    for c in cases:
        elig = eligible_reasons(c["rule"])
        scorable = [(t, r) for t, r in c["rows"] if r not in elig]
        unscored = [(t, r) for t, r in c["rows"] if r in elig]
        print("\n--- %s (%s) ---" % (c["sim"], c["run"]))
        print("  규칙 %d자 · 기준2 자격 사유 %s" % (len(c["rule"]), sorted(elig)))
        print("  채점 가능 행 %d (사유가 목록 **밖** → 정답 False) · 미채점 %d (목록 안)"
              % (len(scorable), len(unscored)))
        if not scorable:
            print("  ⛔채점 가능한 행이 없다 — 이 창은 판정하지 않는다")
            continue
        arms = {"A_asis": c["win"],
                "B_rule": c["win"] + NL + NL + "[policy] " + c["rule"]}
        if c["filler"]:
            arms["N_len"] = c["win"] + NL + NL + "[policy] " + c["filler"]
        else:
            print("  ⚠부정통제 문장을 못 찾았다 — N_len 생략(그 사실을 남긴다)")
        for arm, ctx in arms.items():
            ok = 0
            got = []
            for tid, reason in scorable:
                ans = gen(a.port, ctx + ASK.format(reason=reason))
                v = parse(ans)
                got.append("%s=%s" % (reason[:22], ans.strip()[:6]))
                if v is False:
                    ok += 1
            tally.setdefault(arm, [0, 0])
            tally[arm][0] += ok
            tally[arm][1] += len(scorable)
            print("  %-8s %d/%d   %s" % (arm, ok, len(scorable), " · ".join(got[:4])))
            detail.append({"sim": c["sim"], "arm": arm, "ok": ok, "n": len(scorable)})

    print("\n" + "=" * 100)
    print("합계 (사유가 자격 목록 밖 → 정책만으로 False 가 닫히는 행)")
    print("=" * 100)
    for arm in ("A_asis", "B_rule", "N_len"):
        if arm in tally:
            o, n = tally[arm]
            print("  %-8s **%d/%d**" % (arm, o, n))
    print("\n  판정 규칙: A_asis 가 낮고 B_rule 이 높고 N_len 이 A 수준이면 → **전달 결손**")
    print("             A_asis 가 이미 높으면 → 격리가 라이브를 재현 못 함 = 판정 보류([[62]] 2b)")
    print("             B_rule 도 낮으면 → 규칙이 결정점에 있어도 안 쓴다 = **능력 경계**")
    out = os.path.join(REP, "x551_provisional_credit_iso_2026_08_26.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"tally": tally, "detail": detail,
                             "skipped": skipped}, ensure_ascii=False, indent=1))
    print("\n  → %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
