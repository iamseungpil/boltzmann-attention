# -*- coding: utf-8 -*-
r"""x538 - 085 격리: **Regulation E 책임 한도 표**를 결정점에 되놓으면 값이 표 안으로 들어오는가

## 왜 이 칸인가 (t7354 라이브 실측 · 정본 t2_forensic.action_diff 로 gold 행과 짝지음)

085 grpA1 t0 의 유일한 미달 행은 `file_debit_card_transaction_dispute_6281` 하나이고,
그 행의 인자 차이는 **아홉 칸이 표기, 한 칸이 판단**이다. 오늘 배선한 두 레버가 표기 아홉을
가져가면(불리언 5 = T2_WRITE_ARG_TYPE · 열거 4 = write_arg_enum 값 목록) **남는 것은 하나뿐**:

    customer_max_liability_amount   gold '50'  <->  제출 '0'

`'0'` 은 오답이기 전에 **선언된 집합 밖**이다. 정책 문서가 값을 세 개로 못박는다:

    doc_bank_accounts_bank_accounts_(general)_031 "Internal: Filing a Debit Card Transaction Dispute"
      - Reported within 2 business days of statement: Maximum liability $50
      - Reported within 60 days of statement: Maximum liability $500
      - Reported after 60 days: Unlimited liability - customer may not recover funds
    도구 문서: "Use -1 for unlimited liability."

## 이 프로브가 재는 것

*그 표가 결정점에 있으면 모델이 표 안의 값을 내는가.* 표는 **궤적의 도구 결과에서 찾아**
쓴다 - 코드에 문서 id 도 금액도 적지 않는다([[71]](2)·[[05]]). 못 찾으면 그 sim 은 건너뛰고
그 사실을 남긴다([[25]]).

## 팔 ([[57]] 부정통제 포함)

    A_asis   분쟁 write 직전 창 그대로              <- 라이브 오답('0' 같은 표 밖 값)을 재현해야 한다
    B_rule   창 + **궤적에서 찾은 그 표 축자**        <- 수리 후보
    N_len    창 + 궤적의 **다른 문장**(같은 길이대·무관)  <- 길이가 아니라 내용임을 가른다

A_asis 가 오답을 재현 못 하면 격리가 불공정하고 **판정하지 않는다**([[62]] 2b).

## 채점 - 닫힌 술어 · gold 미접촉

허용 집합은 **찾은 표에서 파싱한다**(코드 상수 아님): 달러 금액 전부 + 'unlimited' 가 있으면 -1.
답이 그 집합에 속하는지만 본다. 어느 티어가 옳은지는 판정하지 않는다 - 그것은 신고 시점을
읽는 일이고 모델 몫이다([[62]](3)(4)). gold 는 보지 않는다([[23]]).

사용: (리모트·cwd=scripts/distill/tau2) py -3 x538_liability_tier_iso.py --port 8141 --n 4
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
TASK = "task_085"
TOOL = "file_debit_card_transaction_dispute_6281"
RUNS = ("bank_t7354_grpA1_20260825", "bank_t7354_grpB2_20260825",
        "bank_t7348_halfB_20260824")

NL = chr(10)
# 닻은 도메인 상품명이 아니라 이 결정의 **술어 이름**이다: 신고 시점 -> 책임 한도.
RE_TIER = re.compile(r"^\s*[-*]?\s*Reported\b[^\n]*liability[^\n]*$", re.I | re.M)
# ★합성 검사용 닻 — 이미 A2 `write_rules` 에 실려 있는 문장(x537 이 산 것). 같은 결정점에
#   두 문장이 함께 실리면 서로 죽이는지를 본다([[19]] 합성-우선 · [[70]] 무엇을 파나).
RE_EARLY = re.compile(r"[^.\n]*\bearliest\b[^.\n]*\bduplicat\w*[^.\n]*\.", re.I)
RE_MONEY = re.compile(r"\$\s*([\d][\d,]*)")
ASK = (NL + NL + "You are filing this debit card dispute now." + NL +
       "Reply with ONLY the number you will pass as `customer_max_liability_amount` "
       "- digits only, no words, no dollar sign, no explanation.")

RE_ID = re.compile(r"^\s*transaction_id:\s*(\S+)\s*$", re.M)


def gen(port, body, maxtok=24):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def find_tiers(msgs):
    """궤적의 **도구 결과**에서 책임 한도 표를 찾는다 - 지어내지 않는다. 못 찾으면 None."""
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        lines = RE_TIER.findall(str(m.get("content") or ""))
        if len(lines) >= 2:
            return NL.join(x.strip() for x in lines)
    return None


def allowed_from(tiers):
    """허용 집합을 **찾은 표에서** 만든다. 코드 상수 0."""
    vals = set()
    for x in RE_MONEY.findall(tiers):
        vals.add(x.replace(",", ""))
    if re.search(r"unlimited", tiers, re.I):
        vals.add("-1")
    return vals


def find_filler(msgs, want_len):
    """같은 궤적의 **다른 문장** - 길이가 비슷하고 이 결정과 무관한 것(부정통제)."""
    best = None
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        for s in re.split(r"(?<=\.)\s+", str(m.get("content") or "")):
            s = s.strip()
            if not (30 < len(s) < 600):
                continue
            low = s.lower()
            if "liabilit" in low or "reported within" in low or "$" in s:
                continue
            d = abs(len(s) - want_len)
            if best is None or d < best[0]:
                best = (d, s)
    return best[1] if best else None


def live_value(msgs):
    """이 궤적이 실제로 보낸 값 - 재현 검사용(계기 생존). 판단 0."""
    got = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            blob = json.dumps(tc, ensure_ascii=False)
            if TOOL not in blob:
                continue
            try:
                args = tc.get("arguments") if isinstance(tc, dict) else {}
                inner = args.get("arguments") if isinstance(args, dict) else None
                if isinstance(inner, str):
                    inner = json.loads(inner)
                if isinstance(inner, dict) and "customer_max_liability_amount" in inner:
                    got.append(str(inner["customer_max_liability_amount"]))
            except Exception:
                pass
    return got


def windows():
    """(sim, 창, 표, 통제문장, 허용집합) - 전부 궤적 축자."""
    W = 12
    cases, skipped = [], []
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            if s.get("task_id") != TASK:
                continue
            ms = s.get("messages") or []
            sim = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            tiers = find_tiers(ms)
            cut = None
            for i, m in enumerate(ms):
                blob = json.dumps(m.get("tool_calls") or [], ensure_ascii=False)
                if TOOL in blob and "unlock" not in blob:
                    cut = i
                    break
            why = None
            if tiers is None:
                why = "책임 한도 표가 이 궤적의 도구 결과에 **없다**"
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
            early = None
            for m in ms:
                if str(m.get("role")) != "tool":
                    continue
                hit = RE_EARLY.search(str(m.get("content") or ""))
                if hit:
                    early = hit.group(0).strip()
                    break
            cases.append({"run": tag, "sim": sim, "win": (NL + NL).join(txt),
                          "tiers": tiers, "filler": find_filler(ms, len(tiers)),
                          "early": early,
                          "allowed": sorted(allowed_from(tiers)),
                          "live": live_value(ms)})
    return cases, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(REP, "x538_liability_tier_2026_08_25.json"))
    a = ap.parse_args(argv)
    cases, skipped = windows()
    for sk in skipped:
        print("건너뜀 %-22s %s" % (sk["sim"], sk["why"]))
    if not cases:
        print("창 0 - 판정하지 않는다")
        io.open(a.out, "w", encoding="utf-8").write(json.dumps(
            {"probe": "x538", "cases": 0, "skipped": skipped}, ensure_ascii=False, indent=1))
        return 1
    print("창 %d개" % len(cases))
    for c in cases:
        print("   %-22s 허용집합 %s · 라이브 제출 %s" % (c["sim"], c["allowed"], c["live"]))
        print("      표 축자: %s" % c["tiers"].replace(NL, " | ")[:200])
    rows, agg = [], {}
    for c in cases:
        arms = {"A_asis": c["win"],
                "B_rule": c["win"] + NL + NL + "[policy] " + c["tiers"]}
        if c["early"]:
            # 합성: A2 `write_rules` 가 **이미 싣고 있는** 문장과 함께 실었을 때.
            arms["B_both"] = (c["win"] + NL + NL + "[policy] " + c["early"] +
                              NL + "[policy] " + c["tiers"])
        if c["filler"]:
            arms["N_len"] = c["win"] + NL + NL + "[policy] " + c["filler"]
        for arm, body in sorted(arms.items()):
            for k in range(a.n):
                try:
                    txt = gen(a.port, body + ASK)
                except Exception as e:
                    txt = "!!%r" % (e,)
                m = re.search(r"-?\d[\d,]*", txt)
                got = m.group(0).replace(",", "") if m else None
                ok = got in c["allowed"]
                rows.append({"sim": c["sim"], "arm": arm, "k": k, "got": got,
                             "allowed": c["allowed"], "ok": ok, "raw": txt[:120]})
                d0 = agg.setdefault(arm, {"n": 0, "ok": 0})
                d0["n"] += 1
                d0["ok"] += 1 if ok else 0
                print("%-7s %-22s k=%d got=%-8s %s" % (arm, c["sim"], k, got,
                                                       "★표 안" if ok else ""), flush=True)
    fair = agg.get("A_asis", {}).get("ok", 0) < agg.get("A_asis", {}).get("n", 1)
    out = {"probe": "x538", "date": "2026-08-25", "task": TASK,
           "arg": "customer_max_liability_amount",
           "rule_source": "궤적의 **도구 결과 본문**에서 닻(Reported ... liability)으로 찾은 표 축자. "
                          "코드에 문서 id 도 금액도 적지 않는다([[71]](2)·[[05]]·[[23]]).",
           "scoring": "허용 집합 = 찾은 표에서 파싱한 달러 금액 + unlimited 이면 -1. "
                      "어느 티어가 옳은지는 판정하지 않는다. gold 미접촉.",
           "agg": agg, "instrument_survives": fair,
           "instrument_note": ("A_asis 가 표 밖 값을 냈다 - 격리가 공정하다" if fair else
                               "A_asis 가 전부 표 안이다 = 이 창은 라이브 실패를 재현 못 한다. "
                               "판정하지 않는다([[62]] 2b)."),
           "skipped": skipped, "rows": rows}
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(NL + "== agg ==")
    for k, v in sorted(agg.items()):
        print("  %-7s %d/%d" % (k, v["ok"], v["n"]))
    print(out["instrument_note"])
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
