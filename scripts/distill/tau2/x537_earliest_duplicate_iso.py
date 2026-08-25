# -*- coding: utf-8 -*-
r"""x537 — 085 격리: **"중복이 여럿이면 가장 이른 것"** 을 결정점에 되놓으면 고르는가 (무료·2026-08-25)

## 관측 (t7354 · `T2_SPEC_AT_WRITE` 가 인자-이름 결손을 닫은 뒤)

085 는 이제 분쟁을 **성사**시킨다(어제까지 유효 분쟁 0건). 두 sim 에서 재현된 산출:

    성사 #1  btxn_a1b2c3d4e501  atm_cash_discrepancy                  ← gold 축자 일치
    성사 #2  btxn_c3d4e5f6g703  duplicate_charge                      ← **늦은 중복을 골랐다**
    성사 #3  btxn_e5f6g7h8i905  atm_cash_discrepancy                  ← 초과
    성사 #4  btxn_f6g7h8i9j006  recurring_charge_after_cancellation    ← gold 축자 일치

⇒ 틀린 하나가 **중복 청구에서 늦은 거래를 고른 것**이다.

## 재료는 이미 대화 안에 있다 — 그것이 이 프로브의 전제다

지시 문장은 **env 가 보낸 문서 본문**에 축자로 들어 있다. 코드에 문서 id 를 적지 않는다
([[71]]②·[[05]]) — **궤적의 도구 결과에서 그 문장을 찾아** 쓴다. 못 찾으면 그 sim 은 건너뛰고
그 사실을 남긴다([[25]] 확인 안 한 것을 단언하지 않는다). 즉 이 프로브는
*"env 가 앞서 보낸 문장을 결정점에 되놓기"* 를 재는 것이고, 오늘 085 를 연 `T2_SPEC_AT_WRITE`
와 **같은 형태**다(그쪽은 도구 명세, 이쪽은 절차 규칙).

## 팔 ([[57]] 부정통제 포함)

    A_asis   분쟁 write 직전 창 그대로            ← 라이브 오답(늦은 중복)을 재현해야 한다
    B_rule   창 + **궤적에서 찾은 그 문장 축자**    ← 수리 후보
    N_len    창 + 궤적의 **다른 문장**(같은 길이대·이 결정과 무관) ← 길이가 아니라 내용임을 가른다

A_asis 가 오답을 재현 못 하면 격리가 불공정하고 **판정하지 않는다**([[62]] 2b).

## 채점 — 닫힌 술어·gold 미접촉

가장 이른 중복은 **원장에서** 정한다: `(amount, description)` 이 같은 묶음 중 날짜 최소.
해석 0·선택 0. gold 는 보지 않는다([[23]]) — 참고 병기도 하지 않는다.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x537_earliest_duplicate_iso.py --port 8141 --n 4
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

# ★규칙을 **궤적에서 찾는 닻**. 코드가 규칙을 쓰지 않는다 — env 문장을 찾아 그대로 옮긴다.
#   닻은 도메인 낱말이 아니라 이 결정의 **술어 이름**(중복·가장 이른)이다.
ANCHOR = re.compile(r"[^.\n]*\bearliest\b[^.\n]*\bduplicat\w*[^.\n]*\.", re.I)
NL = chr(10)
ASK = (NL + NL + "You are about to file a debit card dispute for a DUPLICATE CHARGE." + NL +
       "Reply with ONLY the transaction_id you will dispute - no prose, no JSON, no quotes.")

RE_ID = re.compile(r"^\s*transaction_id:\s*(\S+)\s*$", re.M)
RE_DT = re.compile(r"^\s*date:\s*(\S+)\s*$", re.M)
RE_AM = re.compile(r"^\s*amount:\s*(\S+)\s*$", re.M)
RE_DE = re.compile(r"^\s*description:\s*(.+?)\s*$", re.M)


def gen(port, body, maxtok=60):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def ledger_rows(msgs):
    """대화의 도구 결과에서 (id -> date, amount, desc). env 형식 그대로·판단 0."""
    out = {}
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        c = str(m.get("content") or "")
        if "Record ID:" not in c:
            continue
        for b in re.split(r"\n(?=\s*\d+\.\s+Record ID:)", c):
            i, dt, am, de = RE_ID.search(b), RE_DT.search(b), RE_AM.search(b), RE_DE.search(b)
            if i:
                out[i.group(1)] = (dt.group(1) if dt else "", am.group(1) if am else "",
                                   de.group(1) if de else "")
    return out


def earliest_duplicate(rows):
    """가장 큰 중복 묶음의 **가장 이른 id** — 원장만 본다.

    닫힌 술어: `(amount, description)` 동일 ∧ 2건 이상 → 날짜 최소. 해석·선택 0.
    """
    grp = {}
    for tid, (dt, am, de) in rows.items():
        if not dt or not am:
            continue
        grp.setdefault((am, de), []).append((dt, tid))
    best = None
    for v in grp.values():
        if len(v) >= 2:
            v.sort()
            if best is None or len(v) > best[0]:
                best = (len(v), v[0][1], [t for _, t in v])
    return (best[1], best[2]) if best else (None, [])


def find_rule(msgs):
    """궤적의 **도구 결과**에서 규칙 문장을 찾는다 — 지어내지 않는다. 못 찾으면 None."""
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        hit = ANCHOR.search(str(m.get("content") or ""))
        if hit:
            return hit.group(0).strip()
    return None


def find_filler(msgs, want_len):
    """같은 궤적의 **다른 문장** — 길이가 비슷하고 이 결정과 무관한 것(부정통제)."""
    best = None
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        for s in re.split(r"(?<=\.)\s+", str(m.get("content") or "")):
            s = s.strip()
            if not (30 < len(s) < 400) or "duplicat" in s.lower() or "earliest" in s.lower():
                continue
            d = abs(len(s) - want_len)
            if best is None or d < best[0]:
                best = (d, s)
    return best[1] if best else None


def windows():
    """(sim, 창, 규칙, 통제문장, 가장 이른 중복) — 전부 궤적 축자."""
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
            rule = find_rule(ms)
            early, dups = earliest_duplicate(ledger_rows(ms))
            cut = None
            for i, m in enumerate(ms):
                blob = json.dumps(m.get("tool_calls") or [], ensure_ascii=False)
                if TOOL in blob and "unlock" not in blob and "duplicate" in blob.lower():
                    cut = i
                    break
            why = None
            if rule is None:
                why = "규칙 문장이 이 궤적의 도구 결과에 **없다**"
            elif not early:
                why = "원장에 중복 묶음이 없다"
            elif cut is None:
                why = "중복-청구 분쟁 호출이 없다"
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
                          "earliest": early, "dups": dups})
    return cases, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(REP, "x537_earliest_duplicate_2026_08_25.json"))
    a = ap.parse_args(argv)
    cases, skipped = windows()
    for sk in skipped:
        print("건너뜀 %-22s %s" % (sk["sim"], sk["why"]))
    if not cases:
        print("창 0 — 판정하지 않는다")
        io.open(a.out, "w", encoding="utf-8").write(json.dumps(
            {"probe": "x537", "cases": 0, "skipped": skipped}, ensure_ascii=False, indent=1))
        return 1
    print("창 %d개" % len(cases))
    for c in cases:
        print("   %-22s 중복 %s · 가장 이른 것 %s" % (c["sim"], c["dups"], c["earliest"]))
        print("      규칙 축자: %s" % c["rule"][:150])
    rows, agg = [], {}
    for c in cases:
        arms = {"A_asis": c["win"], "B_rule": c["win"] + NL + NL + "[policy] " + c["rule"]}
        if c["filler"]:
            arms["N_len"] = c["win"] + NL + NL + "[policy] " + c["filler"]
        for arm, body in arms.items():
            for k in range(a.n):
                try:
                    txt = gen(a.port, body + ASK)
                except Exception as e:
                    txt = "!!%r" % (e,)
                m = re.search(r"btxn[a-z0-9_]+", txt)
                got = m.group(0) if m else None
                ok = (got == c["earliest"])
                rows.append({"sim": c["sim"], "arm": arm, "k": k, "got": got,
                             "earliest": c["earliest"], "ok": ok, "raw": txt[:120]})
                d0 = agg.setdefault(arm, {"n": 0, "ok": 0})
                d0["n"] += 1
                d0["ok"] += 1 if ok else 0
                print("%-7s %-22s k=%d got=%-22s %s" % (arm, c["sim"], k, got,
                                                        "★가장 이름" if ok else ""), flush=True)
    fair = agg.get("A_asis", {}).get("ok", 0) < agg.get("A_asis", {}).get("n", 1)
    out = {"probe": "x537", "date": "2026-08-25", "task": TASK,
           "rule_source": "궤적의 **도구 결과 본문**에서 닻(earliest+duplicate)으로 찾은 문장 축자. "
                          "코드에 문서 id 도 규칙 문구도 적지 않는다([[71]]②·[[05]]·[[23]]).",
           "scoring": "가장 이른 중복 = 원장에서 (amount, description) 동일 묶음의 날짜 최소. "
                      "gold 미접촉·닫힌 술어.",
           "agg": agg, "instrument_survives": fair,
           "instrument_note": ("A_asis 가 오답을 재현했다 — 격리가 공정하다" if fair else
                               "A_asis 가 전부 맞혔다 = 이 창은 라이브 실패를 재현 못 한다. "
                               "판정하지 않는다([[62]] 2b)."),
           "skipped": skipped, "rows": rows}
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n== agg ==")
    for k, v in agg.items():
        print("  %-7s %d/%d" % (k, v["ok"], v["n"]))
    print(out["instrument_note"])
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
