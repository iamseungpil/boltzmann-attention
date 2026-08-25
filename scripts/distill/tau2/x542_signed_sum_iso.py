# -*- coding: utf-8 -*-
r"""x542 - 074·072 의 남은 한 칸: **부호를 버리고 절댓값을 더한다** (무료·2026-08-26)

## 관측 (t7356 grpA1 trial0 · 도구 출력에서 직접 계산했다 · gold 미접촉으로 기전이 확정된다)

우리 도구 `get_atm_fee_discrepancies` 가 계좌마다 `difference` 를 **부호와 함께** 낸다.
모델이 제출한 크레딧 금액을 그 값들과 맞대면:

    purple  +2.50 +8.00 +10.50 +3.50 +2.50            부호합 27.00   제출 27.00  ← 음수 없음·유일하게 정확
    lb      +2.50+2.50+2.50+4.00+4.00+1.50, **-2.50** 부호합 14.50   제출 19.50  (= 17.00 + 2.50)
    dg      +1.50+2.00+4.00, **-1.00 -1.75**          부호합  4.75   제출 10.25  (=  7.50 + 2.75)
    ev      +1.50+1.50+3.50, **-1.00 -1.80**          부호합  3.70   제출  9.30  (=  6.50 + 2.80)

차이가 매번 정확히 `2 x |음수합|` 이고 **음수가 하나도 없는 계좌만 맞는다** — 자연 실험이
기전을 확정한다: 절댓값 합이다. 072 도 같은 자리다(부호합 3.50 ↔ 제출 6.50 = 2 x 1.50).

## 배달은 이미 됐다 - 그래서 이 프로브가 필요하다

선언(`return_template`)은 그 출력 안에서 두 번 말한다: *"a fee that is MISSING where one was due
(it shows as a **negative difference**)"* 와 *"the credit policy requires ONE fee_refund credit for
the **net correction** of THIS account"*. 라이브 전수 확인 - 네 계좌 출력 모두 len 1432~1582 로
**두 문장 다 온전히 들어 있다**. 즉 [[55]] 의 배관·문구 단계는 통과했다.

## 팔 - 무엇이 이 한 칸을 사는가

    A_asis  도구 출력 블록 **축자**                      <- 라이브 오답(절댓값 합)을 재현해야 한다
    B_fmt   같은 블록 · `$-2.50` -> `-$2.50` 만 고침      <- **우리 렌더링**이 범인인가
                                                          (`detail_item_template` 이 `${delta:.2f}`
                                                           라 `$` 뒤에 마이너스가 붙는다·[[25]])
    C_sign  블록 + 선언 낱말로 만든 한 줄(부호를 명시)     <- 저작이 필요한가
    D_both  B_fmt + C_sign
    N_len   블록 + 같은 길이 무관 문장                    <- 길이가 아니라 내용임을 가른다([[57]])

A_asis 가 절댓값 합을 재현 못 하면 격리가 불공정하고 **판정하지 않는다**([[62]] 2b).

## 채점 - 닫힌 술어 · gold 미접촉

정답은 **블록에서 파싱한 `difference` 값들의 부호 합**이다. 그 규칙의 출처는 선언 축자
(*"net correction"* · *"it shows as a negative difference"*)이고 gold 는 보지 않는다([[23]]).

사용: (리모트·cwd=scripts/distill/tau2) py -3 x542_signed_sum_iso.py --port 8141 --n 4
"""
import argparse
import collections
import gzip
import io
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F   # noqa: E402  (정본 파서 재사용·[[67]])

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
RUNS = ("bank_t7356_grpA1_20260826", "bank_t7356_grpA3_20260826")
NL = chr(10)
ANCHOR = "whose net charge does NOT match"
RE_DIFF = re.compile(r"difference \$(-?[\d.]+)")
RE_MONEY_NEG = re.compile(r"\$-([\d.]+)")
ASK = (NL + NL + "Reply with ONLY the dollar amount you will pass as the `amount` "
       "argument of the single fee_refund credit for this account - digits and a decimal "
       "point only, no dollar sign, no words.")
# ★C_sign 문면은 **선언에 있는 낱말로만** 만든다([[78]] 이식 대상은 선언 텍스트).
#   출처: return_template 의 "net correction" · "it shows as a negative difference".
SIGN_LINE = ("[policy] The credit is the NET correction: add the differences with their signs - "
             "a negative difference lowers the total, it does not raise it.")


def gen(port, body, maxtok=16):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def signed_sum(block):
    """정답 = 블록의 `difference $X` 를 **부호 그대로** 더한 값. 해석 0."""
    vals = [float(x) for x in RE_DIFF.findall(block)]
    return round(sum(vals), 2), vals


def fix_currency(block):
    """우리 렌더링만 고친다: `$-2.50` -> `-$2.50`. 값·순서·다른 글자 불변."""
    return RE_MONEY_NEG.sub(lambda m: "-$" + m.group(1), block)


def filler(msgs, want_len):
    best = None
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        for s in re.split(r"(?<=\.)\s+", str(m.get("content") or "")):
            s = s.strip()
            if not (60 < len(s) < 900) or "difference" in s or "$" in s:
                continue
            d = abs(len(s) - want_len)
            if best is None or d < best[0]:
                best = (d, s)
    return best[1] if best else None


def windows():
    """**라이브 창**을 쓴다 - 블록만 떼면 모델이 더하지 않는다(2026-08-26 1차 실측: A_asis 가
    10.50 을 냈고 라이브 오답은 19.50 이었다 ⇒ 그 창은 [[62]] 2b 의 공정성 조건을 못 지킨다).

    창 = 크레딧 호출 직전 W 메시지. 블록↔호출 짝짓기는 **닫힌 규칙**이다:
    그 호출의 `amount` 와 **절댓값 합이 일치하는** 블록이 그 계좌의 블록이다(라이브 오답의 정체가
    절댓값 합이라는 것은 도구 출력에서 이미 확정됐다). 짝을 못 찾으면 그 호출은 건너뛴다.
    """
    W = 14
    cases, skipped = [], []
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            ms = s.get("messages") or []
            sim = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            blks = []
            for i, m in enumerate(ms):
                if str(m.get("role")) == "tool" and ANCHOR in str(m.get("content") or ""):
                    c = str(m.get("content") or "")
                    tot, vals = signed_sum(c)
                    if vals:
                        blks.append((i, c, tot, vals, round(sum(abs(v) for v in vals), 2)))
            for i, m in enumerate(ms):
                for tc in (m.get("tool_calls") or []):
                    # ★정규식으로 blob 을 긁지 않는다 — 1차가 그렇게 하다 창 0 이 됐다.
                    #   정본 파서(`t2_forensic.argsof`)로 읽고 중첩 JSON 만 푼다([[67]]).
                    a = F.argsof(tc) or {}
                    inner = a.get("arguments")
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except Exception:
                            inner = {}
                    if not isinstance(inner, dict):
                        inner = {}
                    tool = str(a.get("agent_tool_name") or a.get("user_tool_name")
                               or a.get("discoverable_tool_name") or "")
                    if "apply_checking_account_credit" not in tool:
                        continue
                    amt = inner.get("amount", a.get("amount"))
                    try:
                        live = round(float(str(amt).replace("$", "").replace(",", "")), 2)
                    except Exception:
                        continue
                    hit = [b for b in blks if b[0] < i and abs(b[4] - live) < 0.01]
                    if not hit:
                        skipped.append({"sim": sim, "msg": i,
                                        "why": "amount %.2f 와 절댓값합이 맞는 블록이 없다" % live})
                        continue
                    bi, blk, tot, vals, absum = hit[-1]
                    if not any(v < 0 for v in vals):
                        skipped.append({"sim": sim, "msg": i, "why": "음수 없음(변별력 0)"})
                        continue
                    txt = []
                    for mm in ms[max(0, i - W):i]:
                        cc = str(mm.get("content") or "").strip()
                        if cc:
                            txt.append("[%s] %s" % (mm.get("role"), cc[:1600]))
                    if not txt:
                        skipped.append({"sim": sim, "msg": i, "why": "창이 비었다"})
                        continue
                    cases.append({"sim": sim, "tag": tag, "msg": i, "block": blk,
                                  "win": (NL + NL).join(txt), "want": tot, "vals": vals,
                                  "absum": absum, "live": live,
                                  "filler": filler(ms, len(blk))})
    return cases, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(REP, "x542_signed_sum_2026_08_26.json"))
    a = ap.parse_args(argv)
    cases, skipped = windows()
    for sk in skipped:
        print("건너뜀 %-22s msg%-4s %s" % (sk["sim"], sk["msg"], sk["why"]))
    if not cases:
        io.open(a.out, "w", encoding="utf-8").write(json.dumps(
            {"probe": "x542", "cases": 0, "skipped": skipped}, ensure_ascii=False, indent=1))
        print("창 0 - 판정하지 않는다")
        return 1
    print("창 %d개 (음수를 담은 계좌 · 라이브 호출에 짝지음)" % len(cases))
    for c in cases:
        print("   %-20s msg%-4s 라이브제출 %.2f · 절댓값합 %.2f · **부호합 %.2f** · 창 %d자"
              % (c["sim"], c["msg"], c["live"], c["absum"], c["want"], len(c["win"])))
    rows, agg = [], collections.defaultdict(lambda: {"n": 0, "ok": 0, "abs": 0})
    for c in cases:
        w = c["win"]
        arms = {"A_asis": w,
                "B_fmt": w.replace(c["block"], fix_currency(c["block"])),
                "C_sign": w + NL + NL + SIGN_LINE,
                "D_both": w.replace(c["block"], fix_currency(c["block"])) + NL + NL + SIGN_LINE}
        if c["filler"]:
            arms["N_len"] = w + NL + NL + "[policy] " + c["filler"]
        for arm, body in sorted(arms.items()):
            for k in range(a.n):
                try:
                    txt = gen(a.port, body + ASK)
                except Exception as e:
                    txt = "!!%r" % (e,)
                m = re.search(r"-?\d+(?:\.\d+)?", txt.replace(",", ""))
                got = round(float(m.group(0)), 2) if m else None
                ok = (got == c["want"])
                isabs = (got == c["absum"])
                rows.append({"sim": c["sim"], "msg": c["msg"], "arm": arm, "k": k,
                             "got": got, "want": c["want"], "absum": c["absum"],
                             "ok": ok, "abs_sum_error": isabs,
                             "live": c["live"], "live_match": (got == c["live"]),
                             "raw": txt[:60]})
                d0 = agg[arm]
                d0["n"] += 1
                d0["ok"] += 1 if ok else 0
                d0["abs"] += 1 if isabs else 0
                print("%-7s %-20s msg%-4s k=%d got=%-9s %s"
                      % (arm, c["sim"], c["msg"], k, got,
                         "★부호합" if ok else ("(절댓값합)" if isabs else "")), flush=True)
    live_hit = sum(1 for r in rows if r["arm"] == "A_asis" and r.get("live_match"))
    fair = live_hit > 0
    out = {"probe": "x542", "date": "2026-08-26",
           "scoring": "정답 = 블록의 `difference $X` 를 부호 그대로 더한 값. 규칙 출처는 선언 축자"
                      "('net correction' · 'it shows as a negative difference'). gold 미접촉.",
           "arms": {"A_asis": "도구 출력 축자", "B_fmt": "$-2.50 -> -$2.50 (우리 렌더링만)",
                    "C_sign": "선언 낱말로 만든 부호 명시 한 줄", "D_both": "B+C",
                    "N_len": "같은 길이 무관 문장([[57]])"},
           "agg": {k: dict(v) for k, v in agg.items()},
           "instrument_survives": fair,
           "instrument_note": ("A_asis 가 **라이브 제출값**을 재현했다 - 격리가 공정하다" if fair else
                               "A_asis 가 라이브 오답을 재현하지 못했다 = 이 창은 그 결정을 담지 "
                               "않는다. 판정하지 않는다([[62]] 2b)."),
           "skipped": skipped, "rows": rows}
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(NL + "== agg (ok = 부호합 · abs = 절댓값합 오답) ==")
    for k in sorted(agg):
        v = agg[k]
        print("  %-7s ok %d/%d · 절댓값합 %d" % (k, v["ok"], v["n"], v["abs"]))
    print(out["instrument_note"])
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
