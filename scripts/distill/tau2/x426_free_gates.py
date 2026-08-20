# -*- coding: utf-8 -*-
r"""x426 — **결정기를 짓기 전의 무료 게이트 2종** (사용자 지시 2026-08-20 · [[62]] ① 앞의 앞)

x423/x424 가 남긴 두 물음은 **모델을 안 불러도** 답할 수 있다. 답이 '아니오'면 결정기를 지어도 못 푼다.

    게이트 A (행 유일성)   대화에 손님이 말한 속성만으로 gold 행이 **유일하게** 갈리나
                           → 아니오면 `transaction_id`/`account_id`/`card_id` 는 formalize→filter 로도 못 푼다
    게이트 B (수 도달성)   배달된 숫자들의 **작은 산술 조합**으로 gold 수에 닿나
                           → 아니오면 093·094 의 결손은 계산이 아니라 **수집**이다

## 규율
★**게이트 A 는 신탁(oracle) 상한을 잰다.** 술어에 쓸 토큰을 *gold 행이 실제로 담고 있는 것* 중에서 고른다 —
  즉 **가장 좋은 형식화가 주어졌을 때** 유일해지는가를 묻는다. 이건 방법이 아니라 **천장**이다.
  천장이 낮으면 방법을 지을 이유가 없고, 높아도 그 천장을 실제로 칠지는 별도 측정이다([[62]]).
★엔진이 아니라 **포렌식**이므로 도메인 텍스트를 훑는다([[59]] 는 엔진 규칙이다). 그래도 판단은 안 한다 —
  토큰은 **형태**(날짜·금액·대문자구·인용구)로만 뽑고 뜻을 해석하지 않는다.
★gold 는 **채점·천장 정의**에만 쓴다. 어떤 처방도 여기서 나오지 않는다([[23]]).

사용: py -3 x426_free_gates.py [--max-cases 60]
"""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402
import x423_choice_isolation as I  # noqa: E402

WIN = 320                     # 후보 레코드로 볼 앞뒤 문자 수
NUM_POOL = 70                 # 게이트 B 가 쓸 최근 고유 숫자 수(탐색 폭 제한)

RE_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
RE_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
RE_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
RE_CAPS = re.compile(r"\b(?:[A-Z][a-z]+ ){1,3}(?:Account|Card|Rewards Card)\b")
RE_QUOTE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{3,40})[\"'“”‘’]")


def delivered(sim, upto):
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if i >= upto or m.get("role") != "tool":
            continue
        out.append(" ".join(str(m.get("content") or "").split()))
    return out


def customer_said(sim, upto):
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if i >= upto or m.get("role") != "user":
            continue
        out.append(" ".join(str(m.get("content") or "").split()))
    return " \n".join(out)


def customer_tokens(txt):
    """손님 발화에서 **형태로만** 뽑은 토큰(뜻 해석 0)."""
    t = set()
    for rx in (RE_DATE, RE_MONEY, RE_CAPS):
        t |= {m.group(0).strip() for m in rx.finditer(txt)}
    t |= {m.group(1).strip() for m in RE_QUOTE.finditer(txt)}
    t |= {m.group(0).replace(",", "") for m in RE_NUM.finditer(txt) if len(m.group(0)) >= 2}
    return {x for x in t if len(x) >= 2}


def candidates_like(gold, docs):
    """gold 와 **같은 접두사 꼴**의 토큰 전부 = 그 자리의 후보 행 id 들."""
    if "_" in gold:
        pre = gold.split("_", 1)[0] + "_"
        rx = re.compile(re.escape(pre) + r"[A-Za-z0-9_]+")
    else:
        rx = re.compile(r"\b" + re.escape(gold[:3]) + r"[A-Za-z0-9_]{2,}\b")
    out = set()
    for d in docs:
        out |= set(rx.findall(d))
    return out


def record_of(cand, docs):
    """후보가 등장한 자리의 앞뒤 문맥 = 그 행의 레코드(축자)."""
    buf = []
    for d in docs:
        for m in re.finditer(re.escape(cand), d):
            buf.append(d[max(0, m.start() - WIN):m.end() + WIN])
    return " ".join(buf)


def gate_a(maxn):
    print("=" * 104)
    print("게이트 A — 손님이 말한 속성으로 gold 행이 유일하게 갈리나 (신탁 상한 · LLM 0)")
    print("=" * 104)
    rows = []
    for c in I.cases(maxn):
        if not (c["arg"].endswith("_id") or c["arg"] in ("card_last_4_digits",)):
            continue
        docs = delivered(c["sim"], c["msg_i"])
        cand = candidates_like(c["gold"], docs)
        if c["gold"] not in cand:
            cand.add(c["gold"])
        cust = customer_tokens(customer_said(c["sim"], c["msg_i"]))
        recs = {x: record_of(x, docs) for x in cand}
        gtok = {t for t in cust if t in recs.get(c["gold"], "")}      # 신탁: gold 가 실제로 담은 토큰
        surv = [x for x in cand if all(t in recs.get(x, "") for t in gtok)] if gtok else sorted(cand)
        rows.append({"task": c["task"], "trial": c["trial"], "arg": c["arg"], "gold": c["gold"],
                     "n_cand": len(cand), "n_cust_tok": len(cust), "n_pred_tok": len(gtok),
                     "n_surv": len(surv), "unique_gold": len(surv) == 1 and surv[0] == c["gold"]})
        print("  %-9s t%s %-22s 후보 %3d · 손님토큰 %2d · 술어토큰 %2d → 생존 %3d %s"
              % (c["task"], c["trial"], c["arg"][:22], len(cand), len(cust), len(gtok), len(surv),
                 "✅유일=gold" if rows[-1]["unique_gold"] else ""))
    n = len(rows)
    if n:
        u = sum(x["unique_gold"] for x in rows)
        print("\n  ★신탁 상한: **%d/%d** 사례에서만 손님-발화 속성이 gold 행을 유일하게 지목한다 (%.0f%%)"
              % (u, n, 100.0 * u / n))
        print("  (생존 1 이 아닌 사례는 **최선의 형식화로도** 술어가 행을 못 가른다 ⇒ 그 자리는 결정기 밖)")
    return rows


def numbers_in(docs):
    seen, out = set(), []
    for d in reversed(docs):
        for m in RE_NUM.finditer(d):
            try:
                v = float(m.group(0).replace(",", ""))
            except Exception:
                continue
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= NUM_POOL:
                return out
    return out


def reachable(gold, pool, tol=1e-6):
    """gold 가 배달된 수들의 작은 산술 조합으로 나오나 → 도달 깊이(1·2·3) 또는 None."""
    for v in pool:
        if abs(v - gold) <= tol:
            return 1
    two = {}
    for i, a in enumerate(pool):
        for b in pool:
            for op, f in (("+", a + b), ("-", a - b), ("*", a * b),
                          ("/", (a / b) if b else None)):
                if f is None:
                    continue
                if abs(f - gold) <= tol:
                    return 2
                if abs(f) < 1e12:
                    two[f] = True
    keys = list(two)[:4000]
    for f in keys:
        for b in pool:
            for g in (f + b, f - b, f * b, (f / b) if b else None):
                if g is not None and abs(g - gold) <= tol:
                    return 3
    return None


def gate_b(maxn):
    import x424_formalize_calc as CC
    print("\n" + "=" * 104)
    print("게이트 B — 배달된 숫자의 작은 산술 조합으로 gold 수에 닿나 (LLM 0)")
    print("=" * 104)
    rows = []
    for c in CC.numeric_cases(maxn):
        docs = delivered(c["sim"], c["msg_i"])
        pool = numbers_in(docs)
        d = reachable(float(c["gold"]), pool)
        rows.append({"task": c["task"], "trial": c["trial"], "arg": c["arg"],
                     "gold": c["gold"], "n_pool": len(pool), "depth": d})
        print("  %-9s t%s %-24s gold %-10s 숫자풀 %2d → %s"
              % (c["task"], c["trial"], c["arg"][:24], c["gold"], len(pool),
                 ("깊이 %d 도달" % d) if d else "**도달 불가**"))
    n = len(rows)
    if n:
        ok = sum(1 for x in rows if x["depth"])
        print("\n  ★도달 가능 **%d/%d** (%.0f%%). 도달 불가 사례는 결손이 계산이 아니라 **수집**이다."
              % (ok, n, 100.0 * ok / n))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=60)
    a = ap.parse_args()
    A = gate_a(a.max_cases)
    B = gate_b(max(24, a.max_cases // 2))
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x426_free_gates.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump({"gate_a": A, "gate_b": B}, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
