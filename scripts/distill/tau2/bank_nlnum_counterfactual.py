# -*- coding: utf-8 -*-
"""NLNUM 발화 3/3이 *전부* '맞는 산술'인가 — 전수 판정 (over-block 주장 검증·[[08]] (3)(4)).
KB 정본: 1 point = $0.01 (doc_credit_cards_credit_cards_(general)_006 원문).
발화 금액마다: 그 금액 = (문맥의 points 값)×0.01 인가? 대응 points가 문맥에 있으면 '맞는 산술'.
대조: t019d(다른 arm) 궤적서도 같은 패턴인가.
"""
import gzip, json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_MONEY_RE = re.compile(r"[$€£¥]\s?(\d[\d,]*\.\d{1,2})\b")


def _variants(num):
    s = num.replace(",", "")
    v = {s}
    try:
        v.add(repr(float(s)))
    except ValueError:
        pass
    t = s.rstrip("0").rstrip(".") if "." in s else s
    if t:
        v.add(t)
    return v


def audit(path, tag):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    print("#" * 72)
    print(f"# {tag}")
    for si, sim in enumerate(data.get("simulations") or []):
        msgs = sim.get("messages") or []
        ctx = ""
        fires = []
        for m in msgs:
            role, c = m.get("role"), m.get("content") or ""
            if role == "assistant" and isinstance(c, str) and c.strip():
                cn = ctx.replace(",", "")
                for mm in _MONEY_RE.finditer(c):
                    if not any(v in cn for v in _variants(mm.group(1))):
                        amt = mm.group(1).replace(",", "")
                        # 맞는 산술인가: amt*100 = 문맥의 points 값?
                        try:
                            pts = round(float(amt) * 100)
                        except ValueError:
                            pts = None
                        grounded = pts is not None and re.search(
                            r"\b%d\b" % pts, cn.replace(",", "")) is not None
                        fires.append((mm.group(0), pts, "맞는산술(points×0.01)" if grounded
                                      else "★근거불명"))
            if role in ("user", "tool") and isinstance(c, str):
                ctx += "\n" + c
        seen = set()
        uniq = [f for f in fires if not (f[0] in seen or seen.add(f[0]))]
        ok = sum(1 for f in uniq if f[2].startswith("맞는"))
        print(f"  sim{si}: 발화 고유금액 {len(uniq)} | 맞는산술 {ok} | 근거불명 {len(uniq)-ok}")
        for f in uniq:
            print(f"      {f[0]:>10s}  ->  points={f[1]}  {f[2]}")


B = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
audit(B + r"\bank_t019g_20260716.results.json.gz", "t019g (ASK게이트 arm)")
audit(B + r"\bank_t019d_20260716.results.json.gz", "t019d (대조·게이트 없음)")
