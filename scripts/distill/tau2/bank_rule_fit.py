# -*- coding: utf-8 -*-
"""compute-like 필드가 결정론 규칙에 실제 적합하나 gold 전수 검증([[08]]).
 - provisional_credit_eligible: 날짜식(≤10일?) 적합률 = deterministic vs judgment 판정.
 - amount_difference: (exp-act)/100 × balance 적합률 (balance 추출).
 - customer_max_liability_amount: 기존 liability 규칙 재확인(baseline)."""
import json, glob, re, sys, io
from datetime import datetime
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def Nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}
fam = lambda n: re.sub(r"_\d+$", "", str(n))
def pd(s):
    for f in ("%m/%d/%Y", "%m/%d/%y", "%m/%d"):
        try: return datetime.strptime(str(s).strip()[:10], f)
        except Exception: pass
    return None
def amt(v):
    try: return round(abs(float(re.sub(r"[$,]", "", str(v)))), 2)
    except Exception: return None

# provisional: 여러 날짜-임계 규칙 시도 → 최적 적합률
prov = []       # (days, gold_bool)
liab = []       # (days, disputed, gold_liab)
adiff = []      # (exp, act, gold_diff)
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        for ac in ((s.get("reward_info") or {}).get("action_checks") or []):
            a = ac.get("action") or {}
            outer = Nd(a.get("arguments"))
            atn = outer.get("agent_tool_name", "")
            if not atn or "arguments" not in outer: continue
            ga = Nd(outer.get("arguments"))
            # provisional (debit)
            if "provisional_credit_eligible" in ga and ga.get("provisional_credit_eligible") is not None:
                td, dd = pd(ga.get("transaction_date")), pd(ga.get("discovery_date"))
                if td and dd:
                    prov.append(((dd - td).days, bool(ga.get("provisional_credit_eligible"))))
            # liability
            if ga.get("customer_max_liability_amount") is not None:
                td, dd = pd(ga.get("transaction_date")), pd(ga.get("discovery_date"))
                if td and dd:
                    liab.append(((dd - td).days, amt(ga.get("disputed_amount")), amt(ga.get("customer_max_liability_amount"))))
            # amount_difference
            if ga.get("amount_difference") is not None and ga.get("expected_apy") is not None and ga.get("actual_apy") is not None:
                adiff.append((float(ga["expected_apy"]), float(ga["actual_apy"]), amt(ga["amount_difference"])))

# provisional 규칙 탐색
print("=== provisional_credit_eligible: 날짜-임계 규칙 적합률 (n=%d) ===" % len(prov))
for thr in (2, 5, 10, 30, 60):
    # 규칙: days<=thr → True
    acc = sum(1 for dd, g in prov if (dd <= thr) == g) / max(len(prov), 1)
    print("  days<=%d→True : 적합 %.1f%%" % (thr, 100 * acc))
tr = sum(1 for _, g in prov if g); print("  gold True 비율: %.1f%%" % (100 * tr / max(len(prov), 1)))

# liability 재확인
print("\n=== customer_max_liability: 기존규칙(≤30→50·≤60→500·else disputed) 적합률 (n=%d) ===" % len(liab))
def liab_rule(days, disp): return 50.0 if days <= 30 else (500.0 if days <= 60 else disp)
acc = sum(1 for dd, disp, g in liab if liab_rule(dd, disp) == g) / max(len(liab), 1)
print("  적합 %.1f%%" % (100 * acc))

# amount_difference: balance 역산 후 일관성 (balance = diff/((exp-act)/100))
print("\n=== amount_difference = apy_gap × balance? balance 역산 일관성 (n=%d) ===" % len(adiff))
bals = []
for exp, act, gd in adiff:
    gap = (exp - act) / 100
    if gap and gd is not None:
        bals.append(round(gd / gap, 0))
bc = Counter(bals)
print("  역산 balance 분포 Top8:", dict(bc.most_common(8)))
print("  (깔끔한 round 값=산술규칙 확증·balance=account GET)")
