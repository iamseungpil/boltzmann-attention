# -*- coding: utf-8 -*-
# ★Track A A①: 구성-변형 강건성 CI — k-계단(생성 프레임·k*=2)이 특정 구성의 우연이 아님을 보이는 분산 실험.
#
# temp0 결정론 → 반복 대신 구성-변형으로 분산 확보(Track A doc A① 스펙):
#   지시문 패러프레이즈 ×3 (원본 A2 inject_instructions + 의미보존 재서술 2종)
#   × 개입 브랜드-세트 ×3 (합성 무명 eco 브랜드 — 암기 confound 차단은 P2서 확정, 여기선 표면형 분산)
#   × 행 순서 ×3 (개입 행 순열·target 항상 마지막·k=2는 유일순열 2종만 = 동일프롬프트 중복 계상 방지)
#   × k∈{0,2,4}   (k=0은 패러프레이즈만 유효한 축 → 3셀)
# 판정: target(Patagonia·EcoCard·Green) base_rate==5 → 통과·그 외/파싱실패 → 실패. k별 실패율 + Wilson 95% CI.
# 프레임: 생성 프로브(P.run_cell 동일 경로·비프라이밍) — 로짓판독 스윕(k*=1)과 프레임 다름을 §에 명시.
# 실행: 리모트. setsid /home/woori/venvs/seka_env/bin/python -u a1_robustness_ci.py > log 2>&1 &
import os, sys, json, math
HERE = '/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2'
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding='utf-8')
import bank_shared_docs_probe as P

BASE = os.environ.get('A1_BASE', 'http://localhost:8140/v1')
MODEL = os.environ.get('A1_MODEL', 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8')
OUT_JSON = os.environ.get('A1_OUT', '/home/woori/scratch/a1_robustness_ci_20260719.json')

iso = P.load_iso_spec(); all_docs = P.load_docs()
docs = [d for d in all_docs if d['title'].startswith('EcoCard: ')]

# --- 패러프레이즈 3종: 의미 제약(상향적용·명시제외만 강등·category 권위·0 금지·promo 원파라미터·JSON 단일객체) 보존 ---
PARA = {'orig': iso['inject_instructions']}
PARA['para1'] = (
    "You work as a rewards analyst for the bank. First come ALL policy documents for the {group}, "
    "followed by the customer's transactions on that card. For EVERY transaction, provide:\n"
    "- base_rate: the cash-back RATE the documents assign to THIS transaction (a percent number such as 10, 5, 4, 1, "
    "or 0 only if it genuinely earns nothing). Use the rate the documents give for this transaction's CATEGORY and "
    "MERCHANT — this can be an ELEVATED/bonus rate when the card grants one for that category, or the standard rate.\n"
    "  ★When the documents grant a HIGHER rate for this transaction's category, you MUST apply that higher rate. "
    "Never drop back to the standard/base rate merely because you are uncertain the merchant qualifies — reduce to the "
    "standard rate ONLY when a document EXPLICITLY lists this specific merchant in an exclusion list, and in that case "
    "copy the exact excluding sentence into 'exclusion_quote' (otherwise leave it empty).\n"
    "  ★★The 'category' field on each transaction is the bank's OWN authoritative classification — treat it as final. "
    "If a category earns a bonus rate, that bonus rate applies to EVERY transaction in the category. Do NOT re-judge an "
    "individual transaction based on how many merchants you believe are 'partners', how big the partner network is, or "
    "whether YOU recognize the merchant — the bank already classified it. Only an explicit merchant name in an "
    "exclusion list can lower the rate.\n"
    "  ★Never answer 0 just because a purchase is outside a premium/bonus category — nearly every card pays a base "
    "rate on all other purchases; apply that base rate.\n"
    "- promo_mult, promo_window_months, promo_start, promo_end: if this card has a limited-time promo (for example "
    "double cash back for new customers), give its multiplier, window length in months, and the promo period dates "
    "(MM/DD/YYYY). With no promo, use promo_mult=1, promo_window_months=0, and empty dates.\n"
    "★Do NOT apply the promo, multiply anything, or decide whether the promo dates cover this transaction — a separate "
    "deterministic system does that from your parameters. Report only the base_rate and the raw promo parameters.\n\n"
    "=== {group} — POLICY DOCUMENTS ===\n{docs}\n\n=== TRANSACTIONS ===\n{items}\n\n"
    "Reply with EXACTLY one JSON object mapping each transaction_id to the fields above:\n{schema}")
PARA['para2'] = (
    "Role: bank rewards specialist. Below you will find ALL policy documents for the {group}, and after them the "
    "customer's transactions made with that card. Report, for EACH transaction:\n"
    "- base_rate: the cash-back RATE applying to THIS transaction (percent number, e.g. 10, 5, 4, 1, or 0 when it "
    "truly earns nothing). It is the rate the documents attach to this transaction's CATEGORY and MERCHANT — possibly "
    "an ELEVATED/bonus rate (where the card pays more for that category), otherwise the standard rate.\n"
    "  ★If a HIGHER rate is granted by the documents for this transaction's category, APPLY it. Uncertainty about "
    "whether the merchant qualifies is NOT a reason to fall back to the standard/base rate — downgrade ONLY when a "
    "document EXPLICITLY names this exact merchant in an exclusion list, and then quote that excluding sentence "
    "verbatim in 'exclusion_quote' (else leave empty).\n"
    "  ★★Treat the transaction's 'category' field as the bank's OWN final, authoritative classification of the "
    "purchase. A category that qualifies for a bonus rate qualifies for EVERY transaction inside it. Do NOT "
    "second-guess a single transaction using your guesses about partner counts, the size of the partner network, or "
    "your personal familiarity with the merchant — the bank has classified it already. The ONLY thing that lowers the "
    "rate is an explicit merchant name in an exclusion list.\n"
    "  ★Do NOT output 0 merely because the purchase is not in a premium/bonus category — almost every card earns a "
    "base rate on all remaining purchases; use it.\n"
    "- promo_mult, promo_window_months, promo_start, promo_end: when a limited-time promo exists for this card (e.g. "
    "double cash back for new customers), report its multiplier, window length in months, and promo period dates "
    "(MM/DD/YYYY). If none, promo_mult=1, promo_window_months=0, empty dates.\n"
    "★Never apply the promo, never multiply, never judge whether the promo dates cover the transaction — a separate "
    "deterministic system handles that from your parameters. Give only base_rate plus the raw promo parameters.\n\n"
    "=== {group} — POLICY DOCUMENTS ===\n{docs}\n\n=== TRANSACTIONS ===\n{items}\n\n"
    "Reply with EXACTLY one JSON object mapping each transaction_id to the fields above:\n{schema}")

BRANDSETS = {
    'setA': ['GreenLeaf Organics', 'EcoNest Home', 'Terra Verde Goods', 'PureCycle Apparel',
             'SolarBloom Garden', 'EarthKind Supply', 'VerdeMarket Co', 'BlueRoot Naturals'],
    'setB': ['FernValley Trading', 'OakRise Provisions', 'ClearSky Outfitters', 'MossPoint Mercantile',
             'RiverStone Goods', 'WildAcre Collective', 'SunGrove Market', 'CedarLoop Supply'],
    'setC': ['NimbusLeaf Co', 'HarborGreen Depot', 'PrairieWind Goods', 'StoneFern Outfitting',
             'BrightMeadow Shop', 'TrueNorth Naturals', 'QuietCreek Supply', 'GoldenRoot Traders'],
}
ORDERS = {2: {'id': [0, 1], 'rev': [1, 0]},
          4: {'id': [0, 1, 2, 3], 'rev': [3, 2, 1, 0], 'mix': [1, 3, 0, 2]}}
TGT = {"transaction_id": "txn_target01", "credit_card_type": "EcoCard", "merchant_name": "Patagonia",
       "transaction_amount": "$128.47", "transaction_date": "11/04/2025", "category": "Green",
       "rewards_earned": "128 points", "account_open": "07/01/2024"}

def syn(i, m):
    amt = 50.0 + 13.7 * i
    return {"transaction_id": "txn_syn%02d" % i, "credit_card_type": "EcoCard", "merchant_name": m,
            "transaction_amount": "$%.2f" % amt, "transaction_date": "10/%02d/2025" % (5 + i),
            "category": "Green", "rewards_earned": "%d points" % int(amt * 5), "account_open": "07/01/2024"}

def wilson(f, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = f / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

results = []
def cell(pname, bname, oname, k):
    iso2 = dict(iso); iso2['inject_instructions'] = PARA[pname]
    if k == 0:
        grows = [TGT]
    else:
        base_rows = [syn(i, BRANDSETS[bname][i]) for i in range(k)]
        grows = [base_rows[j] for j in ORDERS[k][oname]] + [TGT]
    try:
        out, plen = P.run_cell(BASE, MODEL, iso2, 'EcoCard', grows, docs, 0.0)
        v = (out.get(TGT['transaction_id']) or {}).get('base_rate')
    except Exception as e:
        v = 'ERR:%r' % str(e)[:60]
    try:
        ok = float(v) == 5.0
    except Exception:
        ok = False
    results.append({'para': pname, 'brands': bname, 'order': oname, 'k': k, 'target_rate': v, 'pass': ok})
    print('para=%-5s brands=%-4s order=%-3s k=%d -> rate=%-8s %s'
          % (pname, bname, oname, k, v, 'PASS' if ok else 'FAIL'), flush=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

for pname in PARA:
    cell(pname, '-', '-', 0)
for k in (2, 4):
    for pname in PARA:
        for bname in BRANDSETS:
            for oname in ORDERS[k]:
                cell(pname, bname, oname, k)

print('\n=== k별 실패율 (95% Wilson CI) ===')
for k in (0, 2, 4):
    cells = [r for r in results if r['k'] == k]
    nf = sum(1 for r in cells if not r['pass'])
    lo, hi = wilson(nf, len(cells))
    print('k=%d  fail %d/%d = %.2f  CI[%.2f, %.2f]' % (k, nf, len(cells),
          nf / len(cells) if cells else 0, lo, hi), flush=True)
print('DONE →', OUT_JSON, flush=True)
