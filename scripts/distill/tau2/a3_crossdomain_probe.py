# -*- coding: utf-8 -*-
# ★Track A A③: 타-도메인 스팟체크 — 항공 수하물 요금 per-item 판단으로 조항-간섭 구조 재현.
#
# 정직 명시: τ² 공식 도메인(retail/telecom/airline)은 절차형 정책이라 문서-기반 per-item 판단 구조가 없음.
#   ⇒ A③는 **합성 도메인·현실적 정책**(항공 수하물 요금표)으로 banking과 동형 구조를 미러링:
#   조항 C(추론-결합): "certified adventure-sports outfitters" 장비 = 수수료 면제(0) / 표준 $35 /
#   명시 제외 리스트(MegaMart·ValueClub). target=유명 브랜드(문서에 미기재=추론 필요)·category=항공사 자체 분류.
#   표면 어휘(수하물/아이템/브랜드)는 banking과 전부 상이 — 도메인-일반성 검정 대상은 *구조*.
# 프레임: 3B 로짓판독(target-first 프라이밍·size_k_sweep 동형). 판정: P('0')(면제) vs P('3')(35 시작 토큰).
# 예측: k-계단(similar 적재서 P(0) 붕괴→35 후퇴) + 유사성-게이팅(dissimilar 유지).
# 실행: setsid /home/woori/venvs/seka_env/bin/python -u a3_crossdomain_probe.py > log 2>&1 &
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.stdout.reconfigure(encoding='utf-8')
import torch
torch.set_num_threads(int(os.environ.get('A3_THREADS', '24')))
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get('A3_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
OUT_JSON = os.environ.get('A3_OUT', '/home/woori/scratch/a3_crossdomain_20260720.json')

DOCS = """### SkyLift Airways: Checked Baggage Fee Schedule
Standard checked baggage fee: $35 per bag on all domestic routes. This standard fee applies to every checked
item that does not qualify for a category waiver below.

### SkyLift Airways: Sports & Adventure Equipment Waiver
Checked sports and adventure equipment qualifies for a full fee waiver (fee $0). Qualifying items are sports
and adventure gear from certified adventure-sports outfitters and pro shops, including skis, snowboards,
surfboards, diving gear, climbing equipment, and camping gear. The airline's baggage system tags qualifying
items with the category "Sports". The waiver applies per item, on all domestic routes, with no annual limit.

### SkyLift Airways: Waiver Exclusions
The Sports & Adventure Equipment Waiver does NOT apply to items purchased from the following general-merchandise
retailers, regardless of item type: MegaMart, ValueClub, BulkBarn Wholesale. Items from these retailers are
charged the standard checked baggage fee. No other merchants are excluded."""

INSTR = ("You are an airline baggage-fee specialist. Below are ALL fee policy documents for SkyLift Airways, "
         "then the passenger's checked items for today's flight. For EACH item report:\n"
         "- fee: the checked-bag FEE in dollars that applies to THIS item (number: e.g. 35 or 0). This is the "
         "fee the documents assign to this item's CATEGORY and BRAND/RETAILER — it may be a WAIVED fee (0) when "
         "the policy grants a category waiver, or the standard fee.\n"
         "  ★If the documents grant a WAIVER for this item's category, APPLY the waiver (fee 0). Do NOT "
         "fall back to the standard fee just because you are unsure whether the brand qualifies — charge the "
         "standard fee ONLY when a document EXPLICITLY names this item's retailer in an exclusion list, and "
         "then copy that exact excluding sentence into 'exclusion_quote' (else leave empty).\n"
         "  ★★The item's 'category' field is the airline's OWN authoritative baggage-system "
         "classification — treat it as final. If a category qualifies for a waiver, apply the waiver to EVERY "
         "item in that category. Do NOT second-guess an individual item using how many outfitters you think "
         "are 'certified', or whether YOU personally recognize the brand — the airline has already classified "
         "it. The ONLY thing that removes the waiver is an explicit retailer name in an exclusion list.\n\n"
         "=== SkyLift Airways — FEE POLICY DOCUMENTS ===\n{docs}\n\n=== CHECKED ITEMS ===\n{items}\n\n"
         "Reply with EXACTLY one JSON object mapping each item_id to the fields above:\n{schema}\n\n"
         "(List the item_id keys in this order: {first} first, then the rest.)")

SCHEMA = {"fee": "<dollar number for this item>", "exclusion_quote": "<exact excluding sentence, or empty>"}

def syn(i, brand, desc):        # 유사: 같은 조항(무명 outfitter·추론-면제) 장비
    return {"item_id": "bag_syn%02d" % i, "brand_retailer": brand, "description": desc,
            "category": "Sports", "declared_value": "$%d" % (180 + 37 * i), "weight_lb": 24 + i}
def dsyn(i, brand, desc, cat):  # 비유사: 타 카테고리(표준 35)
    return {"item_id": "bag_dis%02d" % i, "brand_retailer": brand, "description": desc,
            "category": cat, "declared_value": "$%d" % (180 + 37 * i), "weight_lb": 24 + i}

SIM = [('TrailPeak Outfitters', 'telemark ski set'), ('RiverRun Pro Shop', 'whitewater kayak paddle kit'),
       ('SummitLine Gear', 'ice climbing rack'), ('BlueFin Dive Supply', 'scuba regulator set'),
       ('NorthCairn Equipment', 'mountaineering tent'), ('WaveCrest Boardworks', 'longboard surfboard'),
       ('StoneHollow Outfitting', 'bouldering crash pad'), ('DriftPoint Anglers', 'fly-fishing rod tube')]
DIS = [('Samsonite', 'hardside suitcase', 'Luggage'), ('Apple', 'laptop in padded case', 'Electronics'),
       ('IKEA', 'boxed table lamp', 'Household'), ('Zara', 'garment bag with suits', 'Clothing'),
       ('KitchenAid', 'stand mixer in box', 'Appliances'), ('Fender', 'acoustic guitar in case', 'Instruments'),
       ('Dyson', 'cordless vacuum in box', 'Appliances'), ('LEGO', 'sealed collector sets', 'Toys')]
TGT = {"item_id": "bag_target01", "brand_retailer": "Burton", "description": "snowboard and bindings in board bag",
       "category": "Sports", "declared_value": "$740", "weight_lb": 28}

tok = AutoTokenizer.from_pretrained(MODEL)

def build(rows):
    ids = [r['item_id'] for r in rows]
    schema = json.dumps({i: SCHEMA for i in ids}, ensure_ascii=False)
    user = INSTR.format(docs=DOCS, items=json.dumps(rows, ensure_ascii=False, indent=1),
                        schema=schema, first=TGT['item_id'])
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return text + '{"%s": {"fee": ' % TGT['item_id']

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model.eval()
print('MODEL:', MODEL, '| A3 cross-domain (airline baggage) | threads:', torch.get_num_threads(), flush=True)

results = []
def probe(rows, cond, k):
    ii = torch.tensor([tok(build(rows))['input_ids']])
    with torch.no_grad():
        out = model(ii)
    pr = torch.softmax(out.logits[0, -1].float(), -1)
    def pd(c):
        return float(sum(pr[e] for e in {tok.encode(c)[0], tok.encode(' ' + c)[0]}))
    rec = {'cond': cond, 'k': k, 'S': int(ii.shape[1]), 'P0': pd('0'), 'P3': pd('3')}
    results.append(rec)
    print('%-12s k=%d  P(0)=%.3f  P(35~"3")=%.3f  (S=%d)' % (cond, k, rec['P0'], rec['P3'], rec['S']), flush=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL, 'results': results}, f, ensure_ascii=False)
    del out

for k in (0, 1, 2, 4, 8):
    probe([syn(i, *SIM[i]) for i in range(k)] + [TGT], 'similar', k)
for k in (1, 2, 4, 8):
    probe([dsyn(i, *DIS[i]) for i in range(k)] + [TGT], 'dissimilar', k)
print('DONE →', OUT_JSON, flush=True)
