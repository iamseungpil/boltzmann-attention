# -*- coding: utf-8 -*-
# ★Track B B1b (locus-정제 어텐션 측정): target-행 "구축" 쿼리들의 어텐션 — B2 국소화의 어텐션 대응물.
#
# 배경: B1(readout 쿼리)서 a_C/a_iv/a_tgt 전부 유사/비유사 비특이 → readout-희석 아님(P3 원형 기각).
#   B2는 ko_tgtrow(target-행 쿼리→개입행 차단)만으로 0.96+ 회복 = 간섭 locus는 **target-행 표상 구축**.
#   ⇒ 정제 예측 P3′: target-행 토큰들의 쿼리가 (a) 조항-C에 주는 질량이 similar 적재서 희석되거나
#   (b) 개입-행(에코)에 주는 질량이 similar서 특이적으로 크다. dissimilar 대조가 게이트.
# 방법: 2-pass 분할점을 q0(target-행 시작)로 — pass1=[:q0] sdpa prefill(빠름)·pass2=[q0:] eager
#   (config._attn_implementation 스위치) output_attentions → [L][H, S-q0, S]서 target-행 쿼리 행만 적분.
# 실행: setsid /home/woori/venvs/seka_env/bin/python -u b1b_construction_attn.py > log 2>&1 &
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = '/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2'
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding='utf-8')
import torch
torch.set_num_threads(int(os.environ.get('B1B_THREADS', '24')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import bank_shared_docs_probe as P

MODEL = os.environ.get('B1B_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
OUT_JSON = os.environ.get('B1B_OUT', '/home/woori/scratch/b1b_construction_attn_20260719.json')

iso = P.load_iso_spec(); all_docs = P.load_docs()
docs = [d for d in all_docs if d['title'].startswith('EcoCard: ')]
docstr = "\n\n".join("### %s\n%s" % (x['title'], x['content']) for x in docs)
CLAUSE = 'Certified sustainable retailers and eco-labeled products'
assert CLAUSE in docstr

def syn(i, m):
    amt = 50.0 + 13.7 * i
    return {"transaction_id": "txn_syn%02d" % i, "credit_card_type": "EcoCard", "merchant_name": m,
            "transaction_amount": "$%.2f" % amt, "transaction_date": "10/%02d/2025" % (5 + i),
            "category": "Green", "rewards_earned": "%d points" % int(amt * 5), "account_open": "07/01/2024"}
def dsyn(i, m, cat):
    amt = 50.0 + 13.7 * i
    return {"transaction_id": "txn_dis%02d" % i, "credit_card_type": "EcoCard", "merchant_name": m,
            "transaction_amount": "$%.2f" % amt, "transaction_date": "10/%02d/2025" % (5 + i),
            "category": cat, "rewards_earned": "%d points" % int(amt * 1), "account_open": "07/01/2024"}
BRANDS = ['GreenLeaf Organics', 'EcoNest Home', 'Terra Verde Goods', 'PureCycle Apparel',
          'SolarBloom Garden', 'EarthKind Supply', 'VerdeMarket Co', 'BlueRoot Naturals']
DISSIM = [('Chipotle', 'Dining'), ('Starbucks', 'Dining'), ('Best Buy', 'Shopping'), ('Kroger', 'Groceries'),
          ('Shell', 'Gas'), ('Delta Airlines', 'Travel'), ('AMC Theatres', 'Entertainment'), ('CVS Pharmacy', 'Health')]
TGT = {"transaction_id": "txn_target01", "credit_card_type": "EcoCard", "merchant_name": "Patagonia",
       "transaction_amount": "$128.47", "transaction_date": "11/04/2025", "category": "Green",
       "rewards_earned": "128 points", "account_open": "07/01/2024"}
keep = set(iso.get('row_fields') or [])
tok = AutoTokenizer.from_pretrained(MODEL)

def build(grows):
    raw = [{kk: v for kk, v in r.items() if kk in keep} for r in grows]
    ids = [r['transaction_id'] for r in grows]
    schema = json.dumps({i: iso.get('operand_schema', {}) for i in ids}, ensure_ascii=False)
    user = iso['inject_instructions'].format(group='EcoCard', docs=docstr, schema=schema,
                                             items=json.dumps(raw, ensure_ascii=False, indent=1))
    user += "\n\n(List the transaction_id keys in this order: %s first, then the rest.)" % TGT['transaction_id']
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return text + '{"%s": {"base_rate": ' % TGT['transaction_id']

def tok_span(offs, c0, c1):
    return [t for t, (a, b) in enumerate(offs) if a < c1 and b > c0]

def row_char_spans(text, grows):
    anchors = [text.find('"%s"' % r['transaction_id']) for r in grows]
    assert all(a > 0 for a in anchors)
    ends = anchors[1:] + [text.find(']', anchors[-1])]
    out = []
    for i, a in enumerate(anchors):
        out.append((text.rfind('{', 0, a), text.rfind('}', a, ends[i]) + 1))
    return out

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation='sdpa')
model.eval()
print('MODEL:', MODEL, '| threads:', torch.get_num_threads(), flush=True)

def set_attn(impl):
    model.config._attn_implementation = impl
    for l in model.model.layers:
        if hasattr(l.self_attn, 'config'):
            l.self_attn.config._attn_implementation = impl

results = []
def probe(grows, cond, k):
    text = build(grows)
    enc = tok(text, return_offsets_mapping=True)
    offs = enc['offset_mapping']
    ii = torch.tensor([enc['input_ids']]); S = ii.shape[1]
    spans = row_char_spans(text, grows)
    c0 = text.find(CLAUSE)
    cl = tok_span(offs, c0, c0 + len(CLAUSE))
    iv = sorted({t for (cs, ce) in spans[:-1] for t in tok_span(offs, cs, ce)})
    tgt_toks = tok_span(offs, spans[-1][0], spans[-1][1])
    q0, q1 = min(tgt_toks), max(tgt_toks) + 1
    with torch.no_grad():
        set_attn('sdpa')
        o1 = model(ii[:, :q0], use_cache=True)
        set_attn('eager')
        o2 = model(ii[:, q0:], past_key_values=o1.past_key_values, output_attentions=True)
    pr = torch.softmax(o2.logits[0, -1].float(), -1)
    def pd(c):
        return float(sum(pr[e] for e in {tok.encode(c)[0], tok.encode(' ' + c)[0]}))
    nq = q1 - q0                                   # target-행 쿼리 수(pass2 앞부분)
    def masses(idx):
        if not idx:
            return [0.0] * len(o2.attentions)
        t = torch.tensor(idx)
        # [1,H,S-q0,S] → target-행 쿼리 행(0..nq)·head 평균·span 적분·쿼리 평균
        return [float(a[0, :, :nq, :].float()[:, :, t].sum(-1).mean()) for a in o2.attentions]
    rec = {'cond': cond, 'k': k, 'S': S, 'q0': q0, 'q1': q1,
           'P5': pd('5'), 'P1': pd('1'),
           'row_clause_mass': masses(cl), 'row_iv_mass': masses(iv)}
    results.append(rec)
    cm = sum(rec['row_clause_mass']) / len(rec['row_clause_mass'])
    im = sum(rec['row_iv_mass']) / len(rec['row_iv_mass'])
    print('%-12s k=%d  P(5)=%.3f P(1)=%.3f | row→clause=%.5f  row→iv=%.5f (S=%d·rowTok=%d)'
          % (cond, k, rec['P5'], rec['P1'], cm, im, S, nq), flush=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL, 'results': results}, f, ensure_ascii=False)
    del o1, o2

for k in (0, 1, 2, 4, 8):
    probe([syn(i, BRANDS[i]) for i in range(k)] + [TGT], 'similar', k)
for k in (1, 2, 4, 8):
    probe([dsyn(i, m, c) for i, (m, c) in enumerate(DISSIM[:k])] + [TGT], 'dissimilar', k)
print('DONE →', OUT_JSON, flush=True)
