# -*- coding: utf-8 -*-
# ★Track B B3 (P4): 어텐션 온도 조작 → k* 이동 검정. 이론 k*≈e^{βΔE−θ} ⇒ β↑(T↓) = k* 지수 증가 예측.
#
# 방법: 전 층 self_attn.scaling(=head_dim^-0.5)에 배수 곱 = 어텐션 로짓 전역 β-스케일. sdpa 경로 그대로(빠름).
#   β=1.0 팔이 size_k_sweep 수치 재현 = sanity gate. k=0 P(5)는 전역 조작의 역량-손상 통제(k=0 붕괴 시 그 β는 해석 불가).
# ⚠️전역·조잡 조작 — 방향성 해석만(층/헤드 특이 아님). 로짓판독 프레임(k*=1) 내 비교.
# 실행: setsid /home/woori/venvs/seka_env/bin/python -u b3_temperature.py > log 2>&1 &
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = '/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2'
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding='utf-8')
import torch
torch.set_num_threads(int(os.environ.get('B3_THREADS', '24')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import bank_shared_docs_probe as P

MODEL = os.environ.get('B3_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
OUT_JSON = os.environ.get('B3_OUT', '/home/woori/scratch/b3_temperature_20260719.json')

iso = P.load_iso_spec(); all_docs = P.load_docs()
docs = [d for d in all_docs if d['title'].startswith('EcoCard: ')]
docstr = "\n\n".join("### %s\n%s" % (x['title'], x['content']) for x in docs)

def syn(i, m):
    amt = 50.0 + 13.7 * i
    return {"transaction_id": "txn_syn%02d" % i, "credit_card_type": "EcoCard", "merchant_name": m,
            "transaction_amount": "$%.2f" % amt, "transaction_date": "10/%02d/2025" % (5 + i),
            "category": "Green", "rewards_earned": "%d points" % int(amt * 5), "account_open": "07/01/2024"}
BRANDS = ['GreenLeaf Organics', 'EcoNest Home', 'Terra Verde Goods', 'PureCycle Apparel',
          'SolarBloom Garden', 'EarthKind Supply', 'VerdeMarket Co', 'BlueRoot Naturals']
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

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation='sdpa')
model.eval()
BASE_SCALING = [float(l.self_attn.scaling) for l in model.model.layers]
print('MODEL:', MODEL, '| layers:', len(BASE_SCALING), '| base scaling:', BASE_SCALING[0], flush=True)

def set_beta(mult):
    for l, s in zip(model.model.layers, BASE_SCALING):
        l.self_attn.scaling = s * mult

results = []
for mult in (0.85, 0.93, 1.0, 1.07, 1.15, 1.3):
    set_beta(mult)
    for k in (0, 1, 2, 4):
        grows = [syn(i, BRANDS[i]) for i in range(k)] + [TGT]
        ii = torch.tensor([tok(build(grows))['input_ids']])
        with torch.no_grad():
            out = model(ii)
        pr = torch.softmax(out.logits[0, -1].float(), -1)
        def pd(c):
            return float(sum(pr[e] for e in {tok.encode(c)[0], tok.encode(' ' + c)[0]}))
        rec = {'beta_mult': mult, 'k': k, 'P5': pd('5'), 'P1': pd('1')}
        results.append(rec)
        print('beta=%.2f k=%d  P(5)=%.3f P(1)=%.3f' % (mult, k, rec['P5'], rec['P1']), flush=True)
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump({'model': MODEL, 'results': results}, f, ensure_ascii=False)
        del out
set_beta(1.0)
print('DONE →', OUT_JSON, flush=True)
