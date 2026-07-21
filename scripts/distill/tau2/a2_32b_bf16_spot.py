# -*- coding: utf-8 -*-
# ★Track A A② 보강: 32B bf16(CPU·HF) 스팟체크 — Int8/vLLM 스택 confound 해소용. k∈{0,1,2,4}.
#   size_k_sweep와 동일 build(full-fidelity·target-first 프라이밍). 예측: Int8 서버와 같은 k=1 붕괴면 confound 해소.
# 실행: setsid /home/woori/venvs/seka_env/bin/python -u a2_32b_bf16_spot.py > log 2>&1 &  (다운로드 ~53GB 소요 가능)
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = '/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2'
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding='utf-8')
import torch
torch.set_num_threads(int(os.environ.get('SPOT_THREADS', '40')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import bank_shared_docs_probe as P

MODEL = 'Qwen/Qwen2.5-32B-Instruct'
OUT_JSON = os.environ.get('SPOT_OUT', '/home/woori/scratch/a2_32b_bf16_spot_20260721.json')

iso = P.load_iso_spec(); all_docs = P.load_docs()
docs = [d for d in all_docs if d['title'].startswith('EcoCard: ')]
docstr = "\n\n".join("### %s\n%s" % (x['title'], x['content']) for x in docs)

def syn(i, m):
    amt = 50.0 + 13.7 * i
    return {"transaction_id": "txn_syn%02d" % i, "credit_card_type": "EcoCard", "merchant_name": m,
            "transaction_amount": "$%.2f" % amt, "transaction_date": "10/%02d/2025" % (5 + i),
            "category": "Green", "rewards_earned": "%d points" % int(amt * 5), "account_open": "07/01/2024"}
BRANDS = ['GreenLeaf Organics', 'EcoNest Home', 'Terra Verde Goods', 'PureCycle Apparel']
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

print('loading %s (bf16 CPU)…' % MODEL, flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model.eval()
print('loaded.', flush=True)

results = []
for k in (0, 1, 2, 4):
    grows = [syn(i, BRANDS[i]) for i in range(k)] + [TGT]
    ii = torch.tensor([tok(build(grows))['input_ids']])
    with torch.no_grad():
        out = model(ii)
    pr = torch.softmax(out.logits[0, -1].float(), -1)
    def pd(c):
        return float(sum(pr[e] for e in {tok.encode(c)[0], tok.encode(' ' + c)[0]}))
    rec = {'k': k, 'P5': pd('5'), 'P1': pd('1'), 'S': int(ii.shape[1])}
    results.append(rec)
    print('k=%d  P(5)=%.3f  P(1)=%.3f  (S=%d)' % (k, rec['P5'], rec['P1'], rec['S']), flush=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL, 'results': results}, f, ensure_ascii=False)
    del out
print('DONE →', OUT_JSON, flush=True)
