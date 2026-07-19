# -*- coding: utf-8 -*-
# ★Track A②+B0 전제: 모델크기 × k 로짓판독 스윕 (full-fidelity 프롬프트·target-first 프라이밍)
# 실행: 리모트 CPU(HF bf16) + :8140(32B 서버 로짓판독). 로그=size_k_sweep_20260719.log
# ⚠️프레임 주의: build()가 '{"txn_target01": {"base_rate": ' 로 target-first 출력을 프라이밍
#   → 생성 프로브(k*=2)보다 임계가 1 낮게 관측됨(로짓판독 k*=1). 해석 시 조건 차이 명시 필수.
import os, sys, json, urllib.request
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2')
os.chdir('/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2')
sys.stdout.reconfigure(encoding='utf-8')
import bank_shared_docs_probe as P
iso = P.load_iso_spec(); all_docs = P.load_docs()
users, rows, gold, mustflag = P.gold_and_rows()
docs = [d for d in all_docs if d['title'].startswith('EcoCard: ')]
docstr = "\n\n".join("### %s\n%s" % (x['title'], x['content']) for x in docs)

def syn(i, m):
    amt=50.0+13.7*i
    return {"transaction_id":"txn_syn%02d"%i,"credit_card_type":"EcoCard","merchant_name":m,
            "transaction_amount":"$%.2f"%amt,"transaction_date":"10/%02d/2025"%(5+i),
            "category":"Green","rewards_earned":"%d points"%int(amt*5),"account_open":"07/01/2024"}
BRANDS=['GreenLeaf Organics','EcoNest Home','Terra Verde Goods','PureCycle Apparel',
        'SolarBloom Garden','EarthKind Supply','VerdeMarket Co','BlueRoot Naturals']
TGT={"transaction_id":"txn_target01","credit_card_type":"EcoCard","merchant_name":"Patagonia",
     "transaction_amount":"$128.47","transaction_date":"11/04/2025","category":"Green",
     "rewards_earned":"128 points","account_open":"07/01/2024"}
keep=set(iso.get('row_fields') or [])

from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

def build(k, order_note=True):
    grows=[syn(i,BRANDS[i]) for i in range(k)]+[TGT]
    raw=[{kk:v for kk,v in r.items() if kk in keep} for r in grows]
    ids=[r['transaction_id'] for r in grows]
    schema=json.dumps({i:iso.get('operand_schema',{}) for i in ids}, ensure_ascii=False)
    user = iso['inject_instructions'].format(group='EcoCard', docs=docstr, schema=schema,
                                             items=json.dumps(raw, ensure_ascii=False, indent=1))
    if order_note:
        user += "\n\n(List the transaction_id keys in this order: %s first, then the rest.)" % TGT['transaction_id']
    text = tk.apply_chat_template([{"role":"user","content":user}], tokenize=False, add_generation_prompt=True)
    return text + '{"%s": {"base_rate": ' % TGT['transaction_id']

def server_probs(text):
    req = {"model":"Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8","prompt":text,"max_tokens":1,
           "temperature":0,"logprobs":10}
    r = urllib.request.urlopen(urllib.request.Request('http://localhost:8140/v1/completions',
        data=json.dumps(req).encode(), headers={'Content-Type':'application/json'}), timeout=300)
    out = json.loads(r.read())
    top = out['choices'][0]['logprobs']['top_logprobs'][0]
    import math
    return {t.strip(): math.exp(lp) for t,lp in top.items()}

print('=== 32B(server) ===', flush=True)
for k in (0,1,2,4,8):
    pv = server_probs(build(k))
    print('k=%d  P(5)=%.3f  P(1)=%.3f  top=%s' % (k, pv.get('5',0), pv.get('1',0),
          sorted(pv.items(), key=lambda x:-x[1])[:3]), flush=True)

import torch
torch.set_num_threads(48)
from transformers import AutoModelForCausalLM
for mname in ('Qwen/Qwen2.5-3B-Instruct','Qwen/Qwen2.5-14B-Instruct','Qwen/Qwen2.5-1.5B-Instruct'):
    print('=== %s (CPU) ===' % mname, flush=True)
    tok = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=torch.bfloat16)
    model.eval()
    for k in (0,1,2,4,8):
        text = build(k)
        ii = torch.tensor([tok(text)['input_ids']])
        with torch.no_grad():
            lg = model(ii).logits[0,-1].float()
        pr = torch.softmax(lg,-1)
        def p(c):
            s=0.0
            for enc in {tok.encode(c)[0], tok.encode(' '+c)[0]}: s+=float(pr[enc])
            return s
        print('k=%d  P(5)=%.3f  P(1)=%.3f  (S=%d)' % (k, p('5'), p('1'), ii.shape[1]), flush=True)
    del model
