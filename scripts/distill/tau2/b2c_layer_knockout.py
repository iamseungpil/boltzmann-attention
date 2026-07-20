# -*- coding: utf-8 -*-
# ★Track B B2c: 층별 knockout 국소화 — ko_full(k=4) 회복이 어느 층 구간의 차단에서 오는가.
#
# 방법: 4D causal 마스크를 기본 전달·forward_pre_hook(with_kwargs)으로 **선택 층에만** knockout 마스크 교체.
#   sanity: window=all ⇒ b2 ko_full(0.980) 재현 · window=none ⇒ base(0.120) 재현 — 훅 경로 검증 게이트.
#   구간: 4분위(0-8/9-17/18-26/27-35)+반(0-17/18-35). k=4 similar 고정.
# 실행: setsid /home/woori/venvs/seka_env/bin/python -u b2c_layer_knockout.py > log 2>&1 &
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = '/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2'
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding='utf-8')
import torch
torch.set_num_threads(int(os.environ.get('B2C_THREADS', '16')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import bank_shared_docs_probe as P

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
OUT_JSON = os.environ.get('B2C_OUT', '/home/woori/scratch/b2c_layer_knockout_20260720.json')
K = int(os.environ.get('B2C_K', '4'))

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

def tok_span(offs, c0, c1):
    return [t for t, (a, b) in enumerate(offs) if a < c1 and b > c0]

def row_char_spans(text, grows):
    anchors = [text.find('"%s"' % r['transaction_id']) for r in grows]
    ends = anchors[1:] + [text.find(']', anchors[-1])]
    return [(text.rfind('{', 0, a), text.rfind('}', a, ends[i]) + 1) for i, a in enumerate(anchors)]

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model.eval()
NL = len(model.model.layers)
print('MODEL:', MODEL, '| layers:', NL, '| k=%d' % K, flush=True)

grows = [syn(i, BRANDS[i]) for i in range(K)] + [TGT]
text = build(grows)
enc = tok(text, return_offsets_mapping=True)
offs = enc['offset_mapping']
ii = torch.tensor([enc['input_ids']]); S = ii.shape[1]
spans = row_char_spans(text, grows)
iv_tok = sorted({t for (cs, ce) in spans[:-1] for t in tok_span(offs, cs, ce)})
q0 = min(tok_span(offs, spans[-1][0], spans[-1][1]))
NEG = torch.finfo(torch.bfloat16).min
m_causal = torch.zeros((1, 1, S, S), dtype=torch.bfloat16)
m_causal[0, 0] = torch.triu(torch.full((S, S), NEG, dtype=torch.bfloat16), diagonal=1)
m_ko = m_causal.clone()
m_ko[0, 0, q0:, iv_tok] = NEG

results = []
def run(window, tag):
    wset = set(window)
    hooks = []
    def mk(li):
        def fn(module, args, kwargs):
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                kwargs['attention_mask'] = m_ko if li in wset else m_causal
            return args, kwargs
        return fn
    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(mk(li), with_kwargs=True))
    try:
        with torch.no_grad():
            out = model(ii, attention_mask=m_causal)
        pr = torch.softmax(out.logits[0, -1].float(), -1)
        def pd(c):
            return float(sum(pr[e] for e in {tok.encode(c)[0], tok.encode(' ' + c)[0]}))
        rec = {'k': K, 'window': tag, 'layers': sorted(wset), 'P5': pd('5'), 'P1': pd('1')}
        results.append(rec)
        print('window=%-8s P(5)=%.3f P(1)=%.3f' % (tag, rec['P5'], rec['P1']), flush=True)
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump({'model': MODEL, 'k': K, 'results': results}, f, ensure_ascii=False)
        del out
    finally:
        for h in hooks:
            h.remove()

run([], 'none')                                   # sanity: base 재현 기대
run(list(range(NL)), 'all')                       # sanity: ko_full 재현 기대
run(list(range(0, 18)), 'L0-17')
run(list(range(18, NL)), 'L18-35')
run(list(range(0, 9)), 'L0-8')
run(list(range(9, 18)), 'L9-17')
run(list(range(18, 27)), 'L18-26')
run(list(range(27, NL)), 'L27-35')
print('DONE →', OUT_JSON, flush=True)
