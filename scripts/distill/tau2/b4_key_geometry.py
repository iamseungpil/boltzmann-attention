# -*- coding: utf-8 -*-
# ★Track B B4 (P7·선택): key/value 기하 — 개입-행 토큰의 key·value가 조항-C 방향 성분을 갖는가(유사 특이?).
#
# 방법: 각 층 self_attn.k_proj/v_proj forward hook으로 사영 출력 캡처(१회 forward/조건·pre-RoPE 주의).
#   층별 cosine( mean_{iv-row 토큰} k , mean_{조항 토큰} k ) 및 value 동형·target-행도 병기.
#   조건: k=0 / k=4 similar / k=4 dissimilar. 예측 P7: similar iv의 조항-방향 성분 > dissimilar.
#   (B1b서 라우팅 질량은 비특이였으므로 여기서 갈리면 = 내용/기하 수준의 유사성 발현 증거.)
# 실행: setsid /home/woori/venvs/seka_env/bin/python -u b4_key_geometry.py > log 2>&1 &
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = '/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2'
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('B4_THREADS', '16')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import bank_shared_docs_probe as P

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
OUT_JSON = os.environ.get('B4_OUT', '/home/woori/scratch/b4_key_geometry_20260720.json')

iso = P.load_iso_spec(); all_docs = P.load_docs()
docs = [d for d in all_docs if d['title'].startswith('EcoCard: ')]
docstr = "\n\n".join("### %s\n%s" % (x['title'], x['content']) for x in docs)
CLAUSE = 'Certified sustainable retailers and eco-labeled products'

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
BRANDS = ['GreenLeaf Organics', 'EcoNest Home', 'Terra Verde Goods', 'PureCycle Apparel']
DISSIM = [('Chipotle', 'Dining'), ('Starbucks', 'Dining'), ('Best Buy', 'Shopping'), ('Kroger', 'Groceries')]
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
print('MODEL:', MODEL, '| layers:', NL, '| (pre-RoPE k_proj/v_proj 기하)', flush=True)

cap = {}
hooks = []
for li, layer in enumerate(model.model.layers):
    def mk(li, kind):
        def fn(module, inp, out):
            cap[(li, kind)] = out.detach()[0].float()      # [S, kv_dim]
        return fn
    hooks.append(layer.self_attn.k_proj.register_forward_hook(mk(li, 'k')))
    hooks.append(layer.self_attn.v_proj.register_forward_hook(mk(li, 'v')))

results = []
def probe(grows, cond):
    cap.clear()
    text = build(grows)
    enc = tok(text, return_offsets_mapping=True)
    offs = enc['offset_mapping']
    ii = torch.tensor([enc['input_ids']])
    spans = row_char_spans(text, grows)
    c0 = text.find(CLAUSE)
    cl = tok_span(offs, c0, c0 + len(CLAUSE))
    iv = sorted({t for (cs, ce) in spans[:-1] for t in tok_span(offs, cs, ce)})
    tg = tok_span(offs, spans[-1][0], spans[-1][1])
    with torch.no_grad():
        model(ii)
    rec = {'cond': cond, 'iv_k_cos': [], 'iv_v_cos': [], 'tgt_k_cos': [], 'tgt_v_cos': []}
    for li in range(NL):
        for kind in ('k', 'v'):
            M = cap[(li, kind)]
            c_mean = M[cl].mean(0)
            t_mean = M[tg].mean(0)
            tgt_cos = float(F.cosine_similarity(t_mean, c_mean, dim=0))
            iv_cos = float(F.cosine_similarity(M[iv].mean(0), c_mean, dim=0)) if iv else None
            rec['%s_%s_cos' % ('iv', kind)].append(iv_cos)
            rec['%s_%s_cos' % ('tgt', kind)].append(tgt_cos)
    results.append(rec)
    def mn(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float('nan')
    print('%-12s iv·C: k=%.4f v=%.4f | tgt·C: k=%.4f v=%.4f'
          % (cond, mn(rec['iv_k_cos']), mn(rec['iv_v_cos']), mn(rec['tgt_k_cos']), mn(rec['tgt_v_cos'])), flush=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL, 'results': results}, f, ensure_ascii=False)

probe([TGT], 'k0')
probe([syn(i, BRANDS[i]) for i in range(4)] + [TGT], 'sim4')
probe([dsyn(i, m, c) for i, (m, c) in enumerate(DISSIM)] + [TGT], 'dis4')
for h in hooks:
    h.remove()
print('DONE →', OUT_JSON, flush=True)
