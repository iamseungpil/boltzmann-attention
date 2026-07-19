# -*- coding: utf-8 -*-
# ★Track B B1 (P3 유효 재시도): a_C(k) 어텐션 곡선 — 조항-검색 질량 희석의 직접 측정.
#
# 프레임: full-fidelity 프롬프트(size_k_sweep.build와 동일·target-first 로짓판독 프라이밍).
#   B0 확정(2026-07-19): 이 프레임서 3B가 k-계단 재현(k=0 P(5)=0.983 → k≥1 붕괴) → 3B로 기전 측정 유효.
#   1차 P3(14B·자작 축약 프롬프트)는 k=0부터 실패 = 전제 미충족 무효 — 본 스크립트가 대체.
# ⚠️해석 주의: 로짓판독 프레임 임계 k*=1 (생성 프로브 k*=2와 다름·출력 프라이밍 차이) — 같은 프레임 내 비교만.
#
# 측정(KV 2-pass): pass1=프롬프트[:-1] prefill(use_cache) → pass2=마지막 토큰(판정-쿼리)만
#   output_attentions=True → 어텐션 [L][H,1,S]로 메모리 무시 가능. 층별 head-평균 질량을
#   조항-C 토큰·개입 행·target 행 span별로 적분. per-layer 원자료 전부 JSON 덤프(층 선택은 그림 단계서).
# 대조(P3 예측): 유사(같은 조항 추론-결합·무명 eco 브랜드) k↑ → a_C 감소 + P(5) 붕괴.
#   비유사(타 카테고리 합성·rate1) k↑ → a_C 유지 + P(5) 유지. (rpi.py 행동 해리의 어텐션 대응.)
# 실행: 리모트 CPU. setsid /home/woori/venvs/seka_env/bin/python -u b1_attn_curve.py > log 2>&1 &
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = '/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2'
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding='utf-8')
import torch
torch.set_num_threads(int(os.environ.get('B1_THREADS', '24')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import bank_shared_docs_probe as P

MODEL = os.environ.get('B1_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
OUT_JSON = os.environ.get('B1_OUT', '/home/woori/scratch/b1_attn_curve_20260719.json')

iso = P.load_iso_spec(); all_docs = P.load_docs()
docs = [d for d in all_docs if d['title'].startswith('EcoCard: ')]
docstr = "\n\n".join("### %s\n%s" % (x['title'], x['content']) for x in docs)
CLAUSE = 'Certified sustainable retailers and eco-labeled products'
assert CLAUSE in docstr, 'clause 문자열이 full-fidelity 문서에 없음 — span 정의 재확인 필요'

def syn(i, m):                       # 유사: 같은 조항(무명 eco 소매·추론-5배) — size_k_sweep와 동일
    amt = 50.0 + 13.7 * i
    return {"transaction_id": "txn_syn%02d" % i, "credit_card_type": "EcoCard", "merchant_name": m,
            "transaction_amount": "$%.2f" % amt, "transaction_date": "10/%02d/2025" % (5 + i),
            "category": "Green", "rewards_earned": "%d points" % int(amt * 5), "account_open": "07/01/2024"}

def dsyn(i, m, cat):                 # 비유사: 타 카테고리 합성·rate1 (조항-C 비결합 대조)
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

def build(grows):                    # size_k_sweep.build와 동일 프레임(target-first 프라이밍)
    raw = [{kk: v for kk, v in r.items() if kk in keep} for r in grows]
    ids = [r['transaction_id'] for r in grows]
    schema = json.dumps({i: iso.get('operand_schema', {}) for i in ids}, ensure_ascii=False)
    user = iso['inject_instructions'].format(group='EcoCard', docs=docstr, schema=schema,
                                             items=json.dumps(raw, ensure_ascii=False, indent=1))
    user += "\n\n(List the transaction_id keys in this order: %s first, then the rest.)" % TGT['transaction_id']
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return text + '{"%s": {"base_rate": ' % TGT['transaction_id']

def span_tokens(text, offs, needle, start=0):
    i0 = text.find(needle, start)
    if i0 < 0:
        return [], -1
    i1 = i0 + len(needle)
    return [t for t, (a, b) in enumerate(offs) if a < i1 and b > i0], i0

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation='eager')
model.eval()
print('MODEL:', MODEL, '| threads:', torch.get_num_threads(), '| clause:', repr(CLAUSE), flush=True)

results = []
def probe(grows, cond, k):
    text = build(grows)
    enc = tok(text, return_offsets_mapping=True)
    offs = enc['offset_mapping']
    ii = torch.tensor([enc['input_ids']]); S = ii.shape[1]
    cl, _ = span_tokens(text, offs, CLAUSE)
    tg, _ = span_tokens(text, offs, '"Patagonia"')
    iv_each = []
    for r in grows[:-1]:
        mtok, _ = span_tokens(text, offs, '"%s"' % r['merchant_name'])
        iv_each.append(mtok)
    iv_all = sorted({t for s in iv_each for t in s})
    with torch.no_grad():
        o1 = model(ii[:, :-1], use_cache=True)
        o2 = model(ii[:, -1:], past_key_values=o1.past_key_values, output_attentions=True)
    pr = torch.softmax(o2.logits[0, -1].float(), -1)
    def pd(c):
        return float(sum(pr[e] for e in {tok.encode(c)[0], tok.encode(' ' + c)[0]}))
    def masses(idx):
        if not idx:
            return [0.0] * len(o2.attentions)
        t = torch.tensor(idx)
        return [float(a[0, :, 0, :].float()[:, t].sum(-1).mean()) for a in o2.attentions]
    rec = {'cond': cond, 'k': k, 'S': S, 'P5': pd('5'), 'P1': pd('1'),
           'n_clause_tok': len(cl), 'n_iv_tok': len(iv_all),
           'clause_mass': masses(cl), 'target_merch_mass': masses(tg), 'iv_merch_mass': masses(iv_all)}
    results.append(rec)
    cm = sum(rec['clause_mass']) / len(rec['clause_mass'])
    im = sum(rec['iv_merch_mass']) / len(rec['iv_merch_mass'])
    tm = sum(rec['target_merch_mass']) / len(rec['target_merch_mass'])
    print('%-14s k=%d  P(5)=%.3f P(1)=%.3f | a_C=%.5f  a_iv=%.5f  a_tgt=%.5f (S=%d·clauseTok=%d)'
          % (cond, k, rec['P5'], rec['P1'], cm, im, tm, S, len(cl)), flush=True)
    del o1, o2
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL, 'clause': CLAUSE, 'results': results}, f, ensure_ascii=False)

for k in (0, 1, 2, 4, 8):
    probe([syn(i, BRANDS[i]) for i in range(k)] + [TGT], 'similar', k)
for k in (1, 2, 4, 8):
    probe([dsyn(i, m, c) for i, (m, c) in enumerate(DISSIM[:k])] + [TGT], 'dissimilar', k)
print('DONE →', OUT_JSON, flush=True)
