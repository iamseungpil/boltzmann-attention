# -*- coding: utf-8 -*-
# ★Track B B2 (P6 인과 개입·금표준): 4D attention-mask knockout — 개입-행 어텐션 차단 시 판정 회복 검정.
#
# 설계(Track B doc §B2·위치 id 보존): 토큰 제거가 아니라 **마스크만** — target-행 시작(q0) 이후의 모든 쿼리
#   (target 행·후속 schema/프라이밍·판정 readout)가 개입-행 토큰(key/value)에 어텐션하지 못하게 차단.
#   위치 id·시퀀스 길이 불변 → 위치 효과와 내용 간섭을 분리. 개입 행 자신(q<q0)의 처리는 그대로.
# 팔:
#   base   = 순수 causal 4D 마스크(자체 구성) — size_k_sweep 수치 재현이 sanity gate(마스크 경로 검증).
#   ko_full= q>=q0 쿼리 → 개입-행 토큰 차단 (인과 가설의 본 검정: P(5) 회복 예측).
#   ko_last= 마지막 readout 쿼리만 차단 (readout-희석 채널 단독 기여 분리).
#   ctrl   = q>=q0 쿼리 → 개입-행과 동일 토큰수의 무관 문서구간(조항·target 비포함) 차단 (마스크-크기 통제).
# k∈{2,4,8} similar(붕괴 조건) + k=0 참조. 예측 P6: ko_full서 P(5)→k=0 수준 회복·ctrl은 비회복.
#   회복 실패 시 = 어텐션-희석 아닌 값-혼합/표상 오염 경로(그 자체로 결과·Track B doc 리스크 절).
# ⚠️로짓판독 프레임(k*=1) — 같은 프레임 내 비교만. B1(b1_attn_curve)과 프롬프트·프레임 동일.
# 실행: 리모트 CPU. setsid /home/woori/venvs/seka_env/bin/python -u b2_knockout.py > log 2>&1 &
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = '/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2'
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding='utf-8')
import torch
torch.set_num_threads(int(os.environ.get('B2_THREADS', '24')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import bank_shared_docs_probe as P

MODEL = os.environ.get('B2_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
OUT_JSON = os.environ.get('B2_OUT', '/home/woori/scratch/b2_knockout_20260719.json')

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
    """items JSON 내 각 행의 char span: tid 앵커 → 앞쪽 '{' ~ 다음 앵커 전 '}'."""
    anchors = [text.find('"%s"' % r['transaction_id']) for r in grows]
    assert all(a > 0 for a in anchors)
    ends = anchors[1:] + [text.find(']', anchors[-1])]
    spans = []
    for i, a in enumerate(anchors):
        s = text.rfind('{', 0, a)
        e = text.rfind('}', a, ends[i]) + 1
        spans.append((s, e))
    return spans

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model.eval()
print('MODEL:', MODEL, '| threads:', torch.get_num_threads(), '| attn: default(sdpa)', flush=True)

NEG = torch.finfo(torch.bfloat16).min
results = []

def run(k, arm):
    grows = [syn(i, BRANDS[i]) for i in range(k)] + [TGT]
    text = build(grows)
    enc = tok(text, return_offsets_mapping=True)
    offs = enc['offset_mapping']
    ii = torch.tensor([enc['input_ids']]); S = ii.shape[1]
    spans = row_char_spans(text, grows)
    iv_tok = sorted({t for (cs, ce) in spans[:-1] for t in tok_span(offs, cs, ce)})
    q0 = min(tok_span(offs, spans[-1][0], spans[-1][1]))      # target-행 첫 토큰
    mask = torch.zeros((1, 1, S, S), dtype=torch.bfloat16)
    mask[0, 0] = torch.triu(torch.full((S, S), NEG, dtype=torch.bfloat16), diagonal=1)  # causal
    blocked = []
    if arm == 'ko_full' and iv_tok:
        mask[0, 0, q0:, iv_tok] = NEG; blocked = iv_tok
    elif arm == 'ko_last' and iv_tok:
        mask[0, 0, -1, iv_tok] = NEG; blocked = iv_tok
    elif arm == 'ctrl' and iv_tok:
        c0 = text.find(CLAUSE)
        cl = set(tok_span(offs, c0, c0 + len(CLAUSE)))
        d0 = text.find('=== EcoCard')                           # 문서 섹션 시작
        cand = [t for t, (a, b) in enumerate(offs)
                if a >= d0 and b <= spans[0][0] and t not in cl][:len(iv_tok)]
        mask[0, 0, q0:, cand] = NEG; blocked = cand
    with torch.no_grad():
        out = model(ii, attention_mask=mask)
    pr = torch.softmax(out.logits[0, -1].float(), -1)
    def pd(c):
        return float(sum(pr[e] for e in {tok.encode(c)[0], tok.encode(' ' + c)[0]}))
    rec = {'k': k, 'arm': arm, 'S': S, 'q0': q0, 'n_blocked': len(blocked),
           'P5': pd('5'), 'P1': pd('1'), 'P0': pd('0'), 'P100': pd('100')}
    results.append(rec)
    print('k=%d %-8s P(5)=%.3f P(1)=%.3f (S=%d·q0=%d·blocked=%d)'
          % (k, arm, rec['P5'], rec['P1'], S, q0, len(blocked)), flush=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL, 'results': results}, f, ensure_ascii=False)
    del out

run(0, 'base')                                # 참조(개입 0)
for k in (2, 4, 8):
    for arm in ('base', 'ko_full', 'ko_last', 'ctrl'):
        run(k, arm)
print('DONE →', OUT_JSON, flush=True)
