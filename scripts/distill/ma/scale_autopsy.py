import json, os
from collections import Counter
BASE = 'scripts/distill/ma/results/scale'
SIZES = ['32B', '72B', '235B_fp8']

print("="*70)
print("Lane2 multidomain (⚠️오프라인 op-eval·신뢰불가·정성 패턴만)")
print("="*70)
revcheck = {}
for size in SIZES:
    for dom in ['retail', 'airline']:
        for g in [0, 1]:
            f = f'{BASE}/multidomain_scale/{size}__{dom}_g{g}.json'
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            ov, rec, rows = d['overall'], d['recognition'], d['rows']
            cls = Counter(); miss_attr = Counter()
            for r in rows:
                if r.get('hit'):
                    continue
                gs = r.get('gold_set') or {}; es = r.get('emitted_set') or {}
                if r.get('op') != r.get('case_op'):
                    cls['op_wrong'] += 1; continue
                if not es:
                    cls['no_set'] += 1; continue
                missing = [k for k in gs if k not in es]
                wrong = [k for k in gs if k in es and str(es[k]) != str(gs[k])]
                extra = [k for k in es if k not in gs]
                if missing:
                    cls['miss_key'] += 1
                    for k in missing: miss_attr[k] += 1
                elif wrong:
                    cls['wrong_value'] += 1
                elif extra:
                    cls['extra_key'] += 1
                else:
                    cls['set_ok_resolve_fail'] += 1
            acc = ov[0]/max(ov[1],1)
            revcheck[(size,dom,g)] = acc
            print(f"{size:9} {dom:8} g{g}: acc={ov[0]:2}/{ov[1]:2}={acc:.2f} "
                  f"recog={rec[0]}/{rec[1]} | fails={dict(cls)} | miss_attr={dict(miss_attr.most_common(4))}")

print("\n--- 235B 역전 검정 (retail: 작은 n에서 크기/gloss 비단조) ---")
for dom in ['retail','airline']:
    row = " ".join(f"{s}:g0={revcheck.get((s,dom,0),0):.2f}/g1={revcheck.get((s,dom,1),0):.2f}" for s in SIZES)
    print(f"  {dom}: {row}")
    n = json.load(open(f'{BASE}/multidomain_scale/32B__{dom}_g0.json'))['overall'][1]
    print(f"    (n={n} → 1-case ≈ {1/n:.3f}; 비단조 폭이 1-2 case이면 노이즈)")

print("\n" + "="*70)
print("Lane1 depth (synth·신뢰) — by-op 곡선")
print("="*70)
for size in SIZES:
    for N in [5, 10, 20, 50]:
        f = f'{BASE}/depth_scale/depth_{size}_N{N}.json'
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        bo = d.get('by_op') or {}
        ov = d.get('overall') or {}
        a, b = ov.get('A'), ov.get('B')
        btxt = " ".join(f"{op}={v['A'][0]/v['A'][1]:.2f}" for op, v in bo.items())
        print(f"{size:9} N{N:2}: A(in-head)={a[0]/a[1]:.2f} B(engine)={b[0]/b[1]:.2f} | A_by_op: {btxt}")
