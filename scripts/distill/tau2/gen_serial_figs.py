# -*- coding: utf-8 -*-
"""Paper figures for serial-interference Track A/B (2026-07). Reads persisted sim_results JSONs.
Run locally: py -3 scripts/distill/tau2/gen_serial_figs.py  (repo root 기준 상대경로)
Style: Okabe-Ito CVD-safe fixed-order palette · one axis · direct labels · thin marks."""
import gzip
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                  'ba-frft') if False else None
# repo root 탐색: 이 파일 위치 = <repo>/scripts/distill/tau2/
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
SIM = os.path.join(ROOT, 'reports', 'facet_rft_2026', 'sim_results')
OUT = os.path.join(ROOT, 'reports', 'facet_rft_2026', 'figures', 'serial_interference')
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (CVD-safe) 고정 순서
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'vermil': '#D55E00',
     'sky': '#56B4E9', 'purple': '#CC79A7', 'yellow': '#F0E442', 'black': '#000000'}

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})

def jload(name):
    with gzip.open(os.path.join(SIM, name), 'rt', encoding='utf-8') as f:
        return json.load(f)

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)))
    plt.close(fig)
    print('saved', name)

# ── F: 크기별 계단 (size_k_sweep 로그 수치는 로그에만 있음 → 하드코딩 대신 로그 파싱) ──
def parse_sweep():
    import re
    with gzip.open(os.path.join(SIM, 'size_k_sweep_20260719.log.gz'), 'rt', encoding='utf-8',
                   errors='replace') as f:
        txt = f.read()
    out = {}
    cur = None
    for line in txt.splitlines():
        m = re.match(r'=== (.+?) ===', line)
        if m:
            name = m.group(1)
            cur = ('32B' if '32B' in name else '14B' if '14B' in name
                   else '3B' if '-3B' in name else '1.5B' if '1.5B' in name else name)
            out[cur] = {}
            continue
        m = re.match(r'k=(\d+)\s+P\(5\)=([\d.]+)\s+P\(1\)=([\d.]+)', line)
        if m and cur:
            out[cur][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out

sw = parse_sweep()
fig, ax = plt.subplots(figsize=(3.6, 2.6))
order = [('32B', C['blue']), ('14B', C['orange']), ('3B', C['green']), ('1.5B', C['vermil'])]
for name, col in order:
    ks = sorted(sw[name])
    ax.plot(ks, [sw[name][k][0] for k in ks], '-o', color=col, lw=1.6, ms=4, label=name)
ax.set_xlabel('k (same-clause predecessors)')
ax.set_ylabel('P(correct rate "5")')
ax.set_ylim(-0.03, 1.03)
ax.set_xticks([0, 1, 2, 4, 8])
ax.legend(frameon=False, fontsize=8, title='model', title_fontsize=8)
ax.set_title('Logit-readout staircase across scales', fontsize=9)
save(fig, 'fig_staircase_sizes')

# ── F: knockout 회복 (b2 + fine) ──
b2 = jload('b2_knockout_20260719.json.gz')['results'] + jload('b2_knockout_fine_20260719.json.gz')['results']
arms = ['base', 'ko_full', 'ko_tgtrow', 'ko_post', 'ko_last', 'ctrl']
labels = {'base': 'none', 'ko_full': 'row +\ndownstream', 'ko_tgtrow': 'row\nonly',
          'ko_post': 'post-row\nonly', 'ko_last': 'readout\nonly', 'ctrl': 'control\n(size-matched)'}
cols = {'base': '#888888', 'ko_full': C['blue'], 'ko_tgtrow': C['sky'], 'ko_post': C['orange'],
        'ko_last': C['vermil'], 'ctrl': C['purple']}
k0 = [r for r in b2 if r['k'] == 0][0]['P5']
fig, ax = plt.subplots(figsize=(5.2, 2.6))
ax.set_xlabel('blocked query range (keys = predecessor-row tokens)', fontsize=8)
W = 0.38
for gi, k in enumerate((2, 4)):
    for ai, arm in enumerate(arms):
        rs = [r for r in b2 if r['k'] == k and r['arm'] == arm]
        if not rs:
            continue
        x = ai + (gi - 0.5) * W
        ax.bar(x, rs[0]['P5'], width=W * 0.92, color=cols[arm],
               alpha=1.0 if gi else 0.55, edgecolor='none')
ax.axhline(k0, color='#444444', lw=0.9, ls='--')
ax.text(len(arms) - 0.45, k0 + 0.02, 'k=0 baseline (%.2f)' % k0, fontsize=7, ha='right')
ax.set_xticks(range(len(arms)))
ax.set_xticklabels([labels[a] for a in arms], fontsize=7)
ax.set_ylabel('P(correct rate "5")')
ax.set_ylim(0, 1.05)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc='#666666', alpha=0.55, label='k=2'), Patch(fc='#666666', label='k=4')],
          frameon=False, fontsize=8, loc='upper right')
ax.set_title('Attention-mask knockout: recovery by blocked query range', fontsize=9)
save(fig, 'fig_knockout')

# ── F: 경로 질량 vs 행동 (b1b) ──
b1b = jload('b1b_construction_attn_20260719.json.gz')['results']
def series(cond, field):
    rs = sorted([r for r in b1b if r['cond'] == cond], key=lambda r: r['k'])
    if field == 'row_iv_mass':
        return ([r['k'] for r in rs],
                [sum(r[field]) / len(r[field]) if r['k'] else 0.0 for r in rs])
    return [r['k'] for r in rs], [r[field] for r in rs]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
fig.subplots_adjust(wspace=0.42)
for cond, col in (('similar', C['vermil']), ('dissimilar', C['blue'])):
    ks, p5 = series(cond, 'P5')
    axes[0].plot(ks, p5, '-o', color=col, lw=1.6, ms=4, label=cond)
    ks, im = series(cond, 'row_iv_mass')
    axes[1].plot(ks, im, '-o', color=col, lw=1.6, ms=4, label=cond)
axes[0].set_ylabel('P(correct rate "5")')
axes[0].set_title('(a) behavior separates', fontsize=9)
axes[0].set_ylim(-0.03, 1.03)
axes[1].set_ylabel('target-row query mass\non predecessor rows')
axes[1].set_title('(b) route traffic does not', fontsize=9)
axes[1].set_ylim(0, 0.30)
for ax in axes:
    ax.set_xlabel('k (predecessors)')
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.legend(frameon=False, fontsize=8)
save(fig, 'fig_route_mass')

# ── F: 온도 (b3) ──
b3 = jload('b3_temperature_20260719.json.gz')['results']
fig, ax = plt.subplots(figsize=(3.6, 2.6))
kcols = {0: '#888888', 1: C['green'], 2: C['vermil'], 4: C['purple']}
for k in (0, 1, 2, 4):
    rs = sorted([r for r in b3 if r['k'] == k], key=lambda r: r['beta_mult'])
    ax.plot([r['beta_mult'] for r in rs], [r['P5'] for r in rs], '-o', color=kcols[k],
            lw=1.6, ms=4, label='k=%d' % k)
ax.axvline(1.0, color='#444444', lw=0.8, ls=':')
ax.set_xlabel('attention inverse-temperature multiplier (β×)')
ax.set_ylabel('P(correct rate "5")')
ax.set_ylim(-0.03, 1.03)
ax.legend(frameon=False, fontsize=8)
ax.set_title('Sharpening deepens the collapse', fontsize=9)
save(fig, 'fig_temperature')

print('ALL FIGS →', OUT)
