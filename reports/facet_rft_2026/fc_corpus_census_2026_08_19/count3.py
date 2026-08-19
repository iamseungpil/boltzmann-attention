# -*- coding: utf-8 -*-
import json, re, glob
from collections import Counter
from count import load_glaive, load_hermes, channel, fmt_class

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs)-1, int(p*len(xs)))] if xs else -1

def ntools_glaive(sys_t):
    return sys_t.count('"name"')

def ntools_hermes(sys_t):
    m = re.search(r'<tools>(.*?)</tools>', sys_t or '', re.S)
    body = m.group(1) if m else (sys_t or '')
    return body.count('"type": "function"') or body.count('"name"')

def run(name, convs, kind, ntool_fn):
    dep_tc, dep_tx = [], []
    ctx_tc, ctx_tx = [], []
    prose_before_tc = Counter()
    ntools = []
    tc_total = 0; tc_at_depth0 = 0
    for c in convs:
        turns = c['turns']
        ntools.append(ntool_fn(c['sys']))
        ctx_words = len((c['sys'] or '').split())
        n_prose_ai = 0
        for i,(r,t) in enumerate(turns):
            if r == 'assistant':
                ch = channel(t, kind)
                if ch in ('tool_call','mixed'):
                    dep_tc.append(i); ctx_tc.append(ctx_words); tc_total += 1
                    if i == 0: tc_at_depth0 += 1
                    prose_before_tc[min(n_prose_ai,3)] += 1
                else:
                    dep_tx.append(i); ctx_tx.append(ctx_words); n_prose_ai += 1
            ctx_words += len((t or '').split())
    print('='*72); print(name, ' n_conv=', len(convs))
    print('  tools offered per sample: median=%d p90=%d max=%d' % (pct(ntools,.5), pct(ntools,.9), max(ntools)))
    print('  tool_call turns: n=%d ; turn-index median=%d p90=%d ; at index0 (=first reply) %.3f' % (
        tc_total, pct(dep_tc,.5), pct(dep_tc,.9), tc_at_depth0/max(tc_total,1)))
    print('  text turns:      n=%d ; turn-index median=%d p90=%d' % (len(dep_tx), pct(dep_tx,.5), pct(dep_tx,.9)))
    print('  context words BEFORE turn:  tool_call median=%d p90=%d | text median=%d p90=%d' % (
        pct(ctx_tc,.5), pct(ctx_tc,.9), pct(ctx_tx,.5), pct(ctx_tx,.9)))
    t = sum(prose_before_tc.values())
    if t:
        print('  # of prior PROSE assistant turns before a tool_call: ' + ' | '.join(
            '%s:%.3f' % ('>=3' if k==3 else k, prose_before_tc[k]/t) for k in sorted(prose_before_tc)))

if __name__ == '__main__':
    run('glaive-function-calling-v2', load_glaive(sorted(glob.glob('g2_*.json'))), 'glaive', ntools_glaive)
    run('hermes :: glaive_func_calling', load_hermes(sorted(glob.glob('hgl_*.json'))), 'hermes', ntools_hermes)
    run('hermes :: func_calling', load_hermes(sorted(glob.glob('hfc_*.json'))), 'hermes', ntools_hermes)
