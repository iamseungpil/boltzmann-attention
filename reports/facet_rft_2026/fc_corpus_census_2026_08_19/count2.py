# -*- coding: utf-8 -*-
import json, re, glob
from collections import Counter
from count import load_glaive, load_hermes, fmt_class, channel, args_of, scalars

QUOTED = re.compile(r'["“‘\']([^"”’\']{2,})["”’\']')
CAPS   = re.compile(r'\b[A-Z]{2,}\b')

def quantified(t):
    return bool(re.search(r'\d', t or '')) or bool(QUOTED.search(t or '')) or bool(CAPS.search(t or ''))

REFUSE = re.compile(r"(I'm sorry|I am sorry|I apologize|don't have the (capability|ability)|not able to|unable to|cannot |can't )", re.I)
CLARIFY_Q = re.compile(r'\?\s*$')

def txt_kind(t):
    t = (t or '').strip()
    if REFUSE.search(t): return 'refusal'
    if CLARIFY_Q.search(t): return 'clarify_question'
    if '?' in t: return 'contains_question'
    return 'direct_answer'

def run(name, convs, kind):
    n = len(convs)
    q_cross = Counter(); first_text_kind = Counter(); fmt_x_text_kind = Counter()
    ex = {'a_text': [], 'c_tool': [], 'a_tool': []}
    for c in convs:
        turns = c['turns']
        ai = [i for i,(r,_) in enumerate(turns) if r=='assistant']
        if not ai: continue
        f = ai[0]; pu = ''
        for j in range(f-1,-1,-1):
            if turns[j][0]=='user': pu = turns[j][1]; break
        ch = channel(turns[f][1], kind); fc = fmt_class(pu)
        q_cross[('Q' if quantified(pu) else 'noQ', ch)] += 1
        if ch == 'text':
            k = txt_kind(turns[f][1]); first_text_kind[k]+=1; fmt_x_text_kind[(fc,k)]+=1
            if fc=='a' and len(ex['a_text'])<4: ex['a_text'].append((pu, turns[f][1]))
        else:
            if fc=='c' and len(ex['c_tool'])<3: ex['c_tool'].append((pu, turns[f][1]))
            if fc=='a' and len(ex['a_tool'])<3: ex['a_tool'].append((pu, turns[f][1]))
    print('='*72); print(name, ' n=', n)
    for g in ('Q','noQ'):
        tot = sum(v for (gg,_),v in q_cross.items() if gg==g)
        if tot: print('  user-turn %-4s (숫자/따옴표/대문자코드 포함=Q): tool_call %.3f | text %.3f | n=%d' % (
            g, q_cross[(g,'tool_call')]/tot, q_cross[(g,'text')]/tot, tot))
    t = sum(first_text_kind.values())
    if t:
        print('  first-assistant TEXT breakdown (n=%d): ' % t + ' | '.join('%s %.3f'%(k,v/t) for k,v in first_text_kind.most_common()))
        for fc in 'abc':
            rw = sum(v for (ff,_),v in fmt_x_text_kind.items() if ff==fc)
            if rw: print('     fmt %s (n=%d): '%(fc,rw) + ' | '.join('%s %.3f'%(k,fmt_x_text_kind[(fc,k)]/rw) for k in ('refusal','clarify_question','contains_question','direct_answer')))
    return ex

if __name__ == '__main__':
    G = load_glaive(sorted(glob.glob('g2_*.json')))
    exg = run('glaive-function-calling-v2', G, 'glaive')
    H = load_hermes(sorted(glob.glob('hgl_*.json')))
    run('hermes :: glaive_func_calling', H, 'hermes')
    H2 = load_hermes(sorted(glob.glob('hfc_*.json')))
    run('hermes :: func_calling', H2, 'hermes')
    print('\n\n########## GLAIVE fmt=a (>25w prose) -> TEXT examples ##########')
    for pu, at in exg['a_text']:
        print('---- USER:', pu[:700].replace('\n',' / '))
        print('---- ASSISTANT:', at[:400].replace('\n',' / '))
    print('\n########## GLAIVE fmt=a -> TOOL_CALL examples ##########')
    for pu, at in exg['a_tool']:
        print('---- USER:', pu[:500].replace('\n',' / '))
        print('---- ASSISTANT:', at[:300].replace('\n',' / '))
    # id overlap check
    a = set(); b = set()
    for f in sorted(glob.glob('hfc_*.json')):
        for r in json.load(open(f,encoding='utf-8'))['rows']: a.add(r['row']['id'])
    for f in sorted(glob.glob('hst_*.json')):
        for r in json.load(open(f,encoding='utf-8'))['rows']: b.add(r['row']['id'])
    print('\nhermes func_calling ids=%d, singleturn ids=%d, overlap=%d' % (len(a), len(b), len(a & b)))
