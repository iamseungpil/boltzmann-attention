# -*- coding: utf-8 -*-
import json, re, glob, sys
from collections import Counter, defaultdict

JSONISH = re.compile(r'(\{\s*"|\[\s*\{|\{\s*\')')
TABLE   = re.compile(r'^\s*\|.*\|\s*$', re.M)
FIELDLN = re.compile(r'^\s*(?:[-*\d.]+\s*)?[A-Za-z_][A-Za-z0-9 _/-]{0,40}\s*[:=]\s*\S', re.M)

def fmt_class(text):
    t = (text or '').strip()
    if not t: return 'd'
    if JSONISH.search(t) or len(TABLE.findall(t)) >= 2 or len(FIELDLN.findall(t)) >= 3:
        return 'b'
    w = len(t.split())
    if w <= 25: return 'c'
    return 'a'

TC_G = re.compile(r'<functioncall>')
TC_H = re.compile(r'<tool_call>')

def channel(text, kind):
    t = text or ''
    rx = TC_G if kind == 'glaive' else TC_H
    if not rx.search(t):
        return 'text'
    if kind == 'glaive':
        stripped = re.sub(r'<functioncall>.*?(?:<\|endoftext\|>|$)', ' ', t, flags=re.S)
    else:
        stripped = re.sub(r'<tool_call>.*?</tool_call>', ' ', t, flags=re.S)
    stripped = stripped.replace('<|endoftext|>', ' ')
    return 'mixed' if len(re.sub(r'\s', '', stripped)) >= 20 else 'tool_call'

def args_of(text, kind):
    out = []
    if kind == 'glaive':
        blocks = re.findall(r'<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)', text or '', re.S)
    else:
        blocks = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text or '', re.S)
    for b in blocks:
        o = None
        try:
            o = json.loads(b)
        except Exception:
            b2 = re.sub(r"'(\{.*?\})'", lambda m: m.group(1), b, flags=re.S)
            try:
                o = json.loads(b2)
            except Exception:
                o = None
        if not isinstance(o, dict):
            continue
        a = o.get('arguments')
        if isinstance(a, str):
            try: a = json.loads(a)
            except Exception: a = None
        if isinstance(a, dict):
            out.append(a)
    return out

def scalars(d, acc=None):
    acc = acc if acc is not None else []
    for v in d.values():
        if isinstance(v, dict): scalars(v, acc)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict): scalars(x, acc)
                elif isinstance(x, (str, int, float)): acc.append(str(x))
        elif isinstance(v, (str, int, float)): acc.append(str(v))
    return acc

def load_glaive(files):
    convs = []
    for f in files:
        for r in json.load(open(f, encoding='utf-8'))['rows']:
            row = r['row']; chat = row['chat']
            parts = re.split(r'\n*(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*', chat)
            turns = []
            i = 1
            while i < len(parts):
                tag = parts[i]; body = parts[i+1] if i+1 < len(parts) else ''
                role = {'USER:':'user','ASSISTANT:':'assistant','FUNCTION RESPONSE:':'tool'}[tag]
                body = body.strip() if role == 'assistant' else body.replace('<|endoftext|>','').strip()
                turns.append((role, body))
                i += 2
            convs.append({'sys': row['system'], 'turns': turns, 'kind':'glaive'})
    return convs

ROLE = {'system':'system','human':'user','gpt':'assistant','tool':'tool'}

def load_hermes(files):
    convs = []
    for f in files:
        for r in json.load(open(f, encoding='utf-8'))['rows']:
            row = r['row']; sys_t = ''; turns = []
            for m in row['conversations']:
                role = ROLE.get(m.get('from'), 'other'); v = m.get('value') or ''
                if role == 'system': sys_t = v
                else: turns.append((role, v))
            convs.append({'sys': sys_t, 'turns': turns, 'kind':'hermes', 'id': row.get('id')})
    return convs

def analyse(name, convs):
    kind = convs[0]['kind']
    n = len(convs)
    cross = Counter(); all_cross = Counter()
    multiturn = 0
    prev_prose_next = Counter(); prev_tc_next = Counter()
    has_toolresult = 0; ends_text = 0; ends_tc = 0
    lit_hit = 0; lit_tot = 0; full_lit = 0; full_tot = 0
    firstlen = []
    for c in convs:
        turns = c['turns']
        ai_idx = [i for i,(r,_) in enumerate(turns) if r == 'assistant']
        usr_idx = [i for i,(r,_) in enumerate(turns) if r == 'user']
        if not ai_idx: continue
        f = ai_idx[0]; prev_user = ''
        for j in range(f-1, -1, -1):
            if turns[j][0] == 'user': prev_user = turns[j][1]; break
        cross[(fmt_class(prev_user), channel(turns[f][1], kind))] += 1
        firstlen.append(len(prev_user.split()))
        for i in ai_idx:
            if i > 0 and turns[i-1][0] == 'user':
                all_cross[(fmt_class(turns[i-1][1]), channel(turns[i][1], kind))] += 1
        if len(usr_idx) >= 2 or len(ai_idx) >= 2: multiturn += 1
        for k in range(1, len(ai_idx)):
            pch = channel(turns[ai_idx[k-1]][1], kind)
            nch = channel(turns[ai_idx[k]][1], kind)
            (prev_prose_next if pch == 'text' else prev_tc_next)[nch] += 1
        if any(r == 'tool' for r,_ in turns):
            has_toolresult += 1
            lastch = channel(turns[ai_idx[-1]][1], kind)
            if lastch == 'text': ends_text += 1
            elif lastch == 'tool_call': ends_tc += 1
        for i in ai_idx:
            if channel(turns[i][1], kind) == 'text': continue
            pu = ''
            for j in range(i-1, -1, -1):
                if turns[j][0] == 'user': pu = turns[j][1]; break
            if not pu: continue
            vals = []
            for a in args_of(turns[i][1], kind): vals += scalars(a)
            if not vals: continue
            low = pu.lower()
            hit = sum(1 for v in vals if v.lower() in low)
            lit_hit += hit; lit_tot += len(vals)
            full_tot += 1; full_lit += (hit == len(vals))
            break
    return dict(name=name, n=n, cross=cross, all_cross=all_cross, multiturn=multiturn,
                prev_prose_next=prev_prose_next, prev_tc_next=prev_tc_next,
                has_toolresult=has_toolresult, ends_text=ends_text, ends_tc=ends_tc,
                lit_hit=lit_hit, lit_tot=lit_tot, full_lit=full_lit, full_tot=full_tot,
                firstlen=firstlen)

def show(R):
    print('=' * 72); print(R['name'], '  n_samples =', R['n'])
    fl = sorted(R['firstlen'])
    if fl:
        print('  first-user-turn word count: median=%d mean=%.0f p10=%d p90=%d' % (
            fl[len(fl)//2], sum(fl)/len(fl), fl[int(.1*len(fl))], fl[int(.9*len(fl))]))
    for label, cr in (('FIRST assistant turn', R['cross']), ('ALL user->assistant transitions', R['all_cross'])):
        tot = sum(cr.values()); print('  [%s] total=%d' % (label, tot))
        print('    fmt | tool_call |  mixed  |  text   |  n')
        for f in 'abcd':
            rown = sum(v for (ff,_),v in cr.items() if ff == f)
            if not rown: continue
            print('     %s  |   %.3f   |  %.3f  |  %.3f  | %d' % (
                f, cr[(f,'tool_call')]/rown, cr[(f,'mixed')]/rown, cr[(f,'text')]/rown, rown))
    print('  multiturn: %d/%d = %.3f' % (R['multiturn'], R['n'], R['multiturn']/max(R['n'],1)))
    for lab, c in (('prev-assistant = TEXT(prose)  -> next assistant', R['prev_prose_next']),
                   ('prev-assistant = TOOL_CALL    -> next assistant', R['prev_tc_next'])):
        t = sum(c.values())
        if t:
            print('  %s : tool_call %.3f | mixed %.3f | text %.3f  (n=%d)' % (
                lab, c['tool_call']/t, c['mixed']/t, c['text']/t, t))
    if R['has_toolresult']:
        print('  samples with >=1 tool result: %d ; last assistant turn = text: %.3f ; = tool_call: %.3f' % (
            R['has_toolresult'], R['ends_text']/R['has_toolresult'], R['ends_tc']/R['has_toolresult']))
    if R['lit_tot']:
        print('  arg-literal coverage: scalar-level %.3f (%d/%d) ; all-args-verbatim %.3f (%d/%d)' % (
            R['lit_hit']/R['lit_tot'], R['lit_hit'], R['lit_tot'],
            R['full_lit']/R['full_tot'], R['full_lit'], R['full_tot']))

if __name__ == '__main__':
    G = load_glaive(sorted(glob.glob('g2_*.json')))
    show(analyse('glaiveai/glaive-function-calling-v2 (default)', G))
    for nm, pat in (('func_calling','hfc_*.json'), ('func_calling_singleturn','hst_*.json'),
                    ('glaive_func_calling','hgl_*.json'),
                    ('json_mode_agentic','hja_*.json'), ('json_mode_singleturn','hjs_*.json')):
        H = load_hermes(sorted(glob.glob(pat)))
        show(analyse('hermes-function-calling-v1 :: ' + nm, H))
