# -*- coding: utf-8 -*-
import json, re, glob
from collections import Counter
from count import load_glaive, channel

def hist(name, convs, kind):
    idx = Counter(); first_reply_ch = Counter()
    for c in convs:
        turns = c['turns']
        for i,(r,t) in enumerate(turns):
            if r=='assistant':
                ch = channel(t, kind)
                if ch!='text': idx[i]+=1
        ai=[i for i,(r,_) in enumerate(turns) if r=='assistant']
        if ai: first_reply_ch[(ai[0], channel(turns[ai[0]][1],kind))]+=1
    tot=sum(idx.values())
    print(name,'tool_call turn-index hist (top):', [(k, round(v/tot,3)) for k,v in idx.most_common(8)])
    print('  first-assistant index x channel:', dict(first_reply_ch.most_common(8)))

def ntools(files, field, kind):
    ns=[]
    for f in files:
        for r in json.load(open(f,encoding='utf-8'))['rows']:
            row=r['row']
            if field=='tools':
                try: ns.append(len(json.loads(row['tools'])))
                except Exception:
                    ns.append(row['tools'].count('"type": "function"'))
            else:
                ns.append(row['system'].count('"name"'))
    ns.sort()
    print(kind,'tools/sample: median=%d p90=%d max=%d mean=%.2f n=%d'%(ns[len(ns)//2], ns[int(.9*len(ns))], ns[-1], sum(ns)/len(ns), len(ns)))

if __name__=='__main__':
    G=load_glaive(sorted(glob.glob('g2_*.json'))); hist('glaive-v2', G, 'glaive')
    ntools(sorted(glob.glob('hgl_*.json')),'tools','hermes/glaive_func_calling')
    ntools(sorted(glob.glob('hfc_*.json')),'tools','hermes/func_calling')
