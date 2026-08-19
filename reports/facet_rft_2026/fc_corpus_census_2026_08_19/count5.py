# -*- coding: utf-8 -*-
import glob
from count import load_glaive, load_hermes, channel, fmt_class

def med(x):
    x=sorted(x); return x[len(x)//2] if x else -1

def run(name, convs, kind):
    notool=[]; withtool=[]
    for c in convs:
        turns=c['turns']
        has=any(r=='assistant' and channel(t,kind)!='text' for r,t in turns)
        pu=''
        for r,t in turns:
            if r=='user': pu=t; break
        (withtool if has else notool).append(len(pu.split()))
    n=len(convs)
    print('%s  n=%d | 도구호출 0건 대화 %.3f (n=%d, 첫 user 턴 단어수 median=%d) | 도구호출 있음 %.3f (median=%d)'%(
        name, n, len(notool)/n, len(notool), med(notool), len(withtool)/n, med(withtool)))

if __name__=='__main__':
    run('glaive-v2', load_glaive(sorted(glob.glob('g2_*.json'))), 'glaive')
    run('hermes/glaive_func_calling', load_hermes(sorted(glob.glob('hgl_*.json'))), 'hermes')
    run('hermes/func_calling', load_hermes(sorted(glob.glob('hfc_*.json'))), 'hermes')
