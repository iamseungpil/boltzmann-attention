# -*- coding: utf-8 -*-
"""x611b — t7391_reg12(retail·12 sim) 전체에서 **실행된 write** 앞의 확인 상태를 기계적으로 센다.
닫힌 술어만: ①직전 user 메시지의 CONFIRM_RE 매치 토큰 ②그 user 메시지 바로 앞에
assistant **텍스트** 발화(도구호출 아님)가 있었는가(=행동 세부를 나열할 자리가 있었는가).
"""
import gzip, json, sys
sys.path.insert(0, "C:/workspace/ba-frft/scripts/distill/tau2")
sys.stdout.reconfigure(encoding='utf-8')
from gate_interpreter import CONFIRM_RE
import t2_forensic as F

base = 'C:/workspace/ba-frft/reports/facet_rft_2026/sim_results/'
d = json.load(gzip.open(base + 't7391_reg12.results.json.gz', 'rt', encoding='utf-8'))
mut = F.mutating_tools('retail')
rows = []
for s in sorted(d['simulations'], key=lambda x: int(x['task_id'])):
    ms = s['messages']
    for i, m in enumerate(ms):
        if m.get('role') != 'assistant':
            continue
        for tc in (m.get('tool_calls') or []):
            if tc.get('name') not in mut:
                continue
            nxt = ms[i + 1] if i + 1 < len(ms) else {}
            executed = nxt.get('role') == 'tool' and not str(nxt.get('content', '')).startswith('Error')
            j = max([k for k in range(i) if ms[k].get('role') == 'user' and ms[k].get('content')] or [-1])
            lu = ms[j].get('content') if j >= 0 else ''
            mt = CONFIRM_RE.search(lu or '')
            # 그 user 발화 바로 앞 assistant 가 **텍스트**였는가
            prev_txt = (j - 1 >= 0 and ms[j-1].get('role') == 'assistant'
                        and bool(ms[j-1].get('content')) and not ms[j-1].get('tool_calls'))
            rows.append((s['task_id'], i, tc.get('name'), executed, j,
                         mt.group(0) if mt else None, prev_txt, (lu or '')[:60].replace('\n', ' ')))
print('%4s %4s %-34s %-4s %-5s %-8s %-6s %s' % ('task','msg','tool','exec','lastU','token','prevTxt','user 발화 앞 60자'))
for r in rows:
    print('%4s %4d %-34s %-4s %-5d %-8s %-6s %s' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
ex = [r for r in rows if r[3]]
print('\n실행된 write %d 건 · 그중 CONFIRM 토큰 매치 %d · 그중 직전 assistant 텍스트 부재 %d'
      % (len(ex), sum(1 for r in ex if r[5]), sum(1 for r in ex if r[5] and not r[6])))
