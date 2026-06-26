#!/bin/bash
# 아이디어2 (data-consistency): SOP/TB를 --no_alias(실명)로 재빌드 → base named-tool 호출과 일관.
# 가설: 익명 func_NNNN가 base tool-calling과 충돌→망각. 실명이면 resolve_selection이 additive(덮어쓰기X).
# synth_part(이미 no_alias·resolve_selection 실명)는 재사용. = sft_solo_cons.jsonl.
# light(alias)와 1-변수 차이(데이터 alias만) → 망각곡선 직접 비교. ★학습벤치(SOP/TB)만·tau2 금지(11-transfer).
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
CV=$R/scripts/distill/taskbench
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
TB=${1:-$FC/tb_all_v4.jsonl}
LOG=$S/build_solo_data_cons.log
exec > $LOG 2>&1; set -x; date

# sop+tb = no_alias(실명 유지) — light의 alias와 유일 차이
$PY $CV/fc_build_sft.py --inputs $FC/sop_rand.jsonl $TB --out $FC/soptb_part_noalias.jsonl \
  --max_per_bench 7000 --no_alias --seed 42
# synth_part(no_alias·기존) 재사용 → merge
cat $FC/synth_part.jsonl $FC/soptb_part_noalias.jsonl > $FC/sft_solo_cons.jsonl
echo "=== CONS DATA ==="; wc -l $FC/synth_part.jsonl $FC/soptb_part_noalias.jsonl $FC/sft_solo_cons.jsonl
$PY -c "
import json
from collections import Counter
c=Counter(); names=Counter()
for l in open('$FC/sft_solo_cons.jsonl'):
    d=json.loads(l); c[d['_meta'].get('bench','?')]+=1
    for t in d.get('tools',[]): names[t['function']['name']]+=1
print('bench mix', dict(c))
print('func_NNNN tools remaining (should be 0):', sum(v for k,v in names.items() if k.startswith('func_')))
print('top real tool names:', [k for k,_ in names.most_common(10)])
"
echo CONS_DATA_DONE; date
