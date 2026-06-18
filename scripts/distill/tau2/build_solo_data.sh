#!/bin/bash
# 단독 통합 LoRA 데이터 빌드 (SOP+TaskBench+Synth content-op·CFB 제외). GPU 불요·학습 전 검증용.
# 단독 전환(2026-06-18·C0 근거): facet3 단독은 τ² 맥락서 resolve 0회 → 한 LoRA에 flow+threading+content-op.
# synth는 --no_alias(resolve_selection 실명 유지·τ² 노출명과 일치) / SOP·TB는 alias(grounding 강제).
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$R/scripts/distill; MA=$DIST/ma; CV=$DIST/taskbench
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build; OUT=$S/depth/c8/facet3
TB=${1:-$FC/tb_all_v4.jsonl}   # taskbench native (sft_v7이 쓴 버전·인자로 교체 가능)
LOG=$S/build_solo_data.log
exec > $LOG 2>&1; set -x; date

# 1. synth content-op native (bench=synth meta·anchor_id 제외)
$PY $MA/synth_to_nativefc.py --out $OUT/route_native_bench.jsonl --n_per_op 860 --N 5,10,20 --diverse --seed 0
# 2a. synth part = no_alias(resolve_selection 실명)
$PY $CV/fc_build_sft.py --inputs $OUT/route_native_bench.jsonl --out $FC/synth_part.jsonl --no_alias --seed 42
# 2b. sop+tb part = alias(grounding)
$PY $CV/fc_build_sft.py --inputs $FC/sop_rand.jsonl $TB --out $FC/soptb_part.jsonl --max_per_bench 7000 --seed 42
# 3. merge
cat $FC/synth_part.jsonl $FC/soptb_part.jsonl > $FC/sft_solo_sts.jsonl
echo "=== SOLO DATA ==="; wc -l $FC/synth_part.jsonl $FC/soptb_part.jsonl $FC/sft_solo_sts.jsonl
$PY -c "
import json
from collections import Counter
c=Counter(); ntools=Counter()
for l in open('$FC/sft_solo_sts.jsonl'):
    d=json.loads(l); c[d['_meta'].get('bench','?')]+=1
    ntools['has_resolve']+= any(t['function']['name']=='resolve_selection' for t in d.get('tools',[]))
print('bench mix', dict(c)); print('rows with resolve_selection tool', ntools['has_resolve'])
"
echo SOLO_DATA_DONE; date
