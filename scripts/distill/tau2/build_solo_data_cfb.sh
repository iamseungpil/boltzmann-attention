#!/bin/bash
# solo + cfbsynth(P2b/P4 추상) 데이터. CFB 직접 대신 CFB-참조 추상 synth(structure-only) 추가.
# 근거: solo가 CFB 제외→P2b/P4 미학습→retail 연속홉 fetch 붕괴(order_id 날조). cfbsynth가 그 추상구조 공급.
# = 기존 solo(synth_part + soptb_part) + cfbsynth_part. 1-변수 추가(cfbsynth) → solo_sts와 비교.
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
MA=$R/scripts/distill/ma; CV=$R/scripts/distill/taskbench
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build
N=${1:-6000}
LOG=$S/build_solo_data_cfb.log
exec > $LOG 2>&1; set -x; date

# 1. cfbsynth 추상 P2b/P4 생성 (익명·randomized id·리스트선택)
$PY $MA/synth_fetch_nativefc.py --out $FC/cfbsynth_native.jsonl --n $N --seed 0 --max_hops 3 --max_list 5
# 2. QC + 포맷 정규화 (--no_alias: 이미 per-traj 익명이므로 실명유지)
$PY $CV/fc_build_sft.py --inputs $FC/cfbsynth_native.jsonl --out $FC/cfbsynth_part.jsonl --no_alias --seed 42
# 3. merge: 기존 solo(synth_part + soptb_part·alias) + cfbsynth_part
cat $FC/synth_part.jsonl $FC/soptb_part.jsonl $FC/cfbsynth_part.jsonl > $FC/sft_solo_cfb.jsonl
echo "=== SOLO+CFB DATA ==="; wc -l $FC/synth_part.jsonl $FC/soptb_part.jsonl $FC/cfbsynth_part.jsonl $FC/sft_solo_cfb.jsonl
$PY -c "
import json
from collections import Counter
c=Counter()
for l in open('$FC/sft_solo_cfb.jsonl'):
    d=json.loads(l); c[d.get('_meta',{}).get('bench','?')]+=1
print('bench mix', dict(c))
"
echo SOLO_CFB_DATA_DONE; date
