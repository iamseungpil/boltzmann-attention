#!/bin/bash
# Autonomous 5h batch (AUTONOMOUS_EXPERIMENT_PLAN_2026_06_16.md). Waits for current Sstep
# sweeps (Q1: p8013 7B+14B, p8014 32B), then runs Q2 Snover, Q3 SCv — each as a 2-GPU pair
# (7B+14B on GPU0:8013, 32B-Int8 on GPU1:8014), job-sets sequential. Final aggregate.
# Fire-and-forget; survives ssh drop (setsid). Inference-only (no training).
set -u
S=/home/woori/scratch
REPO=/home/woori/workspace_common/boltzmann-attention-pi
PY=/home/woori/venvs/seka_env/bin/python
LOG=$S/ma_overnight_batch.log
exec > $LOG 2>&1; set -x; date
cd $REPO

wait_done(){  # $1=port; wait for MA_SCALE_DONE in that port's log (cap ~110min)
  for i in $(seq 1 330); do
    grep -q MA_SCALE_DONE $S/ma_eval_scale_p$1.log 2>/dev/null && { echo "p$1 DONE"; return 0; }
    sleep 20
  done
  echo "p$1 TIMEOUT"; return 1
}

run_pair(){  # $1=arms  $2=suffix  — 7B+14B on GPU0:8013, 32B-Int8 on GPU1:8014, wait both
  git pull --ff-only || true
  setsid bash scripts/distill/ma/ma_eval_scale.sh "Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct" "$1" "$2" 0 8013 </dev/null >/dev/null 2>&1 &
  sleep 8
  setsid bash scripts/distill/ma/ma_eval_scale.sh "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8" "$1" "$2" 1 8014 </dev/null >/dev/null 2>&1 &
  sleep 8
  wait_done 8013; wait_done 8014
}

echo "=== WAIT Q1 (current Sstep sweeps) ==="; wait_done 8013; wait_done 8014; date
echo "=== Q2 Snover (Sstep verify OFF — ablation) ==="; run_pair "Snover" "_snover"; date
echo "=== Q3 SCv (self-consistency N=5 majority) ==="; run_pair "SCv" "_scv"; date

echo "=== FINAL AGGREGATE -> ma_overnight_summary.log ==="
$PY - > $S/ma_overnight_summary.log 2>&1 <<'PYEOF'
import json, os, collections
sufs = ['_floor', '_sstep', '_snover', '_scv']
for tag, name in [('Qwen2_5_7B_Instruct', '7B'), ('Qwen2_5_14B_Instruct', '14B'), ('Qwen2_5_32B_Instruct_GPTQ_Int8', '32B-Int8')]:
    print(f'==== {name} ====')
    for suf in sufs:
        f = f'/home/woori/scratch/ma_eval_{tag}{suf}.jsonl'
        if not os.path.exists(f):
            continue
        agg = collections.defaultdict(lambda: [0, 0, 0, 0])
        for l in open(f):
            r = json.loads(l); ic = r.get('item_correct', []); a = agg[r['arm']]
            a[0] += len(ic); a[1] += sum(1 for x in ic if x)
            a[2] += r.get('prompt_tokens', 0) + r.get('completion_tokens', 0); a[3] += r.get('n_calls', 0)
        for q, a in agg.items():
            it, ok, tok, cl = a
            print(f'  [{suf[1:]:6s}] {q:7s} acc={ok/it if it else 0:.3f} ({ok}/{it}) tok/case={tok//29} calls/case={cl/29:.1f}')
PYEOF
cat $S/ma_overnight_summary.log
echo MA_OVERNIGHT_DONE; date
