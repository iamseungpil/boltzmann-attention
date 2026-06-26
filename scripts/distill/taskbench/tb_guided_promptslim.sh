#!/bin/bash
# guided 프롬프트-절감 실험: 마스크가 집행을 보장할 때 도구 목록을 얼마나 빼도 되나.
# arms (MM sub500, dpo2+guided, held-out=의미정보를 프롬프트에 의존하는 최악 케이스):
#   full = 기존 tool_desc 그대로 (통제; 서빙 중인 tb_dpo2_mm_guided 재사용)
#   slim1 = desc 제거 (id+input/output-type+parameters)
#   slim2 = desc+parameters 제거 (id+input/output-type만)
# log: /home/woori/scratch/tb_promptslim.log, sentinel PROMPTSLIM_DONE
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
PORT=8000
exec > $S/tb_promptslim.log 2>&1
set -x

# 0. variant dirs + 프롬프트 크기 측정
$IP - <<'EOF'
import json, shutil, os
TB = "/home/woori/scratch/JARVIS_tb/taskbench"
src = f"{TB}/data_multimedia_sub500"
td = json.load(open(f"{TB}/data_multimedia/tool_desc.json"))
def slim(keep):
    return {"nodes": [{k: v for k, v in t.items() if k in keep} for t in td["nodes"]]}
KEEP1 = {"id", "input-type", "output-type", "parameters"}
KEEP2 = {"id", "input-type", "output-type"}
for name, keep in [("slim1", KEEP1), ("slim2", KEEP2)]:
    dst = f"{TB}/data_multimedia_sub500_{name}"
    os.makedirs(dst, exist_ok=True)
    for f in ["user_requests.json", "graph_desc.json"]:
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    json.dump(slim(keep), open(f"{dst}/tool_desc.json", "w"))
full_len = sum(len(json.dumps(t)) for t in td["nodes"])
s1 = sum(len(json.dumps({k: v for k, v in t.items() if k in KEEP1})) for t in td["nodes"])
s2 = sum(len(json.dumps({k: v for k, v in t.items() if k in KEEP2})) for t in td["nodes"])
print(f"[promptsize chars] full={full_len} slim1={s1} ({100*s1/full_len:.0f}%) slim2={s2} ({100*s2/full_len:.0f}%)")
EOF

# 1. 서버 확인 (tb_dpo2_mm_guided가 떠 있어야 함)
curl -s localhost:$PORT/v1/models | grep -q tb_dpo2_mm_guided || { echo SERVE_MISSING; exit 1; }

# 2. full-통제: 기존 sub500을 guided로 (full-domain은 57.22지만 sub500 직접 통제 필요)
for ARM in full slim1 slim2; do
  case $ARM in
    full) DDIR=data_multimedia_sub500;;
    *) DDIR=data_multimedia_sub500_${ARM};;
  esac
  TAG=tb_dpo2g_${ARM}
  rm -f $TB/$DDIR/predictions/${TAG}.json
  (cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_mm_schema.json \
    $IP inference.py --data_dir $DDIR --api_addr localhost --api_port $PORT \
    --api_key dummy --llm tb_dpo2_mm_guided --multiworker 8 --dependency_type resource)
  mv $TB/$DDIR/predictions/tb_dpo2_mm_guided.json $TB/$DDIR/predictions/${TAG}.json
  $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $TB/$DDIR/predictions/${TAG}.json \
    --dst $TB/data_multimedia_sub500_eval_${TAG} --llm ${TAG}
done

echo "PROMPTSLIM_DONE $(date)"
