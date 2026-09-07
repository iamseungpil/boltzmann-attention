#!/usr/bin/env bash
# 3레인 대시보드 — 읽기 전용
set -u
now=$(date '+%m-%d %H:%M:%S'); echo "===== $now ====="
row(){ # name port queue lanepat
  local NM="$1" PORT="$2" Q="$3" PAT="$4"
  local qn cur el
  qn=$(wc -l < "$Q" 2>/dev/null || echo "?")
  local line; line=$(ps -eo etime,args | grep -E "$PAT" | grep -v grep | grep -oE "(save_to|task_ids) [a-z_0-9]+|^ *[0-9:-]+" | tr '\n' ' ')
  cur=$(ps -eo etime,args | grep -E "$PAT" | grep -v grep | grep -oE "save_to [a-z_0-9]+" | head -1 | cut -d' ' -f2)
  el=$(ps -eo etime,args | grep -E "$PAT" | grep -v grep | grep -oE "^ *[0-9:-]+" | head -1 | tr -d ' ')
  local run wait pre
  local m; m=$(curl -s -m 8 "http://localhost:$PORT/metrics" 2>/dev/null)
  run=$(echo "$m" | grep -E '^vllm:num_requests_running\{' | sed 's/.*} //')
  wait=$(echo "$m" | grep -E '^vllm:num_requests_waiting\{' | sed 's/.*} //')
  pre=$(echo "$m" | grep -E '^vllm:num_preemptions_total\{' | sed 's/.*} //')
  printf "%-6s :%s  현재=%-22s 경과=%-9s 큐=%-3s  running=%-5s wait=%-5s preempt=%s\n" \
    "$NM" "$PORT" "${cur:-없음}" "${el:-–}" "$qn" "${run:-?}" "${wait:-?}" "${pre:-?}"
}
row "rep2"  8141 /home/woori/scratch/x768/q_rep2.txt  "lane_rep2_153|rep2_task"
row "base"  9141 /home/woori/scratch/x768/q_cbase.txt "t2_base_worker.sh CB|bank_x806_base_nt4"
row "rep1"  9143 /home/woori/scratch/x768/q_crep1.txt "lane_rep1_153|rep1_task"
echo
echo "-- 사내 GPU --"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/   /'
echo "-- 완료 누적 --"
for a in rep1 rep2; do
  d=/home/woori/scratch/repo_$a/reports/facet_rft_2026/sim_results
  n=$(ls -1 "$d"/${a}_task_*.results.json.gz 2>/dev/null | wc -l)
  echo "   $a: $n 태스크"
  for f in $(ls -1t "$d"/${a}_task_*.results.json.gz 2>/dev/null | head -6); do
    python3 - "$f" <<'PY'
import sys,gzip,json,os
try:
    j=json.load(gzip.open(sys.argv[1],"rt"))
    rw=[(s.get("reward_info") or {}).get("reward") for s in (j.get("simulations") or [])]
    rw=[x for x in rw if x is not None]
    print("      %-18s %d/%d  %s"%(os.path.basename(sys.argv[1]).split(".")[0],
          sum(1 for v in rw if v==1.0),len(rw),rw))
except Exception as e: print("      (읽기 실패) %s"%e)
PY
  done
done
echo "-- base 완료 누적 --"
ls -d /home/woori/scratch/tau2-bench/data/simulations/bank_x806_base_nt4_task_* /home/woori/iso_tau3/tau2-bench/data/simulations/bank_x806_base_nt4_task_* 2>/dev/null | wc -l | sed 's/^/   run 디렉터리 /'
echo "-- rep2 레버 발화 --"
for m in "RECOMMEND-OFFER" "handover:" ; do
  c=$(grep -h "$m" /home/woori/scratch/logs/rep2_task_*.log /home/woori/scratch/logs/rep2_task_*_drv.log 2>/dev/null | wc -l)
  printf "   %-18s %s회\n" "$m" "$c"
done
