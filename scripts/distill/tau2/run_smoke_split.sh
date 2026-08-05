#!/bin/bash
# Two task lists, one per GPU, with the stack as go_stack.sh defines it.
#
# The fixed six-task smoke could not express "these four on one card, those three on the
# other", which is what the targeted round needs: the four tasks whose causes were read
# one at a time, and the three that stand for the closure cluster the declaration is
# meant to cover in one go.
#
# Launch discipline (learned twice today): the driver must survive the ssh session that
# starts it. Start it with `nohup setsid … & disown` and verify the process tree after,
# or the runners are orphaned and nothing is persisted.
#
#   usage: run_smoke_split.sh <tag_suffix> <gpu0_tasks> <gpu1_tasks>
set -u
SUF="${1:?tag suffix}"; T0="${2:?gpu0 tasks}"; T1="${3:?gpu1 tasks}"
R=/home/woori/workspace_common/boltzmann-attention-pi
LOGD=/home/woori/scratch/logs; mkdir -p "$LOGD"
TRACE=/home/woori/scratch/x30run/smoke_$SUF.jsonl

[ -f /home/woori/.openai_key ] && . /home/woori/.openai_key
[ -f /home/woori/.openrouter_key ] && . /home/woori/.openrouter_key
. $R/scripts/distill/tau2/go_stack.sh
export T2_SG_ISOLATE_TRACE="$TRACE"          # 관측 전용(거동 변화 0)

echo "== 이 런이 가졌다고 믿는 플래그 =="
env | grep -E '^T2_(PROCEDURE|UNINSTRUCTABLE|QUOTE_PIN|MATCH_COUNT|KB_DOCS_DIR)=' | sort

say(){ echo "[smoke $(date +%H:%M:%S)] $*"; }

persist(){
  local tag="$1" src="$GO_TAU2/data/simulations/$1/results.json"
  [ -f "$src" ] || { say "no results.json for $tag"; return; }
  gzip -c "$src" > "$R/reports/facet_rft_2026/sim_results/$tag.results.json.gz"
  gzip -c "$LOGD/$tag.log" > "$R/reports/facet_rft_2026/sim_results/$tag.log.gz" 2>/dev/null
  ( cd "$R" && git pull -q --rebase origin facet-rft-2026 \
    && git add -f "reports/facet_rft_2026/sim_results/$tag."* \
    && git commit -q -m "Persist $tag (targeted smoke)" \
    && git push -q origin facet-rft-2026 ) && say "persisted $tag" || say "PERSIST FAILED $tag"
}

run_block(){  # $1 gpu, $2 port, $3 tasks
  local tag="bank_smk_gpu$1_$SUF"
  rm -rf "$GO_TAU2/data/simulations/$tag"
  say "start gpu$1: $3"
  t2_launch "$tag" "$2" "$3" 1 > "$LOGD/$tag.log" 2>&1
  say "done gpu$1 (exit $?)"
  persist "$tag"
}

run_block 0 8140 "$T0" &
P0=$!
sleep 45                       # git push 충돌 회피용 시차
run_block 1 8141 "$T1" &
P1=$!
wait $P0 $P1

L="$LOGD/bank_smk_gpu0_$SUF.log $LOGD/bank_smk_gpu1_$SUF.log"
echo
echo "== ① 라이브 발화 =="
for m in T2_PROCEDURE T2_UNINSTRUCTABLE; do
  printf '%-22s %s\n' "$m" "$(cat $L 2>/dev/null | grep -c "\[$m\]")"
done
printf '%-22s %s\n' "구 가드(quote-ground)" "$(cat $L 2>/dev/null | grep -c 'quote-ground 불성립')"
printf '%-22s %s\n' "미검증 행 재질의 안내" "$(cat $L 2>/dev/null | grep -c 'unverified row')"
echo
echo "== ② 부작용 =="
printf '%-22s %s\n' "replay ValueError" "$(cat $L 2>/dev/null | grep -c 'ValueError')"
printf '%-22s %s\n' "400/schema" "$(cat $L 2>/dev/null | grep -ciE 'BadRequest|invalid_request')"
echo
echo "== ③ 결과([D]·n=1) =="
for g in 0 1; do
  f="$GO_TAU2/data/simulations/bank_smk_gpu${g}_$SUF/results.json"
  [ -f "$f" ] && python3 -c "
import json
d=json.load(open('$f'))
for s in d.get('simulations',[]):
    ri=s.get('reward_info') or {}
    print('  %-10s reward=%s basis=%s term=%s' % (s['task_id'],ri.get('reward'),ri.get('reward_basis'),s.get('termination_reason')))
"
done
