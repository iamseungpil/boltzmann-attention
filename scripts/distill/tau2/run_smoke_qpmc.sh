#!/bin/bash
# Smoke for the two prescriptions registered on 2026-08-05 — do they fire in a live run?
#
# Both were verified and then left off `go_stack.sh`, so neither has ever run live inside a
# full simulation. Unit tests and replays say they work; this arc produced three separate
# cases of "unit test passes, lever silent in production" ([[30]]), so the question here is
# firing, not score. Six tasks, one trial each.
#
#   task_022   the quote-pin target: the old guard dropped `txn_ba8b473f295d` and burned it
#              in N97B. x30 replay on this HEAD: rate survives, discrepant 10/10, 77 of 77.
#   task_019   quote-pin fires but does NOT recover it (C289 — the block is correct there).
#              Pre-registered as a continued failure; a pass would mean something changed.
#   task_018   the extra-write axis (t0 filed one dispute too many, t1 passed) — watch for
#              a repeat, not for the levers.
#   task_017   passed in N97B t0. Δspurious watch: if it now fails, a lever broke something.
#   task_012   answered from memory after four fruitless searches — where the match-count
#              line should make "nothing matched all these words" visible.
#   task_051   the [READ-FIRST] chain, the busiest KB_search task in the arc.
#
# Judgement is firing + side effects, not pass ([D] at n=1). Expected, pre-registered:
# 022 pass, 019 fail, 017 pass, others unchanged.
#
#   usage:  run_smoke_qpmc.sh [tag_suffix]
set -u
DATE=$(date +%Y%m%d)
SUF="${1:-$DATE}"
R=/home/woori/workspace_common/boltzmann-attention-pi
LOGD=/home/woori/scratch/logs; mkdir -p "$LOGD"
TRACE=/home/woori/scratch/x30run/smoke_qpmc_$SUF.jsonl

[ -f /home/woori/.openai_key ] && . /home/woori/.openai_key        # alltools retrieval
[ -f /home/woori/.openrouter_key ] && . /home/woori/.openrouter_key
. $R/scripts/distill/tau2/go_stack.sh                              # [[19]] the stack itself

# Observation only, no behaviour change: the quote-pin route logs solely when it drops a
# rate, so "no log" is ambiguous between never-ran and ran-and-passed. The trace records
# what the sub declared, which separates the two (C282 ③ had to establish this by hand).
export T2_SG_ISOLATE_TRACE="$TRACE"

echo "== 이 런이 가졌다고 믿는 플래그 =="
env | grep -E '^T2_(QUOTE_PIN|MATCH_COUNT|CALLABLE_HINT|PIN_READ)=' | sort

say(){ echo "[smoke $(date +%H:%M:%S)] $*"; }

persist(){  # $1 = tag
  local tag="$1" src="$GO_TAU2/data/simulations/$1/results.json"
  [ -f "$src" ] || { say "no results.json for $tag"; return; }
  gzip -c "$src" > "$R/reports/facet_rft_2026/sim_results/$tag.results.json.gz"
  gzip -c "$LOGD/$tag.log" > "$R/reports/facet_rft_2026/sim_results/$tag.log.gz" 2>/dev/null
  ( cd "$R" && git pull -q --rebase origin facet-rft-2026 \
    && git add -f "reports/facet_rft_2026/sim_results/$tag."* \
    && git commit -q -m "Persist $tag (quote-pin + match-count firing smoke)" \
    && git push -q origin facet-rft-2026 ) && say "persisted $tag" || say "PERSIST FAILED $tag"
}

run_block(){  # $1 = gpu index, $2 = port, $3 = tasks
  local tag="bank_qpmc_gpu$1_$SUF"
  rm -rf "$GO_TAU2/data/simulations/$tag"
  say "start gpu$1: $3"
  t2_launch "$tag" "$2" "$3" 1 > "$LOGD/$tag.log" 2>&1
  say "done gpu$1 (exit $?)"
  persist "$tag"
}

run_block 0 8140 "task_022,task_019,task_018" &
P0=$!
sleep 60                      # stagger: the two blocks must not reach `git push` together
run_block 1 8141 "task_017,task_012,task_051" &
P1=$!
wait $P0 $P1

L0="$LOGD/bank_qpmc_gpu0_$SUF.log"; L1="$LOGD/bank_qpmc_gpu1_$SUF.log"
echo
echo "== ① 라이브 발화 =="
printf '%-34s %s\n' "구 가드(quote-ground 불성립)" "$(cat $L0 $L1 | grep -c 'quote-ground 불성립')"
echo "   ↑ 0이어야 한다. 0이 아니면 T2_QUOTE_PIN이 적용되지 않은 것이다(A2 미선언 경로)."
printf '%-34s %s\n' "quote-pin 판정 로그(드롭 시에만)" "$(cat $L0 $L1 | grep -c 'quote-pin')"
printf '%-34s %s\n' "서브 핀 선언(trace)" "$(grep -o 'exclusion_pin_kind' $TRACE 2>/dev/null | wc -l)"
printf '%-34s %s\n' "isolate 호출(trace 행)" "$(wc -l < $TRACE 2>/dev/null || echo 0)"
echo
echo "== ② MATCH_COUNT 부착 (궤적에 실재하는가) =="
for t in gpu0 gpu1; do
  f="$R/reports/facet_rft_2026/sim_results/bank_qpmc_${t}_$SUF.results.json.gz"
  [ -f "$f" ] && printf '%-34s %s\n' "matches: 주석 ($t)" "$(zcat $f | grep -o 'matches: ' | wc -l)"
done
echo
echo "== ③ 부작용 감시 =="
printf '%-34s %s\n' "replay ValueError" "$(cat $L0 $L1 | grep -c 'ValueError')"
printf '%-34s %s\n' "400/schema 오류" "$(cat $L0 $L1 | grep -ciE 'BadRequest|invalid_request')"
echo
echo "== ④ 결과(참고·n=1이라 [D]) =="
grep -hoE '"reward": [0-9.]+' $L0 $L1 | sort | uniq -c
