#!/bin/bash
# Full banking sweep — 97 tasks x nt2, two servers, both sides finishing together.
#
# The split comes from x65_gpu_split.py: each server gets an LPT-balanced block of the
# tasks whose cost has been measured, then both drain a shared reserve of the tasks whose
# cost has not. Claiming a reserve batch is `mkdir` — atomic, so the two drivers never
# take the same batch, and the side that finishes its block first simply takes more.
#
#   usage:  run_n97_nt2.sh <gpu-index 0|1> <port> [plan-dir] [tag-date]
#
# Results are persisted after every block, not at the end: simulation output lives in a
# gitignored scratch clone, so a run that dies at hour 11 with nothing pushed has lost
# everything ([[30]] 2026-06-26).
set -u
G="$1"; PORT="$2"; PLAN="${3:-/home/woori/scratch/n97plan}"; DATE="${4:-20260804}"
R=/home/woori/workspace_common/boltzmann-attention-pi
LOGD=/home/woori/scratch/logs
mkdir -p "$LOGD" "$PLAN/claims"

[ -f /home/woori/.openai_key ] && . /home/woori/.openai_key      # alltools retrieval
[ -f /home/woori/.openrouter_key ] && . /home/woori/.openrouter_key
. $R/scripts/distill/tau2/go_stack.sh                            # [[19]] every lever on

say(){ echo "[n97 gpu$G $(date +%H:%M:%S)] $*"; }

persist(){  # $1 = tag. Serialised across the two drivers — git is not concurrent-safe.
  local tag="$1" src="$GO_TAU2/data/simulations/$1/results.json"
  [ -f "$src" ] || { say "no results.json for $tag — nothing to persist"; return; }
  while ! mkdir "$PLAN/claims/.gitlock" 2>/dev/null; do sleep 10; done
  gzip -c "$src" > "$R/reports/facet_rft_2026/sim_results/$tag.results.json.gz"
  gzip -c "$LOGD/$tag.log" > "$R/reports/facet_rft_2026/sim_results/$tag.log.gz" 2>/dev/null
  ( cd "$R" \
    && git pull -q --rebase origin facet-rft-2026 \
    && git add -f "reports/facet_rft_2026/sim_results/$tag."* \
    && git commit -q -m "Persist $tag (banking 97 x nt2 sweep)" \
    && git push -q origin facet-rft-2026 ) && say "persisted $tag" || say "PERSIST FAILED $tag"
  rmdir "$PLAN/claims/.gitlock"
}

run_block(){  # $1 = tag suffix, $2 = comma-separated task ids
  local tag="bank_n97_gpu${G}_$1_$DATE"
  say "start $1 ($(echo "$2" | tr ',' '\n' | wc -l) tasks)"
  rm -rf "$GO_TAU2/data/simulations/$tag"
  # t2_launch, not a copy of its flags — go_stack.sh is the single source of truth for
  # the stack, and a second flag list is how a run silently drifts off it ([[07]]).
  t2_launch "$tag" "$PORT" "$2" 2 > "$LOGD/$tag.log" 2>&1
  say "done $1 (exit $?)"
  persist "$tag"
}

run_block main "$(cat "$PLAN/gpu${G}.tasks")"

# Drain the reserve. Whoever is free claims next; the loop ends when nothing is left.
for f in "$PLAN"/reserve/batch_*; do
  b=$(basename "$f")
  mkdir "$PLAN/claims/$b" 2>/dev/null || continue      # someone else took it
  run_block "$b" "$(cat "$f")"
done
say "★ driver finished — no unclaimed batches left"
