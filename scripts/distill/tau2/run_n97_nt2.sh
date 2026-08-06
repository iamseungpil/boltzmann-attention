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

# ★사이드카는 옵션이 아니다 (2026-08-06 사고). 이 드라이버에는 `T2_FB_SIDECAR`가 없었고
#   `run_smoke_split.sh`에는 있었다. 그 차이 때문에 7시간짜리 전수 런이 **우리 층이 무엇을 언제
#   말했는지 알 수 없는 상태로** 돌았다 — 같은 세션에서 022·017·035의 원인을 전부 사이드카로
#   짚어놓고서다. 비커밋 관측이라 거동 변화는 0이고(기록만), 없으면 포렌식의 절반이 원리적으로
#   불가능하다. 드라이버마다 사람이 기억해서 켜는 방식이 사고의 뿌리이므로 여기 박는다([[07]]).
export T2_FB_SIDECAR="$LOGD/fb_n97_gpu${G}_$DATE.jsonl"          # GPU별 파일(두 드라이버 동시 append 회피)
export T2_FB_SIDECAR_TEXT=1

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
  # An empty id list is not "run nothing" — tau2 reads it as "run every task in the domain",
  # which is a full paid sweep nobody asked for (2026-08-06: an unexpanded glob below sent
  # gpu1 back through all 97 tasks after its block had already been persisted).
  [ -n "$2" ] || { say "empty task list for $1 — refusing to launch"; return; }
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
# nullglob: with an empty reserve the pattern would otherwise survive as the literal
# `batch_*`, get claimed, and be launched with no ids at all.
shopt -s nullglob
for f in "$PLAN"/reserve/batch_*; do
  b=$(basename "$f")
  mkdir "$PLAN/claims/$b" 2>/dev/null || continue      # someone else took it
  run_block "$b" "$(cat "$f")"
done
say "★ driver finished — no unclaimed batches left"
