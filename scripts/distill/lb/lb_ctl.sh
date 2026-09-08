#!/usr/bin/env bash
# Start, stop and watch the LB lanes without pattern matching on process names.
#
#   lb_ctl.sh start <port> <conc>   start a lane; its pid goes in a pid file
#   lb_ctl.sh stop  <port>          put the in-flight task back on the queue and kill the whole
#                                   process group, so no orphan keeps running the old code
#   lb_ctl.sh restart-all           stop every lane, take the current branch, start them again
#   lb_ctl.sh status                what is running, what is queued, where each lane is
#   lb_ctl.sh watch                 run the watcher once
#   lb_ctl.sh watchdog <seconds>    run the watcher on a loop into lb_alerts.log
#   lb_ctl.sh tick <seconds>        sample every task once a minute into lb_tick.log
#
# Killing by `pgrep -f` once matched this script's own command line and took the ssh session with
# it, which orphaned two runs onto init. Everything here uses pid files and process groups.
set -u
R=${LB_REPO:-/home/woori/scratch/repo_lb}; LB="$R/scripts/distill/lb"
RUN=/home/woori/scratch/x768; LOGS=/home/woori/scratch/logs; Q="$RUN/q_lb.txt"
PY=/home/woori/venvs/seka_env/bin/python
mkdir -p "$RUN" "$LOGS"

pidfile() { echo "$RUN/lane_$1.pid"; }

start() {
  local port=$1 conc=${2:-1} pf; pf=$(pidfile "$port")
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then echo "lane $port already running (pid $(cat "$pf"))"; return 0; fi
  sed -i 's/\r$//' "$LB/lane_lb.sh"
  setsid bash "$LB/lane_lb.sh" "$port" "$conc" 4 </dev/null > "$LOGS/lane_lb_$port.log" 2>&1 &
  echo $! > "$pf"
  sleep 8; echo "lane $port started (pid $(cat "$pf")):"; tail -2 "$LOGS/lane_lb_$port.log"
}

stop() {
  local port=$1 pf pid kids task; pf=$(pidfile "$port")
  [ -f "$pf" ] || { echo "lane $port: no pid file"; return 0; }
  pid=$(cat "$pf")
  kids=$(ps -eo pid,ppid --no-headers | awk -v p="$pid" '$2==p {print $1}')
  for k in $kids; do
    task=$(tr '\0' ' ' < /proc/$k/cmdline 2>/dev/null | grep -oE 'task_ids task_[0-9]+' | awk '{print $2}')
    if [ -n "${task:-}" ]; then
      exec 9>"$Q.lock"; flock 9; { echo "$task"; cat "$Q"; } > "$Q.tmp" && mv "$Q.tmp" "$Q"; flock -u 9; exec 9>&-
      echo "requeued $task"
    fi
  done
  kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  sleep 4
  kill -0 "$pid" 2>/dev/null && { kill -KILL -"$pid" 2>/dev/null; echo "forced"; }
  for k in $kids; do kill -0 "$k" 2>/dev/null && kill -KILL "$k" 2>/dev/null; done
  rm -f "$pf"; echo "lane $port stopped"
}

case "${1:-status}" in
  start)   start "$2" "${3:-1}" ;;
  stop)    stop "$2" ;;
  restart-all)
    for pf in "$RUN"/lane_*.pid; do [ -f "$pf" ] && stop "$(basename "$pf" .pid | sed 's/lane_//')"; done
    git -C "$R" fetch -q origin lb && git -C "$R" reset -q --hard origin/lb
    echo "sha $(git -C "$R" log --oneline -1)"
    (cd "$LB" && PYTHONIOENCODING=utf-8 $PY tests/test_lb.py 2>&1 | grep -E '^FAIL|RESULT')
    start 9143 4; start 8141 1 ;;
  status)
    echo "queue $(wc -l < "$Q"): $(head -3 "$Q" | tr '\n' ' ')"
    for pf in "$RUN"/lane_*.pid; do
      [ -f "$pf" ] || continue
      p=$(basename "$pf" .pid | sed 's/lane_//'); pid=$(cat "$pf")
      if kill -0 "$pid" 2>/dev/null; then
        echo "lane $p pid $pid: $(tail -1 "$LOGS/lane_lb_$p.log")"
      else
        echo "lane $p pid $pid: DEAD"
      fi
    done
    ls -t "$LOGS"/lb_task_*_drv.log 2>/dev/null | head -2 | while read -r f; do
      printf '  %s: ' "$(basename "$f" _drv.log)"; grep -a 'Status:' "$f" | tail -1 | cut -c1-90; done ;;
  watch)   (cd "$LB" && PYTHONIOENCODING=utf-8 $PY lb_watch.py --logs "$LOGS" --minutes "${2:-0}") ;;
  watchdog)
    every=${2:-180}
    while true; do
      out=$( (cd "$LB" && PYTHONIOENCODING=utf-8 $PY lb_watch.py --logs "$LOGS") 2>&1 )
      if [ -n "$out" ] && ! echo "$out" | grep -q '^no alerts'; then
        { echo "=== $(date '+%m-%d %H:%M')"; echo "$out"; } >> "$LOGS/lb_alerts.log"
      fi
      sleep "$every"
    done ;;
  tick)
    every=${2:-60}
    echo "$$" > "$RUN/tick.pid"
    while true; do
      (cd "$LB" && PYTHONIOENCODING=utf-8 $PY lb_tick.py --logs "$LOGS") >> "$LOGS/lb_tick.log" 2>&1
      sleep "$every"
    done ;;
  *) sed -n '2,13p' "$0" ;;
esac
