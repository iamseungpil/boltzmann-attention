#!/usr/bin/env bash
# Start, stop and watch the LB lanes without pattern matching on process names.
#
#   lb_ctl.sh start <port> <conc> [k]   start worker k of a lane (two conc-2 workers per A100 keep 4 sims in flight)
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
  # start <port> <conc> [k]: worker k of this engine (a task has only 4 sims, so one conc-4 worker
  # spends most of its time on a straggler; two conc-2 workers keep 4 sims in flight - handoff §6)
  local port=$1 conc=${2:-1} k=${3:-} name pf; name="$port${k:+_$k}"; pf=$(pidfile "$name")
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then echo "lane $name already running (pid $(cat "$pf"))"; return 0; fi
  sed -i 's/\r$//' "$LB/lane_lb.sh"
  setsid bash "$LB/lane_lb.sh" "$port" "$conc" 4 </dev/null > "$LOGS/lane_lb_$name.log" 2>&1 &
  echo $! > "$pf"
  sleep 8; echo "lane $name started (pid $(cat "$pf")):"; tail -2 "$LOGS/lane_lb_$name.log"
}

stop() {
  # stop <port>: every worker of that engine (lane_<port>.pid and lane_<port>_<k>.pid)
  local port=$1 pf
  for pf in "$RUN/lane_$port.pid" "$RUN"/lane_"$port"_*.pid; do [ -f "$pf" ] && stop_one "$(basename "$pf" .pid | sed 's/lane_//')"; done
}

stop_one() {
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
  start)   start "$2" "${3:-1}" "${4:-}" ;;
  probe)   # probe <port> <task...>: nt=1, tag probe_<task>, its own queue; never touches q_lb.txt
    port=$2; shift 2; pf=$(pidfile "$port")
    [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null && { echo "lane $port already running"; exit 1; }
    printf '%s
' "$@" > "$RUN/q_probe.txt"
    sed -i 's/$//' "$LB/lane_lb.sh"
    LB_QUEUE="$RUN/q_probe.txt" setsid bash "$LB/lane_lb.sh" "$port" 1 1 probe </dev/null > "$LOGS/lane_lb_$port.log" 2>&1 &
    echo $! > "$pf"; sleep 8; echo "probe lane $port started (pid $(cat "$pf")):"; tail -2 "$LOGS/lane_lb_$port.log" ;;
  stop)    stop "$2" ;;
  probe-stop) Q="$RUN/q_probe.txt"; stop "$2" ;;      # same, but the in-flight task goes back on the probe queue
  restart-all)
    for pf in "$RUN"/lane_*.pid; do [ -f "$pf" ] && stop "$(basename "$pf" .pid | sed 's/lane_//')"; done
    git -C "$R" fetch -q origin lb && git -C "$R" reset -q --hard origin/lb
    echo "sha $(git -C "$R" log --oneline -1)"
    (cd "$LB" && PYTHONIOENCODING=utf-8 $PY tests/test_lb.py 2>&1 | grep -E '^FAIL|RESULT')
    start 9141 2 1; start 9141 2 2; start 9143 2 1; start 9143 2 2; start 8141 1 ;;
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
  tick)    # tick <seconds> [prefix]: one line per minute per task into lb_tick.log
    every=${2:-60}; prefix=${3:-lb}
    echo "$$" > "$RUN/tick.pid"
    while true; do
      (cd "$LB" && PYTHONIOENCODING=utf-8 $PY lb_tick.py --logs "$LOGS" --prefix "$prefix") >> "$LOGS/lb_tick.log" 2>&1
      sleep "$every"
    done ;;
  *) sed -n '2,13p' "$0" ;;
esac
