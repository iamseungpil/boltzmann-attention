#!/usr/bin/env bash
# chain.sh <기다릴PID> <실행할 명령...>  — 앞 워커가 정상 종료한 뒤 다음 워커를 띄운다.
set -u
W="$1"; shift
while kill -0 "$W" 2>/dev/null; do sleep 20; done
echo "[chain] pid $W 종료 확인 -> $*"
exec "$@"
