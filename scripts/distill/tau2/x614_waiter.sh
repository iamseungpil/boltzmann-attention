#!/usr/bin/env bash
# x614 대기 발사기 — x612(드라이버 PID 490311) 종료를 기다렸다 건다.
# ⛔PID 로 기다린다(패턴 매칭 금지·[[30]]).
WAIT_PID=490311
LOG=/home/woori/scratch/logs/x614_waiter.log
echo "[waiter $(date +%H:%M:%S)] x612(PID $WAIT_PID) 종료 대기 시작" >> $LOG
while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
echo "[waiter $(date +%H:%M:%S)] x612 종료 감지 · flush 대기 90s" >> $LOG
sleep 90
echo "[waiter $(date +%H:%M:%S)] x614 발사" >> $LOG
bash /home/woori/run_x614_32b_usersim52_bank.sh >> /home/woori/scratch/logs/x614_launch.log 2>&1
echo "[waiter $(date +%H:%M:%S)] x614 종료" >> $LOG
