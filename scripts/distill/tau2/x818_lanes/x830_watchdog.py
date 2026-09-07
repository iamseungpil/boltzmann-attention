# -*- coding: utf-8 -*-
r"""x830 — base 레인 드레인 감시자 (2026-09-08 · 사용자 승인)

왜: `t2_base_worker.sh` 는 실패해도 `|| echo FAIL` 만 하고 다음 태스크를 집는다.
    터널/엔진이 흔들리면 큐 전체가 수 분 만에 «실패 완료»로 빠져나간다
    (2026-09-07 실측: 35건 큐가 수초 만에 드레인 — CRLF 로 러너가 즉사했을 때).
    rep1·rep2 레인에는 있는 120초 가드가 base 워커에는 없다.

무엇: 도는 워커를 **건드리지 않고** 큐만 30초마다 스냅샷한다.
    한 폴 구간(30초)에 **3건 이상** 빠지면 = 정상 태스크가 그렇게 빠를 수 없다 = 고장.
    그때만 ① 워커 루프를 cmdline 확인 후 정확 PID 로 종료([[30]] pkill -f 금지)
          ② 결과가 실제로 남지 않은 id 들을 큐 앞으로 되돌린다
          ③ 마커를 남기고 자신도 끝낸다.
    ⛔정상 진행 중에는 아무것도 하지 않는다.
"""
import os,sys,time,json,subprocess,fcntl
Q="/home/woori/scratch/x768/q_cbase.txt"; LOCK=Q+".lock"
SIM="/home/woori/scratch/tau2-bench/data/simulations"
SIM2="/home/woori/iso_tau3/tau2-bench/data/simulations"
MARK="/home/woori/scratch/logs/x830_TRIPPED"
POLL=30; DROP=3
def read_q():
    try: return [l.strip() for l in open(Q) if l.strip()]
    except Exception: return []
def worker_pids():
    out=subprocess.run(["ps","-eo","pid,args"],capture_output=True,text=True).stdout
    pids=[]
    for ln in out.splitlines():
        if "t2_base_worker.sh" in ln and Q in ln and "grep" not in ln:
            pids.append((int(ln.split()[0]),ln.strip()))
    return pids
def run_pids():
    out=subprocess.run(["ps","-eo","pid,args"],capture_output=True,text=True).stdout
    return [int(l.split()[0]) for l in out.splitlines()
            if "t2_run_gated.py" in l and "bank_x806_base_nt4" in l and "grep" not in l]
def completed(t):
    for base in (SIM,SIM2):
        p=os.path.join(base,"bank_x806_base_nt4_%s"%t,"results.json")
        if os.path.exists(p):
            try:
                if (json.load(open(p)).get("simulations") or []): return True
            except Exception: pass
    return False
def log(m): print("[x830 %s] %s"%(time.strftime("%m-%d %H:%M:%S"),m),flush=True)
log("감시 시작 · 큐 %d건 · 임계 = 30초에 %d건 이상 소실"%(len(read_q()),DROP))
prev=read_q()
while True:
    time.sleep(POLL)
    cur=read_q()
    if not worker_pids():
        log("워커 없음 — 감시 종료 (큐 %d건 잔여)"%len(cur)); break
    drop=len(prev)-len(cur)
    if drop>=DROP:
        gone=[t for t in prev if t not in cur]
        log("⛔드레인 감지: 30초에 %d건 소실 %s"%(drop,gone))
        for pid,cmd in worker_pids():
            log("  워커 종료 %d :: %s"%(pid,cmd[:110]))
            try: os.kill(pid,15)
            except Exception as e: log("   실패 %s"%e)
        time.sleep(3)
        for pid in run_pids():
            log("  런 종료 %d"%pid)
            try: os.kill(pid,15)
            except Exception as e: log("   실패 %s"%e)
        time.sleep(5)
        lost=[t for t in gone if not completed(t)]
        log("  결과 없는 id %d건 복구: %s"%(len(lost),lost))
        f=open(LOCK,"w"); fcntl.flock(f,fcntl.LOCK_EX)
        now=read_q()
        open(Q,"w").write("\n".join(lost+[t for t in now if t not in lost])+"\n")
        fcntl.flock(f,fcntl.LOCK_UN); f.close()
        open(MARK,"w").write("tripped %s gone=%s restored=%s\n"%(time.strftime("%F %T"),gone,lost))
        log("복구 완료 · 큐 %d건 · 마커 %s · 감시 종료"%(len(read_q()),MARK))
        break
    if drop: log("정상 진행: %d건 완료 (큐 %d)"%(drop,len(cur)))
    prev=cur
