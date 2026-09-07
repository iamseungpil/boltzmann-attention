# -*- coding: utf-8 -*-
r"""x831 — 레인 감독자 (2026-09-08 · 사용자 승인 "1만 붙여라")

왜: 레인의 120초 가드와 x830 드레인 감시자는 «멈추는» 데까지만 한다. 새벽 2시에
    걸리면 07시까지 클라우드 GPU 가 다섯 시간을 논다.

무엇: 레인이 사라지면 **엔진 건강부터 확인한다** — `/v1/models` id 대조([[30]] 함정:
    같은 포트를 다른 모델이 이어받는다) + 실제 5토큰 생성이 60초 안에 돌아오는지.
    건강할 때만 한 번 되살린다(레인당 최대 3회·백오프 60/300/900초).
    건강하지 않으면 **되살리지 않고 기록만 남긴다** — 가드의 취지(하네스가 고장이면
    큐를 태우지 마라)를 지킨다. 큐가 비어 정상 종료한 경우도 되살리지 않는다.
"""
import os,subprocess,time,json,urllib.request
M="Qwen/Qwen3.8-27B-FP8"
LANES=[
 {"name":"rep1","pat":"lane_rep1_153.sh","port":9143,
  "queue":"/home/woori/scratch/x768/q_crep1.txt",
  "cmd":["bash","/home/woori/scratch/lane_rep1_153.sh","9143"],
  "log":"/home/woori/scratch/logs/laneCR.log"},
 {"name":"base","pat":"t2_base_worker.sh CB","port":9141,
  "queue":"/home/woori/scratch/x768/q_cbase.txt",
  "cmd":["bash","/home/woori/scratch/t2_base_worker.sh","CB","localhost","9141",
         "/home/woori/scratch/x768/q_cbase.txt"],
  "log":"/home/woori/scratch/logs/laneCB.log"},
]
BACKOFF=[60,300,900]; MAXR=3
def log(m): print("[x831 %s] %s"%(time.strftime("%m-%d %H:%M:%S"),m),flush=True)
def alive(pat):
    out=subprocess.run(["ps","-eo","args"],capture_output=True,text=True).stdout
    return any(pat in l and "grep" not in l for l in out.splitlines())
def qlen(p):
    try: return len([l for l in open(p) if l.strip()])
    except Exception: return 0
def healthy(port):
    try:
        with urllib.request.urlopen("http://localhost:%d/v1/models"%port,timeout=20) as r:
            got=json.loads(r.read())["data"][0]["id"]
    except Exception as e:
        return False,"models 무응답 %s"%e
    if got!=M: return False,"서빙 모델 불일치 '%s'"%got
    body={"model":M,"messages":[{"role":"user","content":"say ok"}],"max_tokens":5}
    req=urllib.request.Request("http://localhost:%d/v1/chat/completions"%port,
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    try:
        t0=time.time()
        with urllib.request.urlopen(req,timeout=60) as r: json.loads(r.read())
        return True,"생성 %.1fs"%(time.time()-t0)
    except Exception as e:
        return False,"생성 실패 %s"%e
state={l["name"]:0 for l in LANES}
log("감독 시작 — %s"%", ".join("%s(:%d 큐%d)"%(l["name"],l["port"],qlen(l["queue"])) for l in LANES))
while True:
    time.sleep(60)
    if all(state[l["name"]]>=MAXR or (not alive(l["pat"]) and qlen(l["queue"])==0) for l in LANES):
        log("모든 레인이 소진이거나 재시작 상한 — 감독 종료"); break
    for L in LANES:
        n=L["name"]
        if alive(L["pat"]): continue
        q=qlen(L["queue"])
        if q==0:
            log("%s: 큐 소진 — 되살리지 않음"%n); continue
        if state[n]>=MAXR:
            continue
        ok,why=healthy(L["port"])
        if not ok:
            log("⛔%s: 죽었고 엔진도 불건강(%s) — 되살리지 않음. 큐 %d건 보존"%(n,why,q)); 
            state[n]=MAXR; continue
        d=BACKOFF[min(state[n],len(BACKOFF)-1)]
        log("%s: 죽음 · 엔진 건강(%s) · 큐 %d건 — %d초 후 재시작 (%d/%d회)"%(n,why,q,d,state[n]+1,MAXR))
        time.sleep(d)
        if alive(L["pat"]): log("%s: 대기 중 스스로 살아남 — 취소"%n); continue
        f=open(L["log"],"a")
        subprocess.Popen(L["cmd"],stdout=f,stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL,start_new_session=True,
                         cwd="/home/woori/scratch")
        state[n]+=1
        log("%s: 재시작 발사 (%d/%d)"%(n,state[n],MAXR))
        if n=="base" and not alive("x830_watchdog.py"):
            # x830 은 트립 후 스스로 끝난다 — base 를 되살렸으면 드레인 감시도 같이 되살린다.
            g=open("/home/woori/scratch/logs/x830.log","a")
            subprocess.Popen(["/home/woori/venvs/seka_env/bin/python","-u",
                              "/home/woori/scratch/x830_watchdog.py"],
                             stdout=g,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                             start_new_session=True,cwd="/home/woori/scratch")
            log("  x830 드레인 감시자도 함께 재기동")
