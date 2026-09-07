# -*- coding: utf-8 -*-
"""x926 — 제안 닫힌 술어 검증: '인자 소비자(consumer) 도구를 unlock/call 한 적 있나' 로 바꾸면
   오발(070/088/091)이 죽고 정발(031/053 등)이 사는가."""
import json,io,sys,collections
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
P=r"C:\Users\승원\AppData\Local\Temp\claude\C--workspace\7fc6c1a1-f227-4592-be9c-44f0ba6cffac\scratchpad\x921_slim.jsonl"
CONSUMER="file_credit_card_transaction_dispute"   # 인자 card_last_4_digits 의 소비자(접두)
va={tuple(x[:4]) for x in json.load(open("x923_va.json",encoding="utf-8"))}
sims={}
for ln in open(P,encoding="utf-8"):
    d=json.loads(ln); sims[d["sim"]]=d
rows=[]
for tag,task,sim,rew in sorted(va,key=lambda z:(z[1])):
    d=sims[sim]; touched=False
    for m in d["msgs"]:
        for tc in (m.get("tc") or []):
            blob=(str(tc.get("name") or "")+" "+str(tc.get("args") or "")).lower()
            if CONSUMER in blob: touched=True
    rows.append((task,sim[:8],rew,touched))
print("VA 발화 sim %d개 — 인자 소비자(%s) 접촉 여부"%(len(rows),CONSUMER))
for t,s,r,x in rows: print("  %-9s %s r=%-4s consumer_touched=%s"%(t,s,r,x))
print()
print("touched=True :", sum(1 for r in rows if r[3]), " / touched=False(=제안 술어면 침묵):", sum(1 for r in rows if not r[3]))
print("reward=1.0 중 touched:", [(r[0],r[1]) for r in rows if r[2]==1.0 and r[3]])
print("reward=1.0 중 미접촉(잃음):", [(r[0],r[1]) for r in rows if r[2]==1.0 and not r[3]])
