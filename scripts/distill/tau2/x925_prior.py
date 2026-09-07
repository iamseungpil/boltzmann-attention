# -*- coding: utf-8 -*-
import json,io,sys,re
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
SIG=['last 4','last four','last-4','four digits','4 digits','correct last 4']
P=r"C:\Users\승원\AppData\Local\Temp\claude\C--workspace\7fc6c1a1-f227-4592-be9c-44f0ba6cffac\scratchpad\x921_slim.jsonl"
TARGET={"18a85a4c-":"task_070","448a24bf":"task_088","7fddbff8":"task_091","c2296dc4":"task_090"}
for ln in open(P,encoding="utf-8"):
    d=json.loads(ln)
    if d["task"] not in ("task_070","task_088","task_091","task_090"): continue
    hit=None
    for i,m in enumerate(d["msgs"]):
        if m["r"]!="assistant": continue
        tl=(m["c"] or "").lower()
        s=[x for x in SIG if x in tl]
        if s:
            hit=(i,s,m["c"]); break
    if hit:
        print("== %s %s r=%s  최초 prior 신호 msg%d sig=%s"%(d["task"],d["sim"][:8],d["reward"],hit[0],hit[1]))
        c=hit[2]
        for x in SIG:
            j=c.lower().find(x)
            if j>=0:
                print("   ...%s..."%c[max(0,j-140):j+120].replace("\n"," "))
                break
        print()
