#!/usr/bin/env python
"""tco_cost — 기존 sim 디렉토리에 *비용 열* 부착 (TCO_TABLE_DESIGN #6·새 GPU 0).
실측: duration(latency)·tool_calls·turns. 추정: tokens(누적컨텍스트 chars/4)·$/req.
$/req = frontier: tok×API단가 / on-prem: GPU$_hr ÷ throughput(conc/avg_dur).
Run: PY tco_cost.py <simdir> [--mode frontier|onprem] [--gpu_hr 0.30] [--conc 8] [--in 2.0 --out 8.0]
"""
import argparse, json, os, statistics

def est_tokens(sim):
    msgs=sim.get("messages") or []
    # 누적-컨텍스트: 각 assistant 턴이 그 시점까지 전체를 입력으로 전송
    prompt_chars=0; out_chars=0; ctx=0
    for m in msgs:
        c=m.get("content"); s=c if isinstance(c,str) else (json.dumps(c) if c else "")
        # tool_calls도 컨텍스트
        tcs=json.dumps(m.get("tool_calls") or "")
        sz=len(s)+len(tcs)
        if m.get("role")=="assistant":
            prompt_chars+=ctx        # 이 턴 입력=이전 누적
            out_chars+=sz            # 이 턴 출력
        ctx+=sz
    return prompt_chars/4, out_chars/4  # ~4 chars/token

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("simdir"); ap.add_argument("--mode",default="onprem")
    ap.add_argument("--gpu_hr",type=float,default=0.30); ap.add_argument("--conc",type=int,default=8)
    ap.add_argument("--pin",type=float,default=2.0); ap.add_argument("--pout",type=float,default=8.0)  # $/1M
    a=ap.parse_args()
    sims=json.load(open(os.path.join(a.simdir,"results.json"))).get("simulations",[])
    dur=[s.get("duration") for s in sims if s.get("duration")]
    tc=[sum(len(m.get("tool_calls") or []) for m in (s.get("messages") or [])) for s in sims]
    tr=[sum(1 for m in (s.get("messages") or []) if m.get("role")=="assistant") for s in sims]
    pin=[]; pout=[]
    for s in sims:
        i,o=est_tokens(s); pin.append(i); pout.append(o)
    n=len(sims); adur=statistics.mean(dur) if dur else 0
    ai=statistics.mean(pin); ao=statistics.mean(pout)
    api_cost = ai*a.pin/1e6 + ao*a.pout/1e6
    thr = a.conc/adur if adur else 0          # tasks/sec
    onprem_cost = (a.gpu_hr/3600)/thr if thr else 0
    print(f"== {a.simdir} (n={n}) ==")
    print(f"  latency/req   = {adur:7.1f} s   (실측)")
    print(f"  tool-roundtrips= {statistics.mean(tc):5.1f}     turns={statistics.mean(tr):.1f} (실측)")
    print(f"  est tokens/req = in {ai:8.0f} / out {ao:7.0f} (누적컨텍스트 추정)")
    print(f"  $ frontier-API = ${api_cost:.4f}/req  (gpt-4.1 ${a.pin}/{a.pout} per 1M·추정)")
    print(f"  $ on-prem      = ${onprem_cost:.5f}/req  (A6000 ${a.gpu_hr}/hr ÷ {thr*3600:.0f} req/hr@conc{a.conc})")

if __name__=="__main__": main()
