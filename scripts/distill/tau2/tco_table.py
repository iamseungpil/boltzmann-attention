#!/usr/bin/env python
"""tco_table — 아침 TCO 조립(2026-06-23): compliance + 비용($/latency) 전체 표 + fleet 투영.
compliance=compliance.json(bench/full pass^1). 비용=duration(실측)+token(추정·누적컨텍스트)+$.
fleet=32B g15 per-task pass ∪ gpt-4.1 escalate → blended compliance/$.
Run: cd tau2-bench && PYTHONPATH=src:<T2> PY tco_table.py
"""
import json, os, statistics, collections
SIM = "data/simulations"
PIN, POUT, GPU_HR, CONC = 2.0, 8.0, 0.30, 8     # gpt-4.1 $/1M, A6000 $/hr, concurrency

def _load(d):
    p = os.path.join(SIM, d, "results.json")
    return json.load(open(p)).get("simulations", []) if os.path.exists(p) else None

def _comp(d):
    p = os.path.join(SIM, d, "compliance.json")
    if not os.path.exists(p): return (None, None)
    j = json.load(open(p)); return (j.get("bench", {}).get("pass^1"), j.get("full", {}).get("pass^1"))

def _tokens(sim):
    msgs = sim.get("messages") or []; pin = out = ctx = 0
    for m in msgs:
        c = m.get("content"); s = c if isinstance(c, str) else (json.dumps(c) if c else "")
        sz = len(s) + len(json.dumps(m.get("tool_calls") or ""))
        if m.get("role") == "assistant": pin += ctx; out += sz
        ctx += sz
    return pin/4, out/4

def _cost(d, mode):
    sims = _load(d)
    if not sims: return None
    dur = statistics.mean([s["duration"] for s in sims if s.get("duration")])
    ins = statistics.mean([_tokens(s)[0] for s in sims]); outs = statistics.mean([_tokens(s)[1] for s in sims])
    api = ins*PIN/1e6 + outs*POUT/1e6
    thr = CONC/dur if dur else 0
    onp = (GPU_HR/3600)/thr if thr else 0
    return dict(dur=dur, api=api, onprem=onp, ntok=ins+outs, n=len(sims))

def row(label, d, mode):
    b, f = _comp(d); c = _cost(d, mode)
    if c is None: return f"{label:34s} <none>"
    cost = c["api"] if mode == "api" else c["onprem"]
    return (f"{label:34s} comp(bench/full)={b}/{f}  lat={c['dur']:6.1f}s  "
            f"${cost:.4f}/req({'API' if mode=='api' else 'on-prem'})  tok~{c['ntok']:.0f} n={c['n']}")

print("="*92); print("TCO TABLE — compliance + cost (latency real · token/$ estimated)"); print("="*92)
print("# frontier baseline")
print(row("gpt-4.1 (frontier-API)", "retail_gpt41_nogate", "api"))
print("# on-prem deploy (g15) × scale  [nt3 우선·없으면 nt1]")
for tag in ("n7b","n14b","n32int8"):
    d3, d1 = f"ours_{tag}_g15_retail_t3", f"ours_{tag}_g15_retail"
    d = d3 if os.path.exists(os.path.join(SIM,d3,"results.json")) else d1
    print(row(f"{tag} g15 on-prem", d, "onprem"))
print("# floor × scale (참조)")
for tag in ("n7b","n14b","n32int8"):
    print(row(f"{tag} floor on-prem", f"on_{tag}_floor_retail", "onprem"))

# ===== fleet 투영 (32B g15 ∪ gpt-4.1 escalate) =====
def perfpass(d):
    sims = _load(d)
    if not sims: return None
    by = collections.defaultdict(list)
    for s in sims: by[str(s.get("task_id"))].append((s.get("reward_info") or {}).get("reward",0) or 0)
    return {t: max(v) >= 1 for t, v in by.items()}   # pass-any (모델 능력)
g32 = perfpass("ours_n32int8_g15_retail_t3") or perfpass("ours_n32int8_g15_retail")
fr = perfpass("retail_gpt41_nogate")
c32 = _cost("ours_n32int8_g15_retail_t3","onprem") or _cost("ours_n32int8_g15_retail","onprem")
cfr = _cost("retail_gpt41_nogate","api")
if g32 and fr and c32 and cfr:
    common = set(g32) & set(fr)
    handled = [t for t in common if g32[t]]          # 32B가 처리
    escalate = [t for t in common if not g32[t]]     # frontier로
    blended_pass = sum(1 for t in common if g32[t] or fr[t]) / len(common)
    p = len(handled)/len(common)
    blended_cost = p*c32["onprem"] + (1-p)*cfr["api"]
    print("\n"+"="*92); print("FLEET 투영 (쉬운=32B-on-prem · 어려운=gpt-4.1 escalate · oracle routing=상한)"); print("="*92)
    print(f"  32B 처리율 p={p:.2f}  escalate={1-p:.2f}")
    print(f"  blended compliance={blended_pass:.3f}  (vs 순수 gpt-4.1 {sum(fr[t] for t in common)/len(common):.3f})")
    print(f"  blended $/req=${blended_cost:.4f}  (vs 순수 gpt-4.1 ${cfr['api']:.4f} = {cfr['api']/blended_cost:.1f}× 비쌈)")
    print("  ※ oracle routing(32B 성공여부 사전지)=상한·실 router는 decidable 난이도신호 필요(§10)")
print("\n[done] tco_table")
